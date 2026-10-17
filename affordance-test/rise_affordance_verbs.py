"""
RISE over affordance verbs, comparable to the direct per-part labeling.

region_information_gain.py / rise_attribution.py measure how much each region supports the model's own
free-text description (D0) of the object -- a different quantity from batch_final_labeling.py's per-part
forced-choice affordance label. This script makes the two comparable by changing the measured target:

  * Regions: the same disjoint SAM+SLIC parts (background-filtered) the labeling pipeline uses, read from
    json/affordance_justification/<object>_affordance_justification.json, so part IDs/masks match 1:1.
  * Target: the forced-choice AGD20K question asked about the WHOLE object (no highlight), scored over the full
    37-option vocabulary. Every RISE trial yields a log-probability for every verb, so one run produces a
    region x verb importance matrix with no extra trials.
  * Trials: each part is independently kept or gray-filled with p=0.5 (background always visible); scored
    log p(verb | trial image) is a softmax over the vocabulary, so a part needed to recognize the object shifts
    probability mass between verbs (often toward "none") instead of lowering every verb at once.
  * Importance(part, verb) = mean logp(verb | part visible) - mean logp(verb | part occluded), plus Welch's t
    and the same difference in probability space. The "RISE label" for a part is the verb it supports most.

Usage:
    python rise_affordance_verbs.py object [object ...]      # RISE_M env var overrides trials (default 300)
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, JSON_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, downscale, score_options,
)
from region_information_gain import OCCLUSION_FILL

OBJECTS = sys.argv[1:]

RISE_DIR = os.path.join(JSON_DIR, "rise_affordance_verbs")
os.makedirs(RISE_DIR, exist_ok=True)
VOCAB = "agd20k"
M = int(os.environ.get("RISE_M", 300))
KEEP_PROB = 0.5
RNG_SEED = 0


def build_object_question(options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    return (
        "The image shows an object. Which single affordance is most characteristic of this object as a whole? "
        f"Choose exactly one of: {listed}. If none of these apply, answer \"none\". Answer with the option only."
    )


def occlude_parts(image: Image.Image, masks: list, keep: np.ndarray) -> Image.Image:
    arr = np.array(image).copy()
    for i, m in enumerate(masks):
        if not keep[i]:
            arr[m] = OCCLUSION_FILL
    return Image.fromarray(arr)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def rise_importance(keep_matrix: np.ndarray, trial_lp: np.ndarray) -> dict:
    """RISE estimator. keep_matrix (M, n_parts) bool, True = part visible; trial_lp (M, n_verbs) log-probs.
    Returns per-(part, verb) arrays: bits = mean logp visible - mean logp occluded (in bits), t = Welch t,
    dprob = mean prob visible - mean prob occluded."""
    trial_p = np.exp(trial_lp)
    n, v = keep_matrix.shape[1], trial_lp.shape[1]
    bits, t_stat, dprob = np.zeros((n, v)), np.zeros((n, v)), np.zeros((n, v))
    for i in range(n):
        vis, occ = keep_matrix[:, i], ~keep_matrix[:, i]
        imp_nats = trial_lp[vis].mean(0) - trial_lp[occ].mean(0)
        se = np.sqrt(trial_lp[vis].var(0, ddof=1) / vis.sum() + trial_lp[occ].var(0, ddof=1) / occ.sum())
        bits[i] = imp_nats / np.log(2)
        t_stat[i] = np.where(se > 1e-9, imp_nats / np.maximum(se, 1e-9), 0.0)
        dprob[i] = trial_p[vis].mean(0) - trial_p[occ].mean(0)
    return {"bits": bits, "t": t_stat, "dprob": dprob}


def process_object(model, processor, option_ids, device, object_name: str) -> dict:
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{object_name}_affordance_justification.json")) as f:
        data = json.load(f)
    parts = data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    prompt_text = build_prompt_text(processor, build_object_question(options, glosses))

    full_lp = log_softmax(score_options(model, processor, downscale(image), prompt_text, option_ids, device))
    print(f"  full image top verbs: "
          f"{[(options[i], round(float(np.exp(full_lp[i])), 2)) for i in np.argsort(-full_lp)[:4]]}")

    rng = np.random.default_rng(RNG_SEED)
    keep_matrix = rng.random((M, len(parts))) < KEEP_PROB
    trial_lp = np.zeros((M, len(options)))
    for t in range(M):
        trial = downscale(occlude_parts(image, masks, keep_matrix[t]))
        trial_lp[t] = log_softmax(score_options(model, processor, trial, prompt_text, option_ids, device))
        if t == 0 or (t + 1) % 50 == 0:
            print(f"    trial {t + 1}/{M}")

    imp = rise_importance(keep_matrix, trial_lp)
    regions_out = []
    for i, part in enumerate(parts):
        regions_out.append({
            "region_id": part["region_id"], "area": part["area"],
            "forced_choice_label": part["vocabularies"][VOCAB]["label"],
            "importance_bits": dict(zip(options, imp["bits"][i].tolist())),
            "t_stat": dict(zip(options, imp["t"][i].tolist())),
            "delta_prob": dict(zip(options, imp["dprob"][i].tolist())),
        })

    agree_bits = np.mean([max(r["importance_bits"], key=r["importance_bits"].get) == r["forced_choice_label"]
                          for r in regions_out])
    agree_dp = np.mean([max(r["delta_prob"], key=r["delta_prob"].get) == r["forced_choice_label"]
                        for r in regions_out])
    print(f"\n  {'region':>26s} {'forced-choice':>14s} {'RISE (bits)':>14s} {'RISE (dProb)':>14s}  FC verb rank/importance")
    for r in regions_out:
        by_bits = sorted(r["importance_bits"], key=r["importance_bits"].get, reverse=True)
        by_dp = sorted(r["delta_prob"], key=r["delta_prob"].get, reverse=True)
        fc = r["forced_choice_label"]
        print(f"  {r['region_id']:>26s} {fc:>14s} {by_bits[0]:>14s} {by_dp[0]:>14s}  "
              f"#{by_bits.index(fc) + 1} ({r['importance_bits'][fc]:+.2f} bits)")
    print(f"  agreement with forced-choice label: argmax bits={agree_bits:.2f}  argmax dProb={agree_dp:.2f}")

    return {
        "object": object_name, "model": MODEL_NAME, "M": M, "keep_prob": KEEP_PROB, "rng_seed": RNG_SEED,
        "question": build_object_question(options, glosses),
        "full_image_probs": dict(zip(options, np.exp(full_lp).tolist())),
        "agreement_argmax_bits": float(agree_bits), "agreement_argmax_dprob": float(agree_dp),
        "regions": regions_out,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    options = VOCABULARIES[VOCAB]["options"]
    option_ids = [processor.tokenizer(o + "<|im_end|>", add_special_tokens=False).input_ids for o in options]

    for name in OBJECTS:
        print(f"\n=== {name} ===")
        data = process_object(model, processor, option_ids, device, name)
        out_path = os.path.join(RISE_DIR, f"{name}_rise_affordance_verbs.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
