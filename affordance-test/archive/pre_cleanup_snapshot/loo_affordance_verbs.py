"""
Leave-one-out counterpart to rise_affordance_verbs.py, plus the direct method's full per-part distribution.

RISE hides ~half the parts at once, so it scores many odd partial views. Here each part is hidden ALONE while
every other part stays intact, so every scored image is close to the intact object -- a direct test of whether
RISE's results (e.g. shape_16's "sit on") come from those partial views. Same whole-object 37-verb question as
rise_affordance_verbs.py; importance(part, verb) = logp(verb | intact) - logp(verb | part hidden), in bits, and
the same difference in probability space. Deterministic, n_parts + 1 scored images per object.

It also recomputes the direct forced-choice method's full distribution over the 37 verbs for every part (same
numbered-overlay image and per-part question as batch_final_labeling.py) so the comparison script can use
rank correlation against the whole distribution instead of only the top-1 label. The recomputed top-1 is checked
against the stored label from json/affordance_justification.

Usage:
    python -u loo_affordance_verbs.py object [object ...]
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# region_information_gain.py parses MAX_REGIONS from sys.argv[2] at import time; argv is consumed first.
OBJECTS = sys.argv[1:]
_argv, sys.argv = sys.argv, sys.argv[:1]
from affordance_labeling import (  # noqa: E402
    AFFORDANCE_JUSTIFICATION_DIR, JSON_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, downscale, score_options,
)
from rise_affordance_verbs import build_object_question, log_softmax, occlude_parts  # noqa: E402
from comparative_affordance_labeling import numbered_overlay  # noqa: E402
from comparative_perpart_labeling import build_per_part_question  # noqa: E402
sys.argv = _argv

LOO_DIR = os.path.join(JSON_DIR, "loo_affordance_verbs")
os.makedirs(LOO_DIR, exist_ok=True)
VOCAB = "agd20k"


def process_object(model, processor, option_ids, device, object_name: str) -> dict:
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{object_name}_affordance_justification.json")) as f:
        data = json.load(f)
    parts = data["regions"]
    n = len(parts)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]

    obj_prompt = build_prompt_text(processor, build_object_question(options, glosses))
    full_lp = log_softmax(score_options(model, processor, downscale(image), obj_prompt, option_ids, device))
    print(f"  intact image top verbs: "
          f"{[(options[i], round(float(np.exp(full_lp[i])), 2)) for i in np.argsort(-full_lp)[:4]]}")

    loo_lp = np.zeros((n, len(options)))
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        loo_lp[i] = log_softmax(score_options(
            model, processor, downscale(occlude_parts(image, masks, keep)), obj_prompt, option_ids, device))
        print(f"    hid part {i + 1}/{n}")

    overlay = downscale(numbered_overlay(image, masks))
    fc_lp = np.zeros((n, len(options)))
    for i in range(n):
        prompt = build_prompt_text(processor, build_per_part_question(i + 1, n, options, glosses))
        fc_lp[i] = log_softmax(score_options(model, processor, overlay, prompt, option_ids, device))
        print(f"    direct-method distribution {i + 1}/{n}")

    regions_out = []
    n_match = 0
    for i, part in enumerate(parts):
        stored = part["vocabularies"][VOCAB]["label"]
        recomputed = options[int(np.argmax(fc_lp[i]))]
        n_match += recomputed == stored
        regions_out.append({
            "region_id": part["region_id"], "area": part["area"], "forced_choice_label": stored,
            "fc_probs": dict(zip(options, np.exp(fc_lp[i]).tolist())),
            "importance_bits": dict(zip(options, ((full_lp - loo_lp[i]) / np.log(2)).tolist())),
            "delta_prob": dict(zip(options, (np.exp(full_lp) - np.exp(loo_lp[i])).tolist())),
        })
    print(f"  recomputed direct top-1 matches stored label on {n_match}/{n} parts")

    return {
        "object": object_name, "model": MODEL_NAME, "method": "leave_one_out",
        "question": build_object_question(options, glosses),
        "full_image_probs": dict(zip(options, np.exp(full_lp).tolist())),
        "fc_top1_matches_stored": f"{n_match}/{n}",
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
        out_path = os.path.join(LOO_DIR, f"{name}_loo_affordance_verbs.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
