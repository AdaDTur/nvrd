"""
Does the VLM actually use the region highlight in affordance_labeling.py, or does it answer at the
object level? For each labelled region, compare three views of the same forced-choice question:

    highlight : affordance_labeling.py's view (region lit + outlined, rest darkened), recomputed here
    full      : the same question on the un-highlighted image (object-level control; no region info)
    crop      : a close-up of just the part (bbox crop, everything outside the mask gray)

All images are downscaled to MAX_SIDE px first (the shared GPUs are often too full for full-resolution
vision-tower activations), so highlight numbers here can differ slightly from the full-resolution
ones in <object>_affordance_labels.json; the three conditions are always compared at equal resolution.

Metrics per vocabulary (total-variation distance, 0 = identical distributions, 1 = disjoint):
    highlight_vs_full : mean TV(highlight_r, full)   -- ~0 means the highlight is ignored
    spread_*          : mean pairwise TV between different regions' distributions -- ~0 means every
                        region gets the same answer regardless of which part is shown
Also reports a contrastive label, argmax of log p(option | highlight) - log p(option | full), which
divides out the object-level answer and keeps only what the highlighted part adds.

Requires <object>_affordance_labels.json (run affordance_labeling.py first).

Usage:
    python highlight_ablation.py [object_name]   # default: shape_16
"""
import os
import sys
import json
import itertools

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    MODEL_NAME, REGION_SOURCE, VOCABULARIES, END_TOKEN, build_prompt_text, build_question, downscale,
    highlight_region, labels_path, score_options, softmax,
)
from region_information_gain import resolve_image_path

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
CROP_PAD_FRAC = 0.15
CROP_MIN_SIDE = 224
OUTSIDE_FILL = 127


def build_crop_question(options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    return (
        "The image shows a close-up of one part of an object (the rest of the object is masked out in gray). "
        f"Which single affordance does this part offer? Choose exactly one of: {listed}. "
        'If this part offers none of these, answer "none". Answer with the option only.'
    )


def masked_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(image).copy()
    arr[~mask] = OUTSIDE_FILL
    ys, xs = np.nonzero(mask)
    pad = int(CROP_PAD_FRAC * max(ys.max() - ys.min(), xs.max() - xs.min()))
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    crop = Image.fromarray(arr[y0:y1, x0:x1])
    scale = CROP_MIN_SIDE / min(crop.size)
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
    return crop


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def mean_pairwise_tv(dists: list) -> float:
    pairs = list(itertools.combinations(dists, 2))
    return float(np.mean([tv(a, b) for a, b in pairs])) if pairs else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(labels_path(OBJECT_NAME)) as f:
        labels = json.load(f)
    labelled = labels["regions"]
    image = Image.open(resolve_image_path(OBJECT_NAME)).convert("RGB")

    print(f"Object: {OBJECT_NAME} | {len(labelled)} labelled regions")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    masks = {r["region_id"]: mask_utils.decode(r["segmentation"]).astype(bool) for r in labelled}
    small_full = downscale(image)
    highlights = {rid: downscale(highlight_region(image, m)) for rid, m in masks.items()}
    crops = {rid: downscale(masked_crop(image, m)) for rid, m in masks.items()}
    out = {"object": OBJECT_NAME, "vocabularies": {}}
    for name, vocab in VOCABULARIES.items():
        options = vocab["options"]
        option_ids = [processor.tokenizer(o + END_TOKEN, add_special_tokens=False).input_ids for o in options]
        highlight_prompt = build_prompt_text(processor, build_question(options, vocab["glosses"]))
        crop_prompt = build_prompt_text(processor, build_crop_question(options, vocab["glosses"]))

        full_lp = score_options(model, processor, small_full, highlight_prompt, option_ids, device)
        full_p = softmax(full_lp)
        hl_lp = {
            rid: score_options(model, processor, img, highlight_prompt, option_ids, device)
            for rid, img in highlights.items()
        }
        crop_lp = {
            rid: score_options(model, processor, img, crop_prompt, option_ids, device) for rid, img in crops.items()
        }
        hl_p = {k: softmax(v) for k, v in hl_lp.items()}
        crop_p = {k: softmax(v) for k, v in crop_lp.items()}
        contrast_p = {k: softmax(hl_lp[k] - full_lp) for k in hl_lp}

        def top(p):
            i = int(np.argmax(p))
            return f"{options[i]} ({p[i]:.2f})"

        print(f"\n=== {name} === object-level (no highlight) answer: {top(full_p)}")
        print(f"{'region':>18s} {'area':>7s} | {'highlight':>20s} | {'crop':>20s} | {'contrastive':>20s} | TV(hl,full)")
        rows = []
        for r in labelled:
            rid = r["region_id"]
            row = {
                "region_id": rid, "area": r["area"],
                "highlight": top(hl_p[rid]), "crop": top(crop_p[rid]), "contrastive": top(contrast_p[rid]),
                "tv_highlight_vs_full": tv(hl_p[rid], full_p),
            }
            rows.append(row)
            print(f"{rid:>18s} {r['area']:>7d} | {row['highlight']:>20s} | {row['crop']:>20s} | "
                  f"{row['contrastive']:>20s} | {row['tv_highlight_vs_full']:.3f}")
        summary = {
            "highlight_vs_full": float(np.mean([r["tv_highlight_vs_full"] for r in rows])),
            "spread_highlight": mean_pairwise_tv(list(hl_p.values())),
            "spread_crop": mean_pairwise_tv(list(crop_p.values())),
            "spread_contrastive": mean_pairwise_tv(list(contrast_p.values())),
        }
        print("summary (TV distance):", {k: round(v, 3) for k, v in summary.items()})
        out["vocabularies"][name] = {
            "full_top": top(full_p), "rows": rows, "summary": summary,
            "full_logprobs": dict(zip(options, full_lp.tolist())),
            "crop_logprobs": {k: dict(zip(options, v.tolist())) for k, v in crop_lp.items()},
        }

    out_path = os.path.join(HERE, f"{OBJECT_NAME}_{REGION_SOURCE}_highlight_ablation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
