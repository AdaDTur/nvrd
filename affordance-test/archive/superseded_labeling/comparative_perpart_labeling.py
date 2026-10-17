"""
Hybrid probe: fixes the degenerate-repetition failure found in comparative_affordance_labeling.py's
all-parts-in-one-generation design (e.g. chair generated the identical line "Part N: sit on" for all
16 parts -- once greedy decoding commits to line 1, copying it is the path of least resistance for a
long structured list) while keeping the one thing that design got right for some objects: showing all
parts marked and numbered at once, so the model has real cross-part context to notice differences.

Design: the SAME numbered/colored overlay image (all parts visible at once) is reused for every part,
but each part gets its own independent forced-choice scoring call asking specifically about that one
numbered part ("focusing on part K... does it differ from the other marked parts?"). No call ever sees
another call's answer, so there's no autoregressive line-copying to fall into -- and going back to
per-part calls lets us restore clean forced-choice scoring (a full probability distribution per part)
instead of the free-text list generation and parsing the all-at-once version required.

Requires <object>_affordance_justification.json (isolated single-part labels, for comparison) and
<object>_comparative_labels.json (all-at-once comparative labels, for comparison) --
run affordance_labeling.py/batch_affordance_justification.py and comparative_affordance_labeling.py/
batch_comparative_labeling.py first.

Usage:
    python comparative_perpart_labeling.py [object_name]   # default: chair
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
    AFFORDANCE_JUSTIFICATION_DIR, COMPARATIVE_LABELS_DIR, COMPARATIVE_PERPART_LABELS_DIR, MODEL_NAME,
    VOCABULARIES, build_prompt_text, downscale, score_options, softmax,
)
from comparative_affordance_labeling import numbered_overlay

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "chair"
VOCAB = "agd20k"


def build_per_part_question(part_idx: int, n_parts: int, options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        f"outlined and tinted a different color. Focusing specifically on part {part_idx}: consider "
        "whether its affordance differs from the other numbered parts, then choose the single "
        f"affordance part {part_idx} offers from this list: {listed}. If part {part_idx} offers none "
        'of these, answer "none". Answer with the option only.'
    )


def top1_label(fc_logprobs: dict) -> str:
    return max(fc_logprobs, key=fc_logprobs.get)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{OBJECT_NAME}_affordance_justification.json")) as f:
        data = json.load(f)
    with open(os.path.join(COMPARATIVE_LABELS_DIR, f"{OBJECT_NAME}_comparative_labels.json")) as f:
        comparative = json.load(f)
    parts = data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))

    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    option_ids = None  # built once processor is available

    print(f"Object: {OBJECT_NAME} | {len(parts)} parts, each probed independently with full context")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    option_ids = [processor.tokenizer(o + "<|im_end|>", add_special_tokens=False).input_ids for o in options]

    results = []
    print(f"\n{'part':>6s} {'region_id':>22s} {'isolated':>12s} {'all-at-once':>14s} {'per-part+context':>18s}")
    for i, part in enumerate(parts, start=1):
        prompt_text = build_prompt_text(processor, build_per_part_question(i, len(parts), options, glosses))
        fc = score_options(model, processor, overlay, prompt_text, option_ids, device)
        probs = softmax(fc)
        new_label = top1_label(dict(zip(options, fc.tolist())))
        old_isolated = part["vocabularies"][VOCAB]["label"]
        old_comparative = comparative["regions"][i - 1]["comparative_label"]
        print(f"{i:>6d} {part['region_id']:>22s} {old_isolated:>12s} {old_comparative:>14s} "
              f"{new_label:>15s} ({probs.max():.2f})")
        results.append({
            "region_id": part["region_id"], "area": part["area"],
            "isolated_label": old_isolated, "comparative_label": old_comparative,
            "perpart_context_label": new_label, "perpart_context_prob": float(probs.max()),
        })

    n_isolated = len({r["isolated_label"] for r in results})
    n_comparative = len({r["comparative_label"] for r in results})
    n_new = len({r["perpart_context_label"] for r in results})
    print(f"\ndistinct labels: isolated={n_isolated}  all-at-once-comparative={n_comparative}  "
          f"per-part+context={n_new}  (of {len(parts)} parts)")

    out = {"object": OBJECT_NAME, "model": MODEL_NAME, "regions": results}
    out_path = os.path.join(COMPARATIVE_PERPART_LABELS_DIR, f"{OBJECT_NAME}_comparative_perpart_labels.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
