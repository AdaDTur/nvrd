"""
Batch driver for comparative_perpart_labeling.py, loading Qwen2-VL once across several objects.

Usage:
    python batch_comparative_perpart.py object1 object2 ...
"""
import os
import sys
import json

import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECTS = sys.argv[1:]

# region_information_gain.py parses MAX_REGIONS from sys.argv[2] at import time, which collides with
# this script's own multi-object argv -- argv is already consumed into OBJECTS above, so it's safe to
# blank it out just for these imports.
_argv, sys.argv = sys.argv, sys.argv[:1]
from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, COMPARATIVE_LABELS_DIR, COMPARATIVE_PERPART_LABELS_DIR, MODEL_NAME,
    VOCABULARIES, build_prompt_text, downscale, score_options, softmax,
)
from comparative_affordance_labeling import numbered_overlay
from comparative_perpart_labeling import VOCAB, build_per_part_question, top1_label
sys.argv = _argv


def process_object(model, processor, option_ids, device, object_name: str) -> dict:
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{object_name}_affordance_justification.json")) as f:
        data = json.load(f)
    with open(os.path.join(COMPARATIVE_LABELS_DIR, f"{object_name}_comparative_labels.json")) as f:
        comparative = json.load(f)
    parts = data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]

    results = []
    for i, part in enumerate(parts, start=1):
        prompt_text = build_prompt_text(processor, build_per_part_question(i, len(parts), options, glosses))
        fc = score_options(model, processor, overlay, prompt_text, option_ids, device)
        probs = softmax(fc)
        new_label = top1_label(dict(zip(options, fc.tolist())))
        old_isolated = part["vocabularies"][VOCAB]["label"]
        old_comparative = comparative["regions"][i - 1]["comparative_label"]
        results.append({
            "region_id": part["region_id"], "area": part["area"],
            "isolated_label": old_isolated, "comparative_label": old_comparative,
            "perpart_context_label": new_label, "perpart_context_prob": float(probs.max()),
        })
        print(f"    {part['region_id']:>28s}  isolated={old_isolated:>10s}  all_at_once={old_comparative:>10s}  "
              f"per_part_context={new_label} ({probs.max():.2f})")

    n_isolated = len({r["isolated_label"] for r in results})
    n_comparative = len({r["comparative_label"] for r in results})
    n_new = len({r["perpart_context_label"] for r in results})
    print(f"  {object_name}: distinct labels isolated={n_isolated} all_at_once={n_comparative} "
          f"per_part_context={n_new}  (of {len(parts)} parts)")
    return {"object": object_name, "model": MODEL_NAME, "regions": results}


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
        out_path = os.path.join(COMPARATIVE_PERPART_LABELS_DIR, f"{name}_comparative_perpart_labels.json")
        try:
            data = process_object(model, processor, option_ids, device, name)
        except Exception as e:
            print(f"  {name}: FAILED -- {e}")
            continue
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
