"""
Batch driver for comparative_affordance_labeling.py's probe (all parts shown at once, numbered,
one free-form generation per object) across the full 33-object set, loading Qwen2-VL once instead
of once per object.

Usage:
    python batch_comparative_labeling.py
"""
import os
import json

import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, COMPARATIVE_LABELS_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, downscale,
)
from comparative_affordance_labeling import (
    VOCAB, build_comparative_question, generate_comparative, numbered_overlay, parse_response,
)

HERE = os.path.dirname(os.path.realpath(__file__))

OBJECTS = [
    ("known", ["ladle", "chair", "corkscrew", "backpack", "bow-tie", "clock", "coffee-table", "fedora",
               "fire-alarm", "wallet"]),
    ("novel", ["shape_16", "shape_27", "shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12",
               "shape_13", "shape_14"]),
    ("shape-shape", ["backpack_amplifier", "bonsai_amplifier", "bonsai_crystal", "cactus_camera",
                      "compass_totem", "crystal_telescope", "dagger_amplifier", "dagger_projector",
                      "drone_dagger"]),
    ("shape-texture", ["bird-feathers_headphones", "bird-feathers_saxophone", "checkerboard_tv-remote",
                        "crochet_broom"]),
]


def process_object(model, processor, device, object_name: str) -> dict:
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{object_name}_affordance_justification.json")) as f:
        data = json.load(f)
    parts = data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))

    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    question = build_comparative_question(len(parts), options, glosses)
    prompt_text = build_prompt_text(processor, question)

    max_tokens = 12 * len(parts) + 20
    response = generate_comparative(model, processor, overlay, prompt_text, max_tokens, device)
    parsed = parse_response(response, len(parts), options)

    results = []
    for i, part in enumerate(parts, start=1):
        old_label = part["vocabularies"][VOCAB]["label"]
        new_label = parsed.get(i, "MISSING")
        results.append({"region_id": part["region_id"], "area": part["area"],
                         "isolated_label": old_label, "comparative_label": new_label})
        print(f"    {part['region_id']:>28s}  isolated={old_label:>12s}  comparative={new_label}")

    n_distinct_old = len({r["isolated_label"] for r in results})
    n_distinct_new = len({r["comparative_label"] for r in results})
    print(f"  {object_name}: distinct labels isolated={n_distinct_old} -> comparative={n_distinct_new} "
          f"(of {len(parts)} parts)")

    overlay_path = os.path.join(HERE, "figures", f"{object_name}_comparative_overlay_input.png")
    overlay.save(overlay_path)

    return {"object": object_name, "model": MODEL_NAME, "question": question, "raw_response": response,
            "regions": results}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    for category, names in OBJECTS:
        print(f"\n=== {category} ===")
        for name in names:
            out_path = os.path.join(COMPARATIVE_LABELS_DIR, f"{name}_comparative_labels.json")
            if os.path.isfile(out_path):
                print(f"  {name}: already done, skipping")
                continue
            try:
                data = process_object(model, processor, device, name)
            except Exception as e:
                print(f"  {name}: FAILED -- {e}")
                continue
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
