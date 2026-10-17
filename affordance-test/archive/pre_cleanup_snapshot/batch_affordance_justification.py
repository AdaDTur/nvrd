"""
Batch driver for the "forced-choice raw label + one-sentence justification" figure (AGD20K only)
across many objects in one process, loading Qwen2-VL once instead of once per object/script.
Skips the UMD vocabulary, yes/no scoring, and object-level damped-contrast baseline from
affordance_labeling.py's full pipeline entirely, since this figure only ever uses the AGD20K raw
forced-choice top-1 label -- roughly halves the per-part compute versus running the full pipeline.

Regions are capped to the MAX_REGIONS largest (by area) before partitioning into disjoint parts,
since some shape-texture objects' checkerboard/feather patterns get segmented into 100+ tiny
same-material slivers by SAM+SLIC that are not meaningful parts (e.g. checkerboard_panda: 178
merged regions) -- capping keeps compute and figure clutter bounded across very different objects.

Usage:
    python batch_affordance_justification.py
"""
import os
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, build_question, downscale,
    highlight_region, partition_regions, score_options, softmax,
)
from affordance_justification import build_justification_prompt, generate_justification
from region_information_gain import load_regions, resolve_image_path

HERE = os.path.dirname(os.path.realpath(__file__))
MAX_REGIONS = 30
REGION_SOURCE = "sam_slic"
VOCAB = "agd20k"

OBJECTS = [
    ("known", ["ladle", "chair", "corkscrew", "backpack", "bow-tie", "clock", "coffee-table", "fedora",
               "fire-alarm", "wallet"]),
    ("novel", ["shape_16", "shape_27", "shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12",
               "shape_13", "shape_14"]),
    # Animal-containing objects deliberately excluded (animals don't have designed affordances):
    # shape-shape drops boar_toaster, boar_vacuum, chipmunk_refrigerator, chipmunk_toaster,
    # dolphin_office-chair, dolphin_radio -- only 9 non-animal shape-shape objects exist at all.
    ("shape-shape", ["backpack_amplifier", "bonsai_amplifier", "bonsai_crystal", "cactus_camera",
                      "compass_totem", "crystal_telescope", "dagger_amplifier", "dagger_projector",
                      "drone_dagger"]),
    # shape-texture is overwhelmingly animal-shaped (11/15); only 4 non-animal objects exist at all.
    ("shape-texture", ["bird-feathers_headphones", "bird-feathers_saxophone", "checkerboard_tv-remote",
                        "crochet_broom"]),
]


def top1_label(fc_logprobs: dict) -> str:
    return max(fc_logprobs, key=fc_logprobs.get)


def process_object(model, processor, device, object_name: str) -> dict:
    all_regions = load_regions(object_name, MAX_REGIONS, REGION_SOURCE)
    img_h, img_w = all_regions[0]["segmentation"]["size"]
    image = Image.open(resolve_image_path(object_name)).convert("RGB")

    parts = partition_regions(all_regions, img_h * img_w, image)
    print(f"  {object_name}: {len(all_regions)} regions (capped) -> {len(parts)} disjoint parts")

    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    option_ids = [processor.tokenizer(o + "<|im_end|>", add_special_tokens=False).input_ids for o in options]
    fc_prompt = build_prompt_text(processor, build_question(options, glosses))

    results = []
    for region, mask in parts:
        highlighted = downscale(highlight_region(image, mask))
        fc = score_options(model, processor, highlighted, fc_prompt, option_ids, device)
        label = top1_label(dict(zip(options, fc.tolist())))
        justification_prompt = build_prompt_text(processor, build_justification_prompt(label, glosses))
        justification = generate_justification(model, processor, highlighted, justification_prompt, device)
        results.append({
            "region_id": region["region_id"],
            "area": int(mask.sum()),
            "segmentation": {
                "size": [img_h, img_w],
                "counts": mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8"),
            },
            "vocabularies": {VOCAB: {"label": label, "justification": justification}},
        })
        print(f"    {region['region_id']:>28s}  {label:>14s}: {justification}")

    return {
        "object": object_name, "model": MODEL_NAME, "image_path": resolve_image_path(object_name),
        "region_source": REGION_SOURCE, "regions": results,
    }


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
            out_path = os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{name}_affordance_justification.json")
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
