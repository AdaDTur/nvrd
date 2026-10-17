"""
Final, standard labeling approach: per-part forced-choice scoring AND per-part justification
generation, both against a single shared numbered/colored overlay image showing all of an object's
parts at once (comparative_affordance_labeling.py's numbered_overlay), but each part is still an
independent model call (comparative_perpart_labeling.py's design) so there's no autoregressive
copy-paste between parts -- the failure mode that made the earlier all-parts-in-one-generation
version collapse to a single repeated label for chair, fire-alarm, bonsai_amplifier, bonsai_crystal,
bird-feathers_headphones, and checkerboard_tv-remote. Confirmed on those 6 objects (this session) to
recover real per-part differentiation without reintroducing that collapse.

This replaces both the original isolated single-part-highlighted approach (affordance_labeling.py /
batch_affordance_justification.py) and the broken all-at-once comparative approach
(comparative_affordance_labeling.py / batch_comparative_labeling.py) as the standard pipeline.
Output schema is unchanged from batch_affordance_justification.py's, so
visualize_affordance_justification.py works with it unmodified.

Reuses each object's existing disjoint-part segmentation (region_id/area/segmentation) from its
current <object>_affordance_justification.json rather than recomputing partition_regions -- that
segmentation (background-filtered, capped at 30 regions) is unchanged by this relabeling.

Usage:
    python batch_final_labeling.py
"""
import os
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, downscale, score_options, softmax,
)
from affordance_context import first_sentence
from comparative_affordance_labeling import numbered_overlay
from comparative_perpart_labeling import build_per_part_question

HERE = os.path.dirname(os.path.realpath(__file__))
VOCAB = "agd20k"
MAX_NEW_TOKENS = 64

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


def build_per_part_justification_prompt(part_idx: int, n_parts: int, label: str, glosses: dict) -> str:
    gloss = f" ({glosses[label]})" if label in glosses else ""
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        f"outlined and tinted a different color. You determined that part {part_idx}'s affordance is "
        f'"{label}"{gloss}. In one short, specific sentence, justify this based on what you observe '
        f"about part {part_idx} specifically (its shape, material, or position) -- and why it might "
        "differ from the other numbered parts. Do not mention the colors or numbering. Answer with "
        f"only the justification sentence.\n\n{label}:"
    )


@torch.no_grad()
def generate_justification(model, processor, image: Image.Image, prompt_text: str, label: str, device) -> str:
    forced_prefix = f"{label}:"
    inputs = processor(text=[prompt_text + forced_prefix], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
    )
    continuation = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return first_sentence(continuation)


def top1_label(fc_logprobs: dict) -> str:
    return max(fc_logprobs, key=fc_logprobs.get)


def process_object(model, processor, option_ids, device, object_name: str) -> dict:
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{object_name}_affordance_justification.json")) as f:
        old_data = json.load(f)
    parts = old_data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(old_data["image_path"]).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]

    results = []
    for i, part in enumerate(parts, start=1):
        fc_prompt = build_prompt_text(processor, build_per_part_question(i, len(parts), options, glosses))
        fc = score_options(model, processor, overlay, fc_prompt, option_ids, device)
        label = top1_label(dict(zip(options, fc.tolist())))

        just_prompt = build_prompt_text(
            processor, build_per_part_justification_prompt(i, len(parts), label, glosses)
        )
        justification = generate_justification(model, processor, overlay, just_prompt, label, device)

        results.append({
            "region_id": part["region_id"], "area": part["area"], "segmentation": part["segmentation"],
            "vocabularies": {VOCAB: {"label": label, "justification": justification}},
        })
        print(f"    {part['region_id']:>28s}  {label:>14s}: {justification}")

    return {
        "object": object_name, "model": MODEL_NAME, "image_path": old_data["image_path"],
        "region_source": old_data.get("region_source", "sam_slic"), "method": "per_part_with_comparative_context",
        "regions": results,
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

    for category, names in OBJECTS:
        print(f"\n=== {category} ===")
        for name in names:
            print(f"  {name}:")
            try:
                data = process_object(model, processor, option_ids, device, name)
            except Exception as e:
                print(f"  {name}: FAILED -- {e}")
                continue
            out_path = os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{name}_affordance_justification.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            n_distinct = len({r["vocabularies"][VOCAB]["label"] for r in data["regions"]})
            print(f"  saved {out_path}  (distinct labels: {n_distinct} of {len(data['regions'])} parts)")


if __name__ == "__main__":
    main()
