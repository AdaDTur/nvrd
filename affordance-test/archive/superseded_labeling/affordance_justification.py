"""
For each disjoint SAM+SLIC part, take the raw forced-choice top-1 affordance label (already
scored by affordance_labeling.py) and generate one justification sentence explaining why the
model picked that label, conditioned on the same highlighted-part image.

Usage:
    [REGION_SOURCE=sam_slic|sam] python affordance_justification.py [object_name]   # default: ladle
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
    AFFORDANCE_JUSTIFICATION_DIR, MODEL_NAME, build_prompt_text, downscale, highlight_region, labels_path,
)
from affordance_context import first_sentence

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "ladle"
MAX_NEW_TOKENS = 64
VOCAB_GLOSSES = {"agd20k": {}}


def top1_label(fc_logprobs: dict) -> str:
    return max(fc_logprobs, key=fc_logprobs.get)


def build_justification_prompt(label: str, glosses: dict) -> str:
    gloss = f" ({glosses[label]})" if label in glosses else ""
    return (
        "The image shows an object. One part of it is highlighted: it is at full brightness and "
        "outlined in magenta, while the rest of the image is darkened (use it only as context). "
        f'You determined that the highlighted part\'s affordance is "{label}"{gloss}. '
        "In one sentence, justify this based on what you observe about the highlighted part "
        "(its shape, material, or position). Do not mention the highlighting, outline, or colors. "
        "Answer with only the justification sentence."
    )


@torch.no_grad()
def generate_justification(model, processor, image: Image.Image, prompt_text: str, device) -> str:
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
    )
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return first_sentence(text)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(labels_path(OBJECT_NAME)) as f:
        data = json.load(f)
    image = Image.open(data["image_path"]).convert("RGB")

    print(f"Object: {OBJECT_NAME} | {len(data['regions'])} parts")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for part in data["regions"]:
        mask = mask_utils.decode(part["segmentation"]).astype(bool)
        highlighted = downscale(highlight_region(image, mask))
        rec = {
            "region_id": part["region_id"], "area": part["area"], "segmentation": part["segmentation"],
            "vocabularies": {},
        }
        print(f"\n{part['region_id']} (area={part['area']})")
        for vocab, glosses in VOCAB_GLOSSES.items():
            label = top1_label(part["fc_logprobs"][vocab])
            prompt_text = build_prompt_text(processor, build_justification_prompt(label, glosses))
            justification = generate_justification(model, processor, highlighted, prompt_text, device)
            rec["vocabularies"][vocab] = {"label": label, "justification": justification}
            print(f"  [{vocab:>6s}] {label}: {justification}")
        results.append(rec)

    out = {
        "object": OBJECT_NAME, "model": MODEL_NAME, "image_path": data["image_path"],
        "region_source": data.get("region_source", "sam_slic"), "regions": results,
    }
    out_path = os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{OBJECT_NAME}_affordance_justification.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
