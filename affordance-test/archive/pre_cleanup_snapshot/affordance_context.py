"""
Top-K affordances per part, each paired with an open-ended usage context, e.g. for a ladle's
bowl: "scoop: ladling soup out of a pot", "contain: holding a portion of sauce while plating".

Reuses affordance_labeling.py's raw forced-choice scores (already computed per part, per
vocabulary option) rather than rescoring: for each part, the top K vocabulary options by
softmax probability (excluding "none") are kept, and one open-ended context sentence is
generated for each, conditioned on the same highlighted-part image and that specific
affordance. This keeps the label itself anchored to the fixed vocabulary (so it stays
comparable across parts/objects) while letting the context be free text.

The response is assistant-prefix-forced: generation starts from the literal text
"<affordance>: " (teacher-forced into the input, not just requested in the instructions), so
the model must continue writing *from* that specific affordance rather than being free to
drift back to whatever the part's single dominant use is. An earlier instruction-only version
(ask for a scenario for option X without forcing the prefix) mostly ignored X for anything
but the top-1 or top-2 options and just re-described the part's main use instead -- e.g. a
ladle handle's "wrap-grasp" (rank 5, p=0.01) got "Using the ladle to scoop soup.", not
anything about wrapping a hand around it.

Requires <object>_<region_source>_affordance_labels.json (run affordance_labeling.py first).

Usage:
    [REGION_SOURCE=sam_slic|sam] python affordance_context.py [object_name] [top_k]   # default: shape_16 5
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
    MODEL_NAME, REGION_SOURCE, UMD_GLOSSES, build_prompt_text, downscale, highlight_region, labels_path,
)

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAX_NEW_TOKENS = 64  # headroom to reach a sentence boundary; output is cropped to the first one regardless


def context_path(object_name: str) -> str:
    return os.path.join(HERE, f"{object_name}_{REGION_SOURCE}_affordance_context.json")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def top_options(fc_logprobs: dict, k: int) -> list:
    """[(option, prob)], excluding "none", ranked by softmax probability over the full vocabulary
    (so a part with a confident "none" naturally yields low-probability, uncertain runners-up)."""
    options = list(fc_logprobs)
    probs = softmax(np.array([fc_logprobs[o] for o in options]))
    ranked = sorted(zip(options, probs), key=lambda t: -t[1])
    return [(o, p) for o, p in ranked if o != "none"][:k]


def build_context_prompt(option: str, glosses: dict) -> str:
    gloss = f" ({glosses[option]})" if option in glosses else ""
    return (
        "The image shows an object. One part of it is highlighted: it is at full brightness and "
        "outlined in magenta, while the rest of the image is darkened (use it only as context). "
        f'Assume the highlighted part is being used for "{option}"{gloss}, even if that is not its most '
        "typical use. Complete the sentence below with ONE short, specific, concrete scenario (a single "
        f"clause, under 12 words) of this exact use -- not a different one, and do not mention the "
        "highlighting, outline, or colors. Answer with only the completed sentence.\n\n"
        f"{option}:"
    )


def first_sentence(text: str) -> str:
    """Model output tends to ramble on well past the requested length; cropping to the first
    clause/sentence both enforces brevity and avoids the mid-word cutoffs a hard token budget alone
    can produce (generation still gets a larger MAX_NEW_TOKENS as headroom to reach that boundary)."""
    text = text.strip().strip('"').lstrip(":").strip()
    cut = min((i for i in (text.find(c) for c in ".!?") if i != -1), default=len(text))
    return text[:cut].strip() if cut < len(text) else text


@torch.no_grad()
def generate_context(model, processor, image: Image.Image, prompt_text: str, option: str, device) -> str:
    forced_prefix = f"{option}:"
    inputs = processor(text=[prompt_text + forced_prefix], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
    )
    continuation = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return first_sentence(continuation)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(labels_path(OBJECT_NAME)) as f:
        data = json.load(f)
    vocab_glosses = {"agd20k": {}, "umd": UMD_GLOSSES}
    image = Image.open(data["image_path"]).convert("RGB")

    print(f"Object: {OBJECT_NAME} | {len(data['regions'])} parts | top_k={TOP_K} | region_source={REGION_SOURCE}")
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
        for vocab, glosses in vocab_glosses.items():
            items = []
            for option, prob in top_options(part["fc_logprobs"][vocab], TOP_K):
                prompt_text = build_prompt_text(processor, build_context_prompt(option, glosses))
                context = generate_context(model, processor, highlighted, prompt_text, option, device)
                items.append({"affordance": option, "prob": float(prob), "context": context})
                print(f"  [{vocab:>6s}] {option:>14s} ({prob:.2f}): {context}")
            rec["vocabularies"][vocab] = items
        results.append(rec)

    out = {
        "object": OBJECT_NAME,
        "model": MODEL_NAME,
        "image_path": data["image_path"],
        "region_source": REGION_SOURCE,
        "top_k": TOP_K,
        "regions": results,
    }
    out_path = context_path(OBJECT_NAME)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
