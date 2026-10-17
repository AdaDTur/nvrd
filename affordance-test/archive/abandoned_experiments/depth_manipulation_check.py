"""
Manipulation check for the depth-map identity-hiding diagnostic (grounding test, part 1).

Before trusting any depth-vs-original affordance comparison, we must confirm the depth transform
actually does what it's supposed to: let the VLM still perceive the object's shape, while blocking
it from naming/recognizing the object category. This script asks Qwen2-VL "What object is shown in
this image?" on both the original photo and the depth-only render, for every known object, and
checks whether the object's true name (or an obvious synonym) appears in the free-generated answer.

This is a gate, not the real experiment: if the model still names objects correctly from their
depth maps, the depth condition doesn't isolate anything and the whole diagnostic is invalid for
those objects.

Usage:
    python depth_manipulation_check.py
"""
import os
import json

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from region_information_gain import MODEL_NAME, resolve_image_path
from affordance_labeling import downscale
from depth_diagnostic import KNOWN_OBJECTS, depth_image_path

HERE = os.path.dirname(os.path.realpath(__file__))
NAMING_PROMPT = "What object is shown in this image? Answer with just the object's name, a few words at most."
MAX_NEW_TOKENS = 20

# A few known objects use hyphenated/compound NVRD names that a free-text answer would naturally
# phrase differently (e.g. "coffee-table" -> "coffee table" or just "table"); listing acceptable
# synonyms keeps the automatic check honest without being unfairly strict about phrasing.
SYNONYMS = {
    "coffee-table": ["coffee table", "table"],
    "bow-tie": ["bow tie", "bowtie"],
    "fire-alarm": ["fire alarm", "smoke alarm", "smoke detector"],
}


@torch.no_grad()
def name_object(model, processor, image: Image.Image, device) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": NAMING_PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def matches_true_name(answer: str, object_name: str) -> bool:
    answer_l = answer.lower()
    candidates = [object_name.replace("-", " "), object_name.replace("-", "")] + SYNONYMS.get(object_name, [])
    return any(c in answer_l for c in candidates)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    print(f"{'object':>14s}  {'original answer':>30s}  {'depth answer':>30s}  {'orig_named':>10s} {'depth_named':>11s}")
    for obj in KNOWN_OBJECTS:
        original = downscale(Image.open(resolve_image_path(obj)).convert("RGB"))
        depth = downscale(Image.open(depth_image_path(obj)).convert("RGB"))
        orig_ans = name_object(model, processor, original, device)
        depth_ans = name_object(model, processor, depth, device)
        orig_named = matches_true_name(orig_ans, obj)
        depth_named = matches_true_name(depth_ans, obj)
        results.append({
            "object": obj, "original_answer": orig_ans, "depth_answer": depth_ans,
            "original_named_correctly": orig_named, "depth_named_correctly": depth_named,
        })
        print(f"{obj:>14s}  {orig_ans:>30s}  {depth_ans:>30s}  {str(orig_named):>10s} {str(depth_named):>11s}")

    n_gated = sum(not r["depth_named_correctly"] for r in results)
    n_orig_ok = sum(r["original_named_correctly"] for r in results)
    print(f"\n{n_orig_ok}/{len(results)} named correctly from the original photo "
          f"(sanity check -- should be ~all of them)")
    print(f"{n_gated}/{len(results)} objects successfully de-named by the depth transform "
          f"(these are usable for the diagnostic; the rest are not)")

    out_path = os.path.join(HERE, "depth_manipulation_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
