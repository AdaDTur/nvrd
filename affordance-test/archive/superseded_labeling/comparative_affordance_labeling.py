"""
Comparative multi-part affordance probe: instead of scoring one isolated highlighted part per model
call (affordance_labeling.py's design -- a fresh, blank-slate forward pass per part, with no memory
of or comparison to any other part), show ALL disjoint parts at once, numbered and color-coded in a
single image, and ask the model to assign an affordance to EACH numbered part in one response.

Rationale: repeated diagnosis this session (ladle wrong-name/occlusion tests, fedora's near-identical
per-part justifications) found the model reciting one whole-object-level fact for every part when
each part is probed in isolation. Forcing it to produce several answers side-by-side in a single
generation should make blindly repeating one answer for every part look wrong to the model in a way
isolated single-part calls never surface. The real cost: this can't stay a clean closed-vocabulary
forced-choice softmax the way single-part scoring is (scoring 36 options independently for N parts at
once isn't practical), so this uses free-form generation instead, parsed back against the vocabulary
-- trading exact per-option probabilities for the comparative framing.

Requires <object>_affordance_justification.json (for part masks and the existing isolated-probe
labels to compare against; run batch_affordance_justification.py or affordance_labeling.py first).

Usage:
    python comparative_affordance_labeling.py [object_name]   # default: fedora
"""
import os
import sys
import json
import re

import numpy as np
import torch
import cv2
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, COMPARATIVE_LABELS_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, downscale,
)

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "fedora"
VOCAB = "agd20k"
MAX_NEW_TOKENS_PER_PART = 12
TAB10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
         (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def numbered_overlay(image: Image.Image, masks: list) -> Image.Image:
    """All parts colored + outlined + numbered at once, analogous to the 'Disjoint parts' panel in
    visualize_affordance_justification.py, but as a plain image (no title/axes) for model input."""
    arr = np.array(image).copy()
    overlay = arr.copy()
    for i, m in enumerate(masks):
        color = TAB10[i % 10]
        overlay[m] = (0.5 * np.array(color) + 0.5 * arr[m]).astype(np.uint8)
    for i, m in enumerate(masks):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), max(2, round(min(m.shape) / 300)))
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4),
                      (255, 255, 255), -1)
        cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4),
                      (0, 0, 0), 1)
        cv2.putText(overlay, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return Image.fromarray(overlay)


def build_comparative_question(n_parts: int, options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        "outlined and tinted a different color. For EACH numbered part, choose the single affordance "
        f"it offers from this list: {listed}. If a part offers none of these, answer \"none\" for it. "
        "Different parts of the same object often have different affordances -- do not just repeat "
        "the same answer for every part unless you have a specific reason each one shares it. "
        f"Answer with exactly {n_parts} lines, in the format:\n"
        + "\n".join(f"Part {i}: <option>" for i in range(1, n_parts + 1))
    )


def parse_response(text: str, n_parts: int, options: list) -> dict:
    options_lower = {o.lower(): o for o in options}
    result = {}
    for line in text.strip().splitlines():
        m = re.match(r"\s*Part\s+(\d+)\s*:\s*(.+)", line.strip(), re.IGNORECASE)
        if not m:
            continue
        idx, raw = int(m.group(1)), m.group(2).strip().strip(".").lower()
        if 1 <= idx <= n_parts:
            result[idx] = options_lower.get(raw, f"UNPARSED({raw})")
    return result


@torch.no_grad()
def generate_comparative(model, processor, image: Image.Image, prompt_text: str, max_tokens: int, device) -> str:
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.15)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{OBJECT_NAME}_affordance_justification.json")) as f:
        data = json.load(f)
    parts = data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))

    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    question = build_comparative_question(len(parts), options, glosses)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    prompt_text = build_prompt_text(processor, question)

    print(f"Object: {OBJECT_NAME} | {len(parts)} parts, probed together in one comparative call")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    max_tokens = MAX_NEW_TOKENS_PER_PART * len(parts) + 20
    response = generate_comparative(model, processor, overlay, prompt_text, max_tokens, device)
    print("\nRaw response:\n" + response)

    parsed = parse_response(response, len(parts), options)
    print(f"\n{'part':>8s} {'region_id':>22s} {'isolated (old)':>16s} {'comparative (new)':>18s}  same?")
    results = []
    for i, part in enumerate(parts, start=1):
        old_label = part["vocabularies"][VOCAB]["label"]
        new_label = parsed.get(i, "MISSING")
        same = "same" if old_label == new_label else "DIFFERENT"
        print(f"{i:>8d} {part['region_id']:>22s} {old_label:>16s} {new_label:>18s}  {same}")
        results.append({"region_id": part["region_id"], "area": part["area"],
                         "isolated_label": old_label, "comparative_label": new_label})

    n_distinct_old = len({r["isolated_label"] for r in results})
    n_distinct_new = len({r["comparative_label"] for r in results})
    print(f"\ndistinct labels: isolated={n_distinct_old}  comparative={n_distinct_new}  (out of {len(parts)} parts)")

    out = {"object": OBJECT_NAME, "model": MODEL_NAME, "question": question, "raw_response": response,
           "regions": results}
    out_path = os.path.join(COMPARATIVE_LABELS_DIR, f"{OBJECT_NAME}_comparative_labels.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    overlay_path = os.path.join(HERE, "figures", f"{OBJECT_NAME}_comparative_overlay_input.png")
    overlay.save(overlay_path)
    print(f"Saved {overlay_path} (the numbered image actually shown to the model)")


if __name__ == "__main__":
    main()
