"""
Grounding test 3: cross-region occlusion sensitivity.

If a part's affordance label is actually driven by recognizing the *whole object* (e.g. "this
silhouette is a ladle, so its handle must be for holding") rather than by the part's own local
visual evidence, then hiding the object's OTHER parts -- destroying the cues that let the model
recognize what the whole object is -- should shift the target part's affordance distribution,
even though the target part's own pixels never changed. A part-locally-grounded judgment should
be far more stable to that manipulation.

For each labelled part (target), compares its forced-choice distribution under three levels of
context removal:
    normal        : affordance_labeling.py's default -- target highlighted, everything else
                    (other parts + background) merely dimmed, still visible. Reused from the
                    existing labels JSON, not recomputed.
    other_occluded: every OTHER labelled part is fully gray-filled (erased, not just dimmed);
                    background stays dimmed as in "normal". Removes other-part identity cues
                    while keeping the target's immediate surroundings/background.
    crop          : target isolated in a tight bbox crop, everything else (other parts AND
                    background) gray-filled. The most aggressive removal of object-level context.

Reports, per part per vocabulary: top-1 label under each condition, and total-variation distance
between "normal" and each of the other two (0 = identical distribution, 1 = disjoint).

Requires <object>_<region_source>_affordance_labels.json (run affordance_labeling.py first).

Usage:
    [REGION_SOURCE=sam_slic|sam] python grounding_occlusion.py [object_name]   # default: ladle
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
    MODEL_NAME, VOCABULARIES, build_prompt_text, build_question, downscale, highlight_region, labels_path,
    score_options, softmax,
)
from highlight_ablation import masked_crop
from region_information_gain import OCCLUSION_FILL

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "ladle"


def other_occluded_image(image: Image.Image, masks: list, target_idx: int) -> Image.Image:
    """Gray-fill every part mask except the target, then apply the usual highlight treatment to
    the target on top (dim background, brighten+outline target)."""
    arr = np.array(image).copy()
    for i, m in enumerate(masks):
        if i != target_idx:
            arr[m] = OCCLUSION_FILL
    return highlight_region(Image.fromarray(arr), masks[target_idx])


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def dist_from_logprobs(logprobs: dict, options: list) -> np.ndarray:
    return softmax(np.array([logprobs[o] for o in options]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(labels_path(OBJECT_NAME)) as f:
        data = json.load(f)
    parts = data["regions"]
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(data["image_path"]).convert("RGB")

    print(f"Object: {OBJECT_NAME} | {len(parts)} parts")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for idx, part in enumerate(parts):
        occ_img = downscale(other_occluded_image(image, masks, idx))
        crop_img = downscale(masked_crop(image, masks[idx]))
        rec = {"region_id": part["region_id"], "area": part["area"], "vocabularies": {}}
        print(f"\n{part['region_id']} (area={part['area']})")
        for vocab, vocab_cfg in VOCABULARIES.items():
            options = vocab_cfg["options"]
            option_ids = [processor.tokenizer(o + "<|im_end|>", add_special_tokens=False).input_ids for o in options]
            prompt_text = build_prompt_text(processor, build_question(options, vocab_cfg["glosses"]))

            normal_p = dist_from_logprobs(part["fc_logprobs"][vocab], options)
            occ_p = softmax(score_options(model, processor, occ_img, prompt_text, option_ids, device))
            crop_p = softmax(score_options(model, processor, crop_img, prompt_text, option_ids, device))

            def top(p):
                i = int(np.argmax(p))
                return f"{options[i]} ({p[i]:.2f})"

            rec["vocabularies"][vocab] = {
                "normal_top": top(normal_p), "other_occluded_top": top(occ_p), "crop_top": top(crop_p),
                "tv_normal_vs_other_occluded": tv(normal_p, occ_p),
                "tv_normal_vs_crop": tv(normal_p, crop_p),
            }
            r = rec["vocabularies"][vocab]
            print(f"  [{vocab:>6s}] normal: {r['normal_top']:>18s}  |  other_occluded: {r['other_occluded_top']:>18s} "
                  f"(TV={r['tv_normal_vs_other_occluded']:.2f})  |  crop: {r['crop_top']:>18s} "
                  f"(TV={r['tv_normal_vs_crop']:.2f})")
        results.append(rec)

    for vocab in VOCABULARIES:
        m1 = np.mean([r["vocabularies"][vocab]["tv_normal_vs_other_occluded"] for r in results])
        m2 = np.mean([r["vocabularies"][vocab]["tv_normal_vs_crop"] for r in results])
        print(f"\n{vocab} mean TV: normal-vs-other_occluded={m1:.3f}  normal-vs-crop={m2:.3f}")

    out = {"object": OBJECT_NAME, "model": MODEL_NAME, "regions": results}
    out_path = os.path.join(HERE, f"{OBJECT_NAME}_grounding_occlusion.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
