"""
Size/shape-controlled region information gain.

The naive IG(r) = logP(D0|full) - logP(D0|occluded r) is confounded by region size: a
bigger gray patch is a bigger visual disruption almost regardless of what was under it,
so large regions get inflated IG just for being large. Dividing by area (as
region_information_gain.py's `information_gain_bits_per_1000px` does) is only a
first-order guess at how that confound scales -- it's not verified.

Here we control for it properly: for each region r, take its exact mask (same size,
same shape) and translate it to K random OTHER locations in the image (clamped to stay
in-bounds, resampled if it overlaps the original too much). Score IG for each translated
copy the same way. This gives a null distribution of "IG you'd get from occluding a
patch this size and shape, regardless of content/location." Then:

    excess_IG(r)  = IG(r) - mean(IG_control(r))
    z_IG(r)       = excess_IG(r) / std(IG_control(r))

excess_IG / z_IG isolate the semantic contribution of what's actually at region r's
location, with size and shape held fixed.

Usage:
    python region_information_gain_size_controlled.py [object_name] [K]
    # default object_name=shape_16, K=6 random translations per region
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from region_information_gain import IG_JSON_DIR
from region_information_gain import (
    build_prompt_text, downscale, sequence_logprob, load_nonce_word, load_regions, resolve_image_path,
    OCCLUSION_FILL,
)

HERE = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGIONS_TEST = os.path.join(REPO_ROOT, "regions_test")
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MAX_REGIONS = int(sys.argv[3]) if len(sys.argv) > 3 else None  # must match the value used for region_information_gain.py
MAX_IOU_WITH_ORIGINAL = 0.3
RNG_SEED = 0


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    new_mask = np.zeros_like(mask)
    new_mask[ys + dy, xs + dx] = True
    return new_mask


def sample_translations(mask: np.ndarray, k: int, rng: np.random.Generator):
    """Sample up to k random in-bounds translations of `mask` with low overlap with the
    original. Returns (translations, iou_used, n_attempts).

    For a region that occupies most of one image dimension (e.g. a mask spanning 90%+ of
    the height), there may be very little room to translate it without re-overlapping
    itself -- so a fixed, strict IoU cutoff can legitimately find zero valid candidates.
    Rather than silently returning an empty list (which upstream turns into NaN via
    mean()/std() on an empty array), progressively relax the IoU cutoff until at least
    one candidate is found, and report which cutoff was actually used so this is visible
    rather than a silent failure.
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    dy_lo, dy_hi = -y0, H - 1 - y1
    dx_lo, dx_hi = -x0, W - 1 - x1
    area = mask.sum()

    max_attempts = 3000
    for iou_cutoff in (MAX_IOU_WITH_ORIGINAL, 0.5, 0.7, 0.9, 0.99):
        out = []
        attempts = 0
        while len(out) < k and attempts < max_attempts:
            attempts += 1
            dy = rng.integers(dy_lo, dy_hi + 1)
            dx = rng.integers(dx_lo, dx_hi + 1)
            candidate = translate_mask(mask, dy, dx)
            iou = (candidate & mask).sum() / area
            if iou <= iou_cutoff:
                out.append(candidate)
        if out:
            return out, iou_cutoff, attempts
    # Should be unreachable (iou_cutoff=0.99 accepts virtually any in-bounds shift), but
    # never return an empty list silently.
    return out, iou_cutoff, attempts


def occlude_with_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(image).copy()
    arr[mask] = OCCLUSION_FILL
    return Image.fromarray(arr)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(RNG_SEED)

    with open(os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain.json")) as f:
        base = json.load(f)
    d0 = base["baseline_description"]
    logprob_full = base["logprob_full"]
    base_by_id = {r["region_id"]: r for r in base["regions"]}

    nonce_word = load_nonce_word(OBJECT_NAME)
    all_regions = load_regions(OBJECT_NAME, MAX_REGIONS)
    img_h, img_w = all_regions[0]["segmentation"]["size"]
    regions = [r for r in all_regions if r["area"] < 0.5 * img_h * img_w]
    # excludes the whole-image-sized background region: translating "everything but the
    # object" isn't a meaningful shape-matched control.
    img_path = resolve_image_path(OBJECT_NAME)
    image = Image.open(img_path).convert("RGB")

    print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    prompt_text = build_prompt_text(processor, nonce_word)

    results = []
    for region in regions:
        mask = mask_utils.decode(region["segmentation"]).astype(bool)
        real_ig = logprob_full - base_by_id[region["region_id"]]["logprob_occluded"]

        translations, iou_used, n_attempts = sample_translations(mask, K, rng)
        if iou_used > MAX_IOU_WITH_ORIGINAL:
            print(
                f"  [{region['region_id']}] area covers too much of the image for the "
                f"strict {MAX_IOU_WITH_ORIGINAL} IoU cutoff to find {K} candidates in "
                f"{n_attempts} attempts; relaxed to IoU<={iou_used} "
                f"({len(translations)} controls found)"
            )
        control_igs = []
        for t_mask in translations:
            occ_img = downscale(occlude_with_mask(image, t_mask))
            logprob_ctrl, n_tok = sequence_logprob(model, processor, occ_img, prompt_text, d0, device)
            control_igs.append(logprob_full - logprob_ctrl)

        control_igs = np.array(control_igs)
        if len(control_igs) < 2:
            # Can't estimate a std from 0-1 samples -- z is genuinely undefined here, not
            # just an artifact to paper over. Surface it loudly instead of a silent NaN.
            print(f"  [{region['region_id']}] WARNING: only {len(control_igs)} control(s) "
                  f"found -- z-score undefined for this region, skipping.")
            mean_ctrl = float(control_igs.mean()) if len(control_igs) else float("nan")
            std_ctrl = float("nan")
            excess_ig = float("nan")
            z = float("nan")
        else:
            mean_ctrl = float(control_igs.mean())
            std_ctrl = float(control_igs.std(ddof=1))
            excess_ig = real_ig - mean_ctrl
            z = excess_ig / std_ctrl if std_ctrl > 1e-9 else float("nan")

        rec = {
            "region_id": region["region_id"],
            "area": region["area"],
            "information_gain_nats": real_ig,
            "control_igs_nats": control_igs.tolist(),
            "control_mean_nats": mean_ctrl,
            "control_std_nats": std_ctrl,
            "excess_ig_nats": excess_ig,
            "excess_ig_bits": excess_ig / np.log(2) if excess_ig == excess_ig else float("nan"),
            "z_score": z,
            "n_controls": len(control_igs),
            "iou_cutoff_used": iou_used,
        }
        results.append(rec)
        print(
            f"{region['region_id']:>24s}  area={region['area']:>7d}  "
            f"real_IG={real_ig:+7.3f}  control_mean={mean_ctrl:+7.3f}±{std_ctrl:.3f}  "
            f"excess_IG={excess_ig:+7.3f} nats  z={z:+6.2f}"
        )

    out = {
        "object": OBJECT_NAME,
        "max_regions": MAX_REGIONS,
        "nonce_word": nonce_word,
        "model": MODEL_NAME,
        "baseline_description": d0,
        "logprob_full": logprob_full,
        "k_controls": K,
        "max_iou_with_original": MAX_IOU_WITH_ORIGINAL,
        "regions": results,
    }
    out_path = os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain_size_controlled.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\nRanked by z-score (size/shape-controlled):")
    for r in sorted(results, key=lambda r: r["z_score"], reverse=True):
        print(f"  {r['region_id']:>24s}  z={r['z_score']:+6.2f}  excess_IG={r['excess_ig_bits']:+7.3f} bits")


if __name__ == "__main__":
    main()
