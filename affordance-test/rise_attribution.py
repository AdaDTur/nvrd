"""
RISE (Randomized Input Sampling for Explanation -- Petsiuk et al., BMVC 2018), adapted to
occlude SAM regions instead of raw pixel masks. Reuses the same functionality-prompt D0 and
teacher-forced log-likelihood scoring as region_information_gain.py, but replaces its
single-region-at-a-time occlusion + synthetic "elsewhere" null (region_information_gain_
size_controlled.py) with a fundamentally different design:

    For M random trials, independently keep or occlude EVERY region with probability 0.5,
    gray-filling the occluded ones together in one image. Score logP(D0 | trial_image).
    For each region r:

        importance(r) = mean(logP over trials where r was VISIBLE)
                       - mean(logP over trials where r was OCCLUDED)

Why this instead of the size/shape-controlled z-score approach: that method required
constructing a synthetic "elsewhere" placement for each region's mask (translated, and
later also flipped) to build a null distribution -- and we found a real, structural bug in
it: for a region occupying a large fraction of a small/tight image (e.g. a door leaf at
~31% of the frame), there is provably not enough room to translate the mask far enough from
its own location to get a clean non-overlapping "elsewhere", which silently biased that
region's null distribution upward (see region_information_gain_size_controlled.py's
docstring and the conversation this was debugged in). RISE has no such machinery at all --
there is no synthetic mask placement to construct, so that failure mode cannot occur. Each
region's importance is estimated from many REAL random co-occlusion contexts instead, which
also means (unlike leave-one-out) it captures interaction/context effects between regions,
and the M trials are shared across ALL regions at once (no per-region control loop), so this
is cheaper in total forward passes than the size-controlled method for a comparable region
count.

The whole-image-sized background region (>=50% of the image) is excluded from the random
keep/occlude draw (always kept visible) for the same reason region_information_gain_size_
controlled.py excludes it from its null construction: it isn't a meaningful single "part",
and letting it be randomly occluded half the time would make roughly half of all trials
near-fully-blank images, wasting most of the sampling budget on uninformative extremes.

Usage:
    python rise_attribution.py [object_name] [M] [max_regions]
    # default object_name=shape_16, M=300 random trials
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
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
BACKGROUND_AREA_FRAC = 0.5  # matches region_information_gain_size_controlled.py's threshold
KEEP_PROB = 0.5
RNG_SEED = 0

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 300
MAX_REGIONS = int(sys.argv[3]) if len(sys.argv) > 3 else None


def occlude_union(image: Image.Image, regions: list, masks: list, keep: np.ndarray) -> Image.Image:
    """Gray-fill the union of every region's mask where keep[i] is False."""
    arr = np.array(image).copy()
    any_occluded = np.zeros(masks[0].shape, dtype=bool)
    for i, m in enumerate(masks):
        if not keep[i]:
            any_occluded |= m
    arr[any_occluded] = OCCLUSION_FILL
    return Image.fromarray(arr)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(RNG_SEED)

    with open(os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain.json")) as f:
        base = json.load(f)
    d0 = base["baseline_description"]

    nonce_word = load_nonce_word(OBJECT_NAME)
    all_regions = load_regions(OBJECT_NAME, MAX_REGIONS)
    img_h, img_w = all_regions[0]["segmentation"]["size"]
    regions = [r for r in all_regions if r["area"] < BACKGROUND_AREA_FRAC * img_h * img_w]
    n_excluded = len(all_regions) - len(regions)
    if n_excluded:
        print(f"Excluding {n_excluded} background-sized region(s) (>={BACKGROUND_AREA_FRAC:.0%} "
              f"of image) from the random keep/occlude draw (always kept visible)")

    img_path = resolve_image_path(OBJECT_NAME)
    image = Image.open(img_path).convert("RGB")

    print(f"Object: {OBJECT_NAME} | {len(regions)} regions | M={M} trials | D0 = {d0!r}\n")
    print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    prompt_text = build_prompt_text(processor, nonce_word)

    masks = [mask_utils.decode(r["segmentation"]).astype(bool) for r in regions]
    n_regions = len(regions)

    # keep_matrix[t, i] = True if region i is VISIBLE (not occluded) in trial t.
    keep_matrix = rng.random((M, n_regions)) < KEEP_PROB
    trial_logprobs = np.zeros(M)

    for t in range(M):
        keep = keep_matrix[t]
        trial_image = downscale(occlude_union(image, regions, masks, keep))
        logprob, n_tok = sequence_logprob(model, processor, trial_image, prompt_text, d0, device)
        trial_logprobs[t] = logprob
        if (t + 1) % 25 == 0 or t == 0:
            n_kept = int(keep.sum())
            print(f"  trial {t + 1}/{M}: {n_kept}/{n_regions} regions kept visible, "
                  f"logP(D0)={logprob:.3f}")

    print()
    results = []
    for i, region in enumerate(regions):
        visible_lp = trial_logprobs[keep_matrix[:, i]]
        occluded_lp = trial_logprobs[~keep_matrix[:, i]]
        importance = float(visible_lp.mean() - occluded_lp.mean())
        # Welch's t-statistic (unequal variance two-sample test) as a signal-to-noise
        # measure, analogous in spirit to the z-score from the size-controlled method but
        # with no synthetic-null bias: both groups are real random trials.
        n_v, n_o = len(visible_lp), len(occluded_lp)
        var_v = visible_lp.var(ddof=1) if n_v > 1 else 0.0
        var_o = occluded_lp.var(ddof=1) if n_o > 1 else 0.0
        se = np.sqrt(var_v / max(n_v, 1) + var_o / max(n_o, 1))
        t_stat = float(importance / se) if se > 1e-9 else float("nan")

        rec = {
            "region_id": region["region_id"],
            "area": region["area"],
            "n_visible_trials": n_v,
            "n_occluded_trials": n_o,
            "mean_logprob_visible": float(visible_lp.mean()),
            "mean_logprob_occluded": float(occluded_lp.mean()),
            "importance_nats": importance,
            "importance_bits": importance / np.log(2),
            "t_stat": t_stat,
        }
        results.append(rec)
        print(f"{region['region_id']:>24s}  area={region['area']:>7d}  "
              f"n_vis={n_v:>3d} n_occ={n_o:>3d}  importance={importance:+7.3f} nats  t={t_stat:+6.2f}")

    out = {
        "object": OBJECT_NAME,
        "model": MODEL_NAME,
        "baseline_description": d0,
        "method": "RISE (Petsiuk et al. 2018), region-level",
        "M": M,
        "keep_prob": KEEP_PROB,
        "rng_seed": RNG_SEED,
        "regions": results,
    }
    out_path = os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_rise_attribution.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\nRanked by t-statistic:")
    for r in sorted(results, key=lambda r: r["t_stat"], reverse=True):
        print(f"  {r['region_id']:>24s}  t={r['t_stat']:+6.2f}  importance={r['importance_bits']:+7.3f} bits")


if __name__ == "__main__":
    main()
