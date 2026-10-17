"""
Numbered-parts + highlighted-part images for the Prolific affordance-study's new training phase
(5 known, real-world objects with an unambiguous correct functionality for one specific part each)
-- same visual convention as the novel-object study images (numbered_overlay / highlight_region
from affordance_labeling.py), just pointed at known objects instead of shape_16.

This is deliberately a tiny, standalone script rather than an addition to batch_final_labeling.py:
it renders images for a few hand-picked (object, region_id) training pairs, not a full per-object
labeling pass, so it doesn't share that script's loop/model machinery.

Usage:
    python3 generate_training_images.py
Output: figures/training_images/<object>_numbered.png, <object>_highlight_part<N>.png
"""
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
_VENV_DIR = os.path.join(HERE, "venv_vlm")
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _venv_py = os.path.join(_VENV_DIR, "bin", "python3")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(_venv_py, [_venv_py] + sys.argv)

from PIL import Image

from affordance_labeling import downscale, highlight_region, numbered_overlay, partition_regions
from region_information_gain import load_regions, resolve_image_path

OUT_DIR = os.path.join(HERE, "figures", "training_images")
os.makedirs(OUT_DIR, exist_ok=True)
MAX_REGIONS = 30
REGION_SOURCE = "sam_slic"

# (object, target region_id, correct functionality this part is picked for) -- region_ids and areas
# cross-checked against json/affordance_justification/<object>_affordance_justification.json before
# picking, so each is the large, visually-obvious region a human would actually point to, not a
# stray sliver. See demo.js's TRAINING_ITEMS for the accepted-answer list each maps to (a couple of
# these intentionally accept more than one close synonym).
#
# Re-picked after switching region generation from SAM1(vit_b)+SLIC to SAM2 (see
# generate_regions_sam2.py) -- region_ids are freshly renumbered under the new segmentation, so the
# old ids no longer point at the same parts. Verified each one by actually rendering and looking at
# its highlight_part<N>.png, not just the numbered-overlay's number-LABEL position -- that label
# sits at the mask's pixel centroid, which for a large/irregularly-shaped mask can land visually
# near a completely different feature than the mask itself covers. Caught exactly this on chair:
# the first candidate (by area) looked like "the seat" going by where its number rendered, but was
# actually a thin compound residual strip tracing the backrest/seat-edge/legs/cross-brace, not a
# real "part" at all -- confirmed by rendering its highlight directly. The actual clean seat was
# the next candidate down.
TRAINING_TARGETS = [
    ("chair", "chair_region_40"),
    ("ladle", "ladle_region_4"),
    ("fedora", "fedora_region_7"),
    ("backpack", "backpack_region_15"),
    ("wallet", "wallet_region_4"),
]


def main():
    for obj, target_region_id in TRAINING_TARGETS:
        regions = load_regions(obj, MAX_REGIONS, REGION_SOURCE)
        img_h, img_w = regions[0]["segmentation"]["size"]
        img_path = resolve_image_path(obj)
        image = Image.open(img_path).convert("RGB")
        parts = partition_regions(regions, img_h * img_w, image)
        masks = [m for _, m in parts]  # partition_regions already returns decoded bool masks

        target_idx = None
        for i, (region, _mask) in enumerate(parts):
            if region["region_id"] == target_region_id:
                target_idx = i
                break
        if target_idx is None:
            print(f"  {obj}: region {target_region_id} not found among {len(parts)} parts -- "
                  f"available: {[r['region_id'] for r, _ in parts]}")
            continue

        numbered = downscale(numbered_overlay(image, masks))
        numbered_path = os.path.join(OUT_DIR, f"{obj}_numbered.png")
        numbered.save(numbered_path)

        part_number = target_idx + 1
        highlight = downscale(highlight_region(image, masks[target_idx]))
        highlight_path = os.path.join(OUT_DIR, f"{obj}_highlight_part{part_number}.png")
        highlight.save(highlight_path)

        print(f"  {obj}: part {part_number} ({target_region_id}) of {len(parts)} -- "
              f"saved {numbered_path}, {highlight_path}")


if __name__ == "__main__":
    main()
