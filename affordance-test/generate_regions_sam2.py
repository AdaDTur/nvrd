"""
Replaces the SAM1(vit_b)+SLIC region-generation pipeline with SAM2 automatic mask generation, for
the 20 objects this study actually uses (10 known + 10 novel shape_* -- NOT shape-shape/
shape-texture, a separate never-used-here category).

Why: SAM1's vit_b checkpoint (the weakest official SAM size) badly over-fragments objects with
fine/textured/mechanical detail -- confirmed directly on corkscrew (19 fragmented parts), clock
(15, one per numeral), shape_1/shape_2 (16-20, fragmented rings/textured bodies). SAM2's automatic
mask generator gives dramatically cleaner regions on every object tested (see
sam2_experiment/run_model_comparison.py and its output images) with no engineering overhead --
unlike SAM3, which needed 4 compatibility patches, a hand-rolled automatic-mode implementation,
and still missed some real features that SAM2 caught (see conversation/commit history for the
comparison). SAM3's one genuine edge, text/concept-prompted segmentation, isn't needed here since
novel objects have no nameable parts to prompt with.

Writes output in EXACTLY the same schema partition_regions()/load_regions() already expect
(region_id, area, segmentation as COCO RLE) at the SAME path
(regions_test/output/sam_slic_masks/<obj>.json) the whole rest of the pipeline already reads from
-- so nothing downstream needs to change. partition_regions() already resolves overlapping/nested
candidate masks into disjoint parts (smallest-region-wins per pixel + cleanup), which is exactly
the raw, overlapping candidate-mask list SAM2AutomaticMaskGenerator produces -- so the raw output
is written as-is, with NO extra NMS/dedup step of our own (unlike the quick comparison script,
which hand-rolled its own dedup only because it wasn't using partition_regions downstream).

Usage:
    python3 generate_regions_sam2.py [object ...]   # default: all 20 study objects
Output: regions_test/output/sam_slic_masks/<obj>.json (overwrites the SAM1+SLIC version)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
_VENV_DIR = os.path.join(HERE, "venv_vlm")
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _venv_py = os.path.join(_VENV_DIR, "bin", "python3")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(_venv_py, [_venv_py] + sys.argv)

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

REPO_ROOT = os.path.dirname(HERE)
REGIONS_TEST = os.path.join(REPO_ROOT, "regions_test")
OUT_DIR = os.path.join(REGIONS_TEST, "output", "sam_slic_masks")

SAM2_ROOT = os.path.join(HERE, "sam2_experiment", "sam2")
sys.path.insert(0, SAM2_ROOT)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CHECKPOINT = os.path.join(HERE, "sam2_experiment", "checkpoints", "sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

KNOWN_OBJECTS = ["backpack", "bow-tie", "chair", "clock", "coffee-table",
                 "corkscrew", "fedora", "fire-alarm", "ladle", "wallet"]
NOVEL_OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
                 "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
                 "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
                 "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
                 "shape_39", "shape_40"]  # all 30 novel objects (ids are scattered 1-40, not a
                 # clean 1-30 range -- confirmed via direct dataset query, see list_novel_objects.py
STUDY_OBJECTS = KNOWN_OBJECTS + NOVEL_OBJECTS


def resolve_image_path(obj: str) -> str:
    for split in ("known", "novel"):
        p = os.path.join(REGIONS_TEST, "data", split, f"{obj}.png")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no render found for '{obj}' under regions_test/data/{{known,novel}}/")


def main():
    objects = sys.argv[1:] if len(sys.argv) > 1 else STUDY_OBJECTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    # Full default point density now that we have GPU headroom (the earlier comparison test used
    # a reduced 16x16 grid purely for CPU speed) -- denser sampling catches more small details
    # (confirmed: SAM3's sparser hand-rolled grid missed shape_1's finger-holes and one of
    # shape_2's tube arms that a denser pass would likely have caught too).
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, points_per_side=32, crop_n_layers=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for obj in objects:
        img_path = resolve_image_path(obj)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            masks = mask_generator.generate(image_np)

        regions = []
        for i, m in enumerate(masks):
            seg = m["segmentation"]
            rle = mask_utils.encode(np.asfortranarray(seg.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            regions.append({
                "region_id": f"{obj}_region_{i}",
                "area": int(seg.sum()),
                "segmentation": rle,
            })

        out_path = os.path.join(OUT_DIR, f"{obj}.json")
        with open(out_path, "w") as f:
            json.dump(regions, f)
        print(f"{obj}: {len(regions)} candidate masks -> {out_path}")


if __name__ == "__main__":
    main()
