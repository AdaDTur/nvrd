"""
Full per-region study assets for every novel object the size-annotation groups cover (not just
shape_16, which already had these from earlier work) -- for each object:
  1. <object>_numbered.png              -- all regions outlined/numbered (numbered_overlay)
  2. <object>_highlight_part<N>.png     -- one per region, that region highlighted/dimmed
  3. json/study_manifests/<object>.json -- {vocabulary, models, regions: [{part, region_id, labels}]}
     merging all 4 VLMs' forced-choice labels (json/affordance_justification_<model>/ in this repo),
     same shape as the shape_16 example_manifest.json already used by the Prolific study's demo.js,
     generalized to every object instead of being hand-built for just one.

Usage:
    python3 generate_study_object_assets.py [object ...]   # default: all 10 current novel objects
Output: figures/study_images/<object>_numbered.png, <object>_highlight_part<N>.png
        json/study_manifests/<object>.json
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

from PIL import Image

from affordance_labeling import AGD20K_VERBS, downscale, highlight_region, numbered_overlay, partition_regions
from region_information_gain import load_regions, resolve_image_path

IMG_OUT_DIR = os.path.join(HERE, "figures", "study_images")
MANIFEST_OUT_DIR = os.path.join(HERE, "json", "study_manifests")
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MANIFEST_OUT_DIR, exist_ok=True)

MAX_REGIONS = 30
REGION_SOURCE = "sam_slic"
VOCABULARY = [v.replace("_", " ") for v in AGD20K_VERBS] + ["none"]  # same list batch_final_labeling.py uses

MODELS = ["qwen3-vl-8b", "internvl3-8b", "molmo2-8b"]  # qwen2-vl-7b dropped, outdated
JUSTIFICATION_DIRS = {
    "qwen3-vl-8b": os.path.join(HERE, "json", "affordance_justification_qwen3-vl-8b"),
    "internvl3-8b": os.path.join(HERE, "json", "affordance_justification_internvl3-8b"),
    "molmo2-8b": os.path.join(HERE, "json", "affordance_justification_molmo2-8b"),
}

DEFAULT_OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
                   "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
                   "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
                   "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
                   "shape_39", "shape_40"]  # all 30 novel objects (ids scattered 1-40)


def generate_images(obj: str) -> list:
    """Returns the ordered list of region_ids (index i -> part i+1, matching the *_highlight_part<N>
    filenames) -- the manifest step below needs this exact order/numbering to line up."""
    regions = load_regions(obj, MAX_REGIONS, REGION_SOURCE)
    img_h, img_w = regions[0]["segmentation"]["size"]
    img_path = resolve_image_path(obj)
    image = Image.open(img_path).convert("RGB")
    parts = partition_regions(regions, img_h * img_w, image)
    masks = [m for _, m in parts]
    region_ids = [r["region_id"] for r, _ in parts]

    numbered = downscale(numbered_overlay(image, masks))
    numbered.save(os.path.join(IMG_OUT_DIR, f"{obj}_numbered.png"))

    for i, mask in enumerate(masks, start=1):
        highlight = downscale(highlight_region(image, mask))
        highlight.save(os.path.join(IMG_OUT_DIR, f"{obj}_highlight_part{i}.png"))

    return region_ids


def build_manifest(obj: str, region_ids: list) -> dict:
    labels_by_model = {}
    for model in MODELS:
        path = os.path.join(JUSTIFICATION_DIRS[model], f"{obj}_affordance_justification.json")
        with open(path) as f:
            data = json.load(f)
        labels_by_model[model] = {
            r["region_id"]: r["vocabularies"]["agd20k"]["label"] for r in data["regions"]
        }

    regions = []
    for i, region_id in enumerate(region_ids, start=1):
        regions.append({
            "part": i,
            "region_id": region_id,
            "labels": {m: labels_by_model[m].get(region_id, "none") for m in MODELS},
        })
    return {"vocabulary": VOCABULARY, "models": MODELS, "regions": regions}


def main():
    objects = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_OBJECTS
    for obj in objects:
        print(f"=== {obj} ===")
        region_ids = generate_images(obj)
        print(f"  images: {len(region_ids)} parts -> {IMG_OUT_DIR}/{obj}_numbered.png + "
              f"{len(region_ids)} highlight images")

        manifest = build_manifest(obj, region_ids)
        out_path = os.path.join(MANIFEST_OUT_DIR, f"{obj}.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  manifest: {len(manifest['regions'])} regions -> {out_path}")


if __name__ == "__main__":
    main()
