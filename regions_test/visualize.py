"""
Visualize SAM regions (from generate_regions.py) overlaid on an NVRD image.

Usage:
    python visualize.py [object_name]   # default: shape_16

Expects, relative to this script:
    data/novel/<object_name>.png                    input image
    output/sam_masks/<object_name>.json              SAM masks (COCO RLE, from generate_regions.py)
Writes:
    output/<object_name>_regions_viz.png
"""
import sys, os

HERE = os.path.dirname(os.path.realpath(__file__))
REGIONS_CODE = os.path.join(HERE, "..", "regions", "region_features")
sys.path.insert(0, REGIONS_CODE)
sys.path.insert(0, os.path.join(REGIONS_CODE, "sam_analysis"))

import torch
from torchvision.io import read_image
from pycocotools import mask as mask_utils
import region_utils as utils
from visualize_sam import image_from_masks, show

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"

img_path = os.path.join(HERE, "data", "novel", f"{OBJECT_NAME}.png")
sam_path = os.path.join(HERE, "output", "sam_masks", f"{OBJECT_NAME}.json")

image = read_image(img_path)[:3]  # drop alpha if present
regions = utils.open_file(sam_path)
regions = sorted(regions, key=lambda r: r["area"], reverse=True)

masks = [mask_utils.decode(r["segmentation"]) for r in regions]
masks = torch.stack([torch.tensor(m) for m in masks]).to(torch.bool)

overlay = image_from_masks(masks, combine_as_binary_mask=False, superimpose_on_image=image, cmap="rainbow")
separate = image_from_masks(masks, combine_as_binary_mask=False, superimpose_on_image=None, cmap="rainbow")

fig, axs = show(
    [image, overlay, separate],
    title=f"SAM regions for {OBJECT_NAME}.png ({len(regions)} regions, sorted by area desc)",
    title_y=1.08,
    subplot_titles=["Original", "Regions overlaid on image", "Regions only"],
    fig_kwargs={"figsize": (15, 5)},
)
out_path = os.path.join(HERE, "output", f"{OBJECT_NAME}_regions_viz.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("saved", out_path)

# per-region area/id table
for i, r in enumerate(regions):
    print(i, r["region_id"], "area", r["area"])
