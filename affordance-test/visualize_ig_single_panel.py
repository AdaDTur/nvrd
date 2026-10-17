"""
Single-panel version of visualize_four_panel.py: just the object colored by naive (unnormalized)
information gain, with per-region bit labels. No Original/Regions-separated/z-score panels.

Requires region_information_gain.py to have already been run for `object_name`.

Usage:
    python visualize_ig_single_panel.py [object_name]   # default: shape_16
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from pycocotools import mask as mask_utils

from region_information_gain import IG_JSON_DIR
from region_information_gain import resolve_image_path

HERE = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGIONS_TEST = os.path.join(REPO_ROOT, "regions_test")
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
NA_COLOR = (153, 153, 153)
DIVERGING_CMAP = "RdYlGn"
MIN_LABEL_AREA_FRAC = 0.002
BACKGROUND_AREA_FRAC = 0.5

with open(os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain.json")) as f:
    naive = json.load(f)
naive_by_id = {r["region_id"]: r["information_gain_bits"] for r in naive["regions"]}

img_path = resolve_image_path(OBJECT_NAME)
sam_path = os.path.join(REGIONS_TEST, "output", "sam_masks", f"{OBJECT_NAME}.json")
image_t = read_image(img_path)[:3]
with open(sam_path) as f:
    sam_regions = json.load(f)
sam_regions = sorted(sam_regions, key=lambda r: r["area"], reverse=True)
sam_regions = [r for r in sam_regions if r["region_id"] in naive_by_id]

decoded_masks = [mask_utils.decode(r["segmentation"]).astype(bool) for r in sam_regions]
masks_t = torch.stack([torch.from_numpy(m) for m in decoded_masks])
img_area = masks_t.shape[1] * masks_t.shape[2]

background_ids = {r["region_id"] for r in sam_regions if r["area"] / img_area >= BACKGROUND_AREA_FRAC}


def diverging_colors(values_by_id):
    vals = [v for rid, v in values_by_id.items() if rid not in background_ids]
    vmax = max(abs(min(vals)), abs(max(vals)), 1e-6)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(DIVERGING_CMAP)
    colors = []
    for r in sam_regions:
        rid = r["region_id"]
        if rid in values_by_id and rid not in background_ids:
            colors.append(tuple((np.array(cmap(norm(values_by_id[rid]))[:3]) * 255).astype(int).tolist()))
        else:
            colors.append(NA_COLOR)
    return colors, norm, cmap


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def label_regions(ax, values_by_id, fmt):
    img_h, img_w = masks_t.shape[1], masks_t.shape[2]
    diag = (img_h ** 2 + img_w ** 2) ** 0.5
    min_sep = 0.045 * diag
    placed = []

    def collides(x, y):
        return any((x - px) ** 2 + (y - py) ** 2 < min_sep ** 2 for px, py in placed)

    for r, mask in zip(sam_regions, decoded_masks):
        if r["area"] / img_area < MIN_LABEL_AREA_FRAC:
            continue
        rid = r["region_id"]
        if rid not in values_by_id or rid in background_ids:
            continue
        cx, cy = centroid(mask)
        label_x, label_y = cx, cy
        if collides(cx, cy):
            for radius in (min_sep * 0.9, min_sep * 1.8, min_sep * 2.7, min_sep * 3.6):
                found = False
                for angle_deg in range(0, 360, 45):
                    angle = np.radians(angle_deg)
                    cand_x, cand_y = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
                    if not collides(cand_x, cand_y):
                        label_x, label_y = cand_x, cand_y
                        found = True
                        break
                if found:
                    break
        placed.append((label_x, label_y))
        if (label_x, label_y) != (cx, cy):
            ax.plot([cx, label_x], [cy, label_y], color="black", linewidth=0.6, alpha=0.6)
            ax.plot([cx], [cy], marker="o", markersize=1.5, color="black", alpha=0.8)
        ax.text(
            label_x, label_y, fmt.format(values_by_id[rid]), color="black", fontsize=8,
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.85, pad=0.5, edgecolor="none"),
        )


black_bg = torch.zeros(3, masks_t.shape[1], masks_t.shape[2], dtype=torch.uint8)
naive_colors, naive_norm, naive_cmap = diverging_colors(naive_by_id)
panel_naive = draw_segmentation_masks(black_bg, masks_t, colors=naive_colors, alpha=1.0).permute(1, 2, 0).numpy()

fig, ax = plt.subplots(1, 1, figsize=(6, 6.5))
ax.imshow(panel_naive)
ax.set_title(f"{OBJECT_NAME}  |  information gain (naive, bits)", fontsize=12)
label_regions(ax, naive_by_id, "{:+.1f}")
ax.set_xticks([])
ax.set_yticks([])

cb = fig.colorbar(cm.ScalarMappable(norm=naive_norm, cmap=naive_cmap), ax=ax, fraction=0.046, pad=0.04)
cb.set_label("IG [bits]  (green = informative, red = uninformative/hurts)", fontsize=7.5)

fig.tight_layout()
out_path = os.path.join(FIGURES_DIR, f"{OBJECT_NAME}_ig_single_panel.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("saved", out_path)
