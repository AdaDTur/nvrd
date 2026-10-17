"""
Four-panel visualization for rise_attribution.py's output: Original | Regions separated |
Importance (bits) | t-statistic. Both value panels are signed (diverging colormap), styled
to match visualize_four_panel.py.

Usage:
    python visualize_rise.py [object_name]   # default: shape_16
"""
import os
import sys
import json
import textwrap

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from pycocotools import mask as mask_utils

HERE = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGIONS_TEST = os.path.join(REPO_ROOT, "regions_test")
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(REGIONS_TEST, "..", "regions", "region_features", "sam_analysis"))
sys.path.insert(0, os.path.join(REGIONS_TEST, "..", "regions", "region_features"))
from visualize_sam import get_colors  # noqa: E402
from region_information_gain import IG_JSON_DIR
from region_information_gain import resolve_image_path  # noqa: E402

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
NA_COLOR = (153, 153, 153)
DIVERGING_CMAP = "RdYlGn"
MIN_LABEL_AREA_FRAC = 0.002

with open(os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_rise_attribution.json")) as f:
    data = json.load(f)

importance_by_id = {r["region_id"]: r["importance_bits"] for r in data["regions"]}
t_by_id = {r["region_id"]: r["t_stat"] for r in data["regions"]}

img_path = resolve_image_path(OBJECT_NAME)
sam_path = os.path.join(REGIONS_TEST, "output", "sam_masks", f"{OBJECT_NAME}.json")
image_t = read_image(img_path)[:3]
with open(sam_path) as f:
    sam_regions = json.load(f)
sam_regions = sorted(sam_regions, key=lambda r: r["area"], reverse=True)
# Keep ALL sam regions, including the one excluded from RISE's random draw (background,
# >=50% of image) -- it just renders NA/gray below, but the "regions separated" panel still
# shows the true full segmentation for context.

decoded_masks = [mask_utils.decode(r["segmentation"]).astype(bool) for r in sam_regions]
masks_t = torch.stack([torch.from_numpy(m) for m in decoded_masks])
black_bg = torch.zeros(3, masks_t.shape[1], masks_t.shape[2], dtype=torch.uint8)
img_area = masks_t.shape[1] * masks_t.shape[2]


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def diverging_colors(values_by_id):
    """Diverging, symmetric-log color scale.

    A plain linear scale stretched to fit one outlier (e.g. a handle region at +9.7 bits
    when most regions sit in [-2, +2]) compresses every other, still-meaningful value into
    a narrow band of similar-looking color. The first fix tried here was clipping a linear
    scale at a percentile of |values| -- but that overcorrects whenever there's a *broad
    shoulder* of large values rather than one spike (door_crop has several: 2.3, 4.4, 5.5,
    5.9, 9.7 bits), since a percentile-based vmax then saturates most of that shoulder to
    the *same* maxed-out color -- exactly the "can't tell 2.3 from 9.7" failure this was
    supposed to fix.

    SymLogNorm doesn't have that problem: it's linear within +-linthresh of zero (so small,
    real differences near zero stay visually distinguishable) and logarithmic beyond that
    (so a 2.3 and a 9.7 compress toward each other but are never mapped to the identical
    color the way a hard clip would). No two different values ever render as the same
    color, regardless of how skewed the distribution is -- exact values are still shown via
    the per-region text labels regardless."""
    vals = np.abs(np.array(list(values_by_id.values())))
    vmax = max(float(vals.max()), 1e-6)
    linthresh = max(float(np.median(vals)), vmax * 0.05, 1e-3)
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    cmap = plt.get_cmap(DIVERGING_CMAP)
    colors = []
    for r in sam_regions:
        rid = r["region_id"]
        if rid in values_by_id:
            colors.append(tuple((np.array(cmap(norm(values_by_id[rid]))[:3]) * 255).astype(int).tolist()))
        else:
            colors.append(NA_COLOR)
    return colors, norm, cmap


def to_hwc(img_t):
    return img_t.permute(1, 2, 0).numpy()


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
        if rid not in values_by_id:
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
            label_x, label_y, fmt.format(values_by_id[rid]), color="black", fontsize=7.5,
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.85, pad=0.5, edgecolor="none"),
        )


qualitative_colors = get_colors(len(sam_regions), cmap_name="rainbow", as_tuples=True)
imp_colors, imp_norm, imp_cmap = diverging_colors(importance_by_id)
t_colors, t_norm, t_cmap = diverging_colors(t_by_id)

panel_separated = to_hwc(draw_segmentation_masks(black_bg, masks_t, colors=qualitative_colors, alpha=1.0))
panel_imp = to_hwc(draw_segmentation_masks(black_bg, masks_t, colors=imp_colors, alpha=1.0))
panel_t = to_hwc(draw_segmentation_masks(black_bg, masks_t, colors=t_colors, alpha=1.0))

fig, axs = plt.subplots(1, 4, figsize=(19, 5.5))
fig.suptitle(f"{OBJECT_NAME}  |  RISE, M={data['M']} trials", fontsize=13, y=1.04)

axs[0].imshow(to_hwc(image_t))
axs[0].set_title("Original")
axs[1].imshow(panel_separated)
axs[1].set_title("Regions separated")
axs[2].imshow(panel_imp)
axs[2].set_title("RISE importance (bits)\nmean logP(visible) - mean logP(occluded)")
label_regions(axs[2], importance_by_id, "{:+.1f}")
axs[3].imshow(panel_t)
axs[3].set_title("RISE t-statistic\n(signal-to-noise across trials)")
label_regions(axs[3], t_by_id, "{:+.1f}")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

cb1 = fig.colorbar(cm.ScalarMappable(norm=imp_norm, cmap=imp_cmap), ax=axs[2], fraction=0.046, pad=0.04)
cb1.set_label("importance [bits]", fontsize=8)
cb2 = fig.colorbar(cm.ScalarMappable(norm=t_norm, cmap=t_cmap), ax=axs[3], fraction=0.046, pad=0.04)
cb2.set_label("t-statistic", fontsize=8)
fig.tight_layout()  # reflow axes/colorbars before adding fig-level (transFigure) text below

fig.text(
    0.5, -0.02,
    "green = supports D0, red = hurts/irrelevant  ·  gray = background/excluded from random draw  ·  "
    "color scale is symmetric-log (compresses extremes; exact values in per-region labels)",
    transform=fig.transFigure, ha="center", va="top", fontsize=8,
)


def wrap_caption(text, width):
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(out)


caption = f'Model-generated functionality description:\n"{data["baseline_description"]}"'
fig.text(0.5, -0.06, wrap_caption(caption, width=170), transform=fig.transFigure, ha="center", va="top", fontsize=9)

out_path = os.path.join(FIGURES_DIR, f"{OBJECT_NAME}_rise_four_panel.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("saved", out_path)
