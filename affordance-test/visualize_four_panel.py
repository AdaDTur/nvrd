"""
Four-panel summary: Original | Regions separated | Regions colored by naive/unnormalized
IG | Regions colored by size/shape-controlled (normalized) IG z-score.

Requires all three prior steps to have already been run for `object_name`:
    generate_regions.py            -> regions_test/output/sam_masks/<object_name>.json
    region_information_gain.py                    -> <object_name>_region_information_gain.json
    region_information_gain_size_controlled.py     -> <object_name>_region_information_gain_size_controlled.json

Usage:
    python visualize_four_panel.py [object_name]   # default: shape_16
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
NA_COLOR = (153, 153, 153)  # flat gray for regions not scored by the size-controlled analysis
# Green = informative/supports the model's stated functionality, red = uninformative or
# actively hurts it -- more intuitive than a blue<->red diverging scale, and doesn't waste
# half its range when values are skewed mostly to one sign (common here).
DIVERGING_CMAP = "RdYlGn"
MIN_LABEL_AREA_FRAC = 0.002  # skip text labels on slivers smaller than this fraction of the image

with open(os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain.json")) as f:
    naive = json.load(f)
with open(os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain_size_controlled.json")) as f:
    controlled = json.load(f)

naive_by_id = {r["region_id"]: r["information_gain_bits"] for r in naive["regions"]}
z_by_id = {r["region_id"]: r["z_score"] for r in controlled["regions"]}

img_path = resolve_image_path(OBJECT_NAME)
sam_path = os.path.join(REGIONS_TEST, "output", "sam_masks", f"{OBJECT_NAME}.json")
image_t = read_image(img_path)[:3]
with open(sam_path) as f:
    sam_regions = json.load(f)
sam_regions = sorted(sam_regions, key=lambda r: r["area"], reverse=True)  # largest first, drawn under smaller
# Keep only the regions that were actually scored (region_information_gain.py may have
# been run with a max-regions cap, e.g. for a busy photo with many small SAM regions).
sam_regions = [r for r in sam_regions if r["region_id"] in naive_by_id]

decoded_masks = [mask_utils.decode(r["segmentation"]).astype(bool) for r in sam_regions]
masks_t = torch.stack([torch.from_numpy(m) for m in decoded_masks])
black_bg = torch.zeros(3, masks_t.shape[1], masks_t.shape[2], dtype=torch.uint8)
img_area = masks_t.shape[1] * masks_t.shape[2]

# Matches region_information_gain_size_controlled.py's own background-exclusion threshold.
# A region this large (e.g. a "whole scene" SAM proposal) isn't a meaningful single part,
# and its naive IG is typically enormous just from occluding most of the image -- if left
# in, it dominates the color scale (washing out every real part to a pale, indistinct
# shade) and its label collides with whatever real region happens to share its centroid.
BACKGROUND_AREA_FRAC = 0.5
background_ids = {r["region_id"] for r in sam_regions if r["area"] / img_area >= BACKGROUND_AREA_FRAC}
if background_ids:
    print(f"Excluding {len(background_ids)} background-sized region(s) (>={BACKGROUND_AREA_FRAC:.0%} "
          f"of image) from color scale + labels: {sorted(background_ids)}")


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


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


def to_hwc(img_t: torch.Tensor) -> np.ndarray:
    return img_t.permute(1, 2, 0).numpy()


def label_regions(ax, values_by_id, fmt):
    """Place one text label per region at its centroid, nudging labels apart (with a thin
    leader line back to the true centroid) when their default positions would overlap --
    e.g. several small regions clustered next to one big one (see: the door handle
    hardware sitting on top of the main door panel)."""
    img_h, img_w = masks_t.shape[1], masks_t.shape[2]
    diag = (img_h ** 2 + img_w ** 2) ** 0.5
    min_sep = 0.045 * diag
    placed = []  # (x, y) of already-placed label positions

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
                    cand_x = cx + radius * np.cos(angle)
                    cand_y = cy + radius * np.sin(angle)
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
naive_colors, naive_norm, naive_cmap = diverging_colors(naive_by_id)
z_colors, z_norm, z_cmap = diverging_colors(z_by_id)

panel_separated = to_hwc(draw_segmentation_masks(black_bg, masks_t, colors=qualitative_colors, alpha=1.0))
panel_naive = to_hwc(draw_segmentation_masks(black_bg, masks_t, colors=naive_colors, alpha=1.0))
panel_z = to_hwc(draw_segmentation_masks(black_bg, masks_t, colors=z_colors, alpha=1.0))

fig, axs = plt.subplots(1, 4, figsize=(19, 5.5))
fig.suptitle(OBJECT_NAME, fontsize=13, y=1.04)

axs[0].imshow(to_hwc(image_t))
axs[0].set_title("Original")

axs[1].imshow(panel_separated)
axs[1].set_title("Regions separated")

axs[2].imshow(panel_naive)
axs[2].set_title("Color-coded by IG\n(unnormalized)")
label_regions(axs[2], naive_by_id, "{:+.1f}")

axs[3].imshow(panel_z)
axs[3].set_title("Color-coded by IG\n(normalized, z-score)")
label_regions(axs[3], z_by_id, "{:+.1f}")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

cb1 = fig.colorbar(cm.ScalarMappable(norm=naive_norm, cmap=naive_cmap), ax=axs[2], fraction=0.046, pad=0.04)
cb1.set_label("IG [bits]  (green = informative, red = uninformative/hurts)", fontsize=7.5)
cb2 = fig.colorbar(cm.ScalarMappable(norm=z_norm, cmap=z_cmap), ax=axs[3], fraction=0.046, pad=0.04)
cb2.set_label("z-score  (green = informative, red = uninformative/hurts)", fontsize=7.5)
axs[3].text(
    0.5, -0.08, "gray = background / not scored", transform=axs[3].transAxes,
    ha="center", va="top", fontsize=8,
)


def wrap_caption(text: str, width: int) -> str:
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(out)


nonce_note = f" (nonce='{naive['nonce_word']}')" if naive.get("nonce_word") else ""
caption = f'Model-generated functionality description{nonce_note}:\n"{naive["baseline_description"]}"'
fig.text(
    0.5, -0.02, wrap_caption(caption, width=170),
    transform=fig.transFigure, ha="center", va="top", fontsize=9,
)

out_path = os.path.join(FIGURES_DIR, f"{OBJECT_NAME}_four_panel.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("saved", out_path)

print(f"\n{'region':>24s} {'area':>7s} {'naive IG(bits)':>15s} {'z-score':>10s}")
for r in sam_regions:
    rid = r["region_id"]
    naive_v = naive_by_id.get(rid, float("nan"))
    z_v = z_by_id.get(rid, float("nan"))
    print(f"{rid:>24s} {r['area']:>7d} {naive_v:>15.3f} {'N/A' if z_v != z_v else f'{z_v:>10.2f}'}")
