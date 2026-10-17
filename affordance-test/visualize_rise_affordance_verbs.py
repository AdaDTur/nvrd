"""
Figure for rise_affordance_verbs.py's output: numbered disjoint parts on the left, and a part x verb
heatmap of RISE importance (bits) on the right.

  green = the part being visible raises that verb's probability (part supports the verb)
  red   = the part being visible lowers it
  black box = the part's direct forced-choice label      star = the verb RISE says the part supports most

Columns are the verbs that matter for this object: each part's top-3 RISE verbs, each part's direct label,
and the whole-object top-3 verbs on the full image (ordered by their strongest effect on any part).

Usage:
    python visualize_rise_affordance_verbs.py [--loo] object [object ...]
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from pycocotools import mask as mask_utils

from affordance_labeling import AFFORDANCE_JUSTIFICATION_DIR, JSON_DIR

OBJECTS = sys.argv[1:]

HERE = os.path.dirname(os.path.realpath(__file__))
LOO = "--loo" in sys.argv
OBJECTS = [a for a in OBJECTS if a != "--loo"]
METHOD = "loo" if LOO else "rise"
RISE_DIR = os.path.join(JSON_DIR, "loo_affordance_verbs" if LOO else "rise_affordance_verbs")
FIGURES_DIR = os.path.join(HERE, "figures", "loo_affordance_verbs" if LOO else "rise_affordance_verbs")
os.makedirs(FIGURES_DIR, exist_ok=True)
TOP_K = 3
LINTHRESH = 0.5  # symlog: linear within +-0.5 bits so small effects stay visible, log beyond so one big
                 # effect doesn't wash out the rest


def render(object_name: str):
    with open(os.path.join(RISE_DIR, f"{object_name}_{METHOD}_affordance_verbs.json")) as f:
        rise = json.load(f)
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{object_name}_affordance_justification.json")) as f:
        just = json.load(f)
    seg_by_id = {r["region_id"]: r["segmentation"] for r in just["regions"]}
    parts = rise["regions"]
    masks = [mask_utils.decode(seg_by_id[p["region_id"]]).astype(bool) for p in parts]
    image_t = read_image(just["image_path"])[:3]

    selected = set()
    for p in parts:
        selected.update(sorted(p["importance_bits"], key=p["importance_bits"].get, reverse=True)[:TOP_K])
        selected.add(p["forced_choice_label"])
    selected.update(sorted(rise["full_image_probs"], key=rise["full_image_probs"].get, reverse=True)[:TOP_K])
    verbs = sorted(selected, key=lambda v: -max(abs(p["importance_bits"][v]) for p in parts))
    mat = np.array([[p["importance_bits"][v] for v in verbs] for p in parts])
    vmax = max(float(np.abs(mat).max()), 1.0)
    norm = mcolors.SymLogNorm(linthresh=LINTHRESH, vmin=-vmax, vmax=vmax, base=10)
    cmap = plt.get_cmap("RdYlGn")

    palette = plt.get_cmap("tab10")
    colors = [tuple(int(c * 255) for c in palette(i % 10)[:3]) for i in range(len(parts))]
    overlay = draw_segmentation_masks(
        image_t, torch.stack([torch.from_numpy(m) for m in masks]), colors=colors, alpha=0.5
    ).permute(1, 2, 0).numpy()

    n_rows, n_cols = mat.shape
    fig_w = 5.0 + max(5.0, 0.62 * n_cols + 2.6)
    fig_h = max(5.2, 0.5 * n_rows + 2.6)
    fig, (ax_img, ax_hm) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [5.0, max(5.0, 0.62 * n_cols + 2.6)]}
    )
    ax_img.imshow(overlay)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("Disjoint parts")
    for i, m in enumerate(masks, start=1):
        ys, xs = np.nonzero(m)
        ax_img.text(xs.mean(), ys.mean(), str(i), fontsize=11, fontweight="bold", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.9, pad=1.5, edgecolor="black", linewidth=0.5))

    ax_hm.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax_hm.set_xticks(range(n_cols))
    ax_hm.set_xticklabels(verbs, rotation=45, ha="right", fontsize=9)
    ax_hm.set_yticks(range(n_rows))
    ax_hm.set_yticklabels([f"{i}  ({p['region_id'].replace(object_name + '_', '')})"
                           for i, p in enumerate(parts, start=1)], fontsize=9)
    for r in range(n_rows):
        for c in range(n_cols):
            ax_hm.text(c, r, f"{mat[r, c]:+.1f}", ha="center", va="center", fontsize=7.5, color="black")
        best = int(np.argmax(mat[r]))
        ax_hm.text(best + 0.33, r - 0.28, "★", ha="center", va="center", fontsize=10, color="black")
        fc = parts[r]["forced_choice_label"]
        if fc in verbs:
            c = verbs.index(fc)
            ax_hm.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.2))
    ax_hm.set_title(f"{'Leave-one-out' if LOO else 'RISE'} importance per verb (bits)   ■ box = direct label   "
                    f"★ = top verb", fontsize=10)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_hm, fraction=0.04, pad=0.02,
                 label="bits (symlog)")

    fp = rise["full_image_probs"]
    top_full = ", ".join(f"{v} {fp[v]:.2f}" for v in sorted(fp, key=fp.get, reverse=True)[:3])
    method_name = "leave-one-out over affordance verbs" if LOO else f"RISE over affordance verbs (M={rise['M']})"
    fig.suptitle(f"{object_name}  |  {method_name}   |   whole-object answer on full "
                 f"image: {top_full}", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{object_name}_{METHOD}_affordance_verbs.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


for name in OBJECTS:
    render(name)
