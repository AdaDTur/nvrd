"""
Figure for affordance_context.py's output: image with numbered disjoint parts, plus one text
panel per vocabulary listing each part's top-K "<affordance>: <context>" lines.

Usage:
    [REGION_SOURCE=sam_slic|sam] python visualize_affordance_context.py [object_name]   # default: shape_16
"""
import os
import sys
import json
import textwrap

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from pycocotools import mask as mask_utils

HERE = os.path.dirname(os.path.realpath(__file__))
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
REGION_SOURCE = os.environ.get("REGION_SOURCE", "sam_slic")
OVERLAY_ALPHA = 0.5
VOCAB_TITLES = {"agd20k": "AGD20K (36 verbs)", "umd": "UMD (7 classes)"}

with open(os.path.join(HERE, f"{OBJECT_NAME}_{REGION_SOURCE}_affordance_context.json")) as f:
    data = json.load(f)
parts = data["regions"]
masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
image_t = read_image(data["image_path"])[:3]

palette = plt.get_cmap("tab10")
colors = [tuple(int(c * 255) for c in palette(i % 10)[:3]) for i in range(len(parts))]
overlay = draw_segmentation_masks(
    image_t, torch.stack([torch.from_numpy(m) for m in masks]), colors=colors, alpha=OVERLAY_ALPHA
).permute(1, 2, 0).numpy()


def anchor(mask: np.ndarray) -> tuple:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


vocabs = list(VOCAB_TITLES)
fig = plt.figure(figsize=(7 + 6.5 * len(vocabs), 7))
gs = fig.add_gridspec(1, 1 + len(vocabs), width_ratios=[1] + [1.1] * len(vocabs))

ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(overlay)
ax_img.set_xticks([])
ax_img.set_yticks([])
ax_img.set_title("Disjoint parts")
for i, (p, m) in enumerate(zip(parts, masks), start=1):
    cx, cy = anchor(m)
    ax_img.text(cx, cy, str(i), color="black", fontsize=11, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.9, pad=1.5, edgecolor="black", linewidth=0.5))

HEADER_WIDTH = 60
BODY_WIDTH = 68  # chars; tuned to the panel width set by width_ratios below -- check the render if columns
                 # are added/resized, since matplotlib's own wrap=True can't be trusted with wrapped line counts

for col, vocab in enumerate(vocabs, start=1):
    ax = fig.add_subplot(gs[0, col])
    ax.axis("off")
    ax.set_title(VOCAB_TITLES[vocab], fontsize=12)
    lines = []  # (text, fontsize, bold)
    for i, part in enumerate(parts, start=1):
        header = f"Part {i}  ({part['region_id']}, area={part['area']})"
        lines += [(t, 9.5, True) for t in textwrap.wrap(header, HEADER_WIDTH)]
        for rank, item in enumerate(part["vocabularies"][vocab], start=1):
            full = f"{rank}. {item['affordance']} ({item['prob']:.2f}): {item['context']}"
            wrapped = textwrap.wrap(full, BODY_WIDTH, initial_indent="    ", subsequent_indent="       ")
            lines += [(t, 8.8, False) for t in wrapped]
        lines.append(("", 8.8, False))
    y = 1.0
    line_h = 1.0 / max(len(lines), 1)
    for text, fontsize, bold in lines:
        ax.text(0.0, y, text, transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold" if bold else "normal", va="top", ha="left")
        y -= line_h

fig.suptitle(f"{OBJECT_NAME}  |  top-{data['top_k']} affordances per part, with context "
             f"({data['model'].split('/')[-1]}, regions: {REGION_SOURCE})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(FIGURES_DIR, f"{OBJECT_NAME}_{REGION_SOURCE}_affordance_context.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print("saved", out_path)
