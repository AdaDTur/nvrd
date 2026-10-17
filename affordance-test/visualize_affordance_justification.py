"""
Figure for affordance_justification.py's output: image with numbered disjoint parts on the left,
text panel(s) on the right listing each part's raw forced-choice top-1 label plus its one-sentence
justification (instead of a top-K list).

Usage:
    python visualize_affordance_justification.py [object_name]   # default: ladle
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

from affordance_labeling import AFFORDANCE_JUSTIFICATION_DIR

HERE = os.path.dirname(os.path.realpath(__file__))
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "ladle"
OVERLAY_ALPHA = 0.5
VOCAB_TITLES = {"agd20k": "AGD20K (36 verbs)"}
HEADER_WIDTH = 26
BODY_WIDTH = 26

with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{OBJECT_NAME}_affordance_justification.json")) as f:
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
FIG_WIDTH = 7 + 2.2 * len(vocabs)
VOCAB_RATIO = 0.4
TEXT_LINE_HEIGHT_IN = 0.17  # fixed physical line spacing -- independent of how many lines there are,
                            # so text density never changes how squished individual lines look

# Precompute wrapped lines first: the figure height must be DERIVED from the line count (at a fixed
# inches-per-line), not the other way around. Deriving line spacing from a fixed figure height (as an
# earlier version of this script did) squishes lines together whenever wrapping produces more of them
# (e.g. after narrowing the text column) -- spacing has to stay physically constant instead.
vocab_lines = {}
for vocab in vocabs:
    lines = []
    for i, part in enumerate(parts, start=1):
        v = part["vocabularies"][vocab]
        header = f"Part {i}  ({part['region_id']}, area={part['area']})  -->  {v['label']}"
        lines += [(t, 9.5, True) for t in textwrap.wrap(header, HEADER_WIDTH)]
        wrapped = textwrap.wrap(v["justification"], BODY_WIDTH, initial_indent="    ", subsequent_indent="    ")
        lines += [(t, 9, False) for t in wrapped]
        lines.append(("", 9, False))
    vocab_lines[vocab] = lines

n_lines = max(len(v) for v in vocab_lines.values())
text_height_in = n_lines * TEXT_LINE_HEIGHT_IN + 0.5  # + title margin
image_width_in = FIG_WIDTH * (1 / (1 + VOCAB_RATIO * len(vocabs)))
image_height_in = image_width_in * (overlay.shape[0] / overlay.shape[1]) + 0.5  # keep square-ish aspect
fig_height = max(text_height_in, image_height_in, 3.0)

fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
gs = fig.add_gridspec(1, 1 + len(vocabs), width_ratios=[1] + [VOCAB_RATIO] * len(vocabs))

ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(overlay)
ax_img.set_xticks([])
ax_img.set_yticks([])
ax_img.set_title("Disjoint parts")
for i, (p, m) in enumerate(zip(parts, masks), start=1):
    cx, cy = anchor(m)
    ax_img.text(cx, cy, str(i), color="black", fontsize=11, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.9, pad=1.5, edgecolor="black", linewidth=0.5))

line_h = TEXT_LINE_HEIGHT_IN / fig_height
for col, vocab in enumerate(vocabs, start=1):
    ax = fig.add_subplot(gs[0, col])
    ax.axis("off")
    ax.set_title(VOCAB_TITLES[vocab], fontsize=12)
    y = 1.0
    for text, fontsize, bold in vocab_lines[vocab]:
        ax.text(0.0, y, text, transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold" if bold else "normal", va="top", ha="left")
        y -= line_h

fig.suptitle(f"{OBJECT_NAME}  |  forced-choice raw label + one-sentence justification "
             f"({data['model'].split('/')[-1]}, regions: {data['region_source']})", fontsize=13)
fig.tight_layout()

out_path = os.path.join(FIGURES_DIR, f"{OBJECT_NAME}_affordance_justification.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print("saved", out_path)
