"""
Figure for affordance_labeling.py's output. Rows = vocabularies (AGD20K, UMD). Columns:
    original / disjoint parts | forced choice (raw) | forced choice (damped contrast) |
    yes/no per affordance (raw) | yes/no per affordance (damped contrast)

Damped contrast scores each option by  log p(option | highlighted part) - alpha * log p(option | no
highlight), i.e. it divides out (a fraction alpha of) what the model says about the whole object.
For yes/no it operates on log-odds. alpha = 0 is the raw score; it is applied here, from the stored
scores, so any alpha can be tried without rerunning the model.

Yes/no label = the option with the highest (damped) log-odds; shown as "none" when that option's own
P(Yes) for the highlighted part is below 0.5.

Usage:
    [REGION_SOURCE=sam_slic|sam] python visualize_affordance_labels.py [object_name] [alpha]   # default: shape_16 0.5
"""
import os
import sys
import json

import cv2
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
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
REGION_SOURCE = os.environ.get("REGION_SOURCE", "sam_slic")  # must match the affordance_labeling.py run
OVERLAY_ALPHA = 0.5
VOCAB_TITLES = {"agd20k": "AGD20K\n(36 verbs)", "umd": "UMD\n(7 classes)"}

with open(os.path.join(HERE, f"{OBJECT_NAME}_{REGION_SOURCE}_affordance_labels.json")) as f:
    data = json.load(f)
parts = data["regions"]
masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
image_t = read_image(data["image_path"])[:3]
img_h, img_w = masks[0].shape
diag = (img_h ** 2 + img_w ** 2) ** 0.5

palette = plt.get_cmap("tab10")
colors = [tuple(int(c * 255) for c in palette(i % 10)[:3]) for i in range(len(parts))]
overlay = draw_segmentation_masks(
    image_t, torch.stack([torch.from_numpy(m) for m in masks]), colors=colors, alpha=OVERLAY_ALPHA
).permute(1, 2, 0).numpy()


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def forced_choice(vocab: str, part: dict, alpha: float) -> tuple:
    opts = data["vocabularies"][vocab]
    hl = log_softmax(np.array([part["fc_logprobs"][vocab][o] for o in opts]))
    full = log_softmax(np.array([data["object_level"][vocab]["fc_logprobs"][o] for o in opts]))
    s = hl - alpha * full
    p = np.exp(log_softmax(s))
    i = int(np.argmax(s))
    return f"{opts[i]}\n{p[i]:.2f}", opts[i]


def yes_no(vocab: str, part: dict, alpha: float) -> tuple:
    opts = [o for o in data["vocabularies"][vocab] if o != "none"]
    p_hl = np.array([part["yn_prob"][vocab][o] for o in opts])
    p_full = np.array([data["object_level"][vocab]["yn_prob"][o] for o in opts])
    i = int(np.argmax(logit(p_hl) - alpha * logit(p_full)))
    label = opts[i] if p_hl[i] >= 0.5 else "none"
    return f"{label}\np(yes)={p_hl[i]:.2f}", label


METHODS = [
    ("Forced choice\n(raw)", lambda v, p: forced_choice(v, p, 0.0)),
    (f"Forced choice\n(damped contrast, a={ALPHA})", lambda v, p: forced_choice(v, p, ALPHA)),
    ("Yes/No per affordance\n(raw)", lambda v, p: yes_no(v, p, 0.0)),
    (f"Yes/No per affordance\n(damped contrast, a={ALPHA})", lambda v, p: yes_no(v, p, ALPHA)),
]


def anchor(mask: np.ndarray) -> tuple:
    """Point deepest inside the part -- unlike the centroid, always lies on the part."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x), float(y)


anchors = [anchor(m) for m in masks]


def place_texts(ax, texts: list, fontsize: float):
    min_sep = 0.07 * diag
    placed = []

    def collides(x, y):
        return any((x - px) ** 2 + (y - py) ** 2 < min_sep ** 2 for px, py in placed)

    for (cx, cy), text in zip(anchors, texts):
        lx, ly = cx, cy
        if collides(cx, cy):
            for radius in (0.9, 1.8, 2.7, 3.6):
                free = [
                    (cx + radius * min_sep * np.cos(a), cy + radius * min_sep * np.sin(a))
                    for a in np.radians(range(0, 360, 45))
                ]
                free = [c for c in free if not collides(*c) and 0 <= c[0] < img_w and 0 <= c[1] < img_h]
                if free:
                    lx, ly = free[0]
                    break
        placed.append((lx, ly))
        if (lx, ly) != (cx, cy):
            ax.plot([cx, lx], [cy, ly], color="black", linewidth=0.6, alpha=0.6)
            ax.plot([cx], [cy], marker="o", markersize=1.5, color="black", alpha=0.8)
        ax.text(lx, ly, text, color="black", fontsize=fontsize, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.88, pad=1.0, edgecolor="none"))


vocabs = list(data["vocabularies"])
fig, axs = plt.subplots(len(vocabs), 1 + len(METHODS), figsize=(4.6 * (1 + len(METHODS)), 4.6 * len(vocabs) + 0.6))
fig.suptitle(f"{OBJECT_NAME}  |  zero-shot part affordance labels ({data['model'].split('/')[-1]}, "
             f"{data['max_side']}px, regions: {REGION_SOURCE})", fontsize=13)

axs[0, 0].imshow(image_t.permute(1, 2, 0).numpy())
axs[0, 0].set_title("Original")
axs[1, 0].imshow(overlay)
axs[1, 0].set_title("Disjoint parts (label = source SAM region #)")
place_texts(axs[1, 0], [p["region_id"].rsplit("_", 1)[1] for p in parts], 8)

print(f"\nalpha={ALPHA}")
for row, vocab in enumerate(vocabs):
    print(f"\n=== {vocab} ===  {'region':>22s} {'area':>7s} | " + " | ".join(t.replace(chr(10), ' ') for t, _ in METHODS))
    results = [[fn(vocab, p) for p in parts] for _, fn in METHODS]
    for col, ((title, _), res) in enumerate(zip(METHODS, results), start=1):
        ax = axs[row, col]
        ax.imshow(overlay)
        place_texts(ax, [r[0] for r in res], 7.5)
        if row == 0:
            ax.set_title(title, fontsize=10)
    axs[row, 0].set_ylabel(VOCAB_TITLES.get(vocab, vocab), fontsize=11)
    for i, p in enumerate(parts):
        print(f"{'':>13s}{p['region_id']:>22s} {p['area']:>7d} | " + " | ".join(res[i][1] for res in results))

for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.text(
    0.5, -0.005,
    "Parts are disjoint (each pixel -> smallest containing SAM region); background not labelled. Forced choice: number = "
    "softmax probability over the vocabulary. Yes/No: independent question per affordance, number = P(Yes) of the winning option "
    "(\"none\" if < 0.5). Damped contrast: score - a * (score for the un-highlighted object).",
    ha="center", va="top", fontsize=8.5, wrap=True,
)

out_path = os.path.join(FIGURES_DIR, f"{OBJECT_NAME}_{REGION_SOURCE}_affordance_labels.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print("saved", out_path)
