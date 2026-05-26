"""
Shared constants, data loaders, and plot helpers for NVRD analysis scripts.

Paths are resolved relative to the repo root (parent of this file's directory).
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Repo root is two levels up from this file (analysis/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
HUMAN_CSV = PROJECT_ROOT / "data" / "human_study" / "trial-results-1.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Legacy alias used by some scripts
OUT_DIR = FIGURES_DIR

MODELS = {
    "idefics3-8b":      {"label": "Idefics3 8B",      "color": "#1b9e77", "marker": "o"},
    "molmo2-8b":        {"label": "Molmo2 8B",        "color": "#d95f02", "marker": "s"},
    "qwen2-vl-7b":      {"label": "Qwen2-VL 7B",      "color": "#7570b3", "marker": "^"},
    "gemini-2.5-flash": {"label": "Gemini 2.5 Flash", "color": "#e7298a", "marker": "D"},
    "gpt-4o-mini":      {"label": "GPT-4o-mini",      "color": "#66a61e", "marker": "v"},
}

PROB_MODELS = {k: v for k, v in MODELS.items() if k in ("idefics3-8b", "molmo2-8b", "qwen2-vl-7b")}

PTYPE_LABELS = {
    "add": "Part Addition",
    "background": "Background",
    "color": "Color Shift",
    "jpeg": "JPEG",
    "noise": "Noise",
    "pixelate": "Pixelation",
    "remove": "Part Removal",
    "scale": "Scale",
    "shape": "Shape Deform.",
    "style": "Style Degrad.",
    "texture": "Texture",
}

SPLIT_ORDER = ["known", "shape-texture", "shape-shape", "novel"]
SPLIT_LABELS = {
    "known": "Known", "novel": "Novel",
    "shape-shape": "Shape-Shape", "shape-texture": "Shape-Texture",
}


def load_jsonl(fp):
    with open(fp) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_generation_data():
    rows = []
    seen = set()
    for model_key in MODELS:
        fps = sorted(glob.glob(str(RESULTS_DIR / "generation" / f"{model_key}-visual_similarity-fillin*-outputs.jsonl")))
        for fp in fps:
            for r in load_jsonl(fp):
                if r.get("n") != 4:
                    continue
                key = (model_key, r.get("object", ""), r["perturbation_type"], r["level"])
                if key in seen:
                    continue
                seen.add(key)
                nonce = r.get("nonce_word", r.get("ref", ""))
                response = r.get("model_response", "") or ""
                correct = 1 if nonce and nonce.lower() in response.lower() else 0
                split = r.get("type", r.get("split", "unknown"))
                split = split.replace("modified/", "").replace("/perturbations", "")
                rows.append({
                    "model": model_key,
                    "object": r.get("object", ""),
                    "split": split,
                    "perturbation_type": r["perturbation_type"],
                    "level": r["level"],
                    "correct": correct,
                })
    return pd.DataFrame(rows)


def load_probability_data():
    rows = []
    seen = set()
    for model_key in PROB_MODELS:
        for pattern in [
            f"{model_key}-visual_similarity-fillin_results.jsonl",
            f"{model_key}-visual_similarity-fillin-novel_results.jsonl",
        ]:
            fp = RESULTS_DIR / "probability" / pattern
            if not fp.exists():
                continue
            for r in load_jsonl(str(fp)):
                if r.get("n") != 4:
                    continue
                lp = r.get("nonce_word_log_prob")
                if lp is None or not np.isfinite(lp):
                    continue
                key = (model_key, r.get("object_name", ""), r["perturbation_type"], r["level"])
                if key in seen:
                    continue
                seen.add(key)
                n_tokens = r.get("nonce_word_num_tokens", 1)
                split = r.get("split", "unknown")
                split = split.replace("modified/", "").replace("/perturbations", "")
                rows.append({
                    "model": model_key,
                    "object": r.get("object_name", ""),
                    "split": split,
                    "perturbation_type": r["perturbation_type"],
                    "level": r["level"],
                    "log_prob": lp,
                    "mean_token_log_prob": lp / max(n_tokens, 1),
                })
    return pd.DataFrame(rows)


def load_nonce_vanilla_probability_data():
    """Load both nonce and vanilla per-token-averaged log probs for comparison."""
    rows = []
    seen = set()
    for model_key in PROB_MODELS:
        for pattern in [
            f"{model_key}-visual_similarity-fillin_results.jsonl",
            f"{model_key}-visual_similarity-fillin-novel_results.jsonl",
        ]:
            fp = RESULTS_DIR / "probability" / pattern
            if not fp.exists():
                continue
            for r in load_jsonl(str(fp)):
                if r.get("n") != 4:
                    continue
                nonce_lp = r.get("nonce_word_log_prob")
                if nonce_lp is None or not np.isfinite(nonce_lp):
                    continue
                key = (model_key, r.get("object_name", ""), r["perturbation_type"], r["level"])
                if key in seen:
                    continue
                seen.add(key)
                nonce_ntok = max(r.get("nonce_word_num_tokens", 1), 1)
                split = r.get("split", "unknown").replace("modified/", "").replace("/perturbations", "")
                base = {
                    "model": model_key,
                    "object": r.get("object_name", ""),
                    "split": split,
                    "perturbation_type": r["perturbation_type"],
                    "level": r["level"],
                }
                rows.append({**base, "word_type": "nonce",
                             "mean_token_log_prob": nonce_lp / nonce_ntok})
                for vm in r.get("vanilla_mappings", []):
                    vlp = vm.get("vanilla_word_log_prob")
                    if vlp is None or not np.isfinite(vlp):
                        continue
                    vntok = max(vm.get("vanilla_word_num_tokens", 1), 1)
                    rows.append({**base, "word_type": "vanilla",
                                 "mean_token_log_prob": vlp / vntok})
    return pd.DataFrame(rows)


def load_rating_data():
    rows = []
    seen = set()
    for model_key in MODELS:
        fps = sorted(glob.glob(str(RESULTS_DIR / "prolific_style" / f"{model_key}-prolific-style-ratings*.jsonl")))
        for fp in fps:
            for r in load_jsonl(fp):
                if r.get("rating") is None or r.get("perturbation_type") == "attention_check":
                    continue
                key = (model_key, r.get("object", ""), r.get("perturbation_type", ""), r.get("level", 0))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "model": model_key,
                    "object": r["object"],
                    "perturbation_type": r["perturbation_type"],
                    "level": r["level"],
                    "rating": r["rating"],
                })
    return pd.DataFrame(rows)


def plot_nonce_vanilla_prob_faceted(data_df, models=PROB_MODELS):
    """Faceted line plot comparing nonce vs vanilla mean per-token log-prob."""
    all_ptypes = sorted(PTYPE_LABELS.keys())
    ncols, nrows = 6, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5),
                              sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.08, left=0.06, right=0.99,
                         top=0.85, bottom=0.10)

    for idx, pt in enumerate(all_ptypes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        sub = data_df[data_df["perturbation_type"] == pt]

        for model_key, style in models.items():
            ms = sub[sub["model"] == model_key]
            if ms.empty:
                continue
            for wt, ls in [("nonce", "-"), ("vanilla", "--")]:
                wt_sub = ms[ms["word_type"] == wt]
                if wt_sub.empty:
                    continue
                means = wt_sub.groupby("level")["mean_token_log_prob"].mean()
                levels = sorted(means.index)
                ax.plot(levels, [means[l] for l in levels],
                        color=style["color"], marker=style["marker"],
                        markersize=3.5, linewidth=1.3, linestyle=ls)

        ax.set_title(PTYPE_LABELS[pt], fontweight="bold", pad=5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, alpha=0.15, linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Mean Per-Token Log-Prob")
        if row == nrows - 1:
            ax.set_xlabel("Perturbation Level")

    axes[1, 5].set_visible(False)

    model_handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                             markersize=6, linewidth=1.3, label=s["label"])
                     for s in models.values()]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.3, label="Nonce"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.3, label="Vanilla"),
    ]
    fig.legend(handles=model_handles + style_handles, loc="upper center",
               ncol=len(models) + 2, frameon=False,
               bbox_to_anchor=(0.5, 0.98), fontsize=10)

    return fig


def plot_nonce_vanilla_prob_by_category(data_df, models=PROB_MODELS):
    """Line plot comparing nonce vs vanilla mean per-token log-prob by category."""
    present_splits = [s for s in SPLIT_ORDER if s in data_df["split"].unique()]
    ncols = len(present_splits)

    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.4),
                              sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99,
                         top=0.72, bottom=0.18)

    for idx, split in enumerate(present_splits):
        ax = axes[idx]
        sub = data_df[data_df["split"] == split]

        for model_key, style in models.items():
            ms = sub[sub["model"] == model_key]
            if ms.empty:
                continue
            for wt, ls in [("nonce", "-"), ("vanilla", "--")]:
                wt_sub = ms[ms["word_type"] == wt]
                if wt_sub.empty:
                    continue
                means = wt_sub.groupby("level")["mean_token_log_prob"].mean()
                levels = sorted(means.index)
                ax.plot(levels, [means[l] for l in levels],
                        color=style["color"], marker=style["marker"],
                        markersize=3.5, linewidth=1.3, linestyle=ls)

        ax.set_title(SPLIT_LABELS.get(split, split), fontweight="bold", pad=5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Perturbation Level")

        if idx == 0:
            ax.set_ylabel("Mean Per-Token Log-Prob")

    model_handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                             markersize=6, linewidth=1.3, label=s["label"])
                     for s in models.values()]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.3, label="Nonce"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.3, label="Vanilla"),
    ]
    fig.legend(handles=model_handles + style_handles, loc="upper center",
               ncol=len(models) + 2, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=10)

    return fig


def plot_faceted(data_df, y_col, y_label, models=MODELS):
    all_ptypes = sorted(PTYPE_LABELS.keys())
    ncols, nrows = 6, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5),
                              sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.08, left=0.06, right=0.99,
                         top=0.88, bottom=0.10)

    for idx, pt in enumerate(all_ptypes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        sub = data_df[data_df["perturbation_type"] == pt]

        for model_key, style in models.items():
            ms = sub[sub["model"] == model_key]
            if ms.empty:
                continue
            means = ms.groupby("level")[y_col].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=style["color"], marker=style["marker"],
                    markersize=3.5, linewidth=1.3)

        ax.set_title(PTYPE_LABELS[pt], fontweight="bold", pad=5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, alpha=0.15, linewidth=0.5)

        if col == 0:
            ax.set_ylabel(y_label)
        if row == nrows - 1:
            ax.set_xlabel("Perturbation Level")

    axes[1, 5].set_visible(False)

    handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                       markersize=6, linewidth=1.3, label=s["label"])
               for s in models.values()]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(models), 5),
               frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=11)

    return fig


def plot_faceted_by_category(data_df, y_col, y_label, models=MODELS):
    present_splits = [s for s in SPLIT_ORDER if s in data_df["split"].unique()]
    ncols = len(present_splits)

    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.4),
                              sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99,
                         top=0.75, bottom=0.18)

    for idx, split in enumerate(present_splits):
        ax = axes[idx]
        sub = data_df[data_df["split"] == split]

        for model_key, style in models.items():
            ms = sub[sub["model"] == model_key]
            if ms.empty:
                continue
            means = ms.groupby("level")[y_col].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=style["color"], marker=style["marker"],
                    markersize=3.5, linewidth=1.3)

        ax.set_title(SPLIT_LABELS.get(split, split), fontweight="bold", pad=5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Perturbation Level")

        if idx == 0:
            ax.set_ylabel(y_label)

    handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                       markersize=6, linewidth=1.3, label=s["label"])
               for s in models.values()]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(models), 5),
               frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=11)

    return fig
