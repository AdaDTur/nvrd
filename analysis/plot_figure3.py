"""
Figure 3: Combined 2-row figure showing nonce reference generation accuracy (row 1)
and z-scored log-probability (row 2), broken down by object category.

Corresponds to Figure 3 in the NVRD paper (nonce vs vanilla + log prob by category).

Usage:
    python analysis/plot_figure3.py

Output: figures/combined_gen_prob_by_category.pdf
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_settings import (
    load_generation_data, load_probability_data,
    MODELS, PROB_MODELS, SPLIT_ORDER, SPLIT_LABELS, FIGURES_DIR,
)

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


def plot_combined_by_category(gen_df, prob_df):
    prob_models = PROB_MODELS

    # Z-score log probs per model
    prob_df = prob_df.copy()
    for model_key in prob_models:
        mask = prob_df["model"] == model_key
        vals = prob_df.loc[mask, "mean_token_log_prob"]
        mu, sigma = vals.mean(), vals.std()
        prob_df.loc[mask, "zscore_log_prob"] = (vals - mu) / sigma

    gen_splits = [s for s in SPLIT_ORDER if s in gen_df["split"].unique()]
    prob_splits = [s for s in SPLIT_ORDER if s in prob_df["split"].unique()]
    ncols = max(len(gen_splits), len(prob_splits))

    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 6.4),
                              sharex=True)
    fig.subplots_adjust(hspace=0.30, wspace=0.08, left=0.08, right=0.99,
                         top=0.88, bottom=0.10)

    # Row 0: ref gen accuracy
    for idx, split in enumerate(gen_splits):
        ax = axes[0, idx]
        sub = gen_df[gen_df["split"] == split]
        for model_key, style in MODELS.items():
            ms = sub[sub["model"] == model_key]
            if ms.empty:
                continue
            means = ms.groupby("level")["correct"].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=style["color"], marker=style["marker"],
                    markersize=3.5, linewidth=1.3)
        ax.set_title(SPLIT_LABELS.get(split, split), fontweight="bold", pad=5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xticklabels([])
        if idx == 0:
            ax.set_ylabel("Nonce Reference Usage")
        else:
            ax.set_yticklabels([])

    # Row 1: z-scored log prob
    for idx, split in enumerate(prob_splits):
        ax = axes[1, idx]
        sub = prob_df[prob_df["split"] == split]
        for model_key, style in prob_models.items():
            ms = sub[sub["model"] == model_key]
            if ms.empty:
                continue
            means = ms.groupby("level")["zscore_log_prob"].mean()
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
            ax.set_ylabel("Z-Scored Log-Prob")
        else:
            ax.set_yticklabels([])

    # Hide unused axes if column counts differ
    for row in range(2):
        splits_for_row = gen_splits if row == 0 else prob_splits
        for idx in range(len(splits_for_row), ncols):
            axes[row, idx].set_visible(False)

    # Legend: combine all models used across both rows
    all_model_keys = list(dict.fromkeys(list(MODELS.keys()) + list(prob_models.keys())))
    all_styles = {**MODELS, **prob_models}
    handles = [Line2D([0], [0], color=all_styles[k]["color"],
                       marker=all_styles[k]["marker"],
                       markersize=6, linewidth=1.3, label=all_styles[k]["label"])
               for k in all_model_keys]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=10)

    return fig


if __name__ == "__main__":
    print("Loading data...")
    gen_df = load_generation_data()
    prob_df = load_probability_data()
    print(f"  Gen: {len(gen_df)} rows, Prob: {len(prob_df)} rows")

    fig = plot_combined_by_category(gen_df, prob_df)
    out = FIGURES_DIR / "combined_gen_prob_by_category.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
