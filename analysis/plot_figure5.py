"""
Figure 5: Human vs model Likert rating comparison by object category.

Corresponds to Figure 5 in the NVRD paper.

Usage:
    python analysis/plot_figure5.py

Output (in figures/):
  - fig_cat_model_ratings.pdf          — all-model ratings by category
  - fig_cat_model_ratings_molmo2.pdf   — Molmo2 only
  - fig_cat_human_comparison.pdf       — human + all-model overlay
  - fig_cat_human_comparison_molmo2.pdf
  - scatter_human_vs_model_by_obj_cat.pdf
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_settings import (
    RESULTS_DIR, HUMAN_CSV, FIGURES_DIR,
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

MODELS = {
    "idefics3-8b":      {"label": "Idefics3 8B",      "color": "#1b9e77", "marker": "o"},
    "molmo2-8b":        {"label": "Molmo2 8B",        "color": "#d95f02", "marker": "s"},
    "qwen2-vl-7b":      {"label": "Qwen2-VL 7B",      "color": "#7570b3", "marker": "^"},
    "gemini-2.5-flash": {"label": "Gemini 2.5 Flash", "color": "#e7298a", "marker": "D"},
    "gpt-4o-mini":      {"label": "GPT-4o-mini",      "color": "#66a61e", "marker": "v"},
}
HUMAN_STYLE = {"label": "Humans (n=30)", "color": "#222222", "marker": "X"}
MOLMO_ONLY = {"molmo2-8b": MODELS["molmo2-8b"]}

CATEGORY_ORDER = ["known", "shape-shape", "shape-texture", "novel"]
CATEGORY_LABELS = {
    "known": "Known",
    "shape-shape": "Shape-Shape",
    "shape-texture": "Shape-Texture",
    "novel": "Novel",
}

HUMAN_CAT_MAP = {
    "known": "known",
    "modified/shape-shape": "shape-shape",
    "modified/shape-texture": "shape-texture",
    "novel": "novel",
}


def load_jsonl(fp):
    with open(fp) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model_data():
    rows = []
    for model_key in MODELS:
        fps = sorted(glob.glob(str(RESULTS_DIR / "prolific_style" / f"{model_key}-prolific-style-ratings*.jsonl")))
        if not fps:
            print(f"  [WARN] No files for {model_key}")
            continue
        seen = set()
        for fp in fps:
            for r in load_jsonl(fp):
                if r.get("rating") is None:
                    continue
                key = (model_key, r["object"], r["perturbation_type"], r["level"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "model": model_key,
                    "object": r["object"],
                    "split": r["split"],
                    "perturbation_type": r["perturbation_type"],
                    "level": r["level"],
                    "rating": r["rating"],
                })
    return pd.DataFrame(rows)


def load_human_data():
    df = pd.read_csv(HUMAN_CSV)
    df = df[df["category"].notna()].copy()
    df = df[~df["stimulus_id"].str.startswith("attention_check", na=False)]
    df = df.dropna(subset=["level", "rating"])
    df["level"] = df["level"].astype(int)
    df.rename(columns={"edit_type": "perturbation_type", "stimulus_id": "object"}, inplace=True)
    df["split"] = df["category"].map(HUMAN_CAT_MAP)
    return df


def plot_by_category_models(model_df, models=None):
    if models is None:
        models = MODELS
    ncols = len(CATEGORY_ORDER)
    fig, axes = plt.subplots(1, ncols, figsize=(14, 3.5),
                              sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.99,
                         top=0.82, bottom=0.16)

    for idx, cat in enumerate(CATEGORY_ORDER):
        ax = axes[idx]
        sub = model_df[model_df["split"] == cat]

        for model_key, style in models.items():
            ms = sub[sub["model"] == model_key]
            means = ms.groupby("level")["rating"].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=style["color"], marker=style["marker"],
                    markersize=4, linewidth=1.3)

        ax.set_title(CATEGORY_LABELS[cat], fontweight="bold", pad=5)
        ax.set_ylim(0.5, 7.5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_yticks([1, 3, 5, 7])
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Perturbation Level")

    axes[0].set_ylabel("Mean Rating")

    handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                       markersize=6, linewidth=1.3, label=s["label"])
               for s in models.values()]
    fig.legend(handles=handles, loc="upper center", ncol=len(models),
               frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=11)

    return fig


def plot_by_category_human(model_df, human_df, models=None):
    if models is None:
        models = MODELS
    h_objects = set(human_df["object"].unique())
    h_levels = sorted(human_df["level"].unique())
    h_cats = sorted(human_df["split"].dropna().unique())

    matched_model = model_df[
        (model_df["object"].isin(h_objects)) &
        (model_df["level"].isin(h_levels))
    ]

    cats_with_human = [c for c in CATEGORY_ORDER if c in h_cats]
    n_cats = len(cats_with_human)

    fig, axes = plt.subplots(1, n_cats, figsize=(3.5 * n_cats, 4),
                              sharex=True, sharey=True)
    if n_cats == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.98, top=0.82, bottom=0.16)

    for idx, cat in enumerate(cats_with_human):
        ax = axes[idx]

        hs = human_df[human_df["split"] == cat]
        h_means = hs.groupby("level")["rating"].mean()
        h_lvls = sorted(h_means.index)
        ax.plot(h_lvls, [h_means[l] for l in h_lvls],
                color=HUMAN_STYLE["color"], marker=HUMAN_STYLE["marker"],
                markersize=4.5, linewidth=2, linestyle="--", zorder=10)

        for model_key, style in models.items():
            ms = matched_model[(matched_model["model"] == model_key) &
                               (matched_model["split"] == cat)]
            means = ms.groupby("level")["rating"].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=style["color"], marker=style["marker"],
                    markersize=3.5, linewidth=1.3)

        ax.set_title(CATEGORY_LABELS[cat], fontweight="bold", pad=5)
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks([1, 3, 5, 7])
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Perturbation Level")

        if idx == 0:
            ax.set_ylabel("Mean Rating")
        else:
            ax.set_yticklabels([])

    human_handle = Line2D([0], [0], color=HUMAN_STYLE["color"],
                          marker=HUMAN_STYLE["marker"], markersize=6,
                          linewidth=2, linestyle="--", label=HUMAN_STYLE["label"])
    model_handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                            markersize=6, linewidth=1.3, label=s["label"])
                     for s in models.values()]
    fig.legend(handles=[human_handle] + model_handles, loc="upper center",
               ncol=1 + len(models), frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=11)

    return fig


def plot_scatter_by_obj_cat(model_df, human_df):
    h_objects = set(human_df["object"].unique())
    h_levels = sorted(human_df["level"].unique())
    h_cats = sorted(human_df["split"].dropna().unique())
    cats_with_human = [c for c in CATEGORY_ORDER if c in h_cats]

    matched_model = model_df[
        (model_df["object"].isin(h_objects)) &
        (model_df["level"].isin(h_levels))
    ]

    cmap = plt.cm.Set2
    cat_colors = {c: cmap(i / len(cats_with_human)) for i, c in enumerate(cats_with_human)}

    model_keys = [k for k in MODELS if k in model_df["model"].unique()]
    n = len(model_keys)
    nrows, ncols = 2, 3
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes = axes_grid.ravel()
    fig.subplots_adjust(hspace=0.38, wspace=0.30, left=0.07, right=0.98,
                         top=0.90, bottom=0.08)

    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        style = MODELS[model_key]
        all_hx, all_my = [], []

        for cat in cats_with_human:
            hx_list, my_list = [], []
            for lvl in h_levels:
                h_vals = human_df[(human_df["split"] == cat) &
                                  (human_df["level"] == lvl)]["rating"]
                m_vals = matched_model[(matched_model["model"] == model_key) &
                                       (matched_model["split"] == cat) &
                                       (matched_model["level"] == lvl)]["rating"]
                if len(h_vals) > 0 and len(m_vals) > 0:
                    hx_list.append(h_vals.mean())
                    my_list.append(m_vals.mean())

            ax.scatter(hx_list, my_list, color=cat_colors[cat],
                       marker="o", s=80, alpha=0.75,
                       edgecolors="white", linewidths=0.5)
            all_hx.extend(hx_list)
            all_my.extend(my_list)

        ax.plot([0.5, 7.5], [0.5, 7.5], "k--", linewidth=0.8, alpha=0.35)
        if all_hx:
            rho, _ = spearmanr(all_hx, all_my)
            ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f}",
                    transform=ax.transAxes, fontsize=16, va="top", fontstyle="italic")

        ax.set_title(style["label"], fontweight="bold", fontsize=18, pad=8)
        ax.set_xlabel("Human Mean Rating", fontsize=16)
        ax.set_ylabel("Model Mean Rating", fontsize=16)
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 7.5)
        ax.set_xticks([1, 3, 5, 7])
        ax.set_yticks([1, 3, 5, 7])
        ax.tick_params(labelsize=14)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, linewidth=0.5)

    for idx in range(n, nrows * ncols):
        axes[idx].set_visible(False)

    cat_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cat_colors[c], markersize=11,
                          label=CATEGORY_LABELS[c]) for c in cats_with_human]
    fig.legend(handles=cat_handles, loc="upper center",
               ncol=len(cats_with_human), frameon=False,
               bbox_to_anchor=(0.5, 0.98), fontsize=15)

    return fig


def main():
    print("Loading data...")
    model_df = load_model_data()
    human_df = load_human_data()
    print(f"  Model: {len(model_df)} rows, Human: {len(human_df)} rows")

    for models_subset, suffix in [(MODELS, ""), (MOLMO_ONLY, "_molmo2")]:
        tag = suffix if suffix else " (all models)"
        print(f"Plotting {tag.lstrip('_') or 'all models'}...")

        fig_a = plot_by_category_models(model_df, models_subset)
        fname_a = f"fig_cat_model_ratings{suffix}.pdf"
        fig_a.savefig(FIGURES_DIR / fname_a)
        plt.close(fig_a)
        print(f"  Saved {fname_a}")

        fig_b = plot_by_category_human(model_df, human_df, models_subset)
        fname_b = f"fig_cat_human_comparison{suffix}.pdf"
        fig_b.savefig(FIGURES_DIR / fname_b)
        plt.close(fig_b)
        print(f"  Saved {fname_b}")

    print("Plotting scatter by object category...")
    fig_sc = plot_scatter_by_obj_cat(model_df, human_df)
    fig_sc.savefig(FIGURES_DIR / "scatter_human_vs_model_by_obj_cat.pdf")
    plt.close(fig_sc)
    print("  Saved scatter_human_vs_model_by_obj_cat.pdf")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
