"""
Appendix figures for the NVRD paper.

Produces the following figures in figures/:
  - fig_human_comparison_by_ptype.pdf    (Fig 18: human vs model by perturbation type)
  - scatter_human_vs_model_by_ptype.pdf  (Fig 19: scatter per ptype)
  - ablation_pool_size.pdf               (Fig 25: pool size ablation)
  - ablation_pool_composition.pdf        (Fig 26: pool composition ablation)
  - generation_nonce_vs_vanilla_by_level.pdf  (Fig 20)
  - generation_nonce_vs_vanilla_by_ptype.pdf  (Fig 21)
  - generation_nonce_vs_vanilla_by_obj_cat.pdf (Fig 22)

Usage:
    python analysis/plot_appendix.py
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
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = {
    "molmo2-8b":        {"label": "Molmo2 8B",        "color": "#d95f02", "marker": "s"},
    "qwen2-vl-7b":      {"label": "Qwen2-VL 7B",      "color": "#7570b3", "marker": "^"},
    "idefics3-8b":      {"label": "Idefics3 8B",      "color": "#1b9e77", "marker": "o"},
    "gemini-2.5-flash": {"label": "Gemini 2.5 Flash", "color": "#e7298a", "marker": "D"},
    "gpt-4o-mini":      {"label": "GPT-4o-mini",      "color": "#66a61e", "marker": "v"},
}
HUMAN_STYLE = {"label": "Humans (n=30)", "color": "#222222", "marker": "X"}

PTYPE_LABELS = {
    "add": "Part Addition",
    "background": "Background",
    "color": "Color Shift",
    "remove": "Part Removal",
    "shape": "Shape Deform.",
    "style": "Style Degrad.",
    "texture": "Texture",
}
HUMAN_PTYPES = list(PTYPE_LABELS.keys())

TYPE_ORDER = ["known", "novel", "shape-shape", "shape-texture"]
TYPE_LABELS = {
    "known": "Known",
    "novel": "Novel",
    "shape-shape": "Shape-Shape",
    "shape-texture": "Shape-Texture",
}

CATEGORY_LABELS = {
    "known": "Known",
    "novel": "Novel",
    "modified/shape-shape": "Shape-Shape",
    "modified/shape-texture": "Shape-Texture",
}


def load_jsonl(fp):
    with open(fp) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Human comparison helpers ──────────────────────────────────────────────────

def load_model_rating_data():
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
                    "category": r.get("split", ""),
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
    return df


def plot_human_comparison_by_ptype(model_df, human_df):
    h_objects = set(human_df["object"].unique())
    h_levels = sorted(human_df["level"].unique())
    matched_model = model_df[
        (model_df["object"].isin(h_objects)) &
        (model_df["perturbation_type"].isin(HUMAN_PTYPES)) &
        (model_df["level"].isin(h_levels))
    ]

    pt_order = ["remove", "add", "shape", "style", "texture", "color", "background"]

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.35,
                           left=0.06, right=0.98, top=0.88, bottom=0.10)

    line_axes = []
    for i in range(7):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])
        line_axes.append(ax)

    for idx, pt in enumerate(pt_order):
        ax = line_axes[idx]
        row, col = divmod(idx, 4)

        hs = human_df[human_df["perturbation_type"] == pt]
        h_means = hs.groupby("level")["rating"].mean()
        h_lvls = sorted(h_means.index)
        ax.plot(h_lvls, [h_means[l] for l in h_lvls],
                color=HUMAN_STYLE["color"], marker=HUMAN_STYLE["marker"],
                markersize=4.5, linewidth=2, linestyle="--", zorder=10)

        for model_key, style in MODELS.items():
            ms = matched_model[(matched_model["model"] == model_key) &
                               (matched_model["perturbation_type"] == pt)]
            means = ms.groupby("level")["rating"].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=style["color"], marker=style["marker"],
                    markersize=3.5, linewidth=1.3)

        ax.set_title(PTYPE_LABELS[pt], fontweight="bold", pad=5)
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks([1, 3, 5, 7])
        ax.grid(True, alpha=0.15, linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Mean Rating")
        else:
            ax.set_yticklabels([])
        if row == 1 or idx >= 4:
            ax.set_xlabel("Perturbation Level")
        else:
            ax.set_xticklabels([])

    human_handle = Line2D([0], [0], color=HUMAN_STYLE["color"],
                          marker=HUMAN_STYLE["marker"], markersize=6,
                          linewidth=2, linestyle="--", label=HUMAN_STYLE["label"])
    model_handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                            markersize=6, linewidth=1.3, label=s["label"])
                     for s in MODELS.values()]
    fig.legend(handles=[human_handle] + model_handles, loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=10)

    return fig


def plot_scatter_by_ptype(model_df, human_df):
    h_objects = set(human_df["object"].unique())
    h_levels = sorted(human_df["level"].unique())
    pt_order = ["remove", "add", "shape", "style", "texture", "color", "background"]
    cmap = plt.cm.Set2
    pt_colors = {pt: cmap(i / len(pt_order)) for i, pt in enumerate(pt_order)}

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

        for pt in pt_order:
            hx_list, my_list = [], []
            for lvl in h_levels:
                h_vals = human_df[(human_df["perturbation_type"] == pt) &
                                  (human_df["level"] == lvl)]["rating"]
                m_vals = model_df[(model_df["model"] == model_key) &
                                  (model_df["object"].isin(h_objects)) &
                                  (model_df["perturbation_type"] == pt) &
                                  (model_df["level"] == lvl)]["rating"]
                if len(h_vals) > 0 and len(m_vals) > 0:
                    hx_list.append(h_vals.mean())
                    my_list.append(m_vals.mean())
            ax.scatter(hx_list, my_list, color=pt_colors[pt],
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

    pt_handles = [Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=pt_colors[pt], markersize=11,
                         label=PTYPE_LABELS[pt]) for pt in pt_order]
    fig.legend(handles=pt_handles, loc="upper center",
               ncol=len(pt_order), frameon=False,
               bbox_to_anchor=(0.5, 0.98), fontsize=15)

    return fig


# ── Ablation helpers ──────────────────────────────────────────────────────────

def load_generation_qwen():
    rows = []
    for sim in ["visual_similarity", "color_similarity", "random"]:
        fp = RESULTS_DIR / "generation" / f"qwen2-vl-7b-{sim}-fillin-outputs.jsonl"
        if not fp.exists():
            print(f"[WARN] Missing {fp}")
            continue
        print(f"Loading {fp.name}")
        for r in load_jsonl(str(fp)):
            nonce = r.get("nonce_word", r.get("ref", ""))
            response = r.get("model_response", "") or ""
            correct = 1 if nonce and nonce.lower() in response.lower() else 0
            rows.append({
                "similarity_type": sim,
                "type": r.get("type", "unknown"),
                "n": r["n"],
                "perturbation_type": r["perturbation_type"],
                "level": r["level"],
                "correct": correct,
            })
    return pd.DataFrame(rows)


def plot_faceted_by_type(data_df, y_col, y_label, line_specs, legend_title=None):
    present_types = [t for t in TYPE_ORDER if t in data_df["type"].unique()]
    ncols = len(present_types)

    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.4),
                              sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99,
                         top=0.75, bottom=0.18)

    for idx, obj_type in enumerate(present_types):
        ax = axes[idx]
        sub = data_df[data_df["type"] == obj_type]

        for spec in line_specs:
            mask = pd.Series(True, index=sub.index)
            for k, v in spec["filter"].items():
                mask &= sub[k] == v
            ms = sub[mask]
            if ms.empty:
                continue
            means = ms.groupby("level")[y_col].mean()
            levels = sorted(means.index)
            ax.plot(levels, [means[l] for l in levels],
                    color=spec["color"], marker=spec["marker"],
                    markersize=3.5, linewidth=1.3,
                    linestyle=spec.get("linestyle", "-"))

        ax.set_title(TYPE_LABELS.get(obj_type, obj_type), fontweight="bold", pad=5)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Perturbation Level")

        if idx == 0:
            ax.set_ylabel(y_label)

    handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                       markersize=6, linewidth=1.3,
                       linestyle=s.get("linestyle", "-"),
                       label=s["label"])
               for s in line_specs]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(line_specs), 6),
               frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=10,
               title=legend_title)

    return fig


def plot_pool_size(gen_df):
    sub = gen_df[gen_df["similarity_type"] == "visual_similarity"].copy()
    n_colors = {2: "#e41a1c", 4: "#222222", 8: "#984ea3"}
    n_markers = {2: ">", 4: "h", 8: "<"}
    line_specs = [
        {"filter": {"n": n}, "label": f"n = {n}",
         "color": n_colors[n], "marker": n_markers[n]}
        for n in [2, 4, 8]
    ]
    fig = plot_faceted_by_type(sub, y_col="correct",
                                y_label="Nonce Reference Usage",
                                line_specs=line_specs,
                                legend_title="Pool Size")
    for ax in fig.axes:
        if ax.get_visible():
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    return fig


def plot_pool_composition(gen_df):
    sub = gen_df[gen_df["n"] == 4].copy()
    line_specs = [
        {"filter": {"similarity_type": "visual_similarity"},
         "label": "Visual Similarity", "color": "#7570b3", "marker": "^"},
        {"filter": {"similarity_type": "color_similarity"},
         "label": "Color Similarity", "color": "#e7298a", "marker": "s",
         "linestyle": "--"},
        {"filter": {"similarity_type": "random"},
         "label": "Random", "color": "#1b9e77", "marker": "o",
         "linestyle": ":"},
    ]
    fig = plot_faceted_by_type(sub, y_col="correct",
                                y_label="Nonce Reference Usage",
                                line_specs=line_specs,
                                legend_title="Retrieval Strategy")
    for ax in fig.axes:
        if ax.get_visible():
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    return fig


def classify_response(resp, nonce, obj_name, obj_type):
    resp = (resp or "").lower().strip()
    nonce = nonce.lower()
    if obj_type == "novel":
        has_vanilla = False
    else:
        parts = [p for p in obj_name.lower().replace("-", "_").split("_") if len(p) > 2]
        has_vanilla = any(p in resp for p in parts)
    has_nonce = nonce in resp
    if has_nonce and not has_vanilla:
        return "nonce"
    elif has_vanilla and not has_nonce:
        return "vanilla"
    elif has_nonce and has_vanilla:
        return "both"
    return "other"


def load_nonce_vanilla_data():
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
                obj_type = r.get("type", "unknown")
                cls = classify_response(
                    r.get("model_response", ""),
                    r.get("nonce_word", ""),
                    r.get("object", ""),
                    obj_type,
                )
                rows.append({
                    "model": model_key,
                    "type": obj_type,
                    "object": r.get("object", ""),
                    "perturbation_type": r["perturbation_type"],
                    "level": r["level"],
                    "response_class": cls,
                })
    return pd.DataFrame(rows)


def plot_nonce_vs_vanilla_by_level(df):
    model_keys = [k for k in MODELS if k in df["model"].unique()]
    ncols = len(model_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.98, top=0.82, bottom=0.15)

    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        style = MODELS[model_key]
        sub = df[df["model"] == model_key]

        for cls, ls, color in [
            ("nonce", "-", style["color"]),
            ("vanilla", "--", style["color"]),
            ("other", ":", "#888888"),
        ]:
            props = sub.groupby("level")["response_class"].apply(
                lambda g: (g == cls).mean()
            ).sort_index()
            ax.plot(props.index, props.values, linestyle=ls, color=color,
                    marker="o", markersize=3.5, linewidth=1.5)

        ax.set_title(style["label"], fontweight="bold", pad=8)
        ax.set_xlabel("Perturbation Level")
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        if idx == 0:
            ax.set_ylabel("Proportion of Responses")

    handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Nonce word"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Vanilla name"),
        Line2D([0], [0], color="#888888", linestyle=":", linewidth=1.5, label="Other"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=11)
    return fig


def plot_nonce_vs_vanilla_by_ptype(df):
    model_keys = [k for k in MODELS if k in df["model"].unique()]
    ncols = len(model_keys)
    pt_order = ["remove", "add", "shape", "style", "texture", "color",
                "background", "noise", "jpeg", "pixelate", "scale"]
    pt_labels = {
        "add": "Add", "background": "Bkgd", "color": "Color",
        "jpeg": "JPEG", "noise": "Noise", "pixelate": "Pixel",
        "remove": "Remove", "scale": "Scale", "shape": "Shape",
        "style": "Style", "texture": "Texture",
    }

    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.98, top=0.82, bottom=0.22)

    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        style = MODELS[model_key]
        sub = df[df["model"] == model_key]
        present_pts = [p for p in pt_order if p in sub["perturbation_type"].unique()]
        x = np.arange(len(present_pts))

        nonce_props, vanilla_props, other_props = [], [], []
        for pt in present_pts:
            pt_sub = sub[sub["perturbation_type"] == pt]
            total = len(pt_sub)
            nonce_props.append((pt_sub["response_class"] == "nonce").sum() / total)
            vanilla_props.append((pt_sub["response_class"] == "vanilla").sum() / total)
            other_props.append(1 - nonce_props[-1] - vanilla_props[-1])

        ax.bar(x, nonce_props, 0.7, label="Nonce", color=style["color"], alpha=0.85)
        ax.bar(x, vanilla_props, 0.7, bottom=nonce_props, label="Vanilla",
               color=style["color"], alpha=0.4, edgecolor="white", linewidth=0.5)
        ax.bar(x, other_props, 0.7,
               bottom=[n + v for n, v in zip(nonce_props, vanilla_props)],
               label="Other", color="#cccccc", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([pt_labels.get(p, p) for p in present_pts],
                            rotation=45, ha="right")
        ax.set_title(style["label"], fontweight="bold", pad=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        if idx == 0:
            ax.set_ylabel("Proportion of Responses")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="black", alpha=0.85, label="Nonce word"),
        plt.Rectangle((0, 0), 1, 1, color="black", alpha=0.4, label="Vanilla name"),
        plt.Rectangle((0, 0), 1, 1, color="#cccccc", alpha=0.7, label="Other"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=11)
    return fig


def plot_nonce_vs_vanilla_by_obj_cat(df):
    model_keys = [k for k in MODELS if k in df["model"].unique()]
    ncols = len(model_keys)
    present_types = [t for t in TYPE_ORDER if t in df["type"].unique()]

    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.98, top=0.82, bottom=0.18)

    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        style = MODELS[model_key]
        sub = df[df["model"] == model_key]
        cats = [t for t in present_types if t in sub["type"].unique()]
        x = np.arange(len(cats))

        nonce_props, vanilla_props, other_props = [], [], []
        for cat in cats:
            cat_sub = sub[sub["type"] == cat]
            total = len(cat_sub)
            nonce_props.append((cat_sub["response_class"] == "nonce").sum() / total)
            vanilla_props.append((cat_sub["response_class"] == "vanilla").sum() / total)
            other_props.append(1 - nonce_props[-1] - vanilla_props[-1])

        ax.bar(x, nonce_props, 0.7, label="Nonce", color=style["color"], alpha=0.85)
        ax.bar(x, vanilla_props, 0.7, bottom=nonce_props, label="Vanilla",
               color=style["color"], alpha=0.4, edgecolor="white", linewidth=0.5)
        ax.bar(x, other_props, 0.7,
               bottom=[n + v for n, v in zip(nonce_props, vanilla_props)],
               label="Other", color="#cccccc", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([TYPE_LABELS.get(c, c) for c in cats],
                            rotation=45, ha="right")
        ax.set_title(style["label"], fontweight="bold", pad=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        if idx == 0:
            ax.set_ylabel("Proportion of Responses")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="black", alpha=0.85, label="Nonce word"),
        plt.Rectangle((0, 0), 1, 1, color="black", alpha=0.4, label="Vanilla name"),
        plt.Rectangle((0, 0), 1, 1, color="#cccccc", alpha=0.7, label="Other"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=11)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Appendix figures: human comparison + ablations")
    print("=" * 60)

    print("\nLoading model rating data...")
    model_df = load_model_rating_data()
    print("Loading human data...")
    human_df = load_human_data()
    print(f"  Model: {len(model_df)} rows, Human: {len(human_df)} rows")

    print("\nPlotting human comparison by perturbation type...")
    fig = plot_human_comparison_by_ptype(model_df, human_df)
    fig.savefig(FIGURES_DIR / "fig_human_comparison_by_ptype.pdf")
    plt.close(fig)
    print("  Saved fig_human_comparison_by_ptype.pdf")

    print("Plotting scatter by perturbation type...")
    fig_sc = plot_scatter_by_ptype(model_df, human_df)
    fig_sc.savefig(FIGURES_DIR / "scatter_human_vs_model_by_ptype.pdf")
    plt.close(fig_sc)
    print("  Saved scatter_human_vs_model_by_ptype.pdf")

    print("\nLoading Qwen2 generation data for ablations...")
    gen_df = load_generation_qwen()
    if not gen_df.empty:
        print(f"  {len(gen_df)} rows")

        print("Plotting pool size ablation...")
        fig1 = plot_pool_size(gen_df)
        fig1.savefig(FIGURES_DIR / "ablation_pool_size.pdf")
        plt.close(fig1)
        print("  Saved ablation_pool_size.pdf")

        print("Plotting pool composition ablation...")
        fig2 = plot_pool_composition(gen_df)
        fig2.savefig(FIGURES_DIR / "ablation_pool_composition.pdf")
        plt.close(fig2)
        print("  Saved ablation_pool_composition.pdf")

    print("\nLoading nonce vs vanilla generation data...")
    nv_df = load_nonce_vanilla_data()
    if not nv_df.empty:
        print(f"  {len(nv_df)} rows")

        print("Plotting nonce vs vanilla by level...")
        fig3 = plot_nonce_vs_vanilla_by_level(nv_df)
        fig3.savefig(FIGURES_DIR / "generation_nonce_vs_vanilla_by_level.pdf")
        plt.close(fig3)
        print("  Saved generation_nonce_vs_vanilla_by_level.pdf")

        print("Plotting nonce vs vanilla by perturbation type...")
        fig4 = plot_nonce_vs_vanilla_by_ptype(nv_df)
        fig4.savefig(FIGURES_DIR / "generation_nonce_vs_vanilla_by_ptype.pdf")
        plt.close(fig4)
        print("  Saved generation_nonce_vs_vanilla_by_ptype.pdf")

        print("Plotting nonce vs vanilla by object category...")
        fig5 = plot_nonce_vs_vanilla_by_obj_cat(nv_df)
        fig5.savefig(FIGURES_DIR / "generation_nonce_vs_vanilla_by_obj_cat.pdf")
        plt.close(fig5)
        print("  Saved generation_nonce_vs_vanilla_by_obj_cat.pdf")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
