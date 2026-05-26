"""
Cross-task consistency analysis: Do the three probing paradigms
(generation, Likert rating, log-probability) agree at the condition level?

Aggregates per (perturbation_type, level, model) and computes
pairwise Spearman correlations. Outputs a LaTeX table saved to
figures/cross_task_consistency.tex and printed to stdout.

Usage:
    python analysis/cross_task_consistency.py
"""

import json
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_settings import RESULTS_DIR, FIGURES_DIR


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_generation(directory):
    rows = []
    patterns = [
        f"{directory}/*-visual_similarity-fillin-outputs.jsonl",
        f"{directory}/*-visual_similarity-fillin-novel-outputs.jsonl",
    ]
    for pat in patterns:
        for fp in glob.glob(pat):
            with open(fp) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("n") != 4:
                        continue
                    nonce = rec.get("nonce_word", rec.get("ref", ""))
                    response = rec.get("model_response", "") or ""
                    correct = int(nonce and nonce.lower() in response.lower())
                    rows.append({
                        "perturbation_type": rec["perturbation_type"],
                        "level": rec["level"],
                        "model": rec["model"],
                        "gen_correct": correct,
                    })
    return pd.DataFrame(rows)


def load_likert(directory):
    rows = []
    for fp in glob.glob(f"{directory}/*.jsonl"):
        with open(fp) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("rating") is None or rec.get("perturbation_type") == "attention_check":
                    continue
                rows.append({
                    "perturbation_type": rec["perturbation_type"],
                    "level": rec["level"],
                    "model": rec["model"],
                    "likert_rating": rec["rating"],
                })
    return pd.DataFrame(rows)


def load_probability(directory):
    rows = []
    for fp in glob.glob(f"{directory}/*.jsonl"):
        with open(fp) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("n") != 4:
                    continue
                lp = rec.get("nonce_word_log_prob")
                if lp is None or not np.isfinite(lp):
                    continue
                rows.append({
                    "perturbation_type": rec["perturbation_type"],
                    "level": rec["level"],
                    "model": rec["model"],
                    "nonce_log_prob": lp,
                })
    return pd.DataFrame(rows)


def fmt_rho(rho, p):
    """Format a Spearman result for the LaTeX table."""
    stars = ""
    if p < 0.001:
        stars = "$^{***}$"
    elif p < 0.01:
        stars = "$^{**}$"
    elif p < 0.05:
        stars = "$^{*}$"
    return f"{rho:.2f}{stars}"


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading data...")
gen_df = load_generation(str(RESULTS_DIR / "generation"))
lik_df = load_likert(str(RESULTS_DIR / "prolific_style"))
prob_df = load_probability(str(RESULTS_DIR / "probability"))

print(f"  Generation:  {len(gen_df):,} records, models: {sorted(gen_df['model'].unique()) if not gen_df.empty else []}")
print(f"  Likert:      {len(lik_df):,} records, models: {sorted(lik_df['model'].unique()) if not lik_df.empty else []}")
print(f"  Probability: {len(prob_df):,} records, models: {sorted(prob_df['model'].unique()) if not prob_df.empty else []}")

# Aggregate per (perturbation_type, level, model)
coarse_keys = ["perturbation_type", "level", "model"]
gen_agg = gen_df.groupby(coarse_keys)["gen_correct"].mean().reset_index() if not gen_df.empty else pd.DataFrame()
lik_agg = lik_df.groupby(coarse_keys)["likert_rating"].mean().reset_index() if not lik_df.empty else pd.DataFrame()
prob_agg = prob_df.groupby(coarse_keys)["nonce_log_prob"].mean().reset_index() if not prob_df.empty else pd.DataFrame()

# All models we want in the table, in display order
MODEL_ORDER = [
    ("qwen2-vl-7b",      "Qwen-2 VL 7B"),
    ("idefics3-8b",      "Idefics-3 8B"),
    ("molmo2-8b",        "Molmo-2 8B"),
    ("gpt-4o-mini",      "GPT-4o Mini"),
    ("gemini-2.5-flash", "Gemini-2.5 Flash"),
]

# Task pairs: (col_a, col_b, header label)
PAIRS = [
    ("gen_correct",   "likert_rating",  r"Gen. $\leftrightarrow$ Likert"),
    ("gen_correct",   "nonce_log_prob", r"Gen. $\leftrightarrow$ Log-Prob."),
    ("likert_rating", "nonce_log_prob", r"Likert $\leftrightarrow$ Log-Prob."),
]

gen_models = set(gen_agg["model"].unique()) if not gen_agg.empty else set()
lik_models = set(lik_agg["model"].unique()) if not lik_agg.empty else set()
prob_models = set(prob_agg["model"].unique()) if not prob_agg.empty else set()

task_availability = {}
for model_id, _ in MODEL_ORDER:
    task_availability[model_id] = {
        "gen_correct":   model_id in gen_models,
        "likert_rating": model_id in lik_models,
        "nonce_log_prob": model_id in prob_models,
    }

# Compute correlations
print("\nComputing per-model correlations...\n")
results = {}
for model_id, model_label in MODEL_ORDER:
    results[model_id] = {}
    for col_a, col_b, _ in PAIRS:
        has_a = task_availability[model_id][col_a]
        has_b = task_availability[model_id][col_b]
        if not (has_a and has_b):
            results[model_id][(col_a, col_b)] = None
            continue

        dfs_map = {
            "gen_correct":   gen_agg,
            "likert_rating": lik_agg,
            "nonce_log_prob": prob_agg,
        }
        df_a = dfs_map[col_a][dfs_map[col_a]["model"] == model_id]
        df_b = dfs_map[col_b][dfs_map[col_b]["model"] == model_id]
        m = df_a.merge(df_b, on=coarse_keys, how="inner")

        if len(m) < 5:
            results[model_id][(col_a, col_b)] = None
            continue

        rho, p = spearmanr(m[col_a], m[col_b])
        results[model_id][(col_a, col_b)] = (rho, p)
        print(f"  {model_label:22s}  {col_a:15s} vs {col_b:15s}:  rho={rho:.3f}  p={p:.2e}  (n={len(m)})")

# ── Generate LaTeX table ──────────────────────────────────────────────────────
pair_headers = " & ".join(h for _, _, h in PAIRS)

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(
    r"\caption{Cross-task Spearman correlations ($\rho$) per model, aggregated by "
    r"perturbation type and level. Significance: $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$. "
    r"Dashes indicate task combinations unavailable for closed-source models.}"
)
lines.append(r"\label{tab:cross_task_consistency}")
lines.append(r"\begin{tabular}{l ccc}")
lines.append(r"\toprule")
lines.append(f"Model & {pair_headers} \\\\")
lines.append(r"\midrule")

for model_id, model_label in MODEL_ORDER:
    cells = []
    for col_a, col_b, _ in PAIRS:
        res = results[model_id][(col_a, col_b)]
        if res is None:
            cells.append("---")
        else:
            rho, p = res
            cells.append(fmt_rho(rho, p))
    line = f"{model_label} & " + " & ".join(cells) + r" \\"
    lines.append(line)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex = "\n".join(lines)

out_path = FIGURES_DIR / "cross_task_consistency.tex"
FIGURES_DIR.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    f.write(latex)

print(f"\n{'='*70}")
print("GENERATED LATEX TABLE")
print(f"{'='*70}")
print(latex)
print(f"\nSaved to {out_path}")
