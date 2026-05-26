# Analysis Scripts

All analysis scripts read from `results/` and write figures to `figures/`. They require pre-computed results (either downloaded via `results/download_results.py` or generated from scratch).

---

## Script-to-Figure Mapping

| Script | Paper Figure | Description |
|--------|-------------|-------------|
| `plot_figure3.py` | Figure 3 & 4 | Combined 2-row: generation accuracy (top) and z-scored log-prob (bottom) by object category |
| `plot_figure5.py` | Figure 5 | Human vs model Likert ratings by object category; scatter correlation plots |
| `plot_appendix.py` | Figs 18–26 | Human comparison by perturbation type, bar plots, sycophancy ablation distributions and heatmap |
| `cross_task_consistency.py` | Table 2 | Pairwise Spearman correlations between generation, Likert, and log-prob tasks |

---

## plot_settings.py

Shared module containing:
- Global `MODELS` dict (label, color, marker for each model)
- `PTYPE_LABELS` dict (perturbation type display names)
- `SPLIT_ORDER` and `SPLIT_LABELS` for object categories
- Data loader functions: `load_generation_data()`, `load_probability_data()`
- Shared `RESULTS_DIR` and `OUT_DIR` path constants

All other scripts import from `plot_settings.py`.

---

## Running All Figures

```bash
# From the repo root:
python analysis/plot_figure3.py
python analysis/plot_figure5.py
python analysis/plot_appendix.py
python analysis/cross_task_consistency.py  # prints LaTeX table to stdout
```

---

## Notes on Data Requirements

- `plot_figure3.py`: requires `results/generation/` and `results/probability/`
- `plot_figure5.py`: requires `results/prolific_style/` and `data/human_study/trial-results-1.csv`
- `plot_appendix.py`: requires `results/prolific_style/`, `data/human_study/trial-results-1.csv`, and `results/sycophancy_ablation/` (bundled)
- `cross_task_consistency.py`: requires all three results directories
