# Would you still call this Dax? Novel Visual References in VLMs and Humans

This repository contains the code, data, and pre-computed results for the EMNLP paper:

> **Would you still call this Dax? Novel Visual References in VLMs and Humans**
> *EMNLP 2025 (under review)*

---

## Abstract

We study how vision-language models (VLMs) handle novel visual concept references
under systematic visual perturbation. We introduce **NVRD** (Novel Visual Reference
Dataset), a controlled benchmark of 90 objects across four ontological categories
(Known, Novel, Shape-Shape, Shape-Texture), each perturbed by 11 types of visual
transformation across up to 20 severity levels (19,176 images total). Using three
complementary probing paradigms — nonce word generation (§4.1), log-probability
scoring (§4.2), and dual-image Likert-scale rating (§4.3) — we evaluate five
state-of-the-art VLMs alongside 30 human participants. Our results reveal that
while humans show robust, category-sensitive reference preservation, all VLMs
exhibit substantially different degradation profiles, with notable overreliance on
surface features and limited sensitivity to semantic category boundaries.

---

## Dataset

NVRD is hosted on HuggingFace at [`adadtur/nvrd`](https://huggingface.co/datasets/adadtur/nvrd).

| Property | Value |
|---|---|
| Objects | 90 (across 4 categories) |
| Categories | Known, Novel, Shape-Shape, Shape-Texture |
| Perturbation types | 11 (add, background, color, jpeg, noise, pixelate, remove, scale, shape, style, texture) |
| Levels per perturbation | up to 20 |
| Total images | 19,176 |
| Human participants | 30 (Prolific) |

---

## Quick setup

```bash
git clone https://github.com/your-org/nvrd.git
cd nvrd
python -m venv .venv && source .venv/bin/activate

# Install PyTorch first (match your CUDA version):
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6
# pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8
# pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # CPU only

pip install -r requirements.txt
```

---

## Dataset download

Download all NVRD images from HuggingFace into `data/sample-data/`:

```bash
python data/download_dataset.py
```

This downloads ~19K images from HuggingFace. **The first run downloads ~971 Parquet shard files before extracting images — expect 20–30 minutes before any PNGs appear on disk.** Subsequent runs skip already-downloaded images.

It creates the following layout (which the experiment scripts expect):

```
data/sample-data/
├── known/
│   ├── {object}.png               (original)
│   └── perturbations_comp/
│       └── {ptype}/
│           └── {object}_{level}.png
├── novel/
│   └── ...
└── modified/
    ├── shape-shape/
    │   └── ...
    └── shape-texture/
        └── ...
```

---

## Reproducing figures

All figures are saved to `figures/` (created automatically).

**Figures 3 & 4** — Nonce reference generation accuracy + z-scored log-prob by object category:
```bash
python analysis/plot_figure3.py
```
Output: `figures/combined_gen_prob_by_category.pdf`

**Figure 5** — Human vs model Likert rating comparison:
```bash
python analysis/plot_figure5.py
```
Outputs (all in `figures/`):
- `fig_cat_model_ratings.pdf` — all-model ratings by object category
- `fig_cat_model_ratings_molmo2.pdf` — Molmo2 only
- `fig_cat_human_comparison.pdf` — human + all-model overlay by category
- `fig_cat_human_comparison_molmo2.pdf` — human + Molmo2 overlay
- `scatter_human_vs_model_by_obj_cat.pdf` — scatter correlation by category

**Table 2** — Cross-task Spearman correlations:
```bash
python analysis/cross_task_consistency.py
```
Output: `figures/cross_task_consistency.tex` (also printed to stdout)

**Appendix figures** — Human comparison by perturbation type, ablations, nonce vs vanilla breakdowns:
```bash
python analysis/plot_appendix.py
```
Outputs multiple PDFs in `figures/`.

---

## Running experiments

Experiments require:
- The dataset downloaded to `data/sample-data/` (see above)
- A nonce word mapping file at `data/nonce_word_mapping.json`
- For generation experiments: precomputed ICL pools at `data/precomputed_icl_pools.json`

### Environment variables

| Variable | Required for | Description |
|---|---|---|
| `HF_TOKEN` | Downloading gated HF models | HuggingFace access token |
| `OPENAI_API_KEY` | GPT-4o-mini experiments | OpenAI API key |
| `GEMINI_API_KEY` | Gemini 2.5 Flash experiments | Google Gemini API key |

### Model keys

The following model keys are supported across all experiment scripts:

| Key | Model |
|---|---|
| Key | Model | Notes |
|---|---|---|
| `qwen2-vl-7b` | Qwen/Qwen2-VL-7B-Instruct | Used in paper |
| `qwen2.5-vl-72b` | Qwen/Qwen2.5-VL-72B-Instruct | Large; needs multi-GPU |
| `idefics3-8b` | HuggingFaceM4/Idefics3-8B-Llama3 | Used in paper |
| `molmo2-8b` | allenai/Molmo2-8B | Used in paper |
| `gpt-4o-mini` | gpt-4o-mini | API; requires `OPENAI_API_KEY` |
| `gemini-2.5-flash` | gemini-2.5-flash | API; requires `GEMINI_API_KEY` |

### Generation experiment (§4.1 + §4.2)

Runs both nonce word generation and log-probability scoring (for local models):

```bash
python experiments/run_generation.py qwen2-vl-7b
```

Optional arguments:
```
python experiments/run_generation.py <model_key> [split] [--prob-only]

  split       : known, novel, shape-shape, shape-texture, or all (default)
  --prob-only : skip generation, run probability experiments only
```

Note: Local models (Qwen, Idefics, Molmo) require a GPU with ~16 GB VRAM.
API models (gpt-4o-mini, gemini-2.5-flash) require the corresponding API key.

Results are saved to `results/generation/` and `results/probability/`.

### Likert rating experiment (§4.3)

```bash
python experiments/run_likert.py qwen2-vl-7b
```

Optional arguments:
```
python experiments/run_likert.py <model_key> [split]
```

Results are saved to `results/prolific_style/`.

### Sycophancy ablation (§6.1)

```bash
python experiments/run_sycophancy.py qwen2-vl-7b
```

Optional arguments:
```
python experiments/run_sycophancy.py [model_key] [--n_pairs N] [--seed S]

  --n_pairs : number of cross-object pairs to sample (default: 1000)
  --seed    : random seed (default: 42)
```

Results are saved to `results/sycophancy_ablation/`.

---

## Pre-computed results

Large result files (532 MB total) are **not included** in this Git repository.

- **For server users**: `results/generation/`, `results/probability/`, and
  `results/prolific_style/` are symlinks to the local cluster copy.
- **For external users**: the full result files are available from the authors upon
  request or via HuggingFace dataset card.
- **Sycophancy ablation** results (1.4 MB) are included directly in
  `results/sycophancy_ablation/`.

See `results/README.md` for file format documentation.

---

## Repository structure

```
nvrd/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_dataset.py        download NVRD from HuggingFace
│   ├── nonce_words.txt            nonce word list
│   └── human_study/
│       └── trial-results-1.csv   human participant ratings (n=30)
├── experiments/
│   ├── run_generation.py          §4.1 name generation + §4.2 log-prob
│   ├── run_likert.py              §4.3 Likert rating
│   └── run_sycophancy.py          §6.1 sycophancy ablation
├── analysis/
│   ├── plot_settings.py           shared constants + data loaders
│   ├── plot_figure3.py            Figure 3 (gen accuracy + z-log-prob by category)
│   ├── plot_figure5.py            Figure 5 (human vs model Likert)
│   ├── plot_appendix.py           Appendix figures
│   └── cross_task_consistency.py  Table 2
└── results/
    ├── README.md                  result format documentation
    ├── generation/                symlink -> 331 MB JSONL files
    ├── probability/               symlink -> 102 MB JSONL files
    ├── prolific_style/            symlink -> 99 MB JSONL files
    └── sycophancy_ablation/       1.4 MB, included directly
```

---

## Citation

If you use NVRD or this code in your work, please cite:

```bibtex
@inproceedings{atur2025nvrd,
  title     = {Would you still call this Dax? Novel Visual References in VLMs and Humans},
  author    = {Atur, Aditi and others},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
}
```

---

## License

The code in this repository is released under the MIT License.
The NVRD dataset is released under CC BY 4.0.
