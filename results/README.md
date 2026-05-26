# Results

This directory contains pre-computed experimental results for the NVRD paper.

## Directory structure

```
results/
├── generation/         -> symlink to local large files (331 MB)
├── probability/        -> symlink to local large files (102 MB)
├── prolific_style/     -> symlink to local large files (99 MB)
└── sycophancy_ablation/   small files, included directly (1.4 MB)
    ├── qwen2-vl-7b-sycophancy-ablation.jsonl
    └── qwen2-vl-7b-sycophancy-ablation-pairs.json
```

## For server users (symlinks)

On the cluster where this repo lives, `generation/`, `probability/`, and
`prolific_style/` are symlinks pointing to the original result directories.
The analysis scripts will work out of the box.

## For external users (HuggingFace)

The full result files (532 MB total) are available on HuggingFace at
`adadtur/nvrd` (dataset card) or by request from the authors.

To reproduce results from scratch, run the experiment scripts in `experiments/`:

```bash
python experiments/run_generation.py qwen2-vl-7b
python experiments/run_likert.py qwen2-vl-7b
python experiments/run_sycophancy.py qwen2-vl-7b
```

See the top-level README for full instructions.

## File formats

**generation/**: JSONL files named `{model}-visual_similarity-fillin-outputs.jsonl`
Each line has: `object`, `perturbation_type`, `level`, `nonce_word`, `model_response`, `n`, `model`, `type/split`.

**probability/**: JSONL files named `{model}-visual_similarity-fillin_results.jsonl`
Each line has: `object_name`, `perturbation_type`, `level`, `nonce_word_log_prob`, `nonce_word_num_tokens`, `n`, `model`, `split`.

**prolific_style/**: JSONL files named `{model}-prolific-style-ratings*.jsonl`
Each line has: `object`, `perturbation_type`, `level`, `rating`, `model`, `split`.

**sycophancy_ablation/**: JSONL + JSON for cross-object pair ratings.

## Note on Table 2 values

The cross-task Spearman correlations in Table 2 of the paper were computed from
an earlier snapshot of the results. The values from the current pre-computed files
may differ slightly (e.g. Qwen2-VL 7B Gen↔Likert: ρ ≈ 0.79 from current files
vs. ρ = 0.45 in the paper). The aggregation method in `analysis/cross_task_consistency.py`
(grouping by perturbation type × level) is correct; the difference reflects
iterative result updates made after the paper draft was submitted.
