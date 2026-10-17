# affordance-test

How does a VLM (Qwen2-VL-7B-Instruct) label and infer part-level affordances of NVRD objects (known, novel,
shape-shape, shape-texture)? Regions come from SAM+SLIC ([`../regions`](../regions)); labels come from
forced-choice scoring over the AGD20K verb vocabulary; occlusion attribution (RISE / leave-one-out) is compared
against those labels; the purpose-trained GAT model (Aff-Grasp) is the external reference.

Run everything from this directory in the project venv. The scripts are flat modules that import each other.

## Layout

| file | role |
|---|---|
| `affordance_labeling.py` | shared core (no CLI): vocabulary (`AGD20K_VERBS` + `wear` + `none`), prompts, `partition_regions` (nested SAM regions -> disjoint, background-filtered parts), `score_options` (teacher-forced forced-choice log-probs via a shared KV cache), `numbered_overlay`, `highlight_region`, `downscale` |
| `batch_final_labeling.py` | **standard labeling**: per part, forced-choice label + one-sentence justification, each an independent call against a shared numbered overlay. `python batch_final_labeling.py [object ...]` -> `json/affordance_justification/` |
| `visualize_affordance_justification.py`, `visualize_affordance_labels_only.py` | figures from that JSON (`figures/<category>/`) |
| `rise_affordance_verbs.py` | RISE over the 37 verbs (whole-object question; part x verb importance) -> `json/rise_affordance_verbs/` |
| `loo_affordance_verbs.py` | leave-one-out over verbs, plus the direct method's full per-part distribution -> `json/loo_affordance_verbs/` |
| `visualize_rise_affordance_verbs.py` | part x verb heatmaps (`--loo` for leave-one-out) -> `figures/{rise,loo}_affordance_verbs/` |
| `compare_verb_importance_methods.py` | agreement of RISE / LOO with the direct labels (exact, cluster, rank, Spearman, mismatched-part control); no model needed |
| `verify_methods.py` | correctness checks for forced-choice scoring (`fc`, `fcdiag`) and RISE (`rise`) -> `json/verification/` |
| `run_gat_inference.py` | GAT (Aff-Grasp) inference + comparison figures -> `figures/gat_comparison/` |
| `region_information_gain.py`, `region_information_gain_size_controlled.py`, `rise_attribution.py`, `visualize_four_panel.py`, `visualize_rise.py`, `visualize_ig_single_panel.py` | legacy D0-based information gain / RISE (see below); results in `json/information_gain/` |
| `simple-probing.py` | original functionality-probe script (predates this work; untouched) |
| `VOCABULARY_REVIEW.md` | running list of AGD20K vocabulary gaps (`wear` added; `pull` candidate) |
| `archive/` | superseded / abandoned scripts, moved rather than deleted; `archive/pre_cleanup_snapshot/` is a copy of every script as it was before the cleanup |

## Pipeline

1. **Parts.** `load_regions(sam_slic)` -> the 30 largest regions -> `partition_regions`: each pixel goes to the
   smallest foreground region containing it; regions >=50% of the image and parts that are >=85% background
   coloured (negative space, e.g. the gaps in a chair back) are dropped.
2. **Direct label.** Each part is asked about separately ("Focusing specifically on part i ...") on the same
   numbered/coloured overlay, scored over all 38 options; label = argmax. Independent calls avoid the
   repetition collapse of generating all parts in one answer; the shared overlay avoids object-level collapse.
3. **Justification.** The label is prefix-forced ("{label}:") and the first generated sentence kept.
4. **Occlusion attribution over verbs.** Same 37-verb question about the whole object, on images with parts
   gray-filled: RISE (random keep/drop, p=0.5, M=300) or leave-one-out. importance(part, verb) = logp(verb | part
   visible) - logp(verb | part hidden).

## Known limitations

* Forced-choice labels are sensitive to the order options are listed in the prompt (`verify_methods.py fc`, A6).
* Stored `clock` parts are 1,401 px (0.3% of its largest part) different from what the current partition code
  produces; re-run `python batch_final_labeling.py clock` to sync.
* Direct labels, RISE and LOO ask different questions (per-part on an overlay vs whole-object on the raw image).

## Legacy: D0-based information gain and RISE

Ask the functionality prompt on the full image, greedy-decode a description `D0`, then measure how much each
SAM region supports `D0`: naive IG = logP(D0 | full) - logP(D0 | region gray-filled); a size-controlled z-score
against translated copies of the mask (biased when a region fills much of the frame -- there is not enough room
to translate it away from itself); and RISE (Petsiuk, Das, Saenko, BMVC 2018): keep/occlude every region with
p=0.5 for M trials, importance = mean logP(D0) when visible - when occluded, with a Welch t-statistic. RISE has no
synthetic-mask machinery, so it does not share the translation bias. These measure self-consistency with the
model's own description, not ground-truth affordance.

```bash
python region_information_gain.py <object> [max_regions]
python region_information_gain_size_controlled.py <object> [K] [max_regions]
python rise_attribution.py <object> [M] [max_regions]
python visualize_four_panel.py <object>; python visualize_rise.py <object>; python visualize_ig_single_panel.py <object>
```
