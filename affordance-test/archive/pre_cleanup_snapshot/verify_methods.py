"""
Correctness checks for the two measurement methods everything else rests on: forced-choice option scoring
(affordance_labeling.score_options / region_information_gain.sequence_logprob) and RISE
(rise_affordance_verbs.py). These test the implementation (does the code compute what it claims?) separately
from the methods' validity (does the number mean what we want?) -- the latter is only probed where cheap
(option-order sensitivity, replication noise, permutation null).

Modes:
    python -u verify_methods.py fc      # forced-choice scoring: tokenization, cache-free reference, invariances
    python -u verify_methods.py rise    # RISE: occlusion pixels, estimator recovery, replication, permutation null
Results are printed and written to json/verification/<mode>.json
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODE = sys.argv[1] if len(sys.argv) > 1 else "fc"
_argv, sys.argv = sys.argv, sys.argv[:1]  # region_information_gain parses sys.argv at import time
from affordance_labeling import (  # noqa: E402
    AFFORDANCE_JUSTIFICATION_DIR, END_TOKEN, JSON_DIR, MODEL_NAME, VOCABULARIES, build_prompt_text, downscale,
    score_options, softmax,
)
from comparative_affordance_labeling import numbered_overlay  # noqa: E402
from comparative_perpart_labeling import build_per_part_question  # noqa: E402
from rise_affordance_verbs import (  # noqa: E402
    OCCLUSION_FILL, build_object_question, log_softmax, occlude_parts, rise_importance,
)
from region_information_gain import sequence_logprob  # noqa: E402
sys.argv = _argv

OUT_DIR = os.path.join(JSON_DIR, "verification")
os.makedirs(OUT_DIR, exist_ok=True)
VOCAB = "agd20k"
OPTIONS = VOCABULARIES[VOCAB]["options"]
GLOSSES = VOCABULARIES[VOCAB]["glosses"]
RESULTS = {}


def report(name: str, ok: bool, detail: str, **extra):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    RESULTS[name] = {"pass": bool(ok), "detail": detail, **extra}


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, processor, device


def load_object(name: str):
    with open(os.path.join(AFFORDANCE_JUSTIFICATION_DIR, f"{name}_affordance_justification.json")) as f:
        data = json.load(f)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in data["regions"]]
    return data, masks, Image.open(data["image_path"]).convert("RGB")


@torch.no_grad()
def reference_option_logprobs(model, processor, image, prompt_text, device):
    """Cache-free ground truth: one plain forward pass per option over prompt+option+<|im_end|>, summing the
    teacher-forced log-probs of exactly the option tokens (nothing shared with score_options)."""
    p_len = processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"].shape[1]
    out = []
    for o in OPTIONS:
        inp = processor(text=[prompt_text + o + END_TOKEN], images=[image], return_tensors="pt")
        inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
        logits = model(**inp).logits[0].float()
        lp = F.log_softmax(logits, dim=-1)
        ids = inp["input_ids"][0]
        out.append(sum(lp[j - 1, ids[j]].item() for j in range(p_len, len(ids))))
    return np.array(out)


# ---------------------------------------------------------------------------------------------- forced choice
def run_fc(model, processor, device):
    option_ids = [processor.tokenizer(o + END_TOKEN, add_special_tokens=False).input_ids for o in OPTIONS]
    im_end_id = processor.tokenizer.convert_tokens_to_ids(END_TOKEN)

    # test prompts: per-part questions against the shared numbered overlay (the pipeline's real inputs)
    cases = []
    for obj, part_idxs in (("ladle", (1, 3)), ("shape_16", (2,))):
        data, masks, image = load_object(obj)
        overlay = downscale(numbered_overlay(image, masks))
        for i in part_idxs:
            text = build_prompt_text(processor, build_per_part_question(i, len(masks), OPTIONS, GLOSSES))
            cases.append((f"{obj} part {i}", overlay, text))

    # A1: token boundary -- tokenizing prompt+option+END jointly must equal prompt ids + standalone option ids
    bad = []
    for name, img, text in cases[:1]:
        p_ids = processor(text=[text], images=[img], return_tensors="pt")["input_ids"][0].tolist()
        for o, ids in zip(OPTIONS, option_ids):
            joint = processor(text=[text + o + END_TOKEN], images=[img], return_tensors="pt")["input_ids"][0].tolist()
            if joint != p_ids + ids:
                bad.append(o)
    n_multi = sum(len(i) > 2 for i in option_ids)
    report("A1_tokenization_boundary", not bad and all(i[-1] == im_end_id for i in option_ids),
           f"joint tokenization == prompt + standalone option ids for {len(OPTIONS) - len(bad)}/{len(OPTIONS)} "
           f"options ({n_multi} are multi-token); mismatches: {bad}; every option ends in <|im_end|>")

    # A2: KV-cache scorer vs cache-free full-forward reference (catches rope_deltas / position / crop bugs)
    diffs, top_agree, prob_diffs = [], [], []
    for name, img, text in cases:
        fast = score_options(model, processor, img, text, option_ids, device)
        ref = reference_option_logprobs(model, processor, img, text, device)
        d = np.abs(fast - ref)
        multi = np.array([len(i) > 1 for i in option_ids])
        diffs.append((name, float(d.max()), float(d.mean()), float(d[multi].max() if multi.any() else 0)))
        top_agree.append(OPTIONS[int(np.argmax(fast))] == OPTIONS[int(np.argmax(ref))])
        prob_diffs.append(float(np.abs(softmax(fast) - softmax(ref)).max()))
        print(f"    {name}: max|Δ|={d.max():.3f} mean|Δ|={d.mean():.3f} nats; top1 fast={OPTIONS[int(np.argmax(fast))]}"
              f" ref={OPTIONS[int(np.argmax(ref))]}; max softmax-prob diff={prob_diffs[-1]:.4f}")
    worst = max(x[1] for x in diffs)
    report("A2_cache_scorer_matches_full_forward", worst < 0.5 and all(top_agree),
           f"worst per-option |Δlogp|={worst:.3f} nats over {len(cases)} prompts x {len(OPTIONS)} options "
           f"(bf16 noise expected <~0.3; a position/cache bug would be multi-nat); top-1 identical on "
           f"{sum(top_agree)}/{len(cases)}; worst softmax-prob diff={max(prob_diffs):.4f}", per_case=diffs)

    # A3/A4: scoring order independence (cache contamination) and determinism
    name, img, text = cases[0]
    s0 = score_options(model, processor, img, text, option_ids, device)
    s_rev = score_options(model, processor, img, text, option_ids[::-1], device)[::-1]
    s_rep = score_options(model, processor, img, text, option_ids, device)
    report("A3_no_cache_contamination", float(np.abs(s0 - s_rev).max()) < 0.05,
           f"scoring options in reverse order changes logp by at most {np.abs(s0 - s_rev).max():.4f} nats")
    report("A4_deterministic", float(np.abs(s0 - s_rep).max()) < 1e-6,
           f"repeat call max |Δ|={np.abs(s0 - s_rep).max():.2e}")

    # A5: forced-choice argmax vs what the model actually generates (greedy) for the same prompt
    matches, masses = [], []
    for name, img, text in cases:
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen_text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        s = score_options(model, processor, img, text, option_ids, device)
        masses.append(float(np.exp(s).sum()))
        matches.append((name, gen_text, OPTIONS[int(np.argmax(s))]))
    report("A5_argmax_matches_greedy_generation", all(g == a for _, g, a in matches),
           f"greedy generation == scored argmax on {sum(g == a for _, g, a in matches)}/{len(matches)} prompts; "
           f"probability mass on the 38 listed options (pre-softmax): {[round(m, 3) for m in masses]}", cases=matches)

    # A6 (validity, not a bug check): sensitivity to the ORDER options are listed in the prompt
    rng = np.random.default_rng(0)
    n_perm, stable, agree_canon, rows = 5, 0, [], []
    n_parts_total = 0
    for obj in ("ladle", "shape_16"):
        data, masks, image = load_object(obj)
        overlay = downscale(numbered_overlay(image, masks))
        for i in range(1, len(masks) + 1):
            n_parts_total += 1
            labels = []
            for perm in [list(range(len(OPTIONS)))] + [rng.permutation(len(OPTIONS)).tolist() for _ in range(n_perm)]:
                listed = [OPTIONS[j] for j in perm]
                text = build_prompt_text(processor, build_per_part_question(i, len(masks), listed, GLOSSES))
                labels.append(OPTIONS[int(np.argmax(score_options(model, processor, overlay, text, option_ids, device)))])
            stable += len(set(labels)) == 1
            agree_canon.append(np.mean([l == labels[0] for l in labels[1:]]))
            rows.append((obj, i, labels))
            print(f"    {obj} part {i}: {labels}")
    report("A6_option_order_sensitivity", True,
           f"top-1 label identical across all {n_perm + 1} orderings for {stable}/{n_parts_total} parts; mean fraction "
           f"of shuffled orderings agreeing with the canonical order = {np.mean(agree_canon):.2f} "
           f"(informational -- <1 means the label depends partly on list position)", rows=rows)

    # A7: controls -- a gray image should not yield confident affordances; part index must matter
    gray = Image.new("RGB", (512, 512), OCCLUSION_FILL)
    text = build_prompt_text(processor, build_per_part_question(1, 4, OPTIONS, GLOSSES))
    p = softmax(score_options(model, processor, gray, text, option_ids, device))
    top3 = [(OPTIONS[i], round(float(p[i]), 3)) for i in np.argsort(-p)[:3]]
    data, masks, image = load_object("ladle")
    overlay = downscale(numbered_overlay(image, masks))
    dists = [softmax(score_options(model, processor, overlay, build_prompt_text(
        processor, build_per_part_question(i, len(masks), OPTIONS, GLOSSES)), option_ids, device))
        for i in range(1, len(masks) + 1)]
    tv = [0.5 * np.abs(dists[a] - dists[b]).sum() for a in range(len(dists)) for b in range(a + 1, len(dists))]
    report("A7_controls", True,
           f"blank gray image top-3 (informational): {top3}; ladle: mean total-variation between parts' "
           f"distributions on the same overlay = {np.mean(tv):.2f} (0 would mean the part index is ignored)")

    # A8: sequence_logprob (used by the D0-based IG/RISE scripts) vs HF's own loss on the same tokens
    with open(os.path.join(JSON_DIR, "..", "shape_16_region_information_gain.json")) as f:
        d0 = json.load(f)["baseline_description"]
    data, masks, image = load_object("shape_16")
    from region_information_gain import build_prompt_text as ig_prompt, load_nonce_word
    ptxt = ig_prompt(processor, load_nonce_word("shape_16"))
    img = downscale(image)
    total, n_tok = sequence_logprob(model, processor, img, ptxt, d0, device)
    full = processor(text=[ptxt + d0], images=[img], return_tensors="pt")
    p_len = processor(text=[ptxt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    ids = full["input_ids"].to(device)
    labels = ids.clone()
    labels[:, :p_len] = -100
    with torch.no_grad():
        loss = model(**{k: v.to(device) for k, v in full.items()}, labels=labels).loss.item()
    hf_total = -loss * (ids.shape[1] - p_len)
    joint_prefix_ok = full["input_ids"][0, :p_len].tolist() == processor(
        text=[ptxt], images=[img], return_tensors="pt")["input_ids"][0].tolist()
    report("A8_sequence_logprob_matches_hf_loss", abs(total - hf_total) < 0.5 and joint_prefix_ok and n_tok == ids.shape[1] - p_len,
           f"sequence_logprob={total:.3f} vs HF-loss-derived={hf_total:.3f} over {n_tok} tokens; prompt ids are an "
           f"exact prefix of prompt+D0 ids: {joint_prefix_ok}")



def run_fc_diag(model, processor, device):
    """Where does the cache scorer differ from the full-forward reference? Numerical noise vs a position/cache bug:
    single-token options use only the prefill logits (no cache continuation), multi-token options also use the
    cached continuation. A cache bug would show up as a bias/extra error on the multi-token options only; bf16
    rounding shows up as error growing with |logp| for both groups and with zero mean."""
    option_ids = [processor.tokenizer(o + END_TOKEN, add_special_tokens=False).input_ids for o in OPTIONS]
    multi = np.array([len(i) > 1 for i in option_ids])
    rows = []
    for obj, i in (("ladle", 1), ("ladle", 3), ("shape_16", 2), ("shape_16", 5), ("shape_16", 1)):
        data, masks, image = load_object(obj)
        overlay = downscale(numbered_overlay(image, masks))
        text = build_prompt_text(processor, build_per_part_question(i, len(masks), OPTIONS, GLOSSES))
        fast = score_options(model, processor, overlay, text, option_ids, device)
        ref = reference_option_logprobs(model, processor, overlay, text, device)
        rows.append((fast, ref))
    fast = np.concatenate([r[0] for r in rows]); ref = np.concatenate([r[1] for r in rows])
    m = np.tile(multi, len(rows))
    d = fast - ref
    likely = ref > -6  # options with >0.25% probability -- the ones that can affect a label
    lines = []
    for nm, sel in (("single-token", ~m), ("multi-token", m)):
        lines.append(f"{nm}: n={sel.sum()}, mean signed diff={d[sel].mean():+.4f}, mean |diff|={np.abs(d[sel]).mean():.4f}, "
                     f"max |diff|={np.abs(d[sel]).max():.3f}")
    lines.append(f"options with logp>-6 (n={likely.sum()}): max |diff|={np.abs(d[likely]).max():.3f}, "
                 f"mean |diff|={np.abs(d[likely]).mean():.4f}; options with logp<=-6 (n={(~likely).sum()}): "
                 f"max |diff|={np.abs(d[~likely]).max():.3f}")
    lines.append(f"corr(|diff|, |logp|)={np.corrcoef(np.abs(d), np.abs(ref))[0, 1]:.2f}")
    worst = np.argsort(-np.abs(d))[:5]
    lines.append("worst 5 (option, multi?, ref logp, diff): " + "; ".join(
        f"{OPTIONS[k % len(OPTIONS)]}, {bool(m[k])}, {ref[k]:.1f}, {d[k]:+.2f}" for k in worst))
    for l in lines:
        print("   ", l)
    ok = (np.abs(d[likely]).max() < 0.15 and abs(d[m].mean()) < 0.05)
    report("A2b_diagnose_cache_vs_reference_gap", ok, " | ".join(lines))


# ------------------------------------------------------------------------------------------------------ RISE
def run_rise(model, processor, device):
    rng = np.random.default_rng(123)
    jobs = [(j.split(":")[0], tuple(int(x) for x in j.split(":")[1].split(","))) for j in
            os.environ.get("VERIFY_RISE_JOBS", "ladle:1,2;shape_16:1").split(";")]
    if "VERIFY_RISE_JOBS" not in os.environ:
        run_rise_synthetic(rng)
    run_rise_model(model, processor, device, jobs)


def run_rise_synthetic(rng):
    # B1: occlusion pixels -- kept parts untouched, hidden parts exactly the fill, everything else untouched
    data, masks, image = load_object("ladle")
    base = np.array(image)
    keep = rng.random(len(masks)) < 0.5
    keep[0], keep[-1] = True, False
    out = np.array(occlude_parts(image, masks, keep))
    union_hidden = np.zeros(base.shape[:2], bool)
    for m, k in zip(masks, keep):
        if not k:
            union_hidden |= m
    ok = (np.all(out[union_hidden] == OCCLUSION_FILL) and np.all(out[~union_hidden] == base[~union_hidden])
          and union_hidden.any())
    overlap = int(sum(m.astype(int) for m in masks).max())
    report("B1_occlusion_pixels", bool(ok) and overlap == 1,
           f"hidden pixels == {OCCLUSION_FILL}, all other pixels byte-identical to the original; masks are "
           f"disjoint (max overlap {overlap}); {int(union_hidden.sum())} px hidden for keep={keep.astype(int).tolist()}")

    # B2: estimator recovery on a planted ground truth (no model): logp(v|trial) = c_v + sum_i beta[i,v]*keep_i + noise
    n, V, M = 8, 6, 300
    beta = np.zeros((n, V))
    beta[0, 0], beta[1, 1], beta[2, 2], beta[3, 0] = 2.0, -1.5, 1.0, -1.0  # planted; other cells truly zero
    errs, sign_ok = [], []
    for rep in range(20):
        keep_m = rng.random((M, n)) < 0.5
        lp = keep_m.astype(float) @ beta + rng.normal(0, 0.5, (M, V)) + rng.normal(0, 2, V)
        lp = log_softmax_rows(lp)  # a real scorer's output is normalised over verbs
        est = rise_importance(keep_m, lp)["bits"]
        errs.append(est)
    est_mean = np.mean(errs, 0)
    # softmax normalisation couples verbs, so compare the un-normalised planted effect on the logits: recover the
    # planted contrast up to a per-part additive shift shared by all verbs
    centered_est = est_mean - est_mean.mean(1, keepdims=True)
    centered_true = beta / np.log(2) - (beta / np.log(2)).mean(1, keepdims=True)
    err = float(np.abs(centered_est - centered_true).max())
    # un-normalised recovery (estimator only, without softmax coupling)
    raw_errs = []
    for rep in range(20):
        keep_m = rng.random((M, n)) < 0.5
        lp = keep_m.astype(float) @ beta + rng.normal(0, 0.5, (M, V))
        raw_errs.append(rise_importance(keep_m, lp)["bits"] * np.log(2))
    raw_err = float(np.abs(np.mean(raw_errs, 0) - beta).max())
    report("B2_estimator_recovers_planted_effects", raw_err < 0.05 and err < 0.15,
           f"un-normalised: max |mean estimate - planted| = {raw_err:.3f} nats over 20 replications (sign and size "
           f"recovered); with softmax coupling, per-part-centred error = {err:.3f} bits (softmax makes a part's effect "
           f"on one verb bleed onto the others, which is a property of the target, not a bug)")

    # B2b: estimator's own noise level at M=300 with a null model (no effect) -> how big does |t| get by chance
    null_t = []
    for rep in range(50):
        keep_m = rng.random((M, 11)) < 0.5
        lp = rng.normal(0, 1.0, (M, 37))
        null_t.append(np.abs(rise_importance(keep_m, lp)["t"]).ravel())
    null_t = np.concatenate(null_t)
    report("B2b_null_t_calibration", abs(np.mean(null_t > 1.96) - 0.05) < 0.01,
           f"with no true effect, fraction of cells with |t|>1.96 = {np.mean(null_t > 1.96):.3f} (nominal 0.05); "
           f"99th percentile |t| = {np.percentile(null_t, 99):.2f}")



def run_rise_model(model, processor, device, jobs):
    # B3: real model. (i) all-keep trial == the full image; (ii) replication across seeds + split-half; (iii) permutation null
    replicate_importance = {}
    for obj, seeds in jobs:
        data, masks, image = load_object(obj)
        n = len(masks)
        option_ids = [processor.tokenizer(o + END_TOKEN, add_special_tokens=False).input_ids for o in OPTIONS]
        ptxt = build_prompt_text(processor, build_object_question(OPTIONS, GLOSSES))
        full_lp = log_softmax(score_options(model, processor, downscale(image), ptxt, option_ids, device))
        allkeep_lp = log_softmax(score_options(
            model, processor, downscale(occlude_parts(image, masks, np.ones(n, bool))), ptxt, option_ids, device))
        report(f"B3i_{obj}_allkeep_equals_full", float(np.abs(full_lp - allkeep_lp).max()) < 1e-6,
               f"max |Δlogp| between the untouched image and the all-parts-kept trial = "
               f"{np.abs(full_lp - allkeep_lp).max():.2e}")
        reps = []
        with open(os.path.join(JSON_DIR, "rise_affordance_verbs", f"{obj}_rise_affordance_verbs.json")) as f:
            stored = json.load(f)
        reps.append(("seed0(stored)", np.array([[r["importance_bits"][v] for v in OPTIONS] for r in stored["regions"]]),
                     None))
        for seed in seeds:
            r_seed = np.random.default_rng(seed)
            keep_m = r_seed.random((300, n)) < 0.5
            lp = np.zeros((300, len(OPTIONS)))
            for t in range(300):
                lp[t] = log_softmax(score_options(
                    model, processor, downscale(occlude_parts(image, masks, keep_m[t])), ptxt, option_ids, device))
                if (t + 1) % 100 == 0:
                    print(f"    {obj} seed {seed}: {t + 1}/300")
            np.savez(os.path.join(OUT_DIR, f"rise_{obj}_seed{seed}.npz"), keep=keep_m, lp=lp)
            imp = rise_importance(keep_m, lp)
            reps.append((f"seed{seed}", imp["bits"], (keep_m, lp, imp)))
            # split-half within this seed
            h = 150
            a = rise_importance(keep_m[:h], lp[:h])["bits"]
            b = rise_importance(keep_m[h:], lp[h:])["bits"]
            sh = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            am = float(np.mean(a.argmax(1) == b.argmax(1)))
            print(f"    {obj} seed {seed}: split-half corr={sh:.3f}, per-part argmax-verb agreement={am:.2f}")
            report(f"B3ii_{obj}_seed{seed}_split_half", True,
                   f"halves of 150 trials: correlation of part x verb bits = {sh:.3f}; per-part argmax verb agrees on "
                   f"{am:.2f} of parts (informational: this is the reliability of a single M=150 run)")
            # permutation null: break the link between keep pattern and scores
            perm_t = []
            for _ in range(200):
                perm_t.append(np.abs(rise_importance(keep_m[r_seed.permutation(300)], lp)["t"]).ravel())
            thr = float(np.percentile(np.concatenate(perm_t), 99))
            real_t = np.abs(imp["t"])
            report(f"B3iii_{obj}_seed{seed}_permutation_null", True,
                   f"99th percentile |t| under shuffled keep patterns = {thr:.2f}; {np.mean(real_t > thr):.0%} of the "
                   f"{real_t.size} real part x verb cells exceed it; {int((real_t.max(1) > thr).sum())}/{n} parts have at "
                   f"least one significant verb")
        # cross-replicate agreement (seed0 stored + fresh seeds)
        names = [r[0] for r in reps]
        mats = [r[1] for r in reps]
        for a_i in range(len(mats)):
            for b_i in range(a_i + 1, len(mats)):
                c = float(np.corrcoef(mats[a_i].ravel(), mats[b_i].ravel())[0, 1])
                am = float(np.mean(mats[a_i].argmax(1) == mats[b_i].argmax(1)))
                top3 = float(np.mean([len(set(np.argsort(-mats[a_i][p])[:3]) & set(np.argsort(-mats[b_i][p])[:3])) / 3
                                      for p in range(n)]))
                report(f"B3iv_{obj}_{names[a_i]}_vs_{names[b_i]}", True,
                       f"full-run replicates (M=300, different random masks): corr={c:.3f}, argmax verb agrees on "
                       f"{am:.2f} of parts, top-3 verb overlap={top3:.2f}")
        replicate_importance[obj] = {n_: m.tolist() for n_, m in zip(names, mats)}
    RESULTS["replicates"] = replicate_importance


def log_softmax_rows(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def main():
    model, processor, device = load_model()
    {"fc": run_fc, "fcdiag": run_fc_diag, "rise": run_rise}[MODE](model, processor, device)
    path = os.path.join(OUT_DIR, f"{MODE}{os.environ.get('VERIFY_TAG', '')}.json")
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    n_fail = sum(1 for v in RESULTS.values() if isinstance(v, dict) and v.get("pass") is False)
    print(f"\nsaved {path}; {n_fail} FAIL")


if __name__ == "__main__":
    main()
