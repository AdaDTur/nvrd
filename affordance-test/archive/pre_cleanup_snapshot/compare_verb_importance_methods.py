"""
How well do the occlusion-derived verb importances (RISE, leave-one-out) line up with the direct per-part
forced-choice method? Uses stored JSONs only (no model): json/rise_affordance_verbs, json/loo_affordance_verbs.

For each object and each importance measure (RISE bits / RISE dProb / LOO bits / LOO dProb):
  exact    -- argmax verb of the measure == direct label
  cluster  -- argmax verb is in the same hand-defined cluster as the direct label (CLUSTERS below)
  rank     -- mean rank (1 = best of 38) of the direct label's verb in the measure's ordering
  spearman -- mean over parts of Spearman(direct method's probability over the 38 verbs, measure's importance over the 38)
  mismatch -- the same 'exact' / 'cluster' agreement when each part's measure row is compared with a DIFFERENT
              part's direct label (average over all other parts): if part-specific information is shared, matched
              agreement should exceed this.
Chance for 'exact' is 1/38; for 'cluster' see the per-object printout.

Usage:
    python compare_verb_importance_methods.py object [object ...]
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
JSON_DIR = os.path.join(HERE, "json")

CLUSTERS = {
    "grasp/move": ["hold", "carry", "pick up", "lift", "drag", "catch", "throw", "pack", "push"],
    "strike": ["beat", "boxing", "hit", "kick", "swing"],
    "cut/pierce": ["cut", "cut with", "peel", "stick"],
    "liquid/food": ["drink with", "sip", "pour", "stir", "eat"],
    "device use": ["type on", "text on", "talk on", "take photo", "look out", "open"],
    "tool on surface": ["write", "brush with", "wash"],
    "body/support": ["sit on", "lie on", "ride", "jump", "wear"],
    "none": ["none"],
}
CLUSTER_OF = {v: c for c, vs in CLUSTERS.items() for v in vs}


def ranks_desc(x):
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), int)
    r[order] = np.arange(1, len(x) + 1)
    return r


def spearman(a, b):
    ra, rb = ranks_desc(a).astype(float), ranks_desc(b).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(measure, direct_label, direct_probs, options):
    """measure, direct_probs: (n_parts, n_verbs); direct_label: list of n_parts verbs."""
    n = len(direct_label)
    top = [options[int(np.argmax(measure[i]))] for i in range(n)]
    exact = np.mean([top[i] == direct_label[i] for i in range(n)])
    clus = np.mean([CLUSTER_OF[top[i]] == CLUSTER_OF[direct_label[i]] for i in range(n)])
    rank = np.mean([ranks_desc(measure[i])[options.index(direct_label[i])] for i in range(n)])
    sp = np.nanmean([spearman(direct_probs[i], measure[i]) for i in range(n)])
    mm_exact = np.mean([top[i] == direct_label[j] for i in range(n) for j in range(n) if j != i])
    mm_clus = np.mean([CLUSTER_OF[top[i]] == CLUSTER_OF[direct_label[j]] for i in range(n) for j in range(n) if j != i])
    return {"exact": float(exact), "cluster": float(clus), "mean_rank": float(rank), "spearman": float(sp),
            "mismatch_exact": float(mm_exact), "mismatch_cluster": float(mm_clus)}


def main(objects):
    all_out = {}
    for obj in objects:
        with open(os.path.join(JSON_DIR, "loo_affordance_verbs", f"{obj}_loo_affordance_verbs.json")) as f:
            loo = json.load(f)
        with open(os.path.join(JSON_DIR, "rise_affordance_verbs", f"{obj}_rise_affordance_verbs.json")) as f:
            rise = json.load(f)
        options = list(loo["regions"][0]["fc_probs"])
        assert [r["region_id"] for r in loo["regions"]] == [r["region_id"] for r in rise["regions"]]
        label = [r["forced_choice_label"] for r in loo["regions"]]
        probs = np.array([[r["fc_probs"][v] for v in options] for r in loo["regions"]])

        def mat(d, key):
            return np.array([[r[key][v] for v in options] for r in d["regions"]])

        measures = {"RISE bits": mat(rise, "importance_bits"), "RISE dProb": mat(rise, "delta_prob"),
                    "LOO bits": mat(loo, "importance_bits"), "LOO dProb": mat(loo, "delta_prob")}
        n = len(label)
        cluster_chance = np.mean([CLUSTER_OF[options[k]] == CLUSTER_OF[label[i]] for i in range(n)
                                  for k in range(len(options))])
        print(f"\n=== {obj}: {n} parts; direct labels = {label}")
        print(f"    chance: exact={1 / len(options):.3f}, cluster={cluster_chance:.3f}")
        rows = {}
        for name, m in measures.items():
            rows[name] = evaluate(m, label, probs, options)
            e = rows[name]
            print(f"    {name:11s} exact={e['exact']:.2f} (mismatched-part {e['mismatch_exact']:.2f})  "
                  f"cluster={e['cluster']:.2f} (mismatched {e['mismatch_cluster']:.2f})  "
                  f"mean rank of direct verb={e['mean_rank']:.1f}/38  Spearman={e['spearman']:+.2f}")
            tops = [options[int(np.argmax(m[i]))] for i in range(n)]
            print(f"                 argmax per part: {tops}")
        # LOO vs RISE agree with each other?
        for kind in ("bits", "dProb"):
            a, b = measures[f"RISE {kind}"], measures[f"LOO {kind}"]
            same = np.mean([np.argmax(a[i]) == np.argmax(b[i]) for i in range(n)])
            corr = np.nanmean([spearman(a[i], b[i]) for i in range(n)])
            print(f"    RISE vs LOO ({kind}): same argmax verb on {same:.2f} of parts; mean per-part Spearman={corr:+.2f}")
        all_out[obj] = {"labels": label, "metrics": rows}
    path = os.path.join(JSON_DIR, "verification", "compare_importance_vs_direct.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"clusters": CLUSTERS, "results": all_out}, f, indent=2)
    print("\nsaved", path)


if __name__ == "__main__":
    main(sys.argv[1:])
