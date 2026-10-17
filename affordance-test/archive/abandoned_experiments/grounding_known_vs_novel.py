"""
Grounding test 1: known (real-named) vs novel (nameless/synthetic NVRD) objects.

"VLMs Need Words" (arXiv 2604.02486) shows VLMs bypass visual comparison and answer from a
language prior whenever the entity in view maps to a known name, and only actually look at
pixels when no such name-shortcut exists. Our own affordance labeling never injects an object
name into the prompt, but the VLM can still silently recognize a named object (e.g. "ladle")
from its own training-time visual knowledge and answer from "what ladles are for" rather than
from the specific highlighted pixels. A synthetic NVRD "novel" shape (e.g. shape_16) has no
real-world category to fall back on, so it can't benefit from that shortcut.

If per-part label confidence and cross-part differentiation hold up as well on novel objects as
on known ones, that's evidence the pipeline is actually reading the highlighted region. If
novel objects collapse to low-confidence, near-uniform-across-parts labels while known objects
look sharp and object-appropriate, that's the language-prior signature.

Per object, per vocabulary, computes:
    mean_top1_prob   : mean of the winning option's softmax probability, averaged over parts
                        -- how confident the model is, on average
    label_diversity   : (# distinct top-1 labels used) / (# parts) -- 1.0 if every part gets a
                        different label, ->0 if the same label is repeated for every part
                        (the "it's a ladle, so everything scoops" collapse)
    mean_entropy_bits : mean Shannon entropy (bits) of each part's full softmax distribution,
                        averaged over parts -- low = confident/peaked, high = spread out/unsure

Requires <object>_<region_source>_affordance_labels.json for each object listed below (run
affordance_labeling.py first).

Usage:
    [REGION_SOURCE=sam_slic|sam] python grounding_known_vs_novel.py
"""
import os
import json

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
REGION_SOURCE = os.environ.get("REGION_SOURCE", "sam_slic")
KNOWN_OBJECTS = ["ladle", "chair", "corkscrew"]
NOVEL_OBJECTS = ["shape_16", "shape_27"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def analyze(object_name: str) -> dict:
    path = os.path.join(HERE, f"{object_name}_{REGION_SOURCE}_affordance_labels.json")
    with open(path) as f:
        data = json.load(f)
    out = {}
    for vocab in data["vocabularies"]:
        top1_probs, entropies, top1_labels = [], [], []
        for part in data["regions"]:
            options = list(part["fc_logprobs"][vocab])
            probs = softmax(np.array([part["fc_logprobs"][vocab][o] for o in options]))
            i = int(np.argmax(probs))
            top1_probs.append(probs[i])
            top1_labels.append(options[i])
            entropies.append(entropy_bits(probs))
        out[vocab] = {
            "n_parts": len(data["regions"]),
            "mean_top1_prob": float(np.mean(top1_probs)),
            "label_diversity": len(set(top1_labels)) / len(top1_labels),
            "mean_entropy_bits": float(np.mean(entropies)),
            "top1_labels": top1_labels,
        }
    return out


def main():
    rows = []
    for group, objects in [("known", KNOWN_OBJECTS), ("novel", NOVEL_OBJECTS)]:
        for obj in objects:
            path = os.path.join(HERE, f"{obj}_{REGION_SOURCE}_affordance_labels.json")
            if not os.path.isfile(path):
                print(f"skipping {obj}: {path} not found (run affordance_labeling.py {obj} first)")
                continue
            rows.append((group, obj, analyze(obj)))

    for vocab in ["agd20k", "umd"]:
        print(f"\n=== {vocab} ===")
        print(f"{'group':>6s} {'object':>12s} {'n_parts':>8s} {'mean_top1_prob':>15s} "
              f"{'label_diversity':>16s} {'mean_entropy_bits':>18s}  top1_labels")
        for group, obj, res in rows:
            r = res[vocab]
            print(f"{group:>6s} {obj:>12s} {r['n_parts']:>8d} {r['mean_top1_prob']:>15.3f} "
                  f"{r['label_diversity']:>16.3f} {r['mean_entropy_bits']:>18.3f}  {r['top1_labels']}")
        for group in ["known", "novel"]:
            vals = [res[vocab] for g, o, res in rows if g == group]
            if vals:
                print(f"  {group} mean: top1_prob={np.mean([v['mean_top1_prob'] for v in vals]):.3f}  "
                      f"diversity={np.mean([v['label_diversity'] for v in vals]):.3f}  "
                      f"entropy={np.mean([v['mean_entropy_bits'] for v in vals]):.3f}")


if __name__ == "__main__":
    main()
