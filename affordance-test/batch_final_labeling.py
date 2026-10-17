"""
Standard labeling pipeline: SAM+SLIC regions -> disjoint background-filtered parts -> per-part forced-choice AGD20K
label + one-sentence justification.

Each part is an independent model call (no autoregressive copy-paste between parts -- the failure mode that made an
all-parts-in-one-generation version collapse to a single repeated label), but every call sees the same numbered,
coloured overlay of all the object's parts, so the model can contrast part i with the others (the fix for
object-level language-prior collapse, e.g. corkscrew "open" x19). Confirmed on the objects that had collapsed.

Parts are recomputed from the region files (load_regions -> partition_regions), capped to the MAX_REGIONS largest
regions since some shape-texture objects segment into 100+ same-material slivers. Output goes to
json/affordance_justification/<object>_affordance_justification.json; visualize_affordance_justification.py and
visualize_affordance_labels_only.py read it, as do rise/loo_affordance_verbs.py (which reuse its parts).

Usage:
    python batch_final_labeling.py [object ...]                   # default model: Qwen2-VL-7B (all 33 objects)
    python batch_final_labeling.py --model qwen3-vl-8b [object ...]   # or internvl3-8b -- see MODELS below
    python batch_final_labeling.py --model qwen3-vl-8b --resume    # skip objects that already have an output
Every model in MODELS runs the identical forced-choice-over-options prompt/scoring; the default
model's outputs stay at json/affordance_justification/ (what every downstream script reads), other
models write to their own json/affordance_justification_<model_key>/ sibling directory.
Without --resume (the default), existing outputs are overwritten, same as before --resume existed.
With --resume, any object that already has an output file in the target dir is skipped instead --
for long/full-dataset runs under labeling_watchdog.sh (which restarts on an unexpected kill and
would otherwise redo completed objects from scratch every time).
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2VLForConditionalGeneration

from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, JSON_DIR, MODEL_NAME, VOCABULARIES, build_per_part_question, build_prompt_text,
    downscale, first_sentence, numbered_overlay, partition_regions, score_options,
)
from region_information_gain import load_regions, resolve_image_path

HERE = os.path.dirname(os.path.realpath(__file__))
VOCAB = "agd20k"
MAX_NEW_TOKENS = 64
MAX_REGIONS = 30
REGION_SOURCE = "sam_slic"

# Additional open-source VLMs run through the IDENTICAL forced-choice-over-options pipeline as the
# default Qwen2-VL-7B baseline -- prompted zero-shot on this affordance vocabulary, not fine-tuned on
# it, same as the default -- so GAT (the trained specialist, run_gat_inference.py) can be compared
# against more than one untrained VLM instead of just one. GLM-5.2's vision line ("GLM-5V") turned out
# not to have an open release at all, so it's not in this roster; Kimi-VL-A3B was tried and dropped
# after 5 separate transformers-version compatibility shims still didn't clear it (each fix exposed a
# new legacy Cache-API usage in its bundled trust_remote_code modeling file) -- not worth chasing
# further with 4 other models already clean.
# trust_remote_code: a config.json listing a plausible-looking "...ForConditionalGeneration"
# architecture name does NOT mean that class is actually registered in the installed transformers --
# molmo2-8b looked natively supported by that signal and wasn't; transformers itself raised "contains
# custom code which must be executed". It's from Allen AI's official org, so running its bundled code
# is an acceptable risk here -- but don't assume the next addition is native without actually trying
# to load it first.
MODELS = {
    "qwen2-vl-7b": {"repo_id": MODEL_NAME, "model_cls": Qwen2VLForConditionalGeneration},
    "qwen3-vl-8b": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "model_cls": AutoModelForImageTextToText},
    "internvl3-8b": {"repo_id": "OpenGVLab/InternVL3-8B-hf", "model_cls": AutoModelForImageTextToText},
    # safe_generate: repetition_penalty/no_repeat_ngram_size's logits processors gather over the FULL
    # input_ids history (prompt + image placeholder tokens included) as indices into the vocab-sized
    # logits tensor. Molmo2's image placeholder token ids fall outside its text vocab range, so that
    # gather reads out of bounds and crashes CUDA with a device-side assert -- confirmed via a direct
    # repro isolating the failure to this exact call. Dropping those two knobs for this model only
    # (plain greedy decoding instead) avoids it; every other model keeps the repetition controls.
    "molmo2-8b": {
        "repo_id": "allenai/Molmo2-8B", "model_cls": AutoModelForImageTextToText, "trust_remote_code": True,
        "safe_generate": True,
    },
}

OBJECTS = [
    ("known", ["ladle", "chair", "corkscrew", "backpack", "bow-tie", "clock", "coffee-table", "fedora",
               "fire-alarm", "wallet"]),
    ("novel", ["shape_16", "shape_27", "shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12",
               "shape_13", "shape_14", "shape_15", "shape_17", "shape_18", "shape_19", "shape_20",
               "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_28",
               "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38", "shape_39",
               "shape_40"]),  # all 30 novel objects (ids scattered 1-40, see list_novel_objects.py
               # in regions_test/ for how the full set was determined)
    ("shape-shape", ["backpack_amplifier", "bonsai_amplifier", "bonsai_crystal", "cactus_camera",
                      "compass_totem", "crystal_telescope", "dagger_amplifier", "dagger_projector",
                      "drone_dagger"]),
    ("shape-texture", ["bird-feathers_headphones", "bird-feathers_saxophone", "checkerboard_tv-remote",
                        "crochet_broom"]),
]


def build_per_part_justification_prompt(part_idx: int, n_parts: int, label: str, glosses: dict) -> str:
    gloss = f" ({glosses[label]})" if label in glosses else ""
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        f"outlined and tinted a different color. You determined that part {part_idx}'s affordance is "
        f'"{label}"{gloss}. In one short, specific sentence, justify this based on what you observe '
        f"about part {part_idx} specifically (its shape, material, or position) -- and why it might "
        "differ from the other numbered parts. Do not mention the colors or numbering. Answer with "
        f"only the justification sentence.\n\n{label}:"
    )


@torch.no_grad()
def generate_justification(
    model, processor, image: Image.Image, prompt_text: str, label: str, device, safe_generate: bool = False,
) -> str:
    forced_prefix = f"{label}:"
    inputs = processor(text=[prompt_text + forced_prefix], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    if not safe_generate:  # see MODELS["molmo2-8b"]'s comment for why this is skipped for that model
        gen_kwargs.update(repetition_penalty=1.3, no_repeat_ngram_size=3)
    out = model.generate(**inputs, **gen_kwargs)
    continuation = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return first_sentence(continuation)


def top1_label(fc_logprobs: dict) -> str:
    return max(fc_logprobs, key=fc_logprobs.get)


def compute_parts(object_name: str) -> tuple:
    """Disjoint parts as (region_id, area, RLE segmentation) dicts, plus the image path."""
    regions = load_regions(object_name, MAX_REGIONS, REGION_SOURCE)
    img_h, img_w = regions[0]["segmentation"]["size"]
    img_path = resolve_image_path(object_name)
    parts = partition_regions(regions, img_h * img_w, Image.open(img_path).convert("RGB"))
    out = [{
        "region_id": region["region_id"], "area": int(mask.sum()),
        "segmentation": {"size": [img_h, img_w],
                         "counts": mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")},
    } for region, mask in parts]
    print(f"    {len(regions)} regions (capped) -> {len(out)} disjoint parts")
    return out, img_path


def process_object(
    model, processor, option_ids, device, object_name: str, repo_id: str, safe_generate: bool = False,
) -> dict:
    parts, img_path = compute_parts(object_name)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(img_path).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]

    results = []
    for i, part in enumerate(parts, start=1):
        fc_prompt = build_prompt_text(processor, build_per_part_question(i, len(parts), options, glosses))
        fc = score_options(model, processor, overlay, fc_prompt, option_ids, device)
        label = top1_label(dict(zip(options, fc.tolist())))

        just_prompt = build_prompt_text(
            processor, build_per_part_justification_prompt(i, len(parts), label, glosses)
        )
        justification = generate_justification(
            model, processor, overlay, just_prompt, label, device, safe_generate=safe_generate,
        )

        results.append({**part, "vocabularies": {VOCAB: {"label": label, "justification": justification}}})
        print(f"    {part['region_id']:>28s}  {label:>14s}: {justification}")

    return {
        "object": object_name, "model": repo_id, "image_path": img_path,
        "region_source": REGION_SOURCE, "method": "per_part_with_comparative_context",
        "regions": results,
    }


def main():
    args = sys.argv[1:]
    resume = "--resume" in args  # skip objects whose output file already exists, instead of overwriting
    args = [a for a in args if a != "--resume"]
    model_key = "qwen2-vl-7b"
    if args and args[0] == "--model":
        model_key = args[1]
        args = args[2:]
    if model_key not in MODELS:
        print(f"Unknown --model {model_key!r}; choices: {sorted(MODELS)}")
        return
    repo_id, model_cls = MODELS[model_key]["repo_id"], MODELS[model_key]["model_cls"]
    trust_remote_code = MODELS[model_key].get("trust_remote_code", False)
    safe_generate = MODELS[model_key].get("safe_generate", False)

    # The default model's outputs stay at the original path (everything downstream -- rise/loo scripts,
    # visualize_affordance_*.py -- reads from there); every other model gets its own sibling directory
    # so multi-model runs don't clobber each other or the default.
    out_dir = AFFORDANCE_JUSTIFICATION_DIR if model_key == "qwen2-vl-7b" else os.path.join(
        JSON_DIR, f"affordance_justification_{model_key}"
    )
    os.makedirs(out_dir, exist_ok=True)

    requested = args
    todo = [
        (category, name) for category, names in OBJECTS for name in names
        if (not requested or name in requested)
        and not (resume and os.path.isfile(os.path.join(out_dir, f"{name}_affordance_justification.json")))
    ]
    if not todo:
        print(f"Nothing to do (--resume and every requested object already has an output in {out_dir}).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_key} ({repo_id})...")
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    model = model_cls.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code
    )
    model.eval()
    options = VOCABULARIES[VOCAB]["options"]
    # Not hardcoded to Qwen's "<|im_end|>" -- read off each model's own tokenizer so this generalizes
    # to the other models in MODELS (verified all three currently use "<|im_end|>" as eos_token, but
    # this doesn't assume that holds for whatever gets added next).
    end_token = processor.tokenizer.eos_token
    option_ids = [processor.tokenizer(o + end_token, add_special_tokens=False).input_ids for o in options]

    last_category = None
    for category, name in todo:
        if category != last_category:
            print(f"\n=== {category} ===")
            last_category = category
        print(f"  {name}:")
        try:
            data = process_object(model, processor, option_ids, device, name, repo_id, safe_generate=safe_generate)
        except Exception as e:
            print(f"  {name}: FAILED -- {e}")
            continue
        out_path = os.path.join(out_dir, f"{name}_affordance_justification.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_distinct = len({r["vocabularies"][VOCAB]["label"] for r in data["regions"]})
        print(f"  saved {out_path}  (distinct labels: {n_distinct} of {len(data['regions'])} parts)")


if __name__ == "__main__":
    main()
