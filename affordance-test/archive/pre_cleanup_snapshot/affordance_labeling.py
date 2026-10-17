"""
Training-free per-part affordance labelling with a VLM.

1. Partition. SAM regions are often nested (a whole-object region containing its parts), so a
   "handle" never exists as its own region. Each pixel is assigned to the smallest foreground
   region containing it, giving disjoint parts (e.g. whole ladle minus bowl minus hole = handle).
   Regions covering >=50% of the image (background) are not labelled.
2. Forced choice. Show Qwen2-VL the image with one part highlighted (rest darkened, part outlined in
   magenta) and score every option of a closed vocabulary as the answer (teacher-forced log-prob,
   so we get a full distribution instead of one sampled guess).
3. Yes/No per affordance. For every option, an independent question "does the highlighted part afford
   X?"; P(Yes) is read from the Yes/No token logits. No competition between options.
4. Object-level scores. The same forced-choice and Yes/No questions on the un-highlighted image, stored
   so the visualization can form a damped contrast  log p(part) - alpha * log p(object)  for any alpha.

Two vocabularies: agd20k (36 AGD20K verbs, from OOAL's SEEN_AFF; + "none") and umd (7 UMD Part
Affordance classes, Myers et al. 2015; + "none"; glosses paraphrased and object-agnostic).

Images are downscaled to MAX_SIDE px: the shared GPUs are often too full for full-resolution
vision-tower activations, and answers are somewhat resolution-sensitive, so compare runs at equal MAX_SIDE.

Usage:
    [REGION_SOURCE=sam_slic|sam] python affordance_labeling.py [object_name] [max_regions]   # default: shape_16

Regions come from SAM+SLIC by default (SLIC superpixels fill the pixels SAM leaves unmasked; see
regions/region_features/slic/); set REGION_SOURCE=sam for SAM masks only. Outputs are tagged by source.
"""
import os
import sys
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, DynamicCache, Qwen2VLForConditionalGeneration

from region_information_gain import MODEL_NAME, load_regions, resolve_image_path, sequence_logprob

HERE = os.path.dirname(os.path.realpath(__file__))
# Output JSONs are organized by which pipeline stage produced them, not left flat in HERE --
# imported by every script that reads or writes one of these, so the layout only needs to change
# in one place.
JSON_DIR = os.path.join(HERE, "json")
AFFORDANCE_JUSTIFICATION_DIR = os.path.join(JSON_DIR, "affordance_justification")  # final, standard results
COMPARATIVE_LABELS_DIR = os.path.join(JSON_DIR, "comparative_labels")  # all-at-once (superseded; kept as a record)
COMPARATIVE_PERPART_LABELS_DIR = os.path.join(JSON_DIR, "comparative_perpart_labels")  # fix-validation run
for _d in (AFFORDANCE_JUSTIFICATION_DIR, COMPARATIVE_LABELS_DIR, COMPARATIVE_PERPART_LABELS_DIR):
    os.makedirs(_d, exist_ok=True)
BACKGROUND_AREA_FRAC = 0.5
MIN_PART_FRAC = 0.001
DIM_FACTOR = 0.3
CONTOUR_RGB = (255, 0, 255)
END_TOKEN = "<|im_end|>"
MAX_SIDE = 512

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
MAX_REGIONS = int(sys.argv[2]) if len(sys.argv) > 2 else None
REGION_SOURCE = os.environ.get("REGION_SOURCE", "sam_slic")  # 'sam_slic' (default) or 'sam'


def labels_path(object_name: str) -> str:
    return os.path.join(HERE, f"{object_name}_{REGION_SOURCE}_affordance_labels.json")

AGD20K_VERBS = [
    "beat", "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", "drink_with", "eat",
    "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel", "pick_up",
    "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo", "talk_on",
    "text_on", "throw", "type_on", "wash", "write",
    # Additions beyond the standard AGD20K/OOAL 36-verb set, made for this project -- see
    # VOCABULARY_REVIEW.md for the running list of gaps found and reasoning behind each addition.
    "wear",
]
UMD_GLOSSES = {
    "grasp": "can be picked up or held with the fingers",
    "wrap-grasp": "can be grasped by wrapping the whole hand around it",
    "cut": "can separate one object into two",
    "scoop": "has a curved surface for scooping material up",
    "contain": "has a cavity that holds things inside",
    "pound": "can be used to strike or pound",
    "support": "can hold up other objects",
}

VOCABULARIES = {
    "agd20k": {"options": [v.replace("_", " ") for v in AGD20K_VERBS] + ["none"], "glosses": {}},
    "umd": {"options": list(UMD_GLOSSES) + ["none"], "glosses": UMD_GLOSSES},
}

HIGHLIGHT_INTRO = (
    "The image shows an object. One part of it is highlighted: it is at full brightness and "
    "outlined in magenta, while the rest of the image is darkened (use it only as context). "
)


def build_question(options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    return (
        HIGHLIGHT_INTRO
        + "Which single affordance does the highlighted part offer? "
        + f"Choose exactly one of: {listed}. "
        + 'If the highlighted part offers none of these (for example plain background), answer "none". '
        + "Answer with the option only."
    )


def build_yes_no_question(option: str, glosses: dict) -> str:
    gloss = f" ({glosses[option]})" if option in glosses else ""
    return HIGHLIGHT_INTRO + f'Does the highlighted part afford the action "{option}"{gloss}? Answer Yes or No.'


def build_prompt_text(processor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def downscale(img: Image.Image) -> Image.Image:
    scale = MAX_SIDE / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def highlight_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(image).astype(np.float32)
    arr[~mask] *= DIM_FACTOR
    out = np.ascontiguousarray(arr.astype(np.uint8))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, round(min(mask.shape) / 300))
    cv2.drawContours(out, contours, -1, CONTOUR_RGB, thickness)
    return Image.fromarray(out)


BG_COLOR_TOLERANCE = 20  # max per-pixel RGB distance from the image's own background to still count
                         # as "background-colored"
BG_COLOR_FRAC_THRESHOLD = 0.85  # a part is dropped as negative space if at least this fraction of its
                                # pixels are background-colored


def estimate_background_color(image: Image.Image) -> np.ndarray:
    """These are studio product photos on a plain, near-uniform backdrop -- a thin border strip is
    almost always pure background (the object is centered and never spans the full frame), so its
    median color is a robust background estimate without needing any segmentation."""
    arr = np.array(image)
    h, w = arr.shape[:2]
    m = max(2, round(min(h, w) * 0.02))
    border = np.concatenate([arr[:m].reshape(-1, 3), arr[-m:].reshape(-1, 3),
                              arr[:, :m].reshape(-1, 3), arr[:, -m:].reshape(-1, 3)])
    return np.median(border, axis=0)


def is_background_like(arr: np.ndarray, mask: np.ndarray, bg_color: np.ndarray) -> bool:
    """True if a part is mostly background color showing through a hole/gap in the object (e.g. the
    open space in a chair's back, or a corkscrew's finger-holes) rather than actual object material --
    SAM+SLIC segments these the same as any other region since they're just as geometrically coherent,
    but they aren't parts of the object at all and should never be scored for affordance."""
    dist = np.linalg.norm(arr[mask].astype(np.float32) - bg_color, axis=-1)
    return float((dist < BG_COLOR_TOLERANCE).mean()) >= BG_COLOR_FRAC_THRESHOLD


def partition_regions(regions: list, img_area: int, image: Image.Image) -> list:
    """Nested/overlapping SAM masks -> disjoint parts: every pixel goes to the smallest foreground
    region containing it. Residuals are cleaned by a morphological opening (removes the thin slivers
    left where nested masks don't line up exactly), components under MIN_PART_FRAC of the image are
    dropped, and components that are mostly background color (negative space, not material) are
    dropped regardless of size. Returns [(source_region, part_mask)], largest source region first."""
    foreground = sorted(
        (r for r in regions if r["area"] < BACKGROUND_AREA_FRAC * img_area), key=lambda r: r["area"], reverse=True
    )
    decoded = [mask_utils.decode(r["segmentation"]).astype(bool) for r in foreground]
    label_map = np.full(decoded[0].shape, -1, dtype=np.int32)
    for i, m in enumerate(decoded):  # largest first, so smaller regions overwrite their parents
        label_map[m] = i
    k = max(3, round(min(label_map.shape) * 0.006)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    arr = np.array(image)
    bg_color = estimate_background_color(image)
    parts = []
    for i, region in enumerate(foreground):
        opened = cv2.morphologyEx((label_map == i).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, comp = cv2.connectedComponents(opened)
        keep = np.zeros(opened.shape, dtype=bool)
        for c in range(1, n):
            component = comp == c
            if component.sum() >= MIN_PART_FRAC * img_area and not is_background_like(arr, component, bg_color):
                keep |= component
        if keep.any():
            parts.append((region, keep))
    return parts


@torch.no_grad()
def score_options(model, processor, image: Image.Image, prompt_text: str, option_ids: list, device) -> np.ndarray:
    """log P(option + <|im_end|> | prompt, image) for every option. The image+prompt prefix is run
    once; each option's continuation reuses that KV cache, which is cropped back afterwards."""
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    cache = DynamicCache()
    out = model(**inputs, past_key_values=cache, use_cache=True)
    first_logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)

    scores = []
    for ids in option_ids:
        lp = first_logp[ids[0]].item()
        if len(ids) > 1:
            cont = torch.tensor([ids[:-1]], device=device)
            n = cont.shape[1]
            o = model(
                input_ids=cont,
                attention_mask=torch.ones(1, prompt_len + n, dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(prompt_len, prompt_len + n, device=device),
            )
            logp = F.log_softmax(o.logits[0].float(), dim=-1)
            lp += sum(logp[j, ids[j + 1]].item() for j in range(n))
            cache.crop(prompt_len)
        scores.append(lp)
    return np.array(scores)


@torch.no_grad()
def yes_prob(model, processor, image: Image.Image, prompt_text: str, yes_no_ids: tuple, device) -> float:
    """P(Yes) renormalised over just the Yes/No tokens at the first answer position."""
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    return torch.softmax(logits[list(yes_no_ids)], dim=0)[0].item()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def top_label(options: list, scores: np.ndarray) -> str:
    probs = softmax(scores)
    i = int(np.argmax(probs))
    return f"{options[i]} ({probs[i]:.2f})"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_regions = load_regions(OBJECT_NAME, MAX_REGIONS, REGION_SOURCE)
    img_h, img_w = all_regions[0]["segmentation"]["size"]
    img_path = resolve_image_path(OBJECT_NAME)
    image = Image.open(img_path).convert("RGB")
    small_full = downscale(image)

    parts = partition_regions(all_regions, img_h * img_w, image)
    print(f"Object: {OBJECT_NAME} | {len(all_regions)} regions ({REGION_SOURCE}) -> {len(parts)} disjoint labelled parts")
    for region, mask in parts:
        print(f"  {region['region_id']:>24s}  source area={region['area']:>7d} -> part area={int(mask.sum()):>7d}")

    print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    yes_no_ids = tuple(processor.tokenizer.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
    assert all(len(t) == 1 for t in yes_no_ids), yes_no_ids
    yes_no_ids = tuple(t[0] for t in yes_no_ids)

    setup, object_level = {}, {}
    for name, vocab in VOCABULARIES.items():
        options = vocab["options"]
        yn_options = [o for o in options if o != "none"]
        fc_prompt = build_prompt_text(processor, build_question(options, vocab["glosses"]))
        yn_prompts = [build_prompt_text(processor, build_yes_no_question(o, vocab["glosses"])) for o in yn_options]
        option_ids = [processor.tokenizer(o + END_TOKEN, add_special_tokens=False).input_ids for o in options]
        setup[name] = (options, yn_options, fc_prompt, yn_prompts, option_ids)
        fc_full = score_options(model, processor, small_full, fc_prompt, option_ids, device)
        yn_full = [yes_prob(model, processor, small_full, p, yes_no_ids, device) for p in yn_prompts]
        object_level[name] = {
            "fc_logprobs": dict(zip(options, fc_full.tolist())),
            "yn_prob": dict(zip(yn_options, yn_full)),
        }
        best = sorted(zip(yn_options, yn_full), key=lambda t: -t[1])[:3]
        print(f"[object-level, no highlight] {name}: forced-choice={top_label(options, fc_full)} | "
              f"yes/no top={[(o, round(p, 2)) for o, p in best]}")

    results = []
    checked = False
    for region, mask in parts:
        highlighted = downscale(highlight_region(image, mask))
        rec = {
            "region_id": region["region_id"],
            "area": int(mask.sum()),
            "source_area": region["area"],
            "segmentation": {
                "size": [img_h, img_w],
                "counts": mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8"),
            },
            "fc_logprobs": {},
            "yn_prob": {},
        }
        line = f"{region['region_id']:>24s}  area={rec['area']:>7d} "
        for name, (options, yn_options, fc_prompt, yn_prompts, option_ids) in setup.items():
            fc = score_options(model, processor, highlighted, fc_prompt, option_ids, device)
            if not checked:
                ref = sequence_logprob(model, processor, highlighted, fc_prompt, options[0] + END_TOKEN, device)[0]
                print(f"[check] KV-cache vs full-forward logP({options[0]!r}): {fc[0]:.3f} vs {ref:.3f}")
                checked = True
            yn = [yes_prob(model, processor, highlighted, p, yes_no_ids, device) for p in yn_prompts]
            rec["fc_logprobs"][name] = dict(zip(options, fc.tolist()))
            rec["yn_prob"][name] = dict(zip(yn_options, yn))
            best_yn = yn_options[int(np.argmax(yn))]
            line += f"| {name}: FC={top_label(options, fc)} YN-top={best_yn} (p={max(yn):.2f}) "
        print(line)
        results.append(rec)

    out = {
        "object": OBJECT_NAME,
        "model": MODEL_NAME,
        "image_path": img_path,
        "region_source": REGION_SOURCE,
        "max_side": MAX_SIDE,
        "vocabularies": {k: v["options"] for k, v in VOCABULARIES.items()},
        "forced_choice_questions": {k: build_question(v["options"], v["glosses"]) for k, v in VOCABULARIES.items()},
        "object_level": object_level,
        "regions": results,
    }
    out_path = labels_path(OBJECT_NAME)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
