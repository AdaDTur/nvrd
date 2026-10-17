"""
Per-region information gain for affordance/functionality understanding.

Idea: `simple-probing.py` asks a VLM "Describe what functionality this object serves."
for the full image and gets a description D0 (greedy decoding). For each SAM region r
(from regions_test/output/sam_masks/<object>.json), we occlude that region (gray fill)
and ask: how much less likely is the model to produce that *same* description D0 when it
can't see region r?

    IG(r) = log P(D0 | image)  -  log P(D0 | image with region r occluded)

This is a leave-one-region-out log-likelihood attribution (an occlusion-based estimate of
the pointwise mutual information between region r and the model's stated functionality
judgment, conditioned on the rest of the image). IG(r) > 0 means region r made the model's
own functionality description more likely -- i.e. that part of the object is where the
model is "reading" the affordance from. IG(r) <= 0 means the region was irrelevant (or
even made the description less likely, e.g. a distracting background region).

Usage:
    python region_information_gain.py [object_name]   # default: shape_16
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

HERE = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGIONS_TEST = os.path.join(REPO_ROOT, "regions_test")

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
FUNCTIONALITY_PROMPT = (
    "What is the potential functionality/purpose/affordance of this object, "
    "assuming it is NOT a decorative piece?"
)
NONCE_CAPTION = "This image is best described by the reference: {ref}."
MAX_NEW_TOKENS = 256
IG_JSON_DIR = os.path.join(HERE, "json", "information_gain")  # legacy D0-based IG / RISE results
os.makedirs(IG_JSON_DIR, exist_ok=True)
OCCLUSION_FILL = (127, 127, 127)  # neutral gray erasure baseline
MAX_SIDE = 512  # this shared machine often has other jobs holding most of each GPU's memory; scoring
                # at full native resolution (this script predates that constraint) reliably OOMs, so
                # images are downscaled right before being fed to the model -- after occlusion, so the
                # region masks (native resolution) are still applied at full precision either way.


def downscale(img: Image.Image) -> Image.Image:
    scale = MAX_SIDE / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def load_nonce_word(object_name: str) -> "str | None":
    """Returns the NVRD nonce reference for this object, or None if it isn't an NVRD
    object (e.g. an arbitrary test image) -- in which case the functionality prompt is
    used on its own, without the 'This image is best described by...' framing sentence."""
    mapping_path = os.path.join(REPO_ROOT, "data", "nonce_word_mapping.json")
    with open(mapping_path) as f:
        data = json.load(f)
    for entry in data:
        if os.path.splitext(os.path.basename(entry["fp"]))[0] == object_name:
            return entry["ref"]
    return None


def resolve_image_path(object_name: str) -> str:
    """Looks for object_name's image first among NVRD splits (regions_test/data/<split>/),
    then in test_images/, trying common extensions."""
    search_dirs = [
        os.path.join(REGIONS_TEST, "data", split)
        for split in ("novel", "known", "shape-shape", "shape-texture")
    ] + [os.path.join(REPO_ROOT, "test_images")]
    for d in search_dirs:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(d, f"{object_name}{ext}")
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not find an image for '{object_name}' under regions_test/data/*/ or test_images/"
    )


MASK_SUBDIRS = {"sam": "sam_masks", "sam_slic": "sam_slic_masks"}


def load_regions(object_name: str, max_regions: "int | None" = None, source: str = "sam"):
    """source: 'sam' (SAM masks only) or 'sam_slic' (SAM masks + SLIC superpixels filling the pixels
    SAM left unmasked, from regions/region_features/slic/merge_sam_and_slic_regions.py)."""
    sam_path = os.path.join(REGIONS_TEST, "output", MASK_SUBDIRS[source], f"{object_name}.json")
    with open(sam_path) as f:
        regions = json.load(f)
    regions = sorted(regions, key=lambda r: r["area"], reverse=True)
    if max_regions is not None:
        regions = regions[:max_regions]
    return regions


def occlude_region(image: Image.Image, region: dict) -> Image.Image:
    mask = mask_utils.decode(region["segmentation"]).astype(bool)
    arr = np.array(image).copy()
    arr[mask] = OCCLUSION_FILL
    return Image.fromarray(arr)


def build_prompt_text(processor, nonce_word: "str | None") -> str:
    # tokenize=False here only renders the chat template to a string (image placeholder
    # tokens get inserted based on content type=="image"; the actual pixels are supplied
    # later, per-image, via processor(text=..., images=...))
    content = [{"type": "image"}]
    if nonce_word is not None:
        content.append({"type": "text", "text": NONCE_CAPTION.format(ref=nonce_word)})
    content.append({"type": "text", "text": FUNCTIONALITY_PROMPT})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_baseline(model, processor, image: Image.Image, prompt_text: str, device) -> str:
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return text.strip()


@torch.no_grad()
def sequence_logprob(model, processor, image: Image.Image, prompt_text: str, response_text: str, device) -> tuple:
    """log P(response_text | prompt_text, image) via teacher forcing. Returns (total_logprob, n_tokens)."""
    full_text = prompt_text + response_text

    inputs_prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
    prompt_len = inputs_prompt["input_ids"].shape[1]

    inputs_full = processor(text=[full_text], images=[image], return_tensors="pt")
    inputs_full = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_full.items()}
    input_ids = inputs_full["input_ids"]

    out = model(**inputs_full)
    logits = out.logits[:, :-1].float()  # positions 0..seq_len-2, predict token at position+1
    target = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    response_logprobs = token_logprobs[:, prompt_len - 1:]
    total = response_logprobs.sum().item()
    n_tokens = response_logprobs.shape[1]
    return total, n_tokens


def main():
    OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
    MAX_REGIONS = int(sys.argv[2]) if len(sys.argv) > 2 else None  # cap to the N largest regions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nonce_word = load_nonce_word(OBJECT_NAME)
    regions = load_regions(OBJECT_NAME, MAX_REGIONS)
    img_path = resolve_image_path(OBJECT_NAME)
    image = Image.open(img_path).convert("RGB")

    print(f"Object: {OBJECT_NAME} | nonce word: {nonce_word} | {len(regions)} regions")
    print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    prompt_text = build_prompt_text(processor, nonce_word)

    small_full = downscale(image)

    print("\nGenerating baseline functionality description (full image)...")
    d0 = generate_baseline(model, processor, small_full, prompt_text, device)
    print(f"D0 = {d0!r}\n")

    print("Scoring log P(D0 | full image)...")
    logprob_full, n_tokens = sequence_logprob(model, processor, small_full, prompt_text, d0, device)
    print(f"  logP(D0 | full) = {logprob_full:.3f} over {n_tokens} tokens "
          f"({logprob_full / n_tokens:.4f} nats/token)\n")

    results = []
    for i, region in enumerate(regions):
        occluded = downscale(occlude_region(image, region))
        logprob_occ, n_tokens_occ = sequence_logprob(
            model, processor, occluded, prompt_text, d0, device
        )
        assert n_tokens_occ == n_tokens
        ig_nats = logprob_full - logprob_occ
        rec = {
            "region_id": region["region_id"],
            "area": region["area"],
            "logprob_full": logprob_full,
            "logprob_occluded": logprob_occ,
            "information_gain_nats": ig_nats,
            "information_gain_bits": ig_nats / np.log(2),
            "information_gain_bits_per_1000px": (ig_nats / np.log(2)) / (region["area"] / 1000.0),
        }
        results.append(rec)
        print(
            f"[{i}] {region['region_id']:>24s}  area={region['area']:>7d}  "
            f"logP(D0|occluded)={logprob_occ:9.3f}  IG={ig_nats:+7.3f} nats "
            f"({rec['information_gain_bits']:+7.3f} bits)"
        )

    out = {
        "object": OBJECT_NAME,
        "max_regions": MAX_REGIONS,
        "nonce_word": nonce_word,
        "model": MODEL_NAME,
        "prompt": FUNCTIONALITY_PROMPT,
        "baseline_description": d0,
        "logprob_full": logprob_full,
        "n_tokens": n_tokens,
        "occlusion_fill": OCCLUSION_FILL,
        "regions": results,
    }
    out_path = os.path.join(IG_JSON_DIR, f"{OBJECT_NAME}_region_information_gain.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
