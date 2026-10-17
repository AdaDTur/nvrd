import os
import gc
from pathlib import Path
import json
import time
from PIL import Image
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEBUG = True
MAX_RETRIES = 3
MAX_NEW_TOKENS = 256
MAX_IMAGE_PIXELS = 1280 * 720

FUNCTIONALITY_PROMPT = "Describe what functionality this object serves."
NONCE_CAPTION = "This image is best described by the reference: {ref}."

_MODELS_KEYS = [
    "qwen2-vl-7b",
    #"qwen2.5-vl-72b",
    #"idefics3-8b",
    #"molmo2-8b",
]
SPLIT_FILTER = "novel"
OBJ_RANGE = 1  # first N unperturbed pairs after split filter

MODELS = {
    "qwen2-vl-7b": {
        "name": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2vl",
    },
    "qwen2.5-vl-72b": {
        "name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "type": "qwen2_5vl",
    },
    "idefics3-8b": {
        "name": "HuggingFaceM4/Idefics3-8B-Llama3",
        "type": "idefics3",
    },
    "molmo2-8b": {
        "name": "allenai/Molmo2-8B",
        "type": "molmo2",
    },
}

MODEL_KEY = None
MODEL_NAME = None
MODEL_TYPE = None

vlm_model = None
vlm_processor = None
vlm_tokenizer = None


def set_active_model(model_key: str):
    global MODEL_KEY, MODEL_NAME, MODEL_TYPE
    unload_vlm_model()
    MODEL_KEY = model_key
    config = MODELS[model_key]
    MODEL_NAME = config["name"]
    MODEL_TYPE = config["type"]


def unload_vlm_model():
    global vlm_model, vlm_processor, vlm_tokenizer
    if vlm_model is None:
        return
    del vlm_model
    del vlm_processor
    del vlm_tokenizer
    vlm_model = None
    vlm_processor = None
    vlm_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h <= MAX_IMAGE_PIXELS:
        return img
    scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_image(source) -> Optional[Image.Image]:
    if isinstance(source, Image.Image):
        return _resize_if_needed(source.convert("RGB"))
    try:
        return _resize_if_needed(Image.open(source).convert("RGB"))
    except Exception as e:
        print(f"Failed to load local image {source}: {e}")
        return None


def ensure_parent_dir(fp: str):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str, obj: Dict):
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_existing_keys(path: str) -> set:
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("image_path") or rec.get("fp")
            if key:
                completed.add(key)
    return completed


def split_from_fp(fp: str) -> str:
    parts = Path(fp).parts
    if "modified" in parts:
        idx = parts.index("modified")
        return parts[idx + 1] if idx + 1 < len(parts) else "modified"
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def load_unperturbed_pairs(
    mapping_file: str,
    data_root: str,
    split_filter: str = "all",
) -> List[Dict]:
    """Load unperturbed image + nonce reference pairs from nonce_word_mapping.json."""
    print(f"Loading unperturbed image/nonce pairs from {mapping_file}...")
    with open(mapping_file, "r") as f:
        data = json.load(f)

    pairs = []
    skipped_missing = 0
    skipped_split = 0
    skipped_baseline = 0
    for entry in data:
        fp = entry["fp"]
        nonce_word = entry["ref"]
        split = split_from_fp(fp)
        if split == "baseline":
            skipped_baseline += 1
            continue
        if split_filter != "all" and split != split_filter:
            skipped_split += 1
            continue
        image_path = os.path.join(data_root, fp)
        if not os.path.isfile(image_path):
            skipped_missing += 1
            continue
        pairs.append({
            "fp": fp,
            "image_path": image_path,
            "object": Path(fp).stem,
            "nonce_word": nonce_word,
            "split": split,
        })

    pairs.sort(key=lambda p: (p["split"], p["object"]))
    print(f"  Loaded {len(pairs)} unperturbed pairs")
    print(f"  Skipped baseline: {skipped_baseline}")
    print(f"  Skipped by split filter: {skipped_split}")
    print(f"  Skipped missing image: {skipped_missing}")
    return pairs


def _get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return model.device


def load_vlm_model(model_type: str, model_name: str, device: str = "auto"):
    global vlm_model, vlm_processor, vlm_tokenizer

    if vlm_model is not None:
        print(f"Model already loaded: {model_name}")
        return vlm_model, vlm_processor, vlm_tokenizer

    print(f"Loading {model_name}...")

    if model_type in ("qwen2vl", "qwen2_5vl"):
        vlm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        vlm_processor = AutoProcessor.from_pretrained(model_name)
        model_cls = (
            Qwen2_5_VLForConditionalGeneration
            if model_type == "qwen2_5vl"
            else Qwen2VLForConditionalGeneration
        )
        vlm_model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    elif model_type in ("idefics3", "molmo2"):
        kwargs = {"trust_remote_code": True} if model_type == "molmo2" else {}
        vlm_processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **kwargs,
        )
        vlm_tokenizer = vlm_processor.tokenizer
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loaded {model_name}")
    return vlm_model, vlm_processor, vlm_tokenizer


def generate_with_vlm(
    image: Image.Image,
    nonce_caption: str,
    final_prompt: str,
    model_type: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    debug: bool = False,
) -> Optional[str]:
    global vlm_model, vlm_processor, vlm_tokenizer

    if vlm_model is None:
        load_vlm_model(model_type, MODEL_NAME)

    if debug:
        print(f"\n[DEBUG] Generating with {model_type}")
        print(f"[DEBUG] Nonce caption: {nonce_caption}")
        print(f"[DEBUG] Final prompt: {final_prompt}")

    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*generation flags.*")

        if model_type in ("qwen2vl", "qwen2_5vl"):
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": nonce_caption},
                {"type": "text", "text": final_prompt},
            ]
            messages = [{"role": "user", "content": content}]
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vlm_processor(
                text=[text], images=[image], return_tensors="pt", padding=True
            )
            inputs = {
                k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = vlm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated_text = vlm_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]

        elif model_type == "idefics3":
            content = [
                {"type": "image"},
                {"type": "text", "text": nonce_caption},
                {"type": "text", "text": final_prompt},
            ]
            messages = [{"role": "user", "content": content}]
            text = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = vlm_processor(
                text=text, images=[image], return_tensors="pt", padding=True
            )
            inputs = {
                k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = vlm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated_text = vlm_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]

        elif model_type == "molmo2":
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": nonce_caption},
                {"type": "text", "text": final_prompt},
            ]
            messages = [{"role": "user", "content": content}]
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            inputs = {
                k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = vlm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated_text = vlm_processor.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        torch.cuda.empty_cache()
        cleaned = generated_text.strip()
        if debug:
            print(f"[DEBUG] Generation: '{cleaned}'")
        return cleaned

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM during generation: {e}")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"VLM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_response(image_path: str, nonce_word: str, debug: bool = False) -> Optional[str]:
    nonce_caption = NONCE_CAPTION.format(ref=nonce_word)
    final_prompt = FUNCTIONALITY_PROMPT

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            img = load_image(image_path)
            if img is None:
                return None
            return generate_with_vlm(
                image=img,
                nonce_caption=nonce_caption,
                final_prompt=final_prompt,
                model_type=MODEL_TYPE,
                debug=debug,
            )
        except Exception as e:
            last_err = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(attempt)
    print(f"All retries failed: {last_err}")
    return None


def run_probe(pairs: List[Dict], out_path: str):
    print(f"\n{'=' * 60}")
    print(f"AFFORDANCE PROBE: model={MODEL_KEY} | split={SPLIT_FILTER}")
    print(f"Prompt: {FUNCTIONALITY_PROMPT}")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}\n")

    ensure_parent_dir(out_path)
    completed = load_existing_keys(out_path)
    print(f"Loaded {len(completed)} existing results from {out_path}")

    load_vlm_model(MODEL_TYPE, MODEL_NAME)

    total = len(pairs)
    n_ok = 0
    n_fail = 0
    n_skip = 0

    for i, pair in enumerate(pairs):
        if pair["image_path"] in completed or pair["fp"] in completed:
            n_skip += 1
            continue

        print(
            f"[{i + 1}/{total}] {pair['object']} "
            f"(split={pair['split']}, nonce={pair['nonce_word']})"
        )
        response = generate_response(
            pair["image_path"], pair["nonce_word"], debug=DEBUG and i < 3
        )
        if response is None:
            print("  [FAIL] empty response")
            n_fail += 1
            continue

        rec = {
            "object": pair["object"],
            "split": pair["split"],
            "nonce_word": pair["nonce_word"],
            "image_path": pair["image_path"],
            "fp": pair["fp"],
            "prompt": FUNCTIONALITY_PROMPT,
            "nonce_caption": NONCE_CAPTION.format(ref=pair["nonce_word"]),
            "model_response": response,
            "model": MODEL_KEY,
        }
        append_jsonl(out_path, rec)
        n_ok += 1
        preview = response.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"  {preview}")

    print(f"\n{'=' * 60}")
    print(f"Done. success={n_ok} fail={n_fail} skipped={n_skip} total={total}")
    print(f"Results: {out_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "data")
    mapping_file = os.path.join(data_root, "nonce_word_mapping.json")

    pairs = load_unperturbed_pairs(mapping_file, data_root, SPLIT_FILTER)
    pairs = pairs[:OBJ_RANGE]
    print(f"Using first {len(pairs)} pair(s) from split={SPLIT_FILTER}")

    if not pairs:
        raise FileNotFoundError(
            f"No unperturbed image/nonce pairs found.\n"
            f"  Mapping: {mapping_file}\n"
            f"  Run: python data/download_dataset.py"
        )

    for model_key in _MODELS_KEYS:
        set_active_model(model_key)
        out_path = os.path.join(
            repo_root,
            "results",
            "affordance",
            f"{model_key}-affordance-probe-{SPLIT_FILTER}-n{OBJ_RANGE}.jsonl",
        )
        run_probe(pairs, out_path)
    unload_vlm_model()