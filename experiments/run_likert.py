import os
import sys

_MODELS_KEYS = ["qwen2-vl-7b", "qwen2.5-vl-72b", "idefics3-8b", "molmo2-8b", "gpt-4o-mini", "gemini-2.5-flash"]

if len(sys.argv) >= 2 and sys.argv[1] in _MODELS_KEYS:
    _cli_model_key = sys.argv[1]
    _cli_split = "all"
    _cli_obj_range = None  # e.g. "0-14" to process objects 0..14 (by sorted index)
    for arg in sys.argv[2:]:
        if arg in ("known", "novel", "shape-shape", "shape-texture", "all"):
            _cli_split = arg
        elif "-" in arg and arg.replace("-", "").isdigit():
            _cli_obj_range = arg
else:
    print(f"Usage: python3 {sys.argv[0]} <model_key> [split_filter] [obj_range]")
    print(f"  model_key: {_MODELS_KEYS}")
    print(f"  split_filter: known, novel, shape-shape, shape-texture, or all (default)")
    print(f"  obj_range: e.g. 0-14 to process only objects at sorted indices 0..14")
    sys.exit(1)

import openai
from openai import OpenAI
from pathlib import Path
import numpy as np
import random
import base64
import requests
from io import BytesIO
from PIL import Image
import json
import time
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from google import genai
from google.genai import types as gtypes

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SEED = 42
DEBUG = True
MAX_RETRIES = 3

MODELS = {
    "qwen2-vl-7b": {
        "name": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2vl"
    },
    "qwen2.5-vl-72b": {
        "name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "type": "qwen2_5vl"
    },
    "idefics3-8b": {
        "name": "HuggingFaceM4/Idefics3-8B-Llama3",
        "type": "idefics3"
    },
    "molmo2-8b": {
        "name": "allenai/Molmo2-8B",
        "type": "molmo2"
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "type": "openai"
    },
    "gemini-2.5-flash": {
        "name": "gemini-2.5-flash",
        "type": "gemini"
    },
}

MODEL_KEY = _cli_model_key
MODEL_CONFIG = MODELS[MODEL_KEY]
MODEL_NAME = MODEL_CONFIG["name"]
MODEL_TYPE = MODEL_CONFIG["type"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

gemini_client = None
openai_client = None
vlm_model = None
vlm_processor = None
vlm_tokenizer = None

if MODEL_TYPE == "gemini":
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
elif MODEL_TYPE == "openai":
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ─── Prompt templates ────────────────────────────────────────────────────────

INSTRUCTION_TEMPLATE = (
    'You will be shown two images. '
    'The first image shows an original object — let\'s call it "{nonce_word}." '
    'The second image shows a variation of it. '
    'Your task is to judge whether the object in the second image can also be '
    'called "{nonce_word}."'
)

ORIGINAL_CAPTION_TEMPLATE = 'Let\'s call the object in this image "{nonce_word}."'

FINAL_PROMPT_TEMPLATE = (
    'The object on the right can also be called "{nonce_word}." '
    'Respond with a single integer from 1 to 7:\n'
    '  1 = Strongly Disagree\n'
    '  2 = Disagree\n'
    '  3 = Somewhat Disagree\n'
    '  4 = Neutral\n'
    '  5 = Somewhat Agree\n'
    '  6 = Agree\n'
    '  7 = Strongly Agree\n'
    'Your answer must be just the number.'
)

# ─── Image utilities (identical to in_context_experiments.py) ─────────────────

MAX_IMAGE_PIXELS = 1280 * 720


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def download_and_encode_image(url: str, timeout=10, retries=3, backoff=0.75) -> Optional[str]:
    last_e = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img.verify()
            img = Image.open(BytesIO(response.content))
            buffered = BytesIO()
            img.save(buffered, format=img.format or "PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            last_e = e
            if attempt < retries:
                time.sleep(backoff * attempt)
            continue
    print(f"Failed to download/encode image from {url} after {retries} tries: {last_e}")
    return None


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
    if ("http" in source) or ("www" in source):
        encoded = download_and_encode_image(source)
        if encoded:
            return _resize_if_needed(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
        return None
    else:
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


# ─── Model loading (identical to in_context_experiments.py) ───────────────────

def load_vlm_model(model_type: str, model_name: str, device: str = "auto"):
    global vlm_model, vlm_processor, vlm_tokenizer

    if vlm_model is not None:
        print(f"Model already loaded: {model_name}")
        return vlm_model, vlm_processor, vlm_tokenizer

    print(f"Loading {model_name}...")

    if model_type in ("qwen2vl", "qwen2_5vl"):
        vlm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        vlm_processor = AutoProcessor.from_pretrained(model_name)
        model_cls = Qwen2_5_VLForConditionalGeneration if model_type == "qwen2_5vl" else Qwen2VLForConditionalGeneration
        vlm_model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    elif model_type == "idefics3":
        vlm_processor = AutoProcessor.from_pretrained(model_name)
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        vlm_tokenizer = vlm_processor.tokenizer

    elif model_type == "llava_next":
        vlm_processor = LlavaNextProcessor.from_pretrained(model_name)
        vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        vlm_tokenizer = vlm_processor.tokenizer

    elif model_type == "molmo2":
        vlm_processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        vlm_tokenizer = vlm_processor.tokenizer

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loaded {model_name}")
    return vlm_model, vlm_processor, vlm_tokenizer


# ─── VLM generation (adapted from in_context_experiments.py) ──────────────────
# images = [original_img, perturbed_img]
# captions = [original_caption]  (one caption for the original image)
# instruction = prolific-style instruction with nonce word
# final_prompt = rating request with nonce word

def _get_input_device(model):
    """Get the device to send inputs to (handles multi-GPU sharded models)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return model.device


def generate_with_vlm(
    images: List[Image.Image],
    captions: List[str],
    instruction: str,
    final_prompt: str,
    model_type: str,
    max_new_tokens: int = 10,
    debug: bool = False,
) -> Optional[str]:
    global vlm_model, vlm_processor, vlm_tokenizer

    if vlm_model is None:
        load_vlm_model(model_type, MODEL_NAME)

    if debug:
        print(f"\n[DEBUG] Generating with {model_type}")
        print(f"[DEBUG] Number of images: {len(images)} (1 original + 1 query)")
        print(f"[DEBUG] Instruction: {instruction[:120]}...")
        print(f"[DEBUG] Final prompt: {final_prompt[:120]}...")

    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*generation flags.*")

        if model_type in ("qwen2vl", "qwen2_5vl"):
            content = []
            content.append({"type": "text", "text": instruction})
            for img, caption in zip(images[:-1], captions):
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": caption})
            content.append({"type": "image", "image": images[-1]})
            content.append({"type": "text", "text": final_prompt})

            messages = [{"role": "user", "content": content}]
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vlm_processor(
                text=[text], images=images, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            with torch.no_grad():
                outputs = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = vlm_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]

        elif model_type == "qwen2_5vl":
            content = []
            content.append({"type": "text", "text": instruction})
            for img, caption in zip(images[:-1], captions):
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": caption})
            content.append({"type": "image", "image": images[-1]})
            content.append({"type": "text", "text": final_prompt})

            messages = [{"role": "user", "content": content}]
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vlm_processor(
                text=[text], images=images, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            with torch.no_grad():
                outputs = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = vlm_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]

        elif model_type == "idefics3":
            content = []
            content.append({"type": "text", "text": instruction})
            for caption in captions:
                content.append({"type": "image"})
                content.append({"type": "text", "text": caption})
            content.append({"type": "image"})
            content.append({"type": "text", "text": final_prompt})

            messages = [{"role": "user", "content": content}]
            text = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = vlm_processor(
                text=text, images=images, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            with torch.no_grad():
                outputs = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = vlm_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]

        elif model_type == "molmo2":
            content = []
            content.append({"type": "text", "text": instruction})
            for img, caption in zip(images[:-1], captions):
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": caption})
            content.append({"type": "image", "image": images[-1]})
            content.append({"type": "text", "text": final_prompt})

            messages = [{"role": "user", "content": content}]
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True
            )
            inputs = {k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            with torch.no_grad():
                outputs = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = vlm_processor.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        torch.cuda.empty_cache()

        cleaned_response = generated_text.strip()
        if debug:
            print(f"[DEBUG] Raw generation: '{generated_text}'")
            print(f"[DEBUG] After strip: '{cleaned_response}'")
        return cleaned_response

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM during generation: {e}")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"VLM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _try_encode_image(source):
    if isinstance(source, Image.Image):
        return pil_to_base64(source), None
    is_url = ("http" in source) or ("www" in source)
    if is_url:
        encoded = download_and_encode_image(source)
        return encoded, None
    else:
        try:
            return encode_image(source), None
        except Exception as e:
            print(f"Failed to encode local image {source}: {e}")
            return None, None


def prompt_gemini_model(original_image, perturbed_image, instruction, original_caption, final_prompt):
    if gemini_client is None:
        raise RuntimeError("gemini_client is not initialized.")

    orig_encoded, _ = _try_encode_image(original_image)
    pert_encoded, _ = _try_encode_image(perturbed_image)
    if orig_encoded is None or pert_encoded is None:
        return None

    parts = []
    parts.append(gtypes.Part(text=instruction))
    orig_bytes = base64.b64decode(orig_encoded)
    parts.append(gtypes.Part(inline_data=gtypes.Blob(data=orig_bytes, mime_type="image/png")))
    parts.append(gtypes.Part(text=original_caption))
    pert_bytes = base64.b64decode(pert_encoded)
    parts.append(gtypes.Part(inline_data=gtypes.Blob(data=pert_bytes, mime_type="image/png")))
    parts.append(gtypes.Part(text=final_prompt))

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
        )
        if response and hasattr(response, "text"):
            return (response.text or "").strip()
        elif response and hasattr(response, "parts"):
            text_parts = [p.text for p in response.parts if hasattr(p, "text") and p.text]
            return " ".join(text_parts).strip()
        return str(response).strip()
    except Exception as e:
        print(f"Gemini API request failed: {e}")
        return None


def prompt_openai_model(original_image, perturbed_image, instruction, original_caption, final_prompt):
    if openai_client is None:
        raise RuntimeError("openai_client is not initialized.")

    orig_encoded, _ = _try_encode_image(original_image)
    pert_encoded, _ = _try_encode_image(perturbed_image)
    if orig_encoded is None or pert_encoded is None:
        return None

    content = [
        {"type": "input_text", "text": instruction},
        {"type": "input_image", "image_url": f"data:image/png;base64,{orig_encoded}"},
        {"type": "input_text", "text": original_caption},
        {"type": "input_image", "image_url": f"data:image/png;base64,{pert_encoded}"},
        {"type": "input_text", "text": final_prompt},
    ]

    for attempt in range(5):
        try:
            response = openai_client.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": content}],
            )
            return (response.output_text or "").strip()
        except openai.BadRequestError as e:
            print(f"OpenAI API request failed: {e}")
            return None
        except (openai.RateLimitError, openai.APIStatusError) as e:
            wait = 2 ** attempt
            print(f"Rate limit hit, retrying in {wait}s: {e}")
            time.sleep(wait)


def generate_response_prolific(original_image, perturbed_image, nonce_word, debug=False):
    """Prompt the model in prolific style: two images, no ICL, rate 1-7."""
    instruction = INSTRUCTION_TEMPLATE.format(nonce_word=nonce_word)
    original_caption = ORIGINAL_CAPTION_TEMPLATE.format(nonce_word=nonce_word)
    final_prompt = FINAL_PROMPT_TEMPLATE.format(nonce_word=nonce_word)

    if MODEL_TYPE == "gemini":
        response = prompt_gemini_model(
            original_image, perturbed_image, instruction, original_caption, final_prompt
        )
        return {
            "response": response,
            "instruction_sent": instruction,
            "original_caption_sent": original_caption,
            "final_prompt_sent": final_prompt,
        }
    elif MODEL_TYPE == "openai":
        response = prompt_openai_model(
            original_image, perturbed_image, instruction, original_caption, final_prompt
        )
        return {
            "response": response,
            "instruction_sent": instruction,
            "original_caption_sent": original_caption,
            "final_prompt_sent": final_prompt,
        }
    elif MODEL_TYPE in ("qwen2vl", "qwen2_5vl", "idefics3", "molmo2"):
        orig_img = load_image(original_image)
        pert_img = load_image(perturbed_image)

        if orig_img is None:
            print(f"[ERROR] Failed to load original image: {original_image}")
            return {"response": None, "instruction_sent": instruction,
                    "original_caption_sent": original_caption, "final_prompt_sent": final_prompt}
        if pert_img is None:
            print(f"[ERROR] Failed to load perturbed image: {perturbed_image}")
            return {"response": None, "instruction_sent": instruction,
                    "original_caption_sent": original_caption, "final_prompt_sent": final_prompt}

        # images[0] = original (shown with caption), images[1] = perturbed (query)
        images = [orig_img, pert_img]
        captions = [original_caption]  # one caption for the original image

        response = generate_with_vlm(
            images=images,
            captions=captions,
            instruction=instruction,
            final_prompt=final_prompt,
            model_type=MODEL_TYPE,
            max_new_tokens=10,
            debug=debug,
        )
        return {
            "response": response,
            "instruction_sent": instruction,
            "original_caption_sent": original_caption,
            "final_prompt_sent": final_prompt,
        }
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")


# ─── Response parsing ─────────────────────────────────────────────────────────

def parse_rating(response: Optional[str]) -> Optional[int]:
    """Extract a 1-7 integer rating from the model's raw response."""
    if response is None:
        return None
    import re
    # Look for a standalone digit 1-7
    matches = re.findall(r'\b([1-7])\b', response)
    if matches:
        return int(matches[0])
    # Fallback: first digit in string
    digits = re.findall(r'[1-7]', response)
    if digits:
        return int(digits[0])
    return None


# ─── NVRD dataset loading (identical to in_context_experiments.py) ────────────

class _LocalNVRD:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        r = self._rows[idx]
        return {
            "image": Image.open(r["image_path"]).convert("RGB"),
            "original_image": Image.open(r["original_image_path"]).convert("RGB"),
            "split": r["split"],
            "object": r["object"],
            "perturbation_type": r["perturbation_type"],
            "level": r["level"],
            "image_path": r["image_path"],
            "original_image_path": r["original_image_path"],
        }


def load_nvrd_dataset(
    data_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample-data"),
):
    print(f"Loading NVRD dataset from local directory: {data_root}")

    split_dirs = {
        "known": "known",
        "novel": "novel",
        "shape-shape": os.path.join("modified", "shape-shape"),
        "shape-texture": os.path.join("modified", "shape-texture"),
    }

    rows = []
    nvrd_index = {}
    all_ptypes = set()
    all_levels = set()

    for split_name, rel_dir in split_dirs.items():
        split_root = os.path.join(data_root, rel_dir)
        if not os.path.isdir(split_root):
            continue

        base_images = {}
        for fname in sorted(os.listdir(split_root)):
            if fname.endswith(".png") and not fname.startswith("."):
                obj_name = fname[:-4]
                base_images[obj_name] = os.path.join(split_root, fname)

        obj_names_sorted = sorted(base_images.keys(), key=len, reverse=True)

        pcomp_dir = os.path.join(split_root, "perturbations_comp")
        if not os.path.isdir(pcomp_dir):
            continue

        for ptype in sorted(os.listdir(pcomp_dir)):
            ptype_dir = os.path.join(pcomp_dir, ptype)
            if not os.path.isdir(ptype_dir) or ptype.startswith(".") or ptype.startswith("_"):
                continue

            for fname in sorted(os.listdir(ptype_dir)):
                if not fname.endswith(".png") or fname.startswith("."):
                    continue
                stem = fname[:-4]

                matched_obj = None
                for oname in obj_names_sorted:
                    if stem.startswith(oname + "_"):
                        matched_obj = oname
                        break
                if matched_obj is None:
                    continue

                level_str = stem[len(matched_obj) + 1:]
                try:
                    level = int(level_str)
                except ValueError:
                    continue

                original_path = base_images[matched_obj]
                perturbed_path = os.path.join(ptype_dir, fname)

                idx = len(rows)
                rows.append({
                    "image_path": perturbed_path,
                    "original_image_path": original_path,
                    "split": split_name,
                    "object": matched_obj,
                    "perturbation_type": ptype,
                    "level": level,
                })

                if matched_obj not in nvrd_index:
                    nvrd_index[matched_obj] = {}
                if ptype not in nvrd_index[matched_obj]:
                    nvrd_index[matched_obj][ptype] = {}
                nvrd_index[matched_obj][ptype][level] = idx

                all_ptypes.add(ptype)
                all_levels.add(level)

    perturbation_types = sorted(all_ptypes)
    all_levels = sorted(all_levels)

    ds = _LocalNVRD(rows)
    print(f"  Loaded {len(ds)} rows")
    print(f"  Indexed {len(nvrd_index)} objects")

    if not all_levels:
        data_root = Path(__file__).resolve().parent.parent / "data" / "sample-data"
        raise FileNotFoundError(
            f"No NVRD images found under {data_root}.\n"
            "  Run:  python data/download_dataset.py\n"
            "  to download the dataset from HuggingFace before running experiments."
        )

    print(f"  Perturbation types ({len(perturbation_types)}): {perturbation_types}")
    print(f"  Levels ({len(all_levels)}): {all_levels[0]}-{all_levels[-1]}")

    return ds, nvrd_index, perturbation_types, all_levels


def build_object_to_nonce(training_file: str) -> Dict:
    print(f"Building object-to-nonce mapping from {training_file}...")
    with open(training_file, "r") as f:
        data = json.load(f)
    object_to_nonce = {}
    for entry in data:
        stem = Path(entry["fp"]).stem
        object_to_nonce[stem] = entry["ref"]
    print(f"  Loaded {len(object_to_nonce)} object-to-nonce mappings")
    return object_to_nonce


# ─── Existing results cache ────────────────────────────────────────────────────

def load_existing_results(output_path: str) -> set:
    """Load completed keys (object, perturbation_type, level) from existing JSONL."""
    completed = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (
                            obj.get("object"),
                            obj.get("perturbation_type"),
                            obj.get("level"),
                        )
                        if all(v is not None for v in key):
                            completed.add(key)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
    return completed


# ─── Main experiment loop ─────────────────────────────────────────────────────

def run_prolific_style_experiments(
    nvrd_ds,
    nvrd_index: Dict,
    perturbation_types: List[str],
    all_levels: List[int],
    object_to_nonce: Dict,
    out_path: str,
):
    print(f"\n{'='*60}")
    print(f"PROLIFIC-STYLE RATING EXPERIMENT")
    print(f"Model: {MODEL_KEY} ({MODEL_NAME})")
    print(f"Output: {out_path}")
    print(f"{'='*60}\n")

    ensure_parent_dir(out_path)
    completed = load_existing_results(out_path)
    print(f"Loaded {len(completed)} existing results from {out_path}\n")

    object_names = sorted(nvrd_index.keys())
    # Apply object range filter if specified
    obj_range = getattr(sys.modules[__name__], '_cli_obj_range', None)
    if obj_range:
        start, end = map(int, obj_range.split("-"))
        object_names = object_names[start:end+1]
        print(f"Object range filter: indices {start}-{end} ({len(object_names)} objects)")
    total_objects = len(object_names)
    successful = 0
    failed = 0
    skipped = 0
    skipped_no_nonce = 0

    for obj_idx, object_name in enumerate(object_names):
        nonce_word = object_to_nonce.get(object_name)
        if nonce_word is None:
            print(f"[{obj_idx+1}/{total_objects}] [SKIP] No nonce word for '{object_name}'")
            skipped_no_nonce += 1
            continue

        obj_data = nvrd_index[object_name]

        # Retrieve the original (base) image path from any row for this object
        any_idx = next(iter(next(iter(obj_data.values())).values()))
        any_row = nvrd_ds[any_idx]
        split_type = any_row["split"]
        split_type_norm = split_type.replace("modified/", "").replace("/perturbations", "")
        original_image_path = any_row["original_image_path"]

        # Apply split filter
        if _cli_split != "all" and split_type_norm != _cli_split:
            continue

        print(f"\n[{obj_idx+1}/{total_objects}] '{object_name}' "
              f"(split={split_type}, nonce={nonce_word})")

        for ptype in perturbation_types:
            if ptype not in obj_data:
                continue
            for level in all_levels:
                if level not in obj_data[ptype]:
                    continue

                key = (object_name, ptype, level)
                if key in completed:
                    skipped += 1
                    continue

                row = nvrd_ds[obj_data[ptype][level]]
                perturbed_image_path = row["image_path"]

                print(f"  {ptype} L{level}", end="", flush=True)

                model_response = None
                rating = None
                result_dict = None

                for attempt in range(MAX_RETRIES + 1):
                    result_dict = generate_response_prolific(
                        original_image=original_image_path,
                        perturbed_image=perturbed_image_path,
                        nonce_word=nonce_word,
                        debug=DEBUG,
                    )
                    model_response = result_dict["response"]
                    rating = parse_rating(model_response)

                    if rating is not None:
                        break
                    elif attempt < MAX_RETRIES:
                        print(f" [RETRY {attempt+1}]", end="", flush=True)

                if rating is None:
                    print(f" -> FAILED (response: '{model_response}')")
                    failed += 1
                else:
                    print(f" -> {rating} ('{model_response}')")
                    successful += 1

                out_obj = {
                    "object": object_name,
                    "split": split_type,
                    "perturbation_type": ptype,
                    "level": level,
                    "novel_word": nonce_word,
                    "base_image": original_image_path,
                    "perturbed_image": perturbed_image_path,
                    "model": MODEL_KEY,
                    "model_response": model_response,
                    "rating": rating,
                    "instruction_sent": result_dict["instruction_sent"] if result_dict else None,
                    "original_caption_sent": result_dict["original_caption_sent"] if result_dict else None,
                    "final_prompt_sent": result_dict["final_prompt_sent"] if result_dict else None,
                }
                append_jsonl(out_path, out_obj)
                completed.add(key)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total objects: {total_objects}")
    print(f"Successful (parsed rating): {successful}")
    print(f"Failed (no parseable rating): {failed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Skipped (no nonce word): {skipped_no_nonce}")
    print(f"Results: {out_path}")
    print(f"{'='*60}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    training_file = os.path.join(script_dir, "data", "nonce_word_mapping.json")

    print("=" * 60)
    print(f"Prolific-Style Experiments (full NVRD)")
    print(f"Model: {MODEL_KEY} ({MODEL_NAME})")
    print(f"Model type: {MODEL_TYPE}")
    print("=" * 60)
    print()

    nvrd_ds, nvrd_index, perturbation_types, all_levels = load_nvrd_dataset()
    print()

    object_to_nonce = build_object_to_nonce(training_file)
    print()

    # Load VLM model once
    if MODEL_TYPE not in ("gemini", "openai"):
        load_vlm_model(MODEL_TYPE, MODEL_NAME)

    split_suffix = f"-{_cli_split}" if _cli_split != "all" else ""
    range_suffix = f"-r{_cli_obj_range}" if _cli_obj_range else ""
    out_path = os.path.join(
        script_dir, "results", "prolific_style",
        f"{MODEL_KEY}-prolific-style-ratings{split_suffix}{range_suffix}.jsonl"
    )

    run_prolific_style_experiments(
        nvrd_ds, nvrd_index, perturbation_types, all_levels, object_to_nonce, out_path
    )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Model: {MODEL_KEY}")
    print(f"Results: {out_path}")
    print("=" * 60)
