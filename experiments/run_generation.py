import os
import sys

_MODELS_KEYS = ["qwen2-vl-7b", "qwen2.5-vl-72b", "idefics3-8b", "molmo2-8b", "gpt-4o-mini", "gemini-2.5-flash"]
_ABLATION_MODE = len(sys.argv) >= 2 and sys.argv[1] == "ablation"

if _ABLATION_MODE:
    # ablation: fixed model (qwen2-vl-7b), all similarity types, n=2,4,8
    _cli_model_key = "qwen2-vl-7b"
    _cli_similarity = "all"
    _cli_n_values = [2, 4, 8]
    print("Running in ABLATION mode: qwen2-vl-7b | all similarities | n=2,4,8")
elif len(sys.argv) >= 2 and sys.argv[1] in _MODELS_KEYS:
    # normal: user-specified model, visual_similarity only, n=4
    _cli_model_key = sys.argv[1]
    _cli_similarity = "visual_similarity"
    _cli_n_values = [4]
    # Optional split filter, prob-only flag, and object range
    _cli_split = "all"
    _cli_prob_only = False
    _cli_obj_range = None  # e.g. "0-14" to process objects 0..14 (by sorted index)
    for arg in sys.argv[2:]:
        if arg == "--prob-only":
            _cli_prob_only = True
        elif arg in ("known", "novel", "shape-shape", "shape-texture", "all"):
            _cli_split = arg
        elif "-" in arg and arg.replace("-", "").isdigit():
            _cli_obj_range = arg
else:
    print(f"Usage: python3 {sys.argv[0]} <model_key | ablation> [split_filter] [--prob-only]")
    print(f"  model_key: {_MODELS_KEYS}")
    print(f"  split_filter: known, novel, shape-shape, shape-texture, or all (default)")
    print(f"  --prob-only:  skip generation, run probability experiments only")
    print(f"  ablation : run qwen2-vl-7b with all similarities and n=2,4,8")
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
import copy
import json
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any
import glob as globmod
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
BATCH_SIZE = 8
MAX_WORKERS = 8
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
SPLIT_FILTER = getattr(sys.modules[__name__], '_cli_split', 'all')
CLI_SIMILARITY = _cli_similarity
MODEL_CONFIG = MODELS[MODEL_KEY]
MODEL_NAME = MODEL_CONFIG["name"]
MODEL_TYPE = MODEL_CONFIG["type"]

if MODEL_TYPE not in ("gemini", "openai"):
    MAX_WORKERS = 1
    BATCH_SIZE = 1


caption_archs = [
    "This image is best described by the reference: {ref}.",
    "This image shows {ref}.",
    "In this image, we see {ref}.",
    "This image depicts {ref}.",
    "The subject of this image is {ref}.",
]

final_caption_archs_fillin = [
    "This image is best described by the reference: ____.",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

gemini_client = None
openai_client = None
client = None
vlm_model = None
vlm_processor = None
vlm_tokenizer = None

if MODEL_TYPE == "gemini":
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    #client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
elif MODEL_TYPE == "openai":
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    client = openai_client

def encode_image(image_path: str) -> str:
    """Encode local image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def download_and_encode_image(url: str, timeout=10, retries=3, backoff=0.75) -> Optional[str]:
    """Download and encode image from URL."""
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


MAX_IMAGE_PIXELS = 1280 * 720


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """Downscale image if it exceeds MAX_IMAGE_PIXELS, preserving aspect ratio."""
    w, h = img.size
    if w * h <= MAX_IMAGE_PIXELS:
        return img
    scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_image(source) -> Optional[Image.Image]:
    """Load image from PIL Image, file path, or URL. Downscales large images."""
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


def load_precomputed_pools(pool_file: str = "precomputed_icl_pools.json") -> Dict:
    print(f"Loading precomputed pools from {pool_file}...")
    with open(pool_file, "r") as f:
        data = json.load(f)
    pools_dict = {}
    for pool_entry in data["pools"]:
        datapoint_path = pool_entry["datapoint_path"]
        pools_dict[datapoint_path] = pool_entry
    print(f"Loaded {len(pools_dict)} precomputed pools\n")
    return pools_dict


def build_object_to_nonce(training_file: str = "molmo_final_training_run_short.json") -> Dict:
    print(f"Building object-to-nonce mapping from {training_file}...")
    with open(training_file, "r") as f:
        data = json.load(f)
    object_to_nonce = {}
    for entry in data:
        stem = Path(entry["fp"]).stem
        object_to_nonce[stem] = entry["ref"]
    print(f"  Loaded {len(object_to_nonce)} object-to-nonce mappings")
    return object_to_nonce


def load_nonce_word_mapping(mapping_file: str = "molmo_final_training_run_short.json") -> Dict:
    print(f"Loading nonce word mappings from {mapping_file}...")
    with open(mapping_file, "r") as f:
        data = json.load(f)
    nonce_mapping = {}
    for entry in data:
        fp = entry["fp"]
        ref = entry["ref"]
        nonce_mapping[fp] = ref
        abs_path = str(Path(fp).resolve())
        nonce_mapping[abs_path] = ref
    print(f"Loaded {len(nonce_mapping)} nonce word mappings (with {len(data)} unique entries)\n")
    return nonce_mapping


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
            "image_path": r["image_path"],
            "original_image_path": r["original_image_path"],
            "split": r["split"],
            "object": r["object"],
            "perturbation_type": r["perturbation_type"],
            "level": r["level"],
        }


def load_nvrd_dataset(
    data_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample-data"),
):
    print(f"Loading NVRD dataset from local directory: {data_root}")

    # Splits and their subdirectory paths (relative to data_root)
    split_dirs = {
        "known": "known",
        "novel": "novel",
        "shape-shape": os.path.join("modified", "shape-shape"),
        "shape-texture": os.path.join("modified", "shape-texture"),
    }

    rows = []          # flat list consumed by _LocalNVRD
    nvrd_index = {}    # obj -> ptype -> level -> row_idx
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
                stem = fname[:-4]  # e.g. "boar_toaster_10"

                matched_obj = None
                for oname in obj_names_sorted:
                    if stem.startswith(oname + "_"):
                        matched_obj = oname
                        break
                if matched_obj is None:
                    continue

                level_str = stem[len(matched_obj) + 1:]  # part after "{obj}_"
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


def get_pool_for_datapoint_by_name(
    object_name: str,
    precomputed_pools: Dict,
    object_to_nonce: Dict,
    similarity_type: str,
    caption_templates: List[str],
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    pool_entry = None
    for _, entry in precomputed_pools.items():
        if entry.get("datapoint_ref") == object_name:
            pool_entry = entry
            break

    if pool_entry is None:
        print(f"[WARNING] No precomputed pool found for object '{object_name}'")
        return [], None

    pool_items = pool_entry["pools"].get(similarity_type, [])

    pool_samples = []
    for item in pool_items:
        ref = item.get("cleaned_caption", item["caption"])
        caption = random.choice(caption_templates).format(ref=ref)
        pool_samples.append((item["image_path"], caption))

    nonce_word = object_to_nonce.get(object_name)
    if nonce_word is None:
        print(f"[WARNING] No nonce word found for object '{object_name}'")
    else:
        print(f"[NONCE] Object '{object_name}' -> nonce word '{nonce_word}'")

    return pool_samples, nonce_word


def get_pool_for_datapoint_by_path(
    datapoint_path: str,
    precomputed_pools: Dict,
    nonce_mapping: Dict,
    similarity_type: str,
    caption_templates: List[str],
) -> List[Tuple[str, str]]:
    datapoint_path_abs = str(Path(datapoint_path).resolve())

    pool_entry = None
    for path, entry in precomputed_pools.items():
        if Path(path).resolve() == Path(datapoint_path_abs):
            pool_entry = entry
            break

    if pool_entry is None:
        print(f"Warning: No precomputed pool found for {datapoint_path}")
        return []

    pool_items = pool_entry["pools"].get(similarity_type, [])

    pool_samples = []
    for item in pool_items:
        ref = item.get("cleaned_caption", item["caption"])
        caption = random.choice(caption_templates).format(ref=ref)
        pool_samples.append((item["image_path"], caption))

    return pool_samples


def _try_encode_image(source, pool=None):
    """Encode an image from a PIL Image, file path, or URL.
    If encoding fails for a URL, attempts to resample from pool."""
    if isinstance(source, Image.Image):
        return pil_to_base64(source), None

    is_url = ("http" in source) or ("www" in source)

    if is_url:
        encoded = download_and_encode_image(source)
        if encoded:
            return encoded, None
        print(f"Image failed to download ({source[:60]}...), attempting to resample...")
        if pool and len(pool) > 0:
            max_attempts = min(len(pool), 10)
            for _ in range(max_attempts):
                new_url, new_caption = random.choice(pool)
                if isinstance(new_url, str) and ("http" in new_url or "www" in new_url):
                    encoded = download_and_encode_image(new_url)
                    if encoded:
                        print(f"Successfully resampled from pool")
                        return encoded, new_caption
        print(f"Failed to resample image")
        return None, None
    else:
        try:
            encoded = encode_image(source)
            return encoded, None
        except Exception as e:
            print(f"Failed to encode local image {source}: {e}")
            return None, None


def prompt_gemini_model(stimuli, prompt_prefix, final_prompt, pool=None):
    """Build an interleaved prompt for Gemini."""
    if gemini_client is None:
        raise RuntimeError("gemini_client is not initialized but prompt_gemini_model was called.")

    content = [{"type": "input_text", "text": prompt_prefix}]

    for i, (source, ref) in enumerate(stimuli):
        is_query = (i == len(stimuli) - 1)
        encoded, resampled_caption = _try_encode_image(source, pool=pool)
        if encoded is None:
            print(f"Skipping stimulus {i + 1} due to image load failure")
            continue
        caption_to_use = resampled_caption if resampled_caption is not None else ref
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
        if is_query:
            content.append({"type": "input_text", "text": final_prompt})
        else:
            content.append({"type": "input_text", "text": caption_to_use})

    if len(content) < 3:
        return None

    parts = []
    for item in content:
        if item.get("type") == "input_text":
            text = item.get("text", "")
            if text:
                parts.append(gtypes.Part(text=text))
        elif item.get("type") == "input_image":
            image_url = item.get("image_url", "")
            if image_url.startswith("data:image"):
                header, encoded = image_url.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                mime_type = "image/png"
                if "jpeg" in header or "jpg" in header:
                    mime_type = "image/jpeg"
                elif "png" in header:
                    mime_type = "image/png"
                try:
                    blob = gtypes.Blob(data=image_bytes, mime_type=mime_type)
                    parts.append(gtypes.Part(inline_data=blob))
                except (AttributeError, TypeError):
                    parts.append(gtypes.Part(inline_data={"data": image_bytes, "mime_type": mime_type}))

    if len(parts) < 3:
        return None

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
        )
        if response and hasattr(response, "text"):
            cleaned_response = (response.text or "").strip()
        elif response and hasattr(response, "parts"):
            text_parts = [part.text for part in response.parts if hasattr(part, "text") and part.text]
            cleaned_response = " ".join(text_parts).strip()
        else:
            cleaned_response = str(response).strip()
        cleaned_response = cleaned_response.strip(".,!?;:").lower()
        return cleaned_response
    except Exception as e:
        print(f"Gemini API request failed: {e}")
        return None


def prompt_gpt_model(stimuli, prompt_prefix, final_prompt, pool=None):
    """Build an interleaved prompt for OpenAI."""
    if "gemini" in MODEL_NAME.lower():
        return prompt_gemini_model(stimuli, prompt_prefix, final_prompt, pool=pool)

    if openai_client is None:
        raise RuntimeError("openai_client is not initialized but prompt_gpt_model was called.")

    content = [{"type": "input_text", "text": prompt_prefix}]
    for i, (source, ref) in enumerate(stimuli):
        is_query = (i == len(stimuli) - 1)
        encoded, resampled_caption = _try_encode_image(source, pool=pool)
        if encoded is None:
            print(f"Skipping stimulus {i + 1} due to image load failure")
            continue
        caption_to_use = resampled_caption if resampled_caption is not None else ref
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
        if is_query:
            content.append({"type": "input_text", "text": final_prompt})
        else:
            content.append({"type": "input_text", "text": caption_to_use})

    if len(content) < 3:
        return None

    for attempt in range(5):
        try:
            response = openai_client.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": content}],
            )
            cleaned_response = (response.output_text or "").strip()
            cleaned_response = cleaned_response.strip(".,!?;:").lower()
            return cleaned_response
        except openai.BadRequestError as e:
            print(f"API request failed: {e}")
            return None
        except (openai.RateLimitError, openai.APIStatusError) as e:
            wait = 2 ** attempt
            print(f"Rate limit hit, retrying in {wait}s: {e}")
            import time; time.sleep(wait)
    return None


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

    elif model_type == "phi3_vision":
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
    max_new_tokens: int = 50,
    debug: bool = False,
) -> Optional[str]:
    """Generate text from VLM given interleaved images and captions."""
    global vlm_model, vlm_processor, vlm_tokenizer

    if vlm_model is None:
        load_vlm_model(model_type, MODEL_NAME)

    if debug:
        print(f"\n[DEBUG] Generating with {model_type}")
        print(f"[DEBUG] Number of images: {len(images)} ({len(captions)} ICL + 1 query)")
        print(f"[DEBUG] Instruction: {instruction[:120]}...")
        print(f"[DEBUG] Final prompt: {final_prompt}")
        for i, cap in enumerate(captions):
            print(f"[DEBUG]   ICL caption {i+1}: {cap[:80]}")

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

        elif model_type == "llava_next":
            parts = [instruction]
            for caption in captions:
                parts.append(f"<image>\n{caption}")
            parts.append(f"<image>\n{final_prompt}")
            full_prompt = "\n\n".join(parts)

            inputs = vlm_processor(
                text=full_prompt, images=images, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(_get_input_device(vlm_model)) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            with torch.no_grad():
                outputs = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = vlm_processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]

        elif model_type == "phi3_vision":
            parts = [instruction]
            for i, caption in enumerate(captions):
                parts.append(f"<|image_{i+1}|>\n{caption}")
            parts.append(f"<|image_{len(images)}|>\n{final_prompt}")
            full_prompt = "\n\n".join(parts)

            inputs = vlm_processor(
                text=full_prompt, images=images, return_tensors="pt", padding=True
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
        cleaned_response = cleaned_response.strip(".,!?;:").lower()
        if debug:
            print(f"[DEBUG] Final cleaned: '{cleaned_response}'")
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


def generate_response(stimuli_for_model, prompt_prefix, final_prompt, pool=None, debug=False):
    """Route to appropriate generation method based on model type."""
    if MODEL_TYPE == "gemini":
        response = prompt_gemini_model(stimuli_for_model, prompt_prefix, final_prompt, pool=pool)
        return {
            "response": response,
            "actual_stimuli": stimuli_for_model,
            "instruction_sent": prompt_prefix,
            "final_prompt_sent": final_prompt,
        }
    elif MODEL_TYPE == "openai":
        response = prompt_gpt_model(stimuli_for_model, prompt_prefix, final_prompt, pool=pool)
        return {
            "response": response,
            "actual_stimuli": stimuli_for_model,
            "instruction_sent": prompt_prefix,
            "final_prompt_sent": final_prompt,
        }
    elif MODEL_TYPE in ["qwen2vl", "idefics3", "llava_next", "phi3_vision", "molmo2"]:
        images = []
        icl_captions = []
        actual_stimuli = []
        failed_indices = []

        if debug:
            print(f"\n[DEBUG] Loading {len(stimuli_for_model)} images ({len(stimuli_for_model)-1} ICL + 1 query)")

        for i, (img_path, caption) in enumerate(stimuli_for_model):
            img = load_image(img_path)
            if img is not None:
                images.append(img)
                actual_stimuli.append([img_path, caption])
                if i < len(stimuli_for_model) - 1:
                    icl_captions.append(caption)
                    if debug:
                        print(f"[DEBUG] Loaded ICL image {i+1}: {caption[:50] if caption else '(empty)'}")
                else:
                    if debug:
                        print(f"[DEBUG] Loaded QUERY image: {img_path}")
            else:
                failed_indices.append(i)
                if debug:
                    print(f"[DEBUG] Failed to load image {i+1}: {img_path[:80]}")

        if len(images) < 2:
            print(f"Insufficient images loaded ({len(images)}/{len(stimuli_for_model)}), failed: {failed_indices}")
            return {
                "response": None,
                "actual_stimuli": actual_stimuli,
                "instruction_sent": prompt_prefix,
                "final_prompt_sent": final_prompt,
            }

        query_index = len(stimuli_for_model) - 1
        if query_index in failed_indices:
            print(f"[ERROR] Query image failed to load: {stimuli_for_model[-1][0]}")
            return {
                "response": None,
                "actual_stimuli": actual_stimuli,
                "instruction_sent": prompt_prefix,
                "final_prompt_sent": final_prompt,
            }

        if failed_indices:
            print(f"[WARNING] {len(failed_indices)} ICL images failed to load (indices: {failed_indices}). "
                  f"Proceeding with {len(images)}/{len(stimuli_for_model)} images.")

        num_icl = len(icl_captions)
        instruction = (
            f"You will be shown {num_icl} images, each followed by a caption. "
            "Then, you will see one final image and asked to provide its caption."
        )

        if debug:
            print(f"[DEBUG] Instruction: {instruction}")
            print(f"[DEBUG] Final prompt: {final_prompt}")
            print(f"[DEBUG] ICL captions ({num_icl}):")
            for idx, cap in enumerate(icl_captions):
                print(f"[DEBUG]   {idx+1}. {cap[:80]}")

        response = generate_with_vlm(
            images=images,
            captions=icl_captions,
            instruction=instruction,
            final_prompt=final_prompt,
            model_type=MODEL_TYPE,
            max_new_tokens=50,
            debug=debug,
        )
        return {
            "response": response,
            "actual_stimuli": actual_stimuli,
            "instruction_sent": instruction,
            "final_prompt_sent": final_prompt,
        }
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")


def validate_icl_pool(icl_pool, main_example, n, debug=False):
    """Validate ICL pool composition."""
    if debug:
        print(f"\n[DEBUG] Validating ICL pool (expected size: {n})")
        print(f"[DEBUG] Actual size: {len(icl_pool)}")
        print(f"[DEBUG] Main example (nonce word): {main_example}")

        main_in_pool = any(
            item[0] == main_example[0] and item[1] == main_example[1]
            for item in icl_pool
        )
        print(f"[DEBUG] Main example in pool: {main_in_pool}")

        import re
        nonce_match = re.search(r': (\w+)\.$', main_example[1])
        if nonce_match:
            nonce_word = nonce_match.group(1)
            print(f"[DEBUG] Expected nonce word: '{nonce_word}'")


def is_response_valid(response: Optional[str]) -> bool:
    """Check if a model response is usable."""
    if response is None:
        return False
    stripped = response.strip()
    if len(stripped) < 2:
        return False
    if len(stripped) == 1 and not stripped.isalpha():
        return False
    return True


def run_one_datapoint(
    *,
    object_name,
    split_type,
    nonce_word,
    perturbation_type,
    level,
    n,
    icl_pool,
    main_example_image,
    main_example_caption,
    query_image,
    trial_seed=None,
    final_caption_archs_active,
):
    """Run a single generation datapoint: build ICL context and query the model."""
    distractor_sub = list(icl_pool[:n])

    model_response = None
    actual_stimuli = None
    instruction_sent = None
    final_prompt_sent = None
    built_pool = None

    for attempt in range(MAX_RETRIES + 1):
        attempt_seed = trial_seed + attempt if trial_seed is not None else None
        rng = np.random.RandomState(attempt_seed) if attempt_seed is not None else np.random.RandomState(None)

        built_pool = list(distractor_sub) + [(main_example_image, main_example_caption)]

        if DEBUG and attempt == 0:
            validate_icl_pool(built_pool, (main_example_image, main_example_caption), n, debug=True)

        rng.shuffle(built_pool)

        stimuli_for_model = [list(x) for x in built_pool]
        stimuli_for_model.append([query_image, ""])

        prompt_prefix = (
            f"You will be shown {len(built_pool)} images, each followed by a caption. "
            "Then, you will see one final image and asked to provide its caption."
        )
        final_prompt = random.choice(final_caption_archs_active)

        result = generate_response(
            stimuli_for_model,
            prompt_prefix,
            final_prompt,
            pool=icl_pool,
            debug=DEBUG,
        )

        model_response = result["response"]
        actual_stimuli = result["actual_stimuli"]
        instruction_sent = result["instruction_sent"]
        final_prompt_sent = result["final_prompt_sent"]

        if is_response_valid(model_response):
            break
        elif attempt < MAX_RETRIES:
            print(f"  [RETRY] n={n} attempt {attempt+1}/{MAX_RETRIES}: "
                  f"invalid response '{model_response}', retrying with different shuffle...")

    if not is_response_valid(model_response):
        print(f"  [FAILED] n={n} {perturbation_type} L{level}: all {MAX_RETRIES+1} attempts "
              f"produced invalid response '{model_response}'")

    def _serialize_stim(stim_list):
        serialized = []
        for source, caption in stim_list:
            if isinstance(source, Image.Image):
                serialized.append(["<PIL Image>", caption])
            else:
                serialized.append([source, caption])
        return serialized

    out_obj = {
        "object": object_name,
        "type": split_type,
        "nonce_word": nonce_word,
        "perturbation_type": perturbation_type,
        "level": level,
        "n": n,
        "model": MODEL_KEY,
        "model_response": model_response,
        "stimuli_shown": _serialize_stim(actual_stimuli) if actual_stimuli else [],
        "prompt_prefix": instruction_sent,
        "prompt_suffix": final_prompt_sent,
    }
    return out_obj


def load_existing_generation_results(output_path: str) -> set:
    """Load completed generation experiment keys from existing JSONL output."""
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
                            obj.get("n"),
                        )
                        if all(v is not None for v in key):
                            completed.add(key)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
    return completed


def is_datapoint_complete(object_name, completed_set, perturbation_combos, n_values=None):
    """Check if all (perturbation_type, level, n) combos are done for an object."""
    if n_values is None:
        n_values = [2, 4, 8]
    for ptype, level in perturbation_combos:
        for n in n_values:
            if (object_name, ptype, level, n) not in completed_set:
                return False
    return True


def run_datapoints_batched(datapoint_kwargs_list, max_workers=MAX_WORKERS):
    """Runs multiple run_one_datapoint(...) calls concurrently."""
    results = []
    if not datapoint_kwargs_list:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_kwargs = {
            ex.submit(run_one_datapoint, **kwargs): kwargs
            for kwargs in datapoint_kwargs_list
        }
        for fut in as_completed(fut_to_kwargs):
            kwargs = fut_to_kwargs[fut]
            try:
                out_obj = fut.result()
            except Exception as e:
                print(
                    f"[BATCH] datapoint crashed "
                    f"(obj={kwargs.get('object_name')}, {kwargs.get('perturbation_type')} "
                    f"L{kwargs.get('level')}, n={kwargs.get('n')}): {e}"
                )
                continue
            if out_obj is None or out_obj.get("model_response") is None:
                print(
                    f"[BATCH] model_response None "
                    f"(obj={kwargs.get('object_name')}, {kwargs.get('perturbation_type')} "
                    f"L{kwargs.get('level')}, n={kwargs.get('n')})"
                )
                continue
            results.append(out_obj)
    return results


def run_generation_experiments(
    nvrd_ds,
    nvrd_index,
    perturbation_types,
    all_levels,
    precomputed_pools,
    object_to_nonce,
    similarity_type,
    prompt_mode,
    final_archs,
    out_path,
):
    """Run all generation experiments for one similarity_type × prompt_mode."""
    n_values = _cli_n_values

    print(f"\n{'='*60}")
    print(f"GENERATION: similarity={similarity_type} | prompt={prompt_mode}")
    print(f"Output: {out_path}")
    print(f"{'='*60}\n")

    ensure_parent_dir(out_path)
    completed = load_existing_generation_results(out_path)
    print(f"Loaded {len(completed)} existing generation results from {out_path}")

    stats = {
        "total_attempted": 0,
        "successful": 0,
        "failed": 0,
        "skipped_no_pool": 0,
        "skipped_no_nonce": 0,
        "skipped_completed": 0,
        "skipped_missing_data": 0,
        "response_lengths": [],
    }

    object_names = sorted(nvrd_index.keys())
    obj_range = getattr(sys.modules[__name__], '_cli_obj_range', None)
    if obj_range:
        start, end = map(int, obj_range.split("-"))
        object_names = object_names[start:end+1]
        print(f"Object range filter: indices {start}-{end} ({len(object_names)} objects)")
    total_objects = len(object_names)

    for obj_idx, object_name in enumerate(object_names):
        obj_data = nvrd_index[object_name]

        icl_pool, nonce_word = get_pool_for_datapoint_by_name(
            object_name=object_name,
            precomputed_pools=precomputed_pools,
            object_to_nonce=object_to_nonce,
            similarity_type=similarity_type,
            caption_templates=caption_archs,
        )

        if not icl_pool:
            print(f"[{obj_idx+1}/{total_objects}] [SKIP] No pool for '{object_name}'")
            stats["skipped_no_pool"] += 1
            continue

        if nonce_word is None:
            print(f"[{obj_idx+1}/{total_objects}] [SKIP] No nonce word for '{object_name}'")
            stats["skipped_no_nonce"] += 1
            continue

        any_idx = next(iter(next(iter(obj_data.values())).values()))
        any_row = nvrd_ds[any_idx]
        original_image = any_row["original_image"]
        split_type = any_row["split"]
        # Normalize: strip "modified/" prefix and "/perturbations" suffix
        # so e.g. "modified/shape-texture" -> "shape-texture"
        split_type_norm = split_type.replace("modified/", "").replace("/perturbations", "")

        # --- Split filter: skip objects not in the requested split ---
        if SPLIT_FILTER != "all" and split_type_norm != SPLIT_FILTER:
            continue
        split_type = split_type_norm

        main_example_caption = random.choice(caption_archs).format(ref=nonce_word)

        perturbation_combos = []
        for ptype in perturbation_types:
            if ptype not in obj_data:
                continue
            for level in all_levels:
                if level in obj_data[ptype]:
                    perturbation_combos.append((ptype, level))

        print(f"\n[{obj_idx+1}/{total_objects}] Processing '{object_name}' "
              f"(split={split_type}, nonce={nonce_word}, combos={len(perturbation_combos)})")

        if is_datapoint_complete(object_name, completed, perturbation_combos, n_values):
            print(f"  [SKIP] All perturbation/level/n combos already completed")
            continue

        if DEBUG:
            print(f"  [DEBUG] ICL pool size: {len(icl_pool)}")

        pending = []

        for ptype, level in perturbation_combos:
            row = nvrd_ds[obj_data[ptype][level]]
            query_image = row["image"]

            for n in n_values:
                key = (object_name, ptype, level, n)
                if key in completed:
                    stats["skipped_completed"] += 1
                    continue

                stats["total_attempted"] += 1
                trial_seed = hash((object_name, ptype, level, n, SEED)) % (2**32 - 1)

                pending.append(dict(
                    object_name=object_name,
                    split_type=split_type,
                    nonce_word=nonce_word,
                    perturbation_type=ptype,
                    level=level,
                    n=n,
                    icl_pool=icl_pool,
                    main_example_image=original_image,
                    main_example_caption=main_example_caption,
                    query_image=query_image,
                    trial_seed=trial_seed,
                    final_caption_archs_active=final_archs,
                ))

                if len(pending) >= BATCH_SIZE:
                    batch_out = run_datapoints_batched(pending, max_workers=MAX_WORKERS)
                    pending = []

                    for dp in batch_out:
                        key2 = (dp["object"], dp["perturbation_type"], dp["level"], dp["n"])
                        if key2 in completed:
                            continue
                        response = dp.get("model_response")
                        if response:
                            stats["successful"] += 1
                            response_len = len(response.split())
                            stats["response_lengths"].append(response_len)
                            if response_len <= 2:
                                print(f"  [WARNING] Very short response ({response_len} words): '{response}'")
                        else:
                            stats["failed"] += 1
                        append_jsonl(out_path, dp)
                        completed.add(key2)
                        print(f"  [WROTE] {dp['object']} {dp['perturbation_type']} "
                              f"L{dp['level']} n={dp['n']} -> {dp['model_response']}")

        # Flush remaining
        if pending:
            batch_out = run_datapoints_batched(pending, max_workers=MAX_WORKERS)
            for dp in batch_out:
                key2 = (dp["object"], dp["perturbation_type"], dp["level"], dp["n"])
                if key2 in completed:
                    continue
                response = dp.get("model_response")
                if response:
                    stats["successful"] += 1
                    response_len = len(response.split())
                    stats["response_lengths"].append(response_len)
                    if response_len <= 2:
                        print(f"  [WARNING] Very short response ({response_len} words): '{response}'")
                else:
                    stats["failed"] += 1
                append_jsonl(out_path, dp)
                completed.add(key2)
                print(f"  [WROTE] {dp['object']} {dp['perturbation_type']} "
                      f"L{dp['level']} n={dp['n']} -> {dp['model_response']}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERATION STATS: {similarity_type} | {prompt_mode}")
    print(f"{'='*60}")
    print(f"Total objects: {total_objects}")
    print(f"Total attempted: {stats['total_attempted']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (no pool): {stats['skipped_no_pool']}")
    print(f"Skipped (no nonce): {stats['skipped_no_nonce']}")
    print(f"Skipped (completed): {stats['skipped_completed']}")
    if stats["response_lengths"]:
        lengths = np.array(stats["response_lengths"])
        print(f"\nResponse length statistics (words):")
        print(f"  Mean: {lengths.mean():.2f}")
        print(f"  Median: {np.median(lengths):.2f}")
        print(f"  Min: {lengths.min()}")
        print(f"  Max: {lengths.max()}")
        print(f"  Std: {lengths.std():.2f}")
    print(f"{'='*60}")


class VLMProbabilityScorer:
    """Base class for VLM probability scoring."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        raise NotImplementedError

    def _prepare_inputs(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_sequence_probability(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_text: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_cached_probabilities(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_texts: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute log-probs for multiple targets, encoding the prompt only once."""
        if self.model is None:
            self.load_model()

        device = _get_input_device(self.model)
        inputs = self._prepare_inputs(images, captions, instruction, final_prompt)

        prompt_len = inputs["input_ids"].shape[1]

        # Forward pass on prompt with KV caching
        with torch.no_grad():
            prompt_outputs = self.model(
                **inputs,
                use_cache=True,
            )

        past_kv = prompt_outputs.past_key_values
        # Logits at the last prompt position predict the first target token
        last_logits = prompt_outputs.logits[:, -1:, :]
        del prompt_outputs
        torch.cuda.empty_cache()

        results = []
        for target_text in target_texts:
            target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
            target_tensor = torch.tensor([target_ids], device=device)
            target_len = target_tensor.shape[1]

            # Attention mask covers full sequence (past prompt + target tokens)
            attn_mask = torch.ones((1, prompt_len + target_len), dtype=torch.long, device=device)

            # Deep-copy KV cache: DynamicCache.update() mutates in place,
            # so reusing past_kv across targets would corrupt it.
            kv_copy = copy.deepcopy(past_kv)

            with torch.no_grad():
                target_outputs = self.model(
                    input_ids=target_tensor,
                    attention_mask=attn_mask,
                    past_key_values=kv_copy,
                    use_cache=False,
                )

            # last_logits predicts target[0], target_outputs.logits[:, i] predicts target[i+1]
            all_logits = torch.cat([last_logits, target_outputs.logits], dim=1)
            log_probs = torch.log_softmax(all_logits[:, :target_len, :], dim=-1)
            token_logprobs = log_probs.gather(
                dim=-1, index=target_tensor.unsqueeze(-1)
            ).squeeze(-1)

            ref_logprob = token_logprobs.sum().item()
            results.append({
                "ref_logprob": ref_logprob,
                "num_tokens": target_len,
                "ref_token_ids": target_ids,
            })
            del target_outputs, kv_copy
            torch.cuda.empty_cache()

        del past_kv, last_logits
        torch.cuda.empty_cache()

        return results

    # --- Level 1 caching: ICL prefix KV cache across perturbation levels ---

    def _prepare_icl_only_inputs(self, icl_images, captions, instruction):
        """Prepare inputs for ICL context only (no query image, no final prompt).
        Override in subclasses.
        """
        raise NotImplementedError

    def _prepare_standalone_image_inputs(self, image):
        """Process a single image to get its pixel_values/features.
        Override in subclasses. Returns dict of pixel-related kwargs.
        """
        raise NotImplementedError

    def prepare_icl_cache(self, icl_images, captions, instruction,
                          sample_query_image, sample_final_prompt):
        """Cache ICL prefix KV pairs (Level 1 caching).

        Computes the KV cache for the ICL context (distractor images + captions)
        which stays constant across perturbation levels for the same object.
        """
        if self.model is None:
            self.load_model()

        device = _get_input_device(self.model)

        # Process ICL-only and full inputs to find the common prefix
        icl_inputs = self._prepare_icl_only_inputs(icl_images, captions, instruction)
        all_images = icl_images + [sample_query_image]
        full_inputs = self._prepare_inputs(
            all_images, captions, instruction, sample_final_prompt
        )

        # Find longest common token prefix (diverges where query image starts)
        icl_ids = icl_inputs['input_ids'][0]
        full_ids = full_inputs['input_ids'][0]
        prefix_len = 0
        for i in range(min(len(icl_ids), len(full_ids))):
            if icl_ids[i] == full_ids[i]:
                prefix_len = i + 1
            else:
                break

        if prefix_len < 10:
            print(f"[ICL_CACHE] WARNING: very short common prefix ({prefix_len}), "
                  f"falling back to non-cached mode")
            return None

        print(f"[ICL_CACHE] ICL-only: {len(icl_ids)} tokens, Full: {len(full_ids)} tokens, "
              f"Shared prefix: {prefix_len} tokens, "
              f"Query suffix: {len(full_ids) - prefix_len} tokens")

        # Forward pass on prefix using ICL-only pixel_values
        prefix_ids = full_inputs['input_ids'][:, :prefix_len]
        prefix_attn = torch.ones((1, prefix_len), dtype=torch.long, device=device)

        # Collect pixel-related kwargs from ICL-only inputs
        pixel_kwargs = {}
        for key in ['pixel_values', 'image_grid_thw', 'pixel_attention_mask',
                    'image_sizes', 'images', 'image_input_idx', 'image_masks']:
            if key in icl_inputs:
                pixel_kwargs[key] = icl_inputs[key]

        with torch.no_grad():
            outputs = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_attn,
                use_cache=True,
                **pixel_kwargs,
            )

        cache = {
            'past_kv': outputs.past_key_values,
            'prefix_len': prefix_len,
            'rope_deltas': getattr(outputs, 'rope_deltas', None),
            'icl_images': icl_images,
            'captions': captions,
            'instruction': instruction,
        }

        del outputs, icl_inputs, full_inputs
        torch.cuda.empty_cache()
        return cache

    def _evaluate_targets_with_kv(self, past_kv, last_logits, prefix_len,
                                  target_texts, device, rope_deltas=None):
        """Evaluate multiple target texts using a KV cache. Shared by L1 and L2."""
        results = []
        for target_text in target_texts:
            target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
            target_tensor = torch.tensor([target_ids], device=device)
            target_len = target_tensor.shape[1]

            attn_mask = torch.ones((1, prefix_len + target_len), dtype=torch.long, device=device)

            target_kwargs = {}
            if rope_deltas is not None:
                pos_start = attn_mask.shape[1] - target_len + rope_deltas
                position_ids = torch.arange(target_len, device=device).unsqueeze(0) + pos_start
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                target_kwargs['position_ids'] = position_ids

            # Deep-copy KV cache: DynamicCache.update() mutates in place,
            # so reusing past_kv across targets would corrupt it.
            kv_copy = copy.deepcopy(past_kv)

            with torch.no_grad():
                target_outputs = self.model(
                    input_ids=target_tensor,
                    attention_mask=attn_mask,
                    past_key_values=kv_copy,
                    use_cache=False,
                    **target_kwargs,
                )

            all_logits = torch.cat([last_logits, target_outputs.logits], dim=1)
            log_probs = torch.log_softmax(all_logits[:, :target_len, :], dim=-1)
            token_logprobs = log_probs.gather(
                dim=-1, index=target_tensor.unsqueeze(-1)
            ).squeeze(-1)

            results.append({
                "ref_logprob": token_logprobs.sum().item(),
                "num_tokens": target_len,
                "ref_token_ids": target_ids,
            })
            del target_outputs, kv_copy
            torch.cuda.empty_cache()

        return results

    def compute_with_icl_cache(self, icl_cache, query_image, final_prompt, target_texts):
        """Compute log-probs using cached ICL prefix KV (Level 1 + Level 2)."""
        device = _get_input_device(self.model)
        prefix_len = icl_cache['prefix_len']

        # Process full input for correct tokenization
        all_images = icl_cache['icl_images'] + [query_image]
        full_inputs = self._prepare_inputs(
            all_images, icl_cache['captions'], icl_cache['instruction'], final_prompt
        )

        # Extract query suffix tokens
        query_ids = full_inputs['input_ids'][:, prefix_len:]
        query_len = query_ids.shape[1]
        full_len = prefix_len + query_len
        attn_mask = torch.ones((1, full_len), dtype=torch.long, device=device)

        # Get query image pixel_values via standalone processing
        query_pixel_kwargs = self._prepare_standalone_image_inputs(query_image)

        del full_inputs
        torch.cuda.empty_cache()

        # Forward pass query suffix with ICL KV cache
        with torch.no_grad():
            query_outputs = self.model(
                input_ids=query_ids,
                attention_mask=attn_mask,
                past_key_values=icl_cache['past_kv'],
                use_cache=True,
                **query_pixel_kwargs,
            )

        full_kv = query_outputs.past_key_values
        last_logits = query_outputs.logits[:, -1:, :]
        query_rope_deltas = getattr(query_outputs, 'rope_deltas', None)
        del query_outputs
        torch.cuda.empty_cache()

        # Evaluate targets with full KV cache
        rope_deltas = query_rope_deltas or icl_cache.get('rope_deltas')
        results = self._evaluate_targets_with_kv(
            full_kv, last_logits, full_len, target_texts, device, rope_deltas
        )

        del full_kv, last_logits
        torch.cuda.empty_cache()
        return results


class Qwen2VLScorer(VLMProbabilityScorer):
    """Scorer for Qwen2-VL models."""

    def compute_with_icl_cache(self, icl_cache, query_image, final_prompt, target_texts):
        """Qwen2-VL override: use single-stage caching to avoid rope_deltas
        accumulation errors across the two-stage ICL prefix / query split.

        The base class splits the forward pass into ICL-prefix and query-suffix
        stages, each producing its own rope_deltas. Combining them is unreliable
        for Qwen2-VL's 3-D rotary positions, so we reprocess the full prompt in
        one pass via compute_cached_probabilities instead.
        """
        all_images = icl_cache['icl_images'] + [query_image]
        return self.compute_cached_probabilities(
            all_images, icl_cache['captions'], icl_cache['instruction'],
            final_prompt, target_texts
        )

    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        print(f"Loaded {self.model_name}")

    def _prepare_inputs(self, images, captions, instruction, final_prompt):
        device = _get_input_device(self.model)
        content = []
        content.append({"type": "text", "text": instruction})
        for img, caption in zip(images[:-1], captions):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": caption})
        content.append({"type": "image", "image": images[-1]})
        content.append({"type": "text", "text": final_prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        return inputs

    def _prepare_icl_only_inputs(self, icl_images, captions, instruction):
        device = _get_input_device(self.model)
        content = [{"type": "text", "text": instruction}]
        for img, caption in zip(icl_images, captions):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": caption})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text], images=icl_images, return_tensors="pt", padding=True
        )
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _prepare_standalone_image_inputs(self, image):
        device = _get_input_device(self.model)
        content = [{"type": "image", "image": image}, {"type": "text", "text": "x"}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        kwargs = {}
        for key in ['pixel_values', 'image_grid_thw']:
            if key in inputs:
                kwargs[key] = inputs[key].to(device) if isinstance(inputs[key], torch.Tensor) else inputs[key]
        return kwargs

    def compute_sequence_probability(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_text: str,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        inputs = self._prepare_inputs(images, captions, instruction, final_prompt)

        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_ids = torch.tensor(target_ids).unsqueeze(0).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]
        full_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)

        if "attention_mask" in inputs:
            target_mask = torch.ones((1, target_ids.shape[1]), dtype=inputs["attention_mask"].dtype, device=self.model.device)
            full_attention_mask = torch.cat([inputs["attention_mask"], target_mask], dim=1)
            inputs["attention_mask"] = full_attention_mask

        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                labels=labels,
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "labels"]}
            )

        logits = outputs.logits[:, :-1]
        target_token_ids = full_input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1, index=target_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        target_mask = (labels != -100)[:, 1:]
        target_token_logprobs = token_logprobs[target_mask]

        ref_logprob = target_token_logprobs.sum().item()
        return {
            "ref_logprob": ref_logprob,
            "num_tokens": int(target_mask.sum().item()),
            "ref_token_ids": target_ids.squeeze(0).tolist(),
        }


    def compute_cached_probabilities(self, images, captions, instruction, final_prompt, target_texts):
        """Qwen2VL override: pass position_ids and rope_deltas for rotary embeddings."""
        import copy
        if self.model is None:
            self.load_model()

        device = _get_input_device(self.model)
        inputs = self._prepare_inputs(images, captions, instruction, final_prompt)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            prompt_outputs = self.model(
                **inputs,
                use_cache=True,
            )

        past_kv = prompt_outputs.past_key_values
        last_logits = prompt_outputs.logits[:, -1:, :]

        # Get rope_deltas if available (Qwen2VL uses this for position tracking)
        rope_deltas = getattr(prompt_outputs, "rope_deltas", None)
        del prompt_outputs
        torch.cuda.empty_cache()

        results = []
        for target_text in target_texts:
            target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
            target_tensor = torch.tensor([target_ids], device=device)
            target_len = target_tensor.shape[1]

            attn_mask = torch.ones((1, prompt_len + target_len), dtype=torch.long, device=device)

            # Build position_ids for target tokens continuing from prompt
            if rope_deltas is not None:
                # Qwen2VL: position_ids based on attention mask length + rope_deltas
                pos_start = attn_mask.shape[1] - target_len + rope_deltas
                position_ids = torch.arange(target_len, device=device).unsqueeze(0) + pos_start
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = torch.arange(prompt_len, prompt_len + target_len, device=device).unsqueeze(0)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            # Deep-copy KV cache: DynamicCache.update() mutates in place,
            # so reusing past_kv across targets would corrupt it.
            kv_copy = copy.deepcopy(past_kv)

            with torch.no_grad():
                target_outputs = self.model(
                    input_ids=target_tensor,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=kv_copy,
                    use_cache=False,
                )

            all_logits = torch.cat([last_logits, target_outputs.logits], dim=1)
            log_probs = torch.log_softmax(all_logits[:, :target_len, :], dim=-1)
            token_logprobs = log_probs.gather(
                dim=-1, index=target_tensor.unsqueeze(-1)
            ).squeeze(-1)

            ref_logprob = token_logprobs.sum().item()
            results.append({
                "ref_logprob": ref_logprob,
                "num_tokens": target_len,
                "ref_token_ids": target_ids,
            })
            del target_outputs, kv_copy
            torch.cuda.empty_cache()

        del past_kv, last_logits
        torch.cuda.empty_cache()

        return results


class Qwen2_5VLScorer(Qwen2VLScorer):
    """Scorer for Qwen2.5-VL models."""

    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        print(f"Loaded {self.model_name}")


class Idefics3Scorer(VLMProbabilityScorer):
    """Scorer for Idefics3 models."""

    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        # Reduce image resolution to ~3.4x fewer tokens (898 vs 3036 per image)
        self.processor.image_processor.size = {"longest_edge": 728}
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.tokenizer = self.processor.tokenizer
        print(f"Loaded {self.model_name}")

    def _prepare_inputs(self, images, captions, instruction, final_prompt):
        device = _get_input_device(self.model)
        content = []
        content.append({"type": "text", "text": instruction})
        for img, caption in zip(images[:-1], captions):
            content.append({"type": "image"})
            content.append({"type": "text", "text": caption})
        content.append({"type": "image"})
        content.append({"type": "text", "text": final_prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        return inputs

    def _prepare_icl_only_inputs(self, icl_images, captions, instruction):
        device = _get_input_device(self.model)
        content = [{"type": "text", "text": instruction}]
        for caption in captions:
            content.append({"type": "image"})
            content.append({"type": "text", "text": caption})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            text=text, images=icl_images, return_tensors="pt", padding=True
        )
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _prepare_standalone_image_inputs(self, image):
        device = _get_input_device(self.model)
        content = [{"type": "image"}, {"type": "text", "text": "x"}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt", padding=True
        )
        kwargs = {}
        for key in ['pixel_values', 'pixel_attention_mask']:
            if key in inputs:
                kwargs[key] = inputs[key].to(device) if isinstance(inputs[key], torch.Tensor) else inputs[key]
        return kwargs

    def compute_sequence_probability(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_text: str,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        inputs = self._prepare_inputs(images, captions, instruction, final_prompt)
        device = _get_input_device(self.model)

        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_ids = torch.tensor(target_ids).unsqueeze(0).to(device)

        prompt_len = inputs["input_ids"].shape[1]
        full_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)

        if "attention_mask" in inputs:
            target_mask = torch.ones((1, target_ids.shape[1]), dtype=inputs["attention_mask"].dtype, device=self.model.device)
            full_attention_mask = torch.cat([inputs["attention_mask"], target_mask], dim=1)
            inputs["attention_mask"] = full_attention_mask

        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                labels=labels,
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "labels"]}
            )

        logits = outputs.logits[:, :-1]
        target_token_ids = full_input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1, index=target_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        target_mask = (labels != -100)[:, 1:]
        target_token_logprobs = token_logprobs[target_mask]

        ref_logprob = target_token_logprobs.sum().item()
        return {
            "ref_logprob": ref_logprob,
            "num_tokens": int(target_mask.sum().item()),
            "ref_token_ids": target_ids.squeeze(0).tolist(),
        }


class LlavaNextScorer(VLMProbabilityScorer):
    """Scorer for LLaVA-NeXT models."""

    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.tokenizer = self.processor.tokenizer
        print(f"Loaded {self.model_name}")

    def _prepare_inputs(self, images, captions, instruction, final_prompt):
        device = _get_input_device(self.model)
        parts = [instruction]
        for caption in captions:
            parts.append(f"<image>\n{caption}")
        parts.append(f"<image>\n{final_prompt}")
        full_prompt_text = "\n\n".join(parts)
        inputs = self.processor(
            text=full_prompt_text, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        return inputs

    def compute_sequence_probability(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_text: str,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        inputs = self._prepare_inputs(images, captions, instruction, final_prompt)
        device = _get_input_device(self.model)

        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_ids = torch.tensor(target_ids).unsqueeze(0).to(device)

        prompt_len = inputs["input_ids"].shape[1]
        full_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)

        if "attention_mask" in inputs:
            target_mask = torch.ones((1, target_ids.shape[1]), dtype=inputs["attention_mask"].dtype, device=self.model.device)
            full_attention_mask = torch.cat([inputs["attention_mask"], target_mask], dim=1)
            inputs["attention_mask"] = full_attention_mask

        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                labels=labels,
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "labels"]}
            )

        logits = outputs.logits[:, :-1]
        target_token_ids = full_input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1, index=target_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        target_mask = (labels != -100)[:, 1:]
        target_token_logprobs = token_logprobs[target_mask]

        ref_logprob = target_token_logprobs.sum().item()
        return {
            "ref_logprob": ref_logprob,
            "num_tokens": int(target_mask.sum().item()),
            "ref_token_ids": target_ids.squeeze(0).tolist(),
        }


class Phi3VisionScorer(VLMProbabilityScorer):
    """Scorer for Phi-3-Vision models."""

    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        print(f"Loaded {self.model_name}")

    def compute_sequence_probability(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_text: str,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        parts = [instruction]
        for i, caption in enumerate(captions):
            parts.append(f"<|image_{i+1}|>\n{caption}")
        parts.append(f"<|image_{len(images)}|>\n{final_prompt}")
        full_prompt_text = "\n\n".join(parts)

        inputs = self.processor(
            text=full_prompt_text, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_ids = torch.tensor(target_ids).unsqueeze(0).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]
        full_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)

        if "attention_mask" in inputs:
            target_mask = torch.ones((1, target_ids.shape[1]), dtype=inputs["attention_mask"].dtype, device=self.model.device)
            full_attention_mask = torch.cat([inputs["attention_mask"], target_mask], dim=1)
            inputs["attention_mask"] = full_attention_mask

        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                labels=labels,
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "labels"]}
            )

        logits = outputs.logits[:, :-1]
        target_token_ids = full_input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1, index=target_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        target_mask = (labels != -100)[:, 1:]
        target_token_logprobs = token_logprobs[target_mask]

        ref_logprob = target_token_logprobs.sum().item()
        return {
            "ref_logprob": ref_logprob,
            "num_tokens": int(target_mask.sum().item()),
            "ref_token_ids": target_ids.squeeze(0).tolist(),
        }


class Molmo2Scorer(VLMProbabilityScorer):
    """Scorer for Molmo2 models."""

    def compute_with_icl_cache(self, icl_cache, query_image, final_prompt, target_texts):
        """Molmo2 override: use single-stage caching because Molmo2's
        image_input_idx tensors don't align when the token sequence is split
        for two-stage ICL prefix caching."""
        all_images = icl_cache['icl_images'] + [query_image]
        return self.compute_cached_probabilities(
            all_images, icl_cache['captions'], icl_cache['instruction'],
            final_prompt, target_texts
        )

    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        print(f"Loaded {self.model_name}")

    def _prepare_inputs(self, images, captions, instruction, final_prompt):
        device = _get_input_device(self.model)
        content = []
        content.append({"type": "text", "text": instruction})
        for img, caption in zip(images[:-1], captions):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": caption})
        content.append({"type": "image", "image": images[-1]})
        content.append({"type": "text", "text": final_prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        return inputs

    def _prepare_icl_only_inputs(self, icl_images, captions, instruction):
        device = _get_input_device(self.model)
        content = [{"type": "text", "text": instruction}]
        for img, caption in zip(icl_images, captions):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": caption})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors="pt", return_dict=True
        )
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _prepare_standalone_image_inputs(self, image):
        device = _get_input_device(self.model)
        content = [{"type": "image", "image": image}, {"type": "text", "text": "x"}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors="pt", return_dict=True
        )
        kwargs = {}
        for key in ['images', 'image_input_idx', 'image_masks']:
            if key in inputs:
                kwargs[key] = inputs[key].to(device) if isinstance(inputs[key], torch.Tensor) else inputs[key]
        return kwargs

    def compute_sequence_probability(
        self,
        images: List[Image.Image],
        captions: List[str],
        instruction: str,
        final_prompt: str,
        target_text: str,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        inputs = self._prepare_inputs(images, captions, instruction, final_prompt)
        device = _get_input_device(self.model)

        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_ids = torch.tensor(target_ids).unsqueeze(0).to(device)

        prompt_len = inputs["input_ids"].shape[1]
        full_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)

        target_len = target_ids.shape[1]
        if "attention_mask" in inputs:
            target_attn = torch.ones((1, target_len), dtype=inputs["attention_mask"].dtype, device=self.model.device)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], target_attn], dim=1)
        if "token_type_ids" in inputs:
            target_type = torch.zeros((1, target_len), dtype=inputs["token_type_ids"].dtype, device=self.model.device)
            inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], target_type], dim=1)

        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                labels=labels,
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "labels"]}
            )

        logits = outputs.logits[:, :-1]
        target_token_ids = full_input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        vocab_size = logits.shape[-1]
        safe_target_token_ids = target_token_ids.clamp(0, vocab_size - 1)
        token_logprobs = log_probs.gather(
            dim=-1, index=safe_target_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        target_mask = (labels != -100)[:, 1:]
        target_token_logprobs = token_logprobs[target_mask]

        ref_logprob = target_token_logprobs.sum().item()
        return {
            "ref_logprob": ref_logprob,
            "num_tokens": int(target_mask.sum().item()),
            "ref_token_ids": target_ids.squeeze(0).tolist(),
        }


def get_scorer(model_key: str, device: str = "auto") -> VLMProbabilityScorer:
    """Factory function to get appropriate scorer."""
    model_info = MODELS[model_key]
    model_type = model_info["type"]
    model_name = model_info["name"]

    if model_type == "qwen2vl":
        return Qwen2VLScorer(model_name, device)
    elif model_type == "qwen2_5vl":
        return Qwen2_5VLScorer(model_name, device)
    elif model_type == "idefics3":
        return Idefics3Scorer(model_name, device)
    elif model_type == "llava_next":
        return LlavaNextScorer(model_name, device)
    elif model_type == "phi3_vision":
        return Phi3VisionScorer(model_name, device)
    elif model_type == "molmo2":
        return Molmo2Scorer(model_name, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_icl_context(
    icl_examples: List[Tuple[str, str]],
    query_image_path: str,
    n: int,
) -> Tuple[List[Image.Image], List[str], str]:
    """Build ICL context: load images and collect captions for probability scoring."""
    images = []
    captions = []

    for i, (img_path, caption) in enumerate(icl_examples[:n]):
        img = load_image(img_path)
        if img is not None:
            images.append(img)
            captions.append(caption)

    query_img = load_image(query_image_path)
    if query_img is None:
        print(f"[ERROR] Query image failed to load: {query_image_path}")
        return [], [], ""
    images.append(query_img)

    num_icl = len(captions)
    instruction = (
        f"You will be shown {num_icl} images, each followed by a caption. "
        "Then, you will see one final image and asked to provide its caption."
    )

    return images, captions, instruction


def extract_vanilla_mappings(fp: str, object_name: str = None, split: str = None) -> List[str]:
    """Extract vanilla mapping(s) from object name or image filepath."""
    if object_name and split:
        if split in ("known", "novel"):
            return [object_name]
        else:
            # Modified objects like "boar_toaster" -> ["boar", "toaster"]
            parts = object_name.split("_")
            if len(parts) >= 2:
                return parts[:2]
            return [object_name]
    # Fallback to path-based extraction
    if "known" in fp:
        filename = Path(fp).stem
        return [filename]
    elif "modified" in fp:
        filename = Path(fp).stem
        parts = filename.split("_")
        if len(parts) >= 2:
            return parts[:2]
        else:
            return [filename]
    else:
        return []


def run_single_probability_test(
    model_key: str,
    scorer: VLMProbabilityScorer,
    data_entry: Dict,
    precomputed_pools: Dict,
    nonce_mapping: Dict,
    similarity_type: str,
    caption_templates: List[str],
    final_archs: List[str],
    n: int = 8,
    icl_pool_override: Optional[List[Tuple[str, str]]] = None,
) -> Optional[Dict]:
    """Run single probability test for one data entry."""
    try:
        fp = data_entry["fp"]
        ref = data_entry["ref"]
        freq = data_entry["freq"]

        print(f"[{model_key}] Testing {fp} (ref: {ref}, freq: {freq}, n: {n})")

        if icl_pool_override is not None:
            icl_pool = list(icl_pool_override[:n])
        else:
            pool_samples = get_pool_for_datapoint_by_path(
                fp, precomputed_pools, nonce_mapping, similarity_type, caption_templates
            )
            if not pool_samples:
                print(f"[{model_key}] No pool found for {fp}")
                return None
            icl_pool = list(pool_samples[:n])

        rng = np.random.RandomState(hash((fp, n, SEED)) % (2**32 - 1))
        rng.shuffle(icl_pool)

        images, captions, instruction = build_icl_context(icl_pool, fp, n)

        if len(images) < 2:
            print(f"[{model_key}] Insufficient images for {fp}")
            return None

        final_prompt = random.choice(final_archs)

        # Check approximate sequence length to prevent OOM
        caption_tokens = sum(len(c.split()) for c in captions)
        approx_tokens = len(images) * 100 + caption_tokens + len(instruction.split()) + len(final_prompt.split())
        max_tokens = 32000

        if approx_tokens > max_tokens:
            print(f"[{model_key}] Sequence too long (~{approx_tokens} tokens > {max_tokens})")
            print(f"[{model_key}] Skipping {fp} with n={n}")
            return None

        target_arch = random.choice(caption_templates)
        target_sentence = target_arch.format(ref=ref)

        vanilla_mappings = extract_vanilla_mappings(
            fp, object_name=data_entry.get("object_name"), split=data_entry.get("split")
        )

        # Build all target texts for a single cached forward pass
        target_texts = [target_sentence, ref]
        vanilla_meta = []
        for vanilla_word in vanilla_mappings:
            vanilla_sentence = target_arch.format(ref=vanilla_word)
            target_texts.extend([vanilla_sentence, vanilla_word])
            vanilla_meta.append((vanilla_word, vanilla_sentence))

        # One prompt encoding, multiple cheap target evaluations
        all_results = scorer.compute_cached_probabilities(
            images=images, captions=captions, instruction=instruction,
            final_prompt=final_prompt, target_texts=target_texts,
        )

        sentence_result = all_results[0]
        nonce_result = all_results[1]

        vanilla_results = []
        for i, (vanilla_word, vanilla_sentence) in enumerate(vanilla_meta):
            vanilla_sentence_result = all_results[2 + i * 2]
            vanilla_word_result = all_results[2 + i * 2 + 1]
            vanilla_results.append({
                "vanilla_word": vanilla_word,
                "vanilla_sentence": vanilla_sentence,
                "vanilla_sentence_log_prob": vanilla_sentence_result["ref_logprob"],
                "vanilla_sentence_num_tokens": vanilla_sentence_result["num_tokens"],
                "vanilla_word_log_prob": vanilla_word_result["ref_logprob"],
                "vanilla_word_num_tokens": vanilla_word_result["num_tokens"],
                "vanilla_word_token_ids": vanilla_word_result["ref_token_ids"],
            })
            print(f"[{model_key}]   Vanilla '{vanilla_word}': sent_prob={vanilla_sentence_result['ref_logprob']:.4f}, word_prob={vanilla_word_result['ref_logprob']:.4f}")

        result = {
            "model": model_key,
            "fp": fp,
            "ref": ref,
            "freq": freq,
            "n": n,
            "similarity_type": similarity_type,
            "target_sentence": target_sentence,
            "full_sentence_log_prob": sentence_result["ref_logprob"],
            "full_sentence_num_tokens": sentence_result["num_tokens"],
            "nonce_word_log_prob": nonce_result["ref_logprob"],
            "nonce_word_num_tokens": nonce_result["num_tokens"],
            "nonce_word_token_ids": nonce_result["ref_token_ids"],
            "vanilla_mappings": vanilla_results,
            "icl_pool_intended": [(path, cap) for path, cap in icl_pool],
            "num_images_used": len(images),
            "num_icl_loaded": len(captions),
            "instruction_sent": instruction,
            "final_prompt_sent": final_prompt,
        }

        print(f"[{model_key}] Done {fp}: full_prob={sentence_result['ref_logprob']:.4f}, nonce_prob={nonce_result['ref_logprob']:.4f}")
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError as e:
        print(f"[{model_key}] CUDA OOM on {data_entry.get('fp', 'unknown')}: {e}")
        print(f"[{model_key}] Attempting aggressive GPU memory cleanup...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{model_key}] Memory cleaned. Skipping this sample and continuing...")
        return None

    except Exception as e:
        print(f"[{model_key}] Failed on {data_entry.get('fp', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        if "memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"[{model_key}] Detected potential memory issue, cleaning GPU cache...")
            torch.cuda.empty_cache()
            gc.collect()
        return None


def run_probability_experiments(
    nvrd_ds,
    nvrd_index: Dict,
    perturbation_types: List[str],
    all_levels: List[int],
    precomputed_pools: Dict,
    object_to_nonce: Dict,
    similarity_type: str,
    prompt_mode: str,
    caption_archs_local: List[str],
    final_archs: List[str],
    scorer: VLMProbabilityScorer,
    out_path: str,
):
    """Run all probability experiments for one similarity_type x prompt_mode.

    Iterates over NVRD objects/perturbations (same as generation experiments).
    """
    n_values = _cli_n_values

    print(f"\n{'='*60}")
    print(f"PROBABILITY: similarity={similarity_type} | prompt={prompt_mode}")
    print(f"Output: {out_path}")
    print(f"{'='*60}\n")

    ensure_parent_dir(out_path)

    # Load existing results to skip completed
    completed = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("object_name"), obj.get("perturbation_type"),
                               obj.get("level"), obj.get("n"))
                        completed.add(key)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")

    print(f"Loaded {len(completed)} existing probability results, skipping those.\n")

    consecutive_failures = 0
    max_consecutive_failures = 10
    total_attempts = 0
    total_successes = 0

    object_names = sorted(nvrd_index.keys())
    obj_range = getattr(sys.modules[__name__], '_cli_obj_range', None)
    if obj_range:
        start, end = map(int, obj_range.split("-"))
        object_names = object_names[start:end+1]
        print(f"Object range filter: indices {start}-{end} ({len(object_names)} objects)")
    total_objects = len(object_names)

    for obj_idx, object_name in enumerate(object_names):
        obj_data = nvrd_index[object_name]

        icl_pool, nonce_word = get_pool_for_datapoint_by_name(
            object_name=object_name,
            precomputed_pools=precomputed_pools,
            object_to_nonce=object_to_nonce,
            similarity_type=similarity_type,
            caption_templates=caption_archs_local,
        )

        if not icl_pool:
            continue
        if nonce_word is None:
            continue

        any_idx = next(iter(next(iter(obj_data.values())).values()))
        any_row = nvrd_ds[any_idx]
        split_type = any_row["split"]
        split_type_norm = split_type.replace("modified/", "").replace("/perturbations", "")

        if SPLIT_FILTER != "all" and split_type_norm != SPLIT_FILTER:
            continue

        # Load the original (unperturbed) image for the nonce word ICL example
        original_image = any_row["original_image"]
        main_example_caption = random.choice(caption_archs_local).format(ref=nonce_word)

        print(f"\n[{obj_idx+1}/{total_objects}] Probability for '{object_name}' "
              f"(split={split_type_norm}, nonce={nonce_word})")

        # Iterate n → ptype → level so we can cache ICL prefix per (object, n)
        for n in n_values:
            # Check if any work remains for this n
            has_work = False
            for ptype in perturbation_types:
                if ptype not in obj_data:
                    continue
                for level in all_levels:
                    if level not in obj_data[ptype]:
                        continue
                    if (object_name, ptype, level, n) not in completed:
                        has_work = True
                        break
                if has_work:
                    break
            if not has_work:
                continue

            # Build ICL context for this (object, n)
            # Include distractor pool + original image with nonce word caption
            # (matches the generation experiment setup)
            rng_pool = np.random.RandomState(hash((object_name, n, SEED)) % (2**32 - 1))
            built_pool = list(icl_pool[:n]) + [(original_image, main_example_caption)]
            rng_pool.shuffle(built_pool)

            icl_images = []
            icl_captions_list = []
            for item in built_pool:
                img_or_path, caption = item
                if isinstance(img_or_path, Image.Image):
                    icl_images.append(img_or_path)
                    icl_captions_list.append(caption)
                else:
                    img = load_image(img_or_path)
                    if img is not None:
                        icl_images.append(img)
                        icl_captions_list.append(caption)

            if len(icl_images) < 1:
                print(f"  [SKIP] No ICL images loaded for n={n}")
                continue

            num_icl = len(icl_captions_list)
            instruction = (
                f"You will be shown {num_icl} images, each followed by a caption. "
                "Then, you will see one final image and asked to provide its caption."
            )

            # Find a sample query image for ICL cache setup
            sample_row = None
            for ptype in perturbation_types:
                if ptype in obj_data:
                    for level in all_levels:
                        if level in obj_data[ptype]:
                            sample_row = nvrd_ds[obj_data[ptype][level]]
                            break
                if sample_row is not None:
                    break
            sample_query = load_image(sample_row["image_path"])
            sample_final_prompt = random.choice(final_archs)

            # Build ICL prefix cache (Level 1)
            try:
                icl_cache = scorer.prepare_icl_cache(
                    icl_images, icl_captions_list, instruction,
                    sample_query, sample_final_prompt
                )
            except Exception as e:
                print(f"  [WARNING] ICL cache failed: {e}, falling back to non-cached")
                icl_cache = None

            target_arch = random.choice(caption_archs_local)
            target_sentence = target_arch.format(ref=nonce_word)
            vanilla_mappings = extract_vanilla_mappings(
                sample_row["image_path"],
                object_name=object_name, split=split_type_norm
            )

            # Build target texts (same for all perturbation levels)
            target_texts = [target_sentence, nonce_word]
            vanilla_meta = []
            for vanilla_word in vanilla_mappings:
                vanilla_sentence = target_arch.format(ref=vanilla_word)
                target_texts.extend([vanilla_sentence, vanilla_word])
                vanilla_meta.append((vanilla_word, vanilla_sentence))

            for ptype in perturbation_types:
                if ptype not in obj_data:
                    continue
                for level in all_levels:
                    if level not in obj_data[ptype]:
                        continue

                    key = (object_name, ptype, level, n)
                    if key in completed:
                        continue

                    row = nvrd_ds[obj_data[ptype][level]]
                    query_image_path = row["image_path"]
                    total_attempts += 1

                    try:
                        query_image = load_image(query_image_path)
                        if query_image is None:
                            raise ValueError(f"Failed to load {query_image_path}")

                        final_prompt = random.choice(final_archs)

                        if icl_cache is not None:
                            # Level 1 + Level 2 cached
                            all_results = scorer.compute_with_icl_cache(
                                icl_cache, query_image, final_prompt, target_texts
                            )
                        else:
                            # Level 2 only (fallback)
                            all_images = icl_images + [query_image]
                            all_results = scorer.compute_cached_probabilities(
                                all_images, icl_captions_list, instruction,
                                final_prompt, target_texts
                            )

                        sentence_result = all_results[0]
                        nonce_result = all_results[1]

                        vanilla_results = []
                        for i, (vw, vs) in enumerate(vanilla_meta):
                            vsr = all_results[2 + i * 2]
                            vwr = all_results[2 + i * 2 + 1]
                            vanilla_results.append({
                                "vanilla_word": vw,
                                "vanilla_sentence": vs,
                                "vanilla_sentence_log_prob": vsr["ref_logprob"],
                                "vanilla_sentence_num_tokens": vsr["num_tokens"],
                                "vanilla_word_log_prob": vwr["ref_logprob"],
                                "vanilla_word_num_tokens": vwr["num_tokens"],
                                "vanilla_word_token_ids": vwr["ref_token_ids"],
                            })
                            print(f"[{MODEL_KEY}]   Vanilla '{vw}': "
                                  f"sent_prob={vsr['ref_logprob']:.4f}, "
                                  f"word_prob={vwr['ref_logprob']:.4f}")

                        result = {
                            "model": MODEL_KEY,
                            "fp": query_image_path,
                            "ref": nonce_word,
                            "freq": 0,
                            "n": n,
                            "similarity_type": similarity_type,
                            "target_sentence": target_sentence,
                            "full_sentence_log_prob": sentence_result["ref_logprob"],
                            "full_sentence_num_tokens": sentence_result["num_tokens"],
                            "nonce_word_log_prob": nonce_result["ref_logprob"],
                            "nonce_word_num_tokens": nonce_result["num_tokens"],
                            "nonce_word_token_ids": nonce_result["ref_token_ids"],
                            "vanilla_mappings": vanilla_results,
                            "num_images_used": len(icl_images) + 1,
                            "num_icl_loaded": num_icl,
                            "instruction_sent": instruction,
                            "final_prompt_sent": final_prompt,
                            "object_name": object_name,
                            "perturbation_type": ptype,
                            "level": level,
                            "split": split_type_norm,
                        }

                        print(f"[{MODEL_KEY}] Done {query_image_path}: "
                              f"full_prob={sentence_result['ref_logprob']:.4f}, "
                              f"nonce_prob={nonce_result['ref_logprob']:.4f}")

                        append_jsonl(out_path, result)
                        completed.add(key)
                        consecutive_failures = 0
                        total_successes += 1

                    except torch.cuda.OutOfMemoryError:
                        print(f"[{MODEL_KEY}] CUDA OOM on {query_image_path}, cleaning up...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        consecutive_failures += 1

                    except Exception as e:
                        print(f"[{MODEL_KEY}] Failed on {query_image_path}: {e}")
                        consecutive_failures += 1

                    if consecutive_failures > 0:
                        print(f"[WARNING] Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                        if consecutive_failures == 3:
                            print(f"[RECOVERY] Reloading model...")
                            scorer.model = None
                            scorer.processor = None
                            scorer.tokenizer = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            scorer.load_model()
                            icl_cache = None  # Invalidate cache after reload
                            consecutive_failures = 0
                            print(f"[RECOVERY] Model reloaded, cache invalidated")
                        if consecutive_failures >= max_consecutive_failures:
                            print(f"\n{'='*60}")
                            print(f"STOPPING: {max_consecutive_failures} consecutive failures")
                            print(f"Progress: {total_successes}/{total_attempts}")
                            print(f"{'='*60}\n")
                            return

            # Free ICL cache after all perturbation levels for this n
            if icl_cache is not None:
                del icl_cache['past_kv']
                del icl_cache
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"PROBABILITY STATS: {similarity_type} | {prompt_mode}")
    print(f"Success rate: {total_successes}/{total_attempts} ({100*total_successes/max(total_attempts,1):.1f}%)")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}\n")


# #####################################################################
#
#  MAIN ENTRY POINT
#
# #####################################################################

if __name__ == "__main__":
    split_label = SPLIT_FILTER.replace("/", "-")
    print("=" * 60)
    print(f"ICL Experiments — Combined Generation + Probability")
    print(f"Model: {MODEL_KEY} ({MODEL_NAME})")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Split filter: {SPLIT_FILTER}")
    print(f"Similarity filter: {CLI_SIMILARITY}")
    print(f"Ablation mode: {_ABLATION_MODE}")
    print("=" * 60)
    print()

    # --- Load all data once ---
    training_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nonce_word_mapping.json")
    print(f"Loading training data from {training_file}...")
    with open(training_file, "r") as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} training entries\n")

    nvrd_ds, nvrd_index, perturbation_types, all_levels = load_nvrd_dataset()
    print()

    object_to_nonce = build_object_to_nonce(training_file)
    print()

    nonce_mapping = load_nonce_word_mapping(training_file)

    _repo_root_main = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pool_file = os.path.join(_repo_root_main, "data", "precomputed_icl_pools.json")
    if not os.path.exists(pool_file):
        pool_file = "precomputed_icl_pools.json"
        print(f"Warning: precomputed_icl_pools.json not found at expected location; trying current directory")
    precomputed_pools = load_precomputed_pools(pool_file)

    # --- Load model once (shared between generation and probability) ---
    is_api_model = MODEL_TYPE in ("gemini", "openai")
    scorer = None

    if not is_api_model:
        scorer = get_scorer(MODEL_KEY)
        scorer.load_model()

        # Point global model state to scorer's model so generate_with_vlm can use it
        vlm_model = scorer.model
        vlm_processor = scorer.processor
        vlm_tokenizer = scorer.tokenizer

    # --- Experiment grid (filtered by CLI flags) ---
    similarity_types = [CLI_SIMILARITY] if CLI_SIMILARITY != "all" else ["visual_similarity", "color_similarity", "random"]
    prompt_modes = ["fillin"]

    print(f"\nExperiment grid:")
    print(f"  Similarity types: {similarity_types}")
    print(f"  Prompt modes: {prompt_modes}")
    print(f"  n values: [2, 4, 8]")
    print(f"  Perturbation types ({len(perturbation_types)}): {perturbation_types}")
    print(f"  Levels: {all_levels[0]}-{all_levels[-1]} ({len(all_levels)} levels)")
    if is_api_model:
        print(f"  NOTE: Skipping probability experiments (not supported for API models)")
    print()

    for similarity_type in similarity_types:
        for prompt_mode in prompt_modes:
            final_archs = final_caption_archs_fillin

            # --- Generation experiments ---
            split_suffix = f"-{split_label}" if SPLIT_FILTER != "all" else ""
            _obj_range = getattr(sys.modules[__name__], '_cli_obj_range', None)
            range_suffix = f"-r{_obj_range}" if _obj_range else ""
            _prob_only = getattr(sys.modules[__name__], '_cli_prob_only', False)
            if _prob_only:
                print(f"Skipping generation (--prob-only)")
            else:
                _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                gen_out = os.path.join(_repo_root, "results", "generation", f"{MODEL_KEY}-{similarity_type}-{prompt_mode}{split_suffix}{range_suffix}-outputs.jsonl")
                run_generation_experiments(
                    nvrd_ds, nvrd_index, perturbation_types, all_levels,
                    precomputed_pools, object_to_nonce,
                    similarity_type, prompt_mode, final_archs, gen_out,
                )
                print(f"\nFinished generation: {similarity_type} | {prompt_mode}")
                print(f"Results: {gen_out}\n")

            # --- Probability experiments (local models only) ---
            if not is_api_model:
                prob_out = os.path.join(_repo_root, "results", "probability", f"{MODEL_KEY}-{similarity_type}-{prompt_mode}{split_suffix}{range_suffix}_results.jsonl")
                run_probability_experiments(
                    nvrd_ds, nvrd_index, perturbation_types, all_levels,
                    precomputed_pools, object_to_nonce,
                    similarity_type, prompt_mode, caption_archs, final_archs, scorer, prob_out,
                )
                print(f"\nFinished probability: {similarity_type} | {prompt_mode}")
                print(f"Results: {prob_out}\n")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Model: {MODEL_KEY}")
    print(f"Split: {SPLIT_FILTER}")
    print(f"Similarity types: {similarity_types}")
    print(f"Prompt modes: {prompt_modes}")
    print("=" * 60)
