"""
Shared infrastructure for the depth-map identity-hiding diagnostic (grounding test, part 1).

Rationale (see conversation/README context): a monocular depth map preserves an object's 3D
shape -- the geometric cue affordances actually depend on -- while stripping the color/texture/
material cues a VLM uses to recognize *what the object is* and recall a language prior about it.
This is the same justification AGWM's Depth Feature Injector gives for using depth: "they exclude
color information, allowing the model to fully focus on shape information, which is highly
relevant to affordances" (Depth-Anything is the same depth model family cited there).

Depth maps are generated once per object at full original resolution (Depth-Anything's own
pipeline upsamples back to input size), so the existing SAM+SLIC part masks -- computed on the
original photo -- apply unchanged; no re-segmentation needed.

Usage (batch generation, CPU only -- keeps GPUs free for the VLM):
    python depth_diagnostic.py [object_name ...]   # default: all KNOWN_OBJECTS
"""
import os
import sys

import numpy as np
from PIL import Image
from transformers import pipeline

# region_information_gain.py parses MAX_REGIONS from sys.argv[2] at import time, which collides
# with this script's own multi-object argv (e.g. a second object name is not an int) -- argv isn't
# read until main(), so it's safe to blank it out just for the import.
_argv, sys.argv = sys.argv, sys.argv[:1]
from region_information_gain import REGIONS_TEST, resolve_image_path
sys.argv = _argv

HERE = os.path.dirname(os.path.realpath(__file__))
DEPTH_DIR = os.path.join(REGIONS_TEST, "data", "known_depth")
os.makedirs(DEPTH_DIR, exist_ok=True)

DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
KNOWN_OBJECTS = ["ladle", "chair", "corkscrew", "backpack", "bow-tie", "clock", "coffee-table", "fedora",
                 "fire-alarm", "wallet"]


def depth_image_path(object_name: str) -> str:
    return os.path.join(DEPTH_DIR, f"{object_name}.png")


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    """Normalize a raw depth map to 0-255 and replicate to 3 channels -- grayscale, so no color
    cues survive, but still a plain RGB image the rest of the pipeline (PIL/processor) expects."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    gray = (d * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


def main():
    objects = sys.argv[1:] if len(sys.argv) > 1 else KNOWN_OBJECTS
    print(f"Loading {DEPTH_MODEL_NAME} (CPU)...")
    depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_NAME, device="cpu")

    for obj in objects:
        img_path = resolve_image_path(obj)
        image = Image.open(img_path).convert("RGB")
        result = depth_pipe(image)
        depth_arr = np.array(result["depth"])
        out = depth_to_rgb(depth_arr)
        assert out.size == image.size, f"{obj}: depth {out.size} != original {image.size}"
        out_path = depth_image_path(obj)
        out.save(out_path)
        print(f"{obj:14s} {image.size} -> {out_path}")


if __name__ == "__main__":
    main()
