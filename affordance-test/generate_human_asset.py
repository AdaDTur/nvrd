"""
One-off generator for assets/human_silhouette.png -- the reusable human reference figure
generate_size_charts.py composites into every chart. Run this ONLY to (re)create that asset, not
as part of the regular chart-generation flow (generate_size_charts.py just loads the cached PNG).

Unlike the per-object silhouettes in generate_size_charts.py (extracted from existing renders,
never AI-drawn), this one generic "standing person" icon IS generated via Gemini 2.5 Flash Image
-- see that script's module docstring for why this is a fundamentally lower-risk use of a
generative model than editing a specific object's shape. Pure black silhouette on pure white
background, by prompt construction, so extraction is a simple threshold (no gradient/lighting to
fight, unlike the studio-lit object renders -- grabCut isn't needed here).

No manual environment setup needed -- same self-bootstrap as generate_affordance_edits.py:
    python3 generate_human_asset.py
"""
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
_VENV_DIR = os.path.join(HERE, "venv")  # has google-genai, same as generate_affordance_edits.py
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _venv_py = os.path.join(_VENV_DIR, "bin", "python3")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(_venv_py, [_venv_py] + sys.argv)

_ENV_FILE = os.path.join(HERE, ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import io

import cv2
import numpy as np
from PIL import Image

ASSETS_DIR = os.path.join(HERE, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
RAW_PATH = os.path.join(ASSETS_DIR, "human_silhouette_raw.png")
OUT_PATH = os.path.join(ASSETS_DIR, "human_silhouette.png")

PROMPT = (
    "A realistic black silhouette of an adult person standing, cut from a photograph -- like the "
    "stock-photo human silhouettes used in scientific scale-comparison infographics (dinosaur "
    "size charts, animal size charts), NOT a flat geometric icon, NOT a pictogram, NOT a bathroom-"
    "sign figure, NOT made of simple rectangles or circles. Natural, anatomically realistic body "
    "proportions: a visible neck, sloped shoulders, a naturally shaped head with hair, arms that "
    "taper and hang relaxed at the sides with a slight bend at the elbow, visible hand and finger "
    "shapes (not blocky stumps), a natural waist and hip curve, and two separate legs with "
    "realistic calf/ankle tapering, standing casually (weight very slightly on one leg, not a "
    "perfectly rigid symmetrical pose). Facing forward or very slightly turned. Rendered as a "
    "single solid, 100% opaque, smooth-edged black silhouette shape -- no internal lines, no "
    "gradient, no texture, no face features, no clothing pattern, just the outline shape filled "
    "solid black. Centered on a plain, pure white, completely empty background -- no ground line, "
    "no shadow, no other objects, no text, no watermark."
)


def generate_raw():
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set -- add it to .env and rerun.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[PROMPT],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    out_image = None
    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            out_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    if out_image is None:
        print("FAILED -- no image in response")
        sys.exit(1)
    out_image.save(RAW_PATH)
    print(f"saved {RAW_PATH} {out_image.size}")


def threshold_to_asset():
    """Simple black/white threshold (not grabCut -- the prompt already guarantees flat, pure
    black-on-white contrast, so there's no lighting gradient to fight here)."""
    img = cv2.imread(RAW_PATH, cv2.IMREAD_GRAYSCALE)
    mask = (img < 128).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(bool)

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cropped = mask[y0:y1, x0:x1]

    h, w = cropped.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[cropped] = (0, 0, 0, 255)  # recolored per-chart by generate_size_charts.py at paste time
    Image.fromarray(rgba, mode="RGBA").save(OUT_PATH)
    print(f"saved {OUT_PATH} {cropped.shape[1]}x{cropped.shape[0]}, fill {cropped.mean():.1%}")


if __name__ == "__main__":
    generate_raw()
    threshold_to_asset()
