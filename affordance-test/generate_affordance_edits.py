"""
For every NVRD novel object and every affordance in a fixed vocabulary, ask an image-editing model
(Gemini 2.5 Flash Image, aka "nano-banana") to edit the object's image so its form more clearly
affords that action -- e.g. shape_16 edited to grow a cupped, grasping structure for "grasp" -- the
same kind of edit the user hand-tested with Gemini directly (shape_16 -> "cup holder" affordance).

This is a generative counterpart to the rest of the pipeline: instead of asking a VLM to label which
affordance a region already has, it asks an image model to visibly ADD a given affordance to the
object, producing a stimulus set that could later be fed back through the labeling/RISE/LOO pipeline
to check whether the edit actually shifts the model's own forced-choice labels toward the target
affordance (an image-space analogue of the activation-patching idea from earlier in this project).

Vocabulary: GAT's 8-class AFF_LIST (grasp, cut, scoop, pound, support, screw, contain, stick) --
the same vocabulary figures/gat_comparison/ and the Prolific affordance-study demo use -- not
AGD20K's 37 verbs, most of which (ride, type on, brush with, ...) presuppose an object category the
abstract/novel NVRD shapes don't have. Swap AFFORDANCE_VOCAB below for AGD20K_VERBS from
affordance_labeling.py if broader, more object-specific coverage is wanted instead.

No manual environment setup needed -- just:
    python3 generate_affordance_edits.py [object ...]
Run with ANY python3 (even bare system python with nothing installed). The bootstrap block right
below re-execs the process under ./venv (created once via `python3 -m venv venv` + `pip install
openai google-genai pillow` into it) if it's not already running there, and loads GEMINI_API_KEY /
OPENAI_API_KEY from ./.env automatically -- so `python3 <script>.py` is the whole invocation.

Usage:
    python3 generate_affordance_edits.py [object ...]   # default: first 5 novel objects (OBJECT_LIMIT)
    OBJECT_LIMIT=10 python3 generate_affordance_edits.py                 # all 10 local novel objects
    AFFORDANCE_LIMIT=5 python3 generate_affordance_edits.py shape_16     # explicit object, 5 affordances
Already-generated (object, affordance) pairs are skipped automatically, so a rerun resumes rather
than regenerating -- delete the specific output PNG(s) under figures/affordance_edits/ to redo one.
"""
import os
import sys

# --- self-bootstrap: re-exec under ./venv, then auto-load ./.env -- before any third-party import,
# so this works invoked with a bare system python3 that has nothing installed yet. ---
HERE = os.path.dirname(os.path.realpath(__file__))
_VENV_DIR = os.path.join(HERE, "venv")
# Check sys.prefix, not sys.executable -- on this system venv/bin/python3 is a symlink to the same
# system python *binary*, so comparing executable paths never detects "not yet in the venv"; prefix
# reflects whether the interpreter was actually started as the venv (via its pyvenv.cfg), which is
# what actually matters (site-packages, i.e. whether openai/google-genai/pillow are importable).
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _venv_py = os.path.join(_VENV_DIR, "bin", "python3")
    os.environ["PYTHONUNBUFFERED"] = "1"  # execv drops the original -u flag (argv doesn't carry it)
    os.execv(_venv_py, [_venv_py] + sys.argv)

_ENV_FILE = os.path.join(HERE, ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())  # real exported env vars still win

import json
import time
import io

from PIL import Image

REPO_ROOT = os.path.dirname(HERE)
NOVEL_DIR = os.path.join(REPO_ROOT, "regions_test", "data", "novel")
OUT_DIR = os.path.join(HERE, "figures", "affordance_edits")
JSON_DIR = os.path.join(HERE, "json", "affordance_edits")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

MODEL_NAME = os.environ.get("AFFORDANCE_EDIT_MODEL", "gemini-2.5-flash-image")
MAX_RETRIES = 3
RETRY_BACKOFF_S = 10
REQUEST_DELAY_S = 2  # be polite to the API between calls

# GAT's AFF_LIST (run_gat_inference.py), with a one-line gloss each -- the 5 shared with
# affordance_labeling.UMD_GLOSSES reuse its exact wording; screw/stick are new, written to match style.
AFFORDANCE_VOCAB = {
    "grasp": "can be picked up or held with the fingers",
    "cut": "has a thin, flat, wide bladed edge along a straight or curved line, for slicing through "
           "something the way a knife blade does -- a flat plane with a sharp LINE of an edge, not a point",
    "scoop": "has a curved surface for scooping material up",
    "pound": "can be used to strike or pound",
    "support": "can hold up other objects",
    "screw": "can be twisted into or onto something",
    "contain": "has a cavity that holds things inside",
    "stick": "has a slender, narrow, tapering POINT for piercing or spearing into something, like a "
             "needle, nail, or thorn tip -- a sharp point, not a wide flat blade",
}

def _natural_key(name: str):
    # "shape_2" before "shape_10" -- plain sorted() puts "shape_10" before "shape_2" (lexicographic).
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


NOVEL_OBJECTS = sorted(
    (os.path.splitext(f)[0] for f in os.listdir(NOVEL_DIR) if f.endswith(".png")), key=_natural_key,
) if os.path.isdir(NOVEL_DIR) else []

_AFF_LIMIT = os.environ.get("AFFORDANCE_LIMIT")
if _AFF_LIMIT:
    AFFORDANCE_VOCAB = dict(list(AFFORDANCE_VOCAB.items())[:int(_AFF_LIMIT)])

# Default to the first 5 objects rather than all of them -- pass explicit object names on the
# command line, or set OBJECT_LIMIT, to run on more/fewer/different ones.
OBJECT_LIMIT = int(os.environ.get("OBJECT_LIMIT", 5))

PROMPT_TEMPLATE = (
    "This image shows an abstract, non-representational 3D-rendered object. Reshape THIS EXACT object "
    "so it clearly and unmistakably affords \"{affordance}\" ({gloss}) -- but aim for a MODERATE, "
    "well-judged amount of change, not an extreme one: change enough that a person would immediately "
    "agree the object could be used to {affordance} something, while changing as little else as "
    "possible. Think of the result as a later model in the same product line as the original -- "
    "obviously designed by the same team, from the same materials and parts, just adapted for a new "
    "purpose -- not a different object.\n\n"
    "Preserve exactly as they are in the original image, pixel-for-pixel where possible: the object's "
    "overall outline/silhouette outside the one feature you reshape for \"{affordance}\", the number and "
    "arrangement of its distinct limbs/rings/protrusions, its material finish and surface texture, its "
    "exact color palette, the lighting direction and shadows, the camera angle, and the background. "
    "Change only what is necessary to add the \"{affordance}\" feature; leave every other part of the "
    "object untouched rather than regenerating the whole object from scratch.\n\n"
    "Do NOT let it become a generic, instantly-recognizable everyday tool (e.g. scissors, a hammer, a "
    "door handle, a screw) that merely borrows the original's colors -- it should still look like a "
    "distinct alien/sci-fi design, not a real-world object."
)


def resolve_image_path(object_name: str) -> str:
    path = os.path.join(NOVEL_DIR, f"{object_name}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No novel-split image for '{object_name}' at {path}")
    return path


def edit_image(client, image: Image.Image, affordance: str, gloss: str):
    """Returns (PIL.Image | None, text_response | None, error | None)."""
    from google.genai import types

    prompt = PROMPT_TEMPLATE.format(affordance=affordance, gloss=gloss)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            out_image, out_text = None, None
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) is not None:
                    out_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                elif getattr(part, "text", None):
                    out_text = part.text
            if out_image is None:
                last_err = f"no image returned (text: {out_text!r})"
                continue
            return out_image, out_text, None
        except Exception as e:  # noqa: BLE001 -- surfaced to caller, not swallowed
            last_err = str(e)
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return None, None, last_err


def main():
    objects = sys.argv[1:] if len(sys.argv) > 1 else NOVEL_OBJECTS[:OBJECT_LIMIT]
    if not objects:
        print(f"No novel objects found under {NOVEL_DIR}")
        return
    print(f"Objects ({len(objects)}): {objects}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set -- export it (https://aistudio.google.com/apikey) and rerun.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    manifest = []
    for obj in objects:
        try:
            img_path = resolve_image_path(obj)
        except FileNotFoundError as e:
            print(f"  {obj}: {e}")
            continue
        image = Image.open(img_path).convert("RGB")
        obj_out_dir = os.path.join(OUT_DIR, obj)
        os.makedirs(obj_out_dir, exist_ok=True)

        print(f"\n=== {obj} ===")
        for affordance, gloss in AFFORDANCE_VOCAB.items():
            out_path = os.path.join(obj_out_dir, f"{obj}_{affordance}.png")
            record = {
                "object": obj, "affordance": affordance, "gloss": gloss, "model": MODEL_NAME,
                "prompt": PROMPT_TEMPLATE.format(affordance=affordance, gloss=gloss),
                "source_image": img_path, "output_image": out_path,
            }
            if os.path.isfile(out_path):
                print(f"  {affordance}: already done, skipping")
                record["status"] = "skipped_existing"
                manifest.append(record)
                continue

            out_image, out_text, err = edit_image(client, image, affordance, gloss)
            if out_image is None:
                print(f"  {affordance}: FAILED -- {err}")
                record["status"], record["error"] = "failed", err
            else:
                out_image.save(out_path)
                print(f"  {affordance}: saved {out_path}" + (f"  (model also said: {out_text!r})" if out_text else ""))
                record["status"] = "ok"
                if out_text:
                    record["model_text"] = out_text
            manifest.append(record)
            time.sleep(REQUEST_DELAY_S)

    manifest_path = os.path.join(JSON_DIR, "manifest.json")
    existing = []
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
    existing = [r for r in existing if not any(r["object"] == m["object"] and r["affordance"] == m["affordance"] for m in manifest)]
    with open(manifest_path, "w") as f:
        json.dump(existing + manifest, f, indent=2)
    print(f"\nSaved {manifest_path}")


if __name__ == "__main__":
    main()
