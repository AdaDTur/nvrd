"""
Grounding test 2: wrong-name swap.

If the VLM assigns a part's affordance by first silently recognizing the whole object (e.g.
"this is a ladle") and then reciting a language prior about ladles, then telling it -- falsely
-- that the object is something else should drag the labels toward that false object's typical
affordances, even though the actual pixels haven't changed at all. If the labels are instead
read off the highlighted region's true visual shape/material, a false name shouldn't move them
much.

For one object, scores every labelled part under three caption conditions:
    none       : affordance_labeling.py's default -- no object name is stated anywhere
    true_name  : "This object is a <object_name>." prepended to the question
    wrong_name : "This object is <wrong_name>." prepended to the question (same object pixels)

wrong_name is deliberately chosen so its stereotypical affordance has an unambiguous, specific
word in the AGD20K vocabulary that would not plausibly arise from the true object's actual
shape/material (e.g. calling a ladle "a magnifying glass" tests for "look out", a word with no
plausible reading of a metal spoon). --expect-word names that word so its probability is
tracked explicitly across all three conditions, in addition to the full top-5 printout.

Requires <object>_<region_source>_affordance_labels.json (for the part masks; run
affordance_labeling.py first). Does not require a matching context/labels run for wrong_name --
this rescoring is independent.

Usage:
    [REGION_SOURCE=sam_slic|sam] python grounding_wrongname.py [object_name] [wrong_name] [expect_word]
    # default: ladle "a magnifying glass" "look out"
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

HERE = os.path.dirname(os.path.realpath(__file__))
OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "ladle"
WRONG_NAME = sys.argv[2] if len(sys.argv) > 2 else "a magnifying glass"
EXPECT_WORD = sys.argv[3] if len(sys.argv) > 3 else "look out"

# affordance_labeling.py (via region_information_gain.py) parses MAX_REGIONS from sys.argv[2] at
# import time, which collides with WRONG_NAME above ("a magnifying glass" is not an int) -- argv
# is already consumed into the constants above, so it's safe to blank it out just for the import.
_argv, sys.argv = sys.argv, sys.argv[:1]
from affordance_labeling import (
    MODEL_NAME, VOCABULARIES, build_prompt_text, downscale, highlight_region, labels_path, score_options, softmax,
)
sys.argv = _argv

HIGHLIGHT_INTRO = (
    "The image shows an object. One part of it is highlighted: it is at full brightness and "
    "outlined in magenta, while the rest of the image is darkened (use it only as context). "
)


def build_question(caption: "str | None", options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    prefix = f"{caption} " if caption else ""
    return (
        prefix + HIGHLIGHT_INTRO
        + "Which single affordance does the highlighted part offer? "
        + f"Choose exactly one of: {listed}. "
        + 'If the highlighted part offers none of these (for example plain background), answer "none". '
        + "Answer with the option only."
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(labels_path(OBJECT_NAME)) as f:
        data = json.load(f)
    image = Image.open(data["image_path"]).convert("RGB")
    conditions = {
        "none": None,
        "true_name": f"This object is {'an' if OBJECT_NAME[0].lower() in 'aeiou' else 'a'} {OBJECT_NAME}.",
        "wrong_name": f"This object is {WRONG_NAME}.",
    }
    print(f"Object: {OBJECT_NAME} | wrong_name={WRONG_NAME!r} | expect_word={EXPECT_WORD!r}")
    print(f"Conditions: {conditions}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for part in data["regions"]:
        mask = mask_utils.decode(part["segmentation"]).astype(bool)
        highlighted = downscale(highlight_region(image, mask))
        rec = {"region_id": part["region_id"], "area": part["area"], "conditions": {}}
        print(f"\n{part['region_id']} (area={part['area']})")
        for vocab, vocab_cfg in VOCABULARIES.items():
            options = vocab_cfg["options"]
            option_ids = [processor.tokenizer(o + "<|im_end|>", add_special_tokens=False).input_ids for o in options]
            expect_idx = options.index(EXPECT_WORD) if EXPECT_WORD in options else None
            cond_results = {}
            for cond_name, caption in conditions.items():
                prompt_text = build_prompt_text(processor, build_question(caption, options, vocab_cfg["glosses"]))
                logprobs = score_options(model, processor, highlighted, prompt_text, option_ids, device)
                probs = softmax(logprobs)
                i = int(np.argmax(probs))
                cond_results[cond_name] = {
                    "top_label": options[i],
                    "top_prob": float(probs[i]),
                    "expect_word_prob": float(probs[expect_idx]) if expect_idx is not None else None,
                }
            rec["conditions"][vocab] = cond_results
            line = f"  [{vocab:>6s}] "
            for cond_name in conditions:
                r = cond_results[cond_name]
                ew = f", p({EXPECT_WORD})={r['expect_word_prob']:.2f}" if r["expect_word_prob"] is not None else ""
                line += f"{cond_name}: {r['top_label']} ({r['top_prob']:.2f}{ew})  |  "
            print(line)
        results.append(rec)

    out = {
        "object": OBJECT_NAME, "wrong_name": WRONG_NAME, "expect_word": EXPECT_WORD,
        "model": MODEL_NAME, "conditions": conditions, "regions": results,
    }
    out_path = os.path.join(HERE, f"{OBJECT_NAME}_grounding_wrongname.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
