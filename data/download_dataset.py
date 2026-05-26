"""
Download the NVRD dataset from HuggingFace (adadtur/nvrd) into data/sample-data/.

The downloaded images follow this directory structure:

  data/sample-data/{split}/{object}.png                                  (originals)
  data/sample-data/{split}/perturbations_comp/{ptype}/{object}_{level}.png

where split is one of: known, novel, modified/shape-shape, modified/shape-texture.

This mirrors the layout expected by the experiment scripts.

Usage:
    python data/download_dataset.py

Requires: datasets, pyarrow, Pillow  (see requirements.txt)

Environment variables:
    HF_TOKEN  — optional HuggingFace token for gated datasets
"""

import os
import io
from pathlib import Path

HF_REPO = "adadtur/nvrd"
# Download to data/sample-data/ relative to this file's location
OUTPUT_ROOT = Path(__file__).resolve().parent / "sample-data"


def main():
    import datasets

    print(f"Loading dataset {HF_REPO}...")
    print(f"Output directory: {OUTPUT_ROOT}")
    print()
    print("NOTE: HuggingFace will first download ~971 Parquet shard files before")
    print("      extracting images. This initial download can take 20-30 minutes")
    print("      on a typical connection. The HF progress bar will show shard")
    print("      download progress — this is normal, images come after.")
    print()

    ds = datasets.load_dataset(HF_REPO, split="train", verification_mode="no_checks")
    print(f"Loaded {len(ds)} rows\n")

    output_root = str(OUTPUT_ROOT)
    metadata_cols = ["split", "object", "perturbation_type", "level"]
    metadata = ds.select_columns(metadata_cols)

    to_download = []
    skipped = 0
    ignored = 0

    for i in range(len(metadata)):
        row = metadata[i]
        split = row["split"]
        obj = row["object"]
        ptype = row["perturbation_type"]
        level = row["level"]

        # Skip outdated rows from the old flat perturbations/ structure
        if split.endswith("/perturbations"):
            ignored += 1
            continue

        split_dir = os.path.join(output_root, split)
        if ptype == "original":
            out_fp = os.path.join(split_dir, f"{obj}.png")
        else:
            out_fp = os.path.join(split_dir, "perturbations_comp", ptype, f"{obj}_{level}.png")

        if os.path.exists(out_fp):
            skipped += 1
        else:
            to_download.append((i, out_fp))

    print(
        f"Ignored {ignored} outdated rows, "
        f"skipping {skipped} existing images, "
        f"downloading {len(to_download)} new images...\n"
    )

    if not to_download:
        print("Nothing to download. All images already present.")
        return

    # Access the underlying Arrow table to extract raw image bytes
    # without triggering PIL decoding of original_image.
    arrow_table = ds.data

    from PIL import Image

    saved = 0
    errors = 0
    for idx, out_fp in to_download:
        try:
            img_struct = arrow_table.column("image")[idx].as_py()
            img_bytes = img_struct["bytes"]
            if img_bytes is None:
                raise ValueError("image bytes are None (external path reference)")

            os.makedirs(os.path.dirname(out_fp), exist_ok=True)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(out_fp)
            saved += 1
            if saved % 500 == 0:
                print(f"  Saved {saved}/{len(to_download)} images so far...")
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [ERROR] row {idx} -> {out_fp}: {e}")

    print(f"\nDone! Saved {saved} new images, skipped {skipped} existing, {errors} errors.")
    print(f"Images saved to: {output_root}")


if __name__ == "__main__":
    main()
