"""
Download one object's original (unperturbed) image from the adadtur/nvrd HuggingFace
dataset, using the shard/split located by find_shape16.py.

Usage:
    python find_shape16.py shape_16      # first, to locate the shard + split
    python fetch_image.py shape_16       # then, to download data/<split>/<object>.png
"""
import sys
import os
import io
import json
import pyarrow.parquet as pq
import fsspec
from PIL import Image

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
HERE = os.path.dirname(os.path.realpath(__file__))
BASE = "https://huggingface.co/datasets/adadtur/nvrd/resolve/main/data/train-{:05d}-of-00971.parquet"

matches_path = os.path.join(HERE, f"{OBJECT_NAME}_matches.json")
with open(matches_path) as f:
    matches = json.load(f)

originals = [(shard, row) for shard, row in matches if row.get("perturbation_type") == "original"]
if not originals:
    raise SystemExit(f"No 'original' row found for {OBJECT_NAME} in {matches_path}; run find_shape16.py first")

shard, row = originals[0]
split = row["split"]
print(f"{OBJECT_NAME}: shard={shard} split={split}")

fs = fsspec.filesystem("https")
url = BASE.format(shard)
with fs.open(url, "rb") as f:
    pf = pq.ParquetFile(f)
    tbl = pf.read()
df = tbl.to_pandas()
match = df[(df["object"] == OBJECT_NAME) & (df["perturbation_type"] == "original")]
img_bytes = match.iloc[0]["image"]["bytes"]
img = Image.open(io.BytesIO(img_bytes))

out_dir = os.path.join(HERE, "data", split)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{OBJECT_NAME}.png")
img.save(out_path)
print(f"saved {out_path} ({img.size[0]}x{img.size[1]} {img.mode})")
