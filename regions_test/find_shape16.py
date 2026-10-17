"""
Locate a specific NVRD object's rows in the adadtur/nvrd HuggingFace dataset without
downloading the full ~19K-image dataset (971 parquet shards).

Scans every shard's metadata columns only (split/object/perturbation_type/level -- not the
image bytes) in parallel, which is fast because parquet lets you read a subset of columns.

Usage:
    python find_shape16.py [object_name]   # default: shape_16

Writes <object_name>_matches.json next to this script with every matching row's metadata.
"""
import sys
import os
import pyarrow.parquet as pq
import fsspec
import concurrent.futures as cf
import time
import json

OBJECT_NAME = sys.argv[1] if len(sys.argv) > 1 else "shape_16"
HERE = os.path.dirname(os.path.realpath(__file__))

N_SHARDS = 971
BASE = "https://huggingface.co/datasets/adadtur/nvrd/resolve/main/data/train-{:05d}-of-00971.parquet"

fs = fsspec.filesystem("https")

def scan_shard(i):
    url = BASE.format(i)
    try:
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            tbl = pf.read(columns=["split", "object", "perturbation_type", "level"])
        df = tbl.to_pandas()
        matches = df[df["object"] == OBJECT_NAME]
        return i, len(df), matches.to_dict("records")
    except Exception as e:
        return i, -1, [("ERROR", str(e))]

if __name__ == "__main__":
    t0 = time.time()
    all_matches = []
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(scan_shard, i): i for i in range(N_SHARDS)}
        done_count = 0
        for fut in cf.as_completed(futs):
            i, n, matches = fut.result()
            if matches:
                all_matches.extend([(i, m) for m in matches])
                print(f"shard {i}: {len(matches)} matches")
            done_count += 1
            if done_count % 100 == 0:
                print(f"scanned {done_count}/{N_SHARDS} shards, elapsed {time.time()-t0:.1f}s, matches so far {len(all_matches)}")

    print(f"\nTotal scan time: {time.time()-t0:.1f}s")
    print(f"Total matches for object=='{OBJECT_NAME}': {len(all_matches)}")

    out_path = os.path.join(HERE, f"{OBJECT_NAME}_matches.json")
    with open(out_path, "w") as f:
        json.dump(all_matches, f, indent=2, default=str)
    print(f"wrote {out_path}")

    # Print originals specifically
    originals = [(i, m) for i, m in all_matches if m.get("perturbation_type") == "original"]
    print(f"\nOriginal rows for {OBJECT_NAME}: {len(originals)}")
    for i, m in originals:
        print(i, m)
