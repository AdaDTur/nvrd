"""
Batch version of find_shape16.py: scans all 971 parquet shards ONCE and locates rows for several
objects at once, instead of rescanning the whole dataset once per object (wasteful for a one-off
bulk fetch of many novel objects at the same time). Writes the same per-object <name>_matches.json
format find_shape16.py does, so fetch_image.py works unmodified against the output.

Usage:
    python find_novel_batch.py shape_4 shape_5 shape_6 ...
"""
import sys
import os
import pyarrow.parquet as pq
import fsspec
import concurrent.futures as cf
import time
import json

TARGETS = set(sys.argv[1:])
if not TARGETS:
    raise SystemExit("usage: python find_novel_batch.py <object_name> [object_name ...]")

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
        matches = df[df["object"].isin(TARGETS)]
        return i, matches.to_dict("records")
    except Exception as e:
        return i, [{"object": "__ERROR__", "error": str(e)}]


if __name__ == "__main__":
    t0 = time.time()
    all_matches = {name: [] for name in TARGETS}
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(scan_shard, i): i for i in range(N_SHARDS)}
        done_count = 0
        for fut in cf.as_completed(futs):
            i, matches = fut.result()
            for m in matches:
                obj = m.get("object")
                if obj in all_matches:
                    all_matches[obj].append((i, m))
            done_count += 1
            if done_count % 200 == 0:
                print(f"scanned {done_count}/{N_SHARDS} shards, elapsed {time.time()-t0:.1f}s")

    print(f"\nTotal scan time: {time.time()-t0:.1f}s")
    for name, matches in all_matches.items():
        out_path = os.path.join(HERE, f"{name}_matches.json")
        with open(out_path, "w") as f:
            json.dump(matches, f, indent=2, default=str)
        originals = [m for _, m in matches if m.get("perturbation_type") == "original"]
        print(f"{name}: {len(matches)} total rows, {len(originals)} original -> {out_path}")
