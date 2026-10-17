"""
One-off: scans all 971 parquet shards for the full, authoritative list of distinct object names
actually present under split=='novel' with perturbation_type=='original' -- ground truth from the
dataset itself, since the nonce_word_mapping.json name list turned out to include names (shape_4-9,
shape_31-40 confirmed) that aren't actually present as rows in the dataset.
"""
import pyarrow.parquet as pq
import fsspec
import concurrent.futures as cf
import time

N_SHARDS = 971
BASE = "https://huggingface.co/datasets/adadtur/nvrd/resolve/main/data/train-{:05d}-of-00971.parquet"
fs = fsspec.filesystem("https")


def scan_shard(i):
    url = BASE.format(i)
    try:
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            tbl = pf.read(columns=["split", "object", "perturbation_type"])
        df = tbl.to_pandas()
        novel_originals = df[(df["split"] == "novel") & (df["perturbation_type"] == "original")]
        return set(novel_originals["object"].unique())
    except Exception as e:
        print(f"shard {i} error: {e}")
        return set()


if __name__ == "__main__":
    t0 = time.time()
    all_names = set()
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(scan_shard, i) for i in range(N_SHARDS)]
        done = 0
        for fut in cf.as_completed(futs):
            all_names |= fut.result()
            done += 1
            if done % 200 == 0:
                print(f"scanned {done}/{N_SHARDS}, elapsed {time.time()-t0:.1f}s, found {len(all_names)} so far")

    print(f"\nTotal scan time: {time.time()-t0:.1f}s")
    print(f"Total distinct novel objects: {len(all_names)}")

    def sort_key(name):
        try:
            return (0, int(name.split("_")[1]))
        except (IndexError, ValueError):
            return (1, name)

    for name in sorted(all_names, key=sort_key):
        print(name)
