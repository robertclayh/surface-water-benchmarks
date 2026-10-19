#!/usr/bin/env python
"""Download the hand-labelled Sen1Floods11 Sentinel-2 chips and split files.

Only the subset this benchmark uses is fetched (~450 chips of S2Hand +
LabelHand, roughly 0.7 GB) from the public ``gs://sen1floods11`` bucket over
HTTPS, so neither gsutil nor credentials are required.

    python scripts/download_sen1floods11.py --out data/sen1floods11
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS = {
    "train": "splits/flood_handlabeled/flood_train_data.csv",
    "valid": "splits/flood_handlabeled/flood_valid_data.csv",
    "test": "splits/flood_handlabeled/flood_test_data.csv",
    "bolivia": "splits/flood_handlabeled/flood_bolivia_data.csv",
}
LAYERS = ("S2Hand", "LabelHand")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def chip_ids_from_csv(text: str) -> list[str]:
    """Split CSVs list ``<Country>_<id>_S1Hand.tif,<Country>_<id>_LabelHand.tif``."""
    ids = []
    for row in csv.reader(io.StringIO(text)):
        if row:
            ids.append(row[0].strip().rsplit("_", 1)[0])
    return ids


def download_one(chip: str, layer: str, out: Path, retries: int = 3) -> Path:
    dest = out / layer / f"{chip}_{layer}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{BASE}/data/flood_events/HandLabeled/{layer}/{chip}_{layer}.tif"
    for attempt in range(retries):
        try:
            data = _fetch(url)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/sen1floods11"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    (args.out / "splits").mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        (args.out / layer).mkdir(parents=True, exist_ok=True)

    chips: set[str] = set()
    for split, rel in SPLITS.items():
        text = _fetch(f"{BASE}/{rel}").decode()
        ids = chip_ids_from_csv(text)
        (args.out / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
        print(f"{split:8s} {len(ids):4d} chips")
        chips.update(ids)

    jobs = [(c, layer) for c in sorted(chips) for layer in LAYERS]
    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(download_one, c, layer, args.out): (c, layer) for c, layer in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                failed.append((*futs[fut], str(e)))
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files", flush=True)

    if failed:
        for c, layer, err in failed:
            print(f"FAILED {c} {layer}: {err}", file=sys.stderr)
        return 1
    print(f"done: {len(chips)} chips in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
