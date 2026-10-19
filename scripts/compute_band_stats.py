#!/usr/bin/env python
"""Per-band mean/std of the Sen1Floods11 training split (labelled pixels only).

Prints a ``SEN1FLOODS11_STATS`` literal to paste into swbench/data/bands.py.
Only the training split is used so no statistic leaks from val/test/Bolivia.
"""

from __future__ import annotations

import argparse

import numpy as np

from swbench.data.bands import S2_BANDS
from swbench.data.sen1floods11 import IGNORE_INDEX, load_chip, read_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sen1floods11")
    args = ap.parse_args()

    s = np.zeros(len(S2_BANDS))
    ss = np.zeros(len(S2_BANDS))
    n = 0
    for chip in read_split(args.root, "train"):
        image, label = load_chip(args.root, chip)
        valid = (label != IGNORE_INDEX) & np.isfinite(image).all(0)
        px = image[:, valid].astype(np.float64)
        s += px.sum(1)
        ss += (px**2).sum(1)
        n += px.shape[1]
    mean = s / n
    std = np.sqrt(ss / n - mean**2)
    print(f"# {n} valid pixels from the train split")
    print("SEN1FLOODS11_STATS: dict[str, tuple[float, float]] = {")
    for b, m, sd in zip(S2_BANDS, mean, std, strict=True):
        print(f'    "{b}": ({m:.1f}, {sd:.1f}),')
    print("}")


if __name__ == "__main__":
    main()
