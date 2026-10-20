#!/usr/bin/env python
"""Validation water IoU of every learning-rate tuning run, best first per arm.

    python scripts/summarize_lr.py /scratch/$USER/swbench/runs_lr

Only the valid split is read: test and Bolivia must not influence the choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    args = ap.parse_args()

    rows = []
    for result in sorted(args.runs.glob("*/*/*/*/result.json")):
        r = json.loads(result.read_text())
        train = yaml.safe_load((result.parent / "config.yaml").read_text())["train"]
        rows.append((r["arm"], train["lr"], train.get("encoder_lr"), r["valid_iou_water"], r["best_step"]))
    if not rows:
        raise SystemExit(f"no result.json under {args.runs}")

    rows.sort(key=lambda t: (t[0], -t[3]))
    print(f"{'arm':18s} {'lr':>8s} {'encoder_lr':>10s} {'val IoU':>8s} {'best step':>9s}")
    for arm, lr, elr, iou, step in rows:
        elr_s = "-" if elr is None else f"{elr:.0e}"
        print(f"{arm:18s} {lr:8.0e} {elr_s:>10s} {iou:8.4f} {step:9d}")


if __name__ == "__main__":
    main()
