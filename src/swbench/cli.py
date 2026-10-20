"""Command-line entry points: train, sweep, cost, report."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml

from swbench.config import apply_overrides, deep_merge, load_config


def _run_name(arm: str, budget, seed: int) -> str:
    return f"{arm}/N{budget if budget else 'all'}/seed{seed}"


def train_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate one benchmark cell.")
    ap.add_argument("config", help="arm config, e.g. configs/arms/prithvi_finetune.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. label_budget=16 seed=1 train.steps=500")
    args = ap.parse_intermixed_args(argv)

    from swbench.train.engine import run

    cfg = apply_overrides(load_config(args.config), args.overrides)
    out = Path(args.out) / _run_name(cfg["name"], cfg.get("label_budget"), cfg.get("seed", 0))
    row = run(cfg, out)
    print(json.dumps({k: row[k] for k in row if k.endswith("iou_water") or k in ("arm", "label_budget", "seed")}))


def sweep_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the arms x label budgets x seeds grid.")
    ap.add_argument("sweep", help="sweep config, e.g. configs/sweep.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--arms", nargs="*", help="subset of arms to run")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th cell starting at i (one shard per GPU)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args(argv)

    sweep_path = Path(args.sweep)
    sweep = yaml.safe_load(sweep_path.read_text())
    arms = args.arms or sweep["arms"]
    cells = list(itertools.product(arms, sweep["label_budgets"], sweep["seeds"]))
    i, n = (int(v) for v in args.shard.split("/"))
    cells = cells[i::n]
    common = sweep.get("common", {})

    from swbench.train.engine import run

    for arm, budget, seed in cells:
        out = Path(args.out) / _run_name(arm, budget, seed)
        if (out / "result.json").exists():
            print(f"skip  {out} (done)")
            continue
        cfg = load_config(sweep_path.parent / "arms" / f"{arm}.yaml")
        cfg = deep_merge(cfg, common)
        cfg = apply_overrides(cfg, [f"label_budget={budget}", f"seed={seed}", *args.overrides])
        print(f"run   {out}", flush=True)
        if not args.dry_run:
            run(cfg, out)


def cost_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Static FLOP/param profile and inference throughput per arm.")
    ap.add_argument("configs", nargs="+", help="arm configs")
    ap.add_argument("--out", default="results/profiles.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-latency", action="store_true", help="only device-independent params/FLOPs")
    args = ap.parse_args(argv)

    import pandas as pd
    import torch

    from swbench.cost.profile import inference_benchmark, static_profile
    from swbench.data.bands import get_band_spec
    from swbench.models.factory import build_model

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for path in args.configs:
        cfg = load_config(path)
        # Weights do not change FLOPs, memory or latency; skip the download.
        mcfg = {**cfg["model"], "pretrained": False}
        model = build_model(mcfg)
        c = len(get_band_spec(mcfg["bands"]))
        prof = {"arm": cfg["name"], **static_profile(model, c, device="cpu")}
        if not args.skip_latency:
            prof.update(inference_benchmark(model, c, device, batch_size=args.batch_size))
            prof["profile_device"] = str(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        print(prof, flush=True)
        rows.append(prof)
        del model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


def report_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate runs into results/RESULTS.md and figures.")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--baseline", default="unet13")
    ap.add_argument("--profiles", default="results/profiles.csv")
    ap.add_argument("--min-gain", type=float, default=0.02)
    args = ap.parse_args(argv)

    from swbench.analysis.report import write_report

    print(write_report(args.runs, args.out, args.baseline, args.profiles, args.min_gain))
