"""Turn per-run ``result.json`` files into label-efficiency and cost tables.

Nothing in this module states a conclusion in advance: every verdict in the
report is computed from the rows it is given, with thresholds that are
arguments (and documented in docs/COST_ACCOUNTING.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(runs_dir: str | Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).rglob("result.json"))]
    if not rows:
        raise FileNotFoundError(f"no result.json under {runs_dir}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, split: str, metric: str = "iou_water") -> pd.DataFrame:
    """Mean / std / min / max of ``split_metric`` across seeds per arm and budget."""
    col = f"{split}_{metric}"
    g = df.groupby(["arm", "label_budget"])[col]
    out = g.agg(["mean", "std", "min", "max", "count"]).reset_index()
    return out.rename(columns={"count": "n_seeds"})


def paired_delta(df: pd.DataFrame, arm: str, baseline: str, split: str, metric: str = "iou_water") -> pd.DataFrame:
    """Per-budget difference ``arm - baseline`` on matched seeds.

    Seeds fix the labelled chips, so a matched-seed difference removes the
    (large) variance due to *which* chips were labelled.
    """
    col = f"{split}_{metric}"
    a = df[df.arm == arm].set_index(["label_budget", "seed"])[col]
    b = df[df.arm == baseline].set_index(["label_budget", "seed"])[col]
    d = (a - b).dropna().rename("delta").reset_index()
    if d.empty:
        return pd.DataFrame(columns=["label_budget", "delta_mean", "delta_std", "n_pairs", "wins", "delta_lo", "delta_hi"])
    rows = []
    for n, grp in d.groupby("label_budget"):
        vals = grp["delta"].to_numpy()
        lo, hi = _bootstrap_mean_ci(vals)
        rows.append({
            "label_budget": n,
            "delta_mean": vals.mean(),
            "delta_std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "n_pairs": len(vals),
            "wins": int((vals > 0).sum()),
            "delta_lo": lo,
            "delta_hi": hi,
        })
    return pd.DataFrame(rows)


MIN_SEEDS_FOR_CI = 3


def _bootstrap_mean_ci(vals: np.ndarray, n: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    # With fewer than three seeds a bootstrap "interval" is just the range of
    # one or two numbers; report no interval rather than a misleading one.
    if len(vals) < MIN_SEEDS_FOR_CI:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def equivalent_budget(curve_n: np.ndarray, curve_y: np.ndarray, target: float) -> float:
    """Labels the baseline would need to reach ``target`` (log2-linear interpolation).

    Returns ``inf`` if the baseline never reaches it within the tested range
    and ``-inf`` if the baseline already exceeds it at the smallest budget
    (the equivalent budget is somewhere below the tested range, not equal to
    its bottom).
    """
    order = np.argsort(curve_n)
    n, y = np.asarray(curve_n, float)[order], np.asarray(curve_y, float)[order]
    # Make the curve monotone (running max) so interpolation is well-defined;
    # a non-monotone mean curve is seed noise, not a real loss from more labels.
    y = np.maximum.accumulate(y)
    if target < y[0]:
        return float("-inf")
    if target == y[0]:
        return float(n[0])
    if target > y[-1]:
        return float("inf")
    i = int(np.searchsorted(y, target))
    x0, x1 = np.log2(n[i - 1]), np.log2(n[i])
    y0, y1 = y[i - 1], y[i]
    frac = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
    return float(2 ** (x0 + frac * (x1 - x0)))


def label_efficiency(df: pd.DataFrame, arm: str, baseline: str, split: str, metric: str = "iou_water") -> pd.DataFrame:
    """For each budget N of ``arm``: the baseline budget with equal accuracy, and the ratio.

    A multiplier of 4 at N=16 means the baseline needs ~64 labelled chips to
    match what ``arm`` achieves with 16.
    """
    s = summarize(df, split, metric)
    base = s[s.arm == baseline]
    rows = []
    for _, r in s[s.arm == arm].iterrows():
        eq = equivalent_budget(base.label_budget.to_numpy(), base["mean"].to_numpy(), r["mean"])
        rows.append({
            "label_budget": r.label_budget,
            "score": r["mean"],
            "baseline_equivalent_budget": eq,
            "label_multiplier": eq / r.label_budget,
        })
    return pd.DataFrame(rows)


def cost_table(df: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-arm training cost (mean over runs) joined with static profiles."""
    cols = {
        "train_seconds": "mean",
        "run_wall_seconds": "mean",
        "train_peak_mem_gb": "mean",
        "params_total": "first",
        "params_trainable": "first",
        "device": "first",
    }
    if "run_energy_kwh" in df:
        cols["run_energy_kwh"] = "mean"
    if "test_infer_seconds" in df:
        cols["test_infer_seconds"] = "mean"
    out = df.groupby("arm").agg({k: v for k, v in cols.items() if k in df}).reset_index()
    if profiles is not None and not profiles.empty:
        profiles = profiles.drop(columns=[c for c in profiles.columns if c != "arm" and c in out.columns])
        out = out.merge(profiles, on="arm", how="left")
        if "gflops_train_per_tile" in out and {"steps", "batch_size"} <= set(df):
            samples = df.groupby("arm").apply(lambda g: (g.steps * g.batch_size).mean(), include_groups=False)
            out["train_pflops_per_run"] = out.arm.map(samples) * out.gflops_train_per_tile / 1e6
    return out


def compute_bound_warnings(cost: pd.DataFrame, min_flop_ratio: float = 1.5) -> list[str]:
    """Pairs of arms whose measured training time contradicts their FLOPs.

    If an arm needs at least ``min_flop_ratio`` x the training FLOPs of
    another yet trains no slower, wall-clock time is set by something other
    than model compute (data loading, shared-node contention), and seconds
    are not a fair cost comparison between those arms.
    """
    if not {"train_pflops_per_run", "train_seconds"} <= set(cost):
        return []
    c = cost.dropna(subset=["train_pflops_per_run", "train_seconds"]).set_index("arm")
    out = []
    for a, ra in c.iterrows():
        for b, rb in c.iterrows():
            if a != b and ra.train_pflops_per_run >= min_flop_ratio * rb.train_pflops_per_run \
                    and ra.train_seconds <= rb.train_seconds:
                out.append(
                    f"`{a}` needs {ra.train_pflops_per_run / rb.train_pflops_per_run:.1f}x the training FLOPs "
                    f"of `{b}` but trained in {ra.train_seconds:.0f} s vs {rb.train_seconds:.0f} s"
                )
    return out


def verdicts(
    df: pd.DataFrame,
    arm: str,
    baseline: str,
    split: str,
    min_gain: float = 0.02,
    metric: str = "iou_water",
    train_pflops: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Is ``arm`` worth its overhead over ``baseline`` at each budget?

    * ``justified``: the bootstrap CI of the matched-seed gain lies entirely
      above ``min_gain`` (default 2 IoU points).
    * ``not justified``: the CI lies entirely below ``min_gain`` (the gain is
      too small to matter, or negative).
    * ``inconclusive``: the CI straddles ``min_gain``, or fewer than three
      matched seeds exist; more seeds are needed.

    The extra training compute per IoU point is reported alongside, so a
    reader with a different exchange rate between accuracy and compute can
    draw their own line. With ``train_pflops`` (arm -> PFLOPs per run) that
    rate is in FLOPs, which do not depend on data loading or node load;
    otherwise in measured seconds.
    """
    d = paired_delta(df, arm, baseline, split, metric)
    secs = df.groupby(["arm", "label_budget"])["train_seconds"].mean()
    rows = []
    for _, r in d.iterrows():
        n = r.label_budget
        if np.isnan(r.delta_lo):
            verdict = "inconclusive"
        elif r.delta_lo > min_gain:
            verdict = "justified"
        elif r.delta_hi < min_gain:
            verdict = "not justified"
        else:
            verdict = "inconclusive"
        extra = secs.get((arm, n), np.nan) - secs.get((baseline, n), np.nan)
        pts = 100 * r.delta_mean
        row = {
            "label_budget": n,
            "delta_iou": r.delta_mean,
            "delta_lo": r.delta_lo,
            "delta_hi": r.delta_hi,
            "wins": f"{int(r.wins)}/{int(r.n_pairs)}",
            "extra_train_seconds": extra,
        }
        if train_pflops is not None:
            extra_pf = train_pflops.get(arm, np.nan) - train_pflops.get(baseline, np.nan)
            row["extra_train_pflops"] = extra_pf
            row["extra_train_pflops_per_iou_point"] = _per_point(extra_pf, pts)
        else:
            row["extra_train_seconds_per_iou_point"] = _per_point(extra, pts)
        row["verdict"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)


def _per_point(extra: float, pts: float) -> float:
    # Undefined (shown as "–") without a gain to pay for, or without an extra
    # cost: a ratio would be negative or infinite and mean nothing.
    return extra / pts if pts > 0 and extra > 0 else np.nan
