"""Binary water-segmentation metrics.

Scores are computed from confusion counts aggregated over every valid pixel
of a split (the convention of the Sen1Floods11 paper), not averaged per chip,
so chips with little valid data do not dominate. Per-chip counts are kept so
that uncertainty can be estimated by a chip-level bootstrap: pixels within a
chip are strongly correlated, so resampling pixels would give confidence
intervals that are far too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from swbench.data.sen1floods11 import IGNORE_INDEX


def confusion_counts(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Return ``[tp, fp, fn, tn]`` for water (class 1), skipping ignored pixels."""
    valid = target != IGNORE_INDEX
    p = pred[valid].bool()
    t = target[valid].bool()
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    tn = (~p & ~t).sum().item()
    return np.array([tp, fp, fn, tn], dtype=np.int64)


def scores_from_counts(c: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = (float(v) for v in c)
    nan = float("nan")
    iou_w = tp / (tp + fp + fn) if tp + fp + fn else nan
    iou_d = tn / (tn + fp + fn) if tn + fp + fn else nan
    prec = tp / (tp + fp) if tp + fp else nan
    rec = tp / (tp + fn) if tp + fn else nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan
    total = tp + fp + fn + tn
    return {
        "iou_water": iou_w,
        "iou_dry": iou_d,
        "miou": np.nanmean([iou_w, iou_d]) if not (np.isnan(iou_w) and np.isnan(iou_d)) else nan,
        "f1_water": f1,
        "precision_water": prec,
        "recall_water": rec,
        "accuracy": (tp + tn) / total if total else nan,
    }


@dataclass
class SegmentationMeter:
    """Accumulates per-chip confusion counts over an evaluation split."""

    chips: list[str] = field(default_factory=list)
    counts: list[np.ndarray] = field(default_factory=list)

    def update(self, chip: str, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.chips.append(chip)
        self.counts.append(confusion_counts(pred, target))

    @property
    def total(self) -> np.ndarray:
        return np.sum(self.counts, axis=0) if self.counts else np.zeros(4, dtype=np.int64)

    def compute(self) -> dict[str, float]:
        return scores_from_counts(self.total)

    def bootstrap(self, metric: str = "iou_water", n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
        """Percentile CI of ``metric`` from resampling chips with replacement."""
        if not self.counts:
            return float("nan"), float("nan")
        counts = np.stack(self.counts)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(counts), size=(n, len(counts)))
        samples = [scores_from_counts(counts[i].sum(0))[metric] for i in idx]
        lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    def per_chip(self, metric: str = "iou_water") -> dict[str, float]:
        return {c: scores_from_counts(k)[metric] for c, k in zip(self.chips, self.counts, strict=True)}
