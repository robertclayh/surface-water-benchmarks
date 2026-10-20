"""Static figures for the report (matplotlib, PNG + SVG).

Encoding: hue = model family (U-Net / Prithvi / Clay, three slots of a
CVD-validated categorical palette), line style + marker = protocol. Every
line is also direct-labelled at its right end, so identity never rests on
colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from swbench.analysis.aggregate import summarize  # noqa: E402

FAMILY_COLOR = {"unet": "#2a78d6", "prithvi": "#eb6834", "clay": "#1baf7a", "simple_vit": "#6250d6"}
PROTOCOL_STYLE = {
    "scratch": ("-", "o"),
    "finetune": ("-", "o"),
    "frozen": ("--", "s"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLIT_TITLE = {
    "test": "Sen1Floods11 test (in-distribution)",
    "bolivia": "Bolivia (held-out region)",
    "valid": "Validation",
}


def _style(arm: str, df: pd.DataFrame) -> tuple[str, str, str]:
    row = df[df.arm == arm].iloc[0]
    ls, marker = PROTOCOL_STYLE.get(row.protocol, ("-", "o"))
    if row.arch == "unet" and "6" in arm:
        ls, marker = ":", "^"  # band-matched U-Net
    return FAMILY_COLOR.get(row.arch, "#52514e"), ls, marker


def _axes_style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _spread(y, min_gap: float):
    """Label positions near ``y`` that are at least ``min_gap`` apart (order kept)."""
    order = sorted(range(len(y)), key=lambda i: y[i])
    pos = [float(y[i]) for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    shift = (sum(pos) - sum(float(y[i]) for i in order)) / max(len(pos), 1)
    out = [0.0] * len(y)
    for k, i in enumerate(order):
        out[i] = pos[k] - shift  # re-centre so labels stay next to their lines
    return out


def label_efficiency_plot(df: pd.DataFrame, out: str | Path, splits=("test", "bolivia"), metric: str = "iou_water"):
    out = Path(out)
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 4.4), sharey=True)
    axes = [axes] if len(splits) == 1 else list(axes)
    arms = sorted(df.arm.unique())
    line_ends = []
    for ax, split in zip(axes, splits, strict=True):
        s = summarize(df, split, metric)
        for arm in arms:
            a = s[s.arm == arm].sort_values("label_budget")
            if a.empty:
                continue
            color, ls, marker = _style(arm, df)
            ax.plot(a.label_budget, a["mean"], ls, color=color, lw=2, marker=marker, ms=8,
                    mec="white", mew=1.5, label=arm)
            if (a.n_seeds > 1).any():
                ax.fill_between(a.label_budget, a["min"], a["max"], color=color, alpha=0.12, lw=0)
        line_ends.append((ax, s.sort_values("label_budget").groupby("arm").last()))
        ax.set_xscale("log", base=2)
        budgets = sorted(df.label_budget.unique())
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("Labelled training chips (log scale)", color=INK, fontsize=10)
        ax.set_title(SPLIT_TITLE.get(split, split), color=INK, fontsize=11, loc="left")
        _axes_style(ax)
        ax.set_xlim(right=max(budgets) * 2.2)
    # Direct labels last: the shared y-range is only final once every panel is drawn.
    lo, hi = axes[0].get_ylim()
    for ax, ends in line_ends:
        for arm, y in zip(ends.index, _spread(ends["mean"].to_numpy(), 0.045 * (hi - lo)), strict=True):
            x = ends.loc[arm, "label_budget"]
            ax.annotate(arm, (x, ends.loc[arm, "mean"]), xytext=(x * 1.15, y), textcoords="data",
                        va="center", fontsize=8, color=INK)
    axes[0].set_ylabel("Water IoU (mean over seeds; band = min-max)", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def cost_accuracy_plot(
    df: pd.DataFrame,
    out: str | Path,
    split: str = "test",
    budget: int | None = None,
    train_pflops: dict[str, float] | None = None,
):
    """Training cost (log) against water IoU at one label budget.

    Cost is training PFLOPs per run when ``train_pflops`` is given (measured
    seconds can be set by data loading rather than the model; see
    docs/COST_ACCOUNTING.md), otherwise measured training seconds.
    """
    out = Path(out)
    budget = budget or int(df.label_budget.min())
    sub = df[df.label_budget == budget]
    g = sub.groupby("arm").agg(secs=("train_seconds", "mean"), iou=(f"{split}_iou_water", "mean")).reset_index()
    if train_pflops is not None:
        g["cost"] = g.arm.map(train_pflops)
        xlabel = "Training compute per run (PFLOPs, log scale)"
    else:
        g["cost"] = g.secs
        xlabel = "Training seconds per run (log scale)"
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for _, r in g.iterrows():
        color, _, marker = _style(r.arm, df)
        ax.scatter(r.cost, r.iou, s=80, color=color, marker=marker, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(r.arm, (r.cost, r.iou), xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(f"Water IoU, {SPLIT_TITLE.get(split, split)}", color=INK, fontsize=10)
    ax.set_title(f"Cost vs. accuracy at N = {budget} labelled chips", color=INK, fontsize=11, loc="left")
    _axes_style(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
