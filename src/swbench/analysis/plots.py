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


def label_efficiency_plot(df: pd.DataFrame, out: str | Path, splits=("test", "bolivia"), metric: str = "iou_water"):
    out = Path(out)
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 4.4), sharey=True)
    axes = [axes] if len(splits) == 1 else list(axes)
    arms = sorted(df.arm.unique())
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
            last = a.iloc[-1]
            ax.annotate(arm, (last.label_budget, last["mean"]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_xscale("log", base=2)
        budgets = sorted(df.label_budget.unique())
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("Labelled training chips (log scale)", color=INK, fontsize=10)
        ax.set_title(SPLIT_TITLE.get(split, split), color=INK, fontsize=11, loc="left")
        _axes_style(ax)
        ax.set_xlim(right=max(budgets) * 2.2)
    axes[0].set_ylabel("Water IoU (mean over seeds; band = min-max)", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def cost_accuracy_plot(df: pd.DataFrame, out: str | Path, split: str = "test", budget: int | None = None):
    """Training seconds (log) against water IoU at one label budget."""
    out = Path(out)
    budget = budget or int(df.label_budget.min())
    sub = df[df.label_budget == budget]
    g = sub.groupby("arm").agg(secs=("train_seconds", "mean"), iou=(f"{split}_iou_water", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for _, r in g.iterrows():
        color, _, marker = _style(r.arm, df)
        ax.scatter(r.secs, r.iou, s=80, color=color, marker=marker, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(r.arm, (r.secs, r.iou), xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Training seconds per run (log scale)", color=INK, fontsize=10)
    ax.set_ylabel(f"Water IoU, {SPLIT_TITLE.get(split, split)}", color=INK, fontsize=10)
    ax.set_title(f"Cost vs. accuracy at N = {budget} labelled chips", color=INK, fontsize=11, loc="left")
    _axes_style(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
