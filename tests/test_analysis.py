import json
import math

import numpy as np
import pandas as pd

from swbench.analysis.aggregate import equivalent_budget, label_efficiency, paired_delta, verdicts
from swbench.analysis.report import write_report


def _synthetic(tmp_path=None):
    """FM curve is the baseline curve shifted by 2x labels (+ seed noise)."""
    rows = []
    rng = np.random.default_rng(0)
    for arm, arch, proto, shift, secs in [("unet13", "unet", "scratch", 1, 100), ("fm", "prithvi", "finetune", 4, 900)]:
        for n in [8, 16, 32, 64, 128]:
            for seed in range(5):
                iou = 0.4 + 0.08 * math.log2(n * shift) + rng.normal(0, 0.005)
                rows.append({
                    "arm": arm, "arch": arch, "protocol": proto, "label_budget": n, "seed": seed,
                    "test_iou_water": iou, "bolivia_iou_water": iou - 0.05, "train_seconds": secs,
                    "run_wall_seconds": secs, "train_peak_mem_gb": 1.0, "params_total": 1, "params_trainable": 1,
                    "device": "test",
                })
    return pd.DataFrame(rows)


def test_equivalent_budget_interpolates_in_log2():
    n, y = np.array([8, 16, 32]), np.array([0.5, 0.6, 0.7])
    assert equivalent_budget(n, y, 0.6) == 16
    assert np.isclose(equivalent_budget(n, y, 0.55), 8 * 2**0.5)
    assert equivalent_budget(n, y, 0.4) == 8
    assert equivalent_budget(n, y, 0.9) == float("inf")


def test_label_multiplier_recovers_shift():
    le = label_efficiency(_synthetic(), "fm", "unet13", "test")
    inner = le[le.label_budget <= 16]  # N=32 x4 = 128 sits at the edge of the range
    assert np.allclose(inner.label_multiplier, 4, rtol=0.1)


def test_paired_delta_and_verdict():
    df = _synthetic()
    d = paired_delta(df, "fm", "unet13", "test")
    assert np.allclose(d.delta_mean, 0.16, atol=0.01)
    v = verdicts(df, "fm", "unet13", "test", min_gain=0.02)
    assert (v.verdict == "justified").all()
    v = verdicts(df, "fm", "unet13", "test", min_gain=0.5)
    assert (v.verdict == "not justified").all()


def test_report_writes_markdown_and_figures(tmp_path):
    df = _synthetic()
    for i, r in df.iterrows():
        d = tmp_path / "runs" / str(i)
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(r.to_dict(), default=float))
    path = write_report(tmp_path / "runs", tmp_path / "out", baseline="unet13")
    text = path.read_text()
    assert "label_multiplier" in text and "justified" in text
    assert (tmp_path / "out" / "figures" / "label_efficiency.png").exists()
