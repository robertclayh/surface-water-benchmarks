import numpy as np
import torch

from swbench.eval.metrics import SegmentationMeter, confusion_counts, scores_from_counts


def test_confusion_ignores_nodata():
    pred = torch.tensor([1, 1, 0, 0, 1])
    target = torch.tensor([1, 0, 1, 0, -1])
    assert confusion_counts(pred, target).tolist() == [1, 1, 1, 1]


def test_scores():
    s = scores_from_counts(np.array([8, 2, 2, 88]))
    assert np.isclose(s["iou_water"], 8 / 12)
    assert np.isclose(s["f1_water"], 16 / 20)
    assert np.isclose(s["accuracy"], 0.96)
    assert np.isclose(s["iou_dry"], 88 / 92)


def test_scores_empty_is_nan():
    assert np.isnan(scores_from_counts(np.zeros(4))["iou_water"])


def test_aggregate_is_pixel_pooled_not_chip_mean():
    m = SegmentationMeter()
    m.update("a", torch.ones(100, dtype=torch.long), torch.ones(100, dtype=torch.long))  # IoU 1 on 100 px
    m.update("b", torch.tensor([1, 0]), torch.tensor([0, 1]))  # IoU 0 on 2 px
    assert np.isclose(m.compute()["iou_water"], 100 / 102)


def test_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(0)
    m = SegmentationMeter()
    for i in range(30):
        t = torch.from_numpy(rng.integers(0, 2, 200))
        p = torch.where(torch.from_numpy(rng.random(200) < 0.8), t, 1 - t)
        m.update(str(i), p, t)
    lo, hi = m.bootstrap(n=500)
    assert lo <= m.compute()["iou_water"] <= hi
    assert hi - lo < 0.2
