import numpy as np
import pytest

from swbench.data.bands import CLAY_10, PRITHVI_6, S1F11_13, S2_BANDS, BandSpec, get_band_spec
from swbench.data.sen1floods11 import Sen1Floods11, label_budget_subset, load_chip, normalize, read_split


def test_band_specs_select_expected_bands():
    assert PRITHVI_6.indices == [1, 2, 3, 8, 11, 12]
    assert len(CLAY_10) == 10 and "B8A" in CLAY_10.bands and "B10" not in CLAY_10.bands
    assert S1F11_13.bands == S2_BANDS
    assert CLAY_10.wavelengths[0] == pytest.approx(0.493)
    with pytest.raises(KeyError):
        get_band_spec("nope")
    with pytest.raises(ValueError):
        BandSpec("bad", ("B02",), (0.0, 1.0), (1.0,))


def test_budget_subsets_are_nested_and_seeded():
    chips = [f"c{i}" for i in range(100)]
    s8, s16 = label_budget_subset(chips, 8, 0), label_budget_subset(chips, 16, 0)
    assert s16[:8] == s8
    assert label_budget_subset(chips, 8, 0) == s8
    assert label_budget_subset(chips, 8, 1) != s8
    assert label_budget_subset(chips, None, 0) == sorted(chips)
    assert label_budget_subset(chips, 500, 0) == sorted(chips)


def test_normalize_handles_nan():
    img = np.full((13, 4, 4), 1000.0, np.float32)
    img[1, 0, 0] = np.nan
    x = normalize(img, PRITHVI_6)
    assert x.shape == (6, 4, 4) and np.isfinite(x).all()


def test_dataset_crops_and_full_chips(fake_sen1floods11):
    root = fake_sen1floods11
    train = read_split(root, "train")
    img, lab = load_chip(root, train[0])
    assert img.shape == (13, 64, 64) and set(np.unique(lab)) <= {-1, 0, 1}

    ds = Sen1Floods11(root, train, PRITHVI_6, crop_size=32, augment=True, samples_per_epoch=50)
    assert len(ds) == 50
    item = ds[7]
    assert item["image"].shape == (6, 32, 32) and item["mask"].shape == (32, 32)

    full = Sen1Floods11(root, read_split(root, "test"), CLAY_10)
    assert full[0]["image"].shape == (10, 64, 64)
