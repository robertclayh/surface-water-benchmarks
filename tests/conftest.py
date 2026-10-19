import numpy as np
import pytest


def _write_tif(path, array):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2], count=array.shape[0],
        dtype=array.dtype, crs="EPSG:4326", transform=from_origin(0, 0, 1e-4, 1e-4),
    ) as dst:
        dst.write(array)


@pytest.fixture
def fake_sen1floods11(tmp_path):
    """A miniature Sen1Floods11 tree with 64 x 64 chips.

    Water is where B8A (NIR) is low, so a model can actually learn it.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "s1f11"
    (root / "S2Hand").mkdir(parents=True)
    (root / "LabelHand").mkdir()
    (root / "splits").mkdir()
    splits = {"train": 12, "valid": 3, "test": 3, "bolivia": 2}
    k = 0
    for split, n in splits.items():
        ids = []
        for _ in range(n):
            chip = f"{'Bolivia' if split == 'bolivia' else 'Ghana'}_{k}"
            k += 1
            water = np.zeros((64, 64), bool)
            r, c = rng.integers(8, 40, size=2)
            water[r : r + 20, c : c + 20] = True
            img = rng.normal(1500, 300, size=(13, 64, 64))
            img[8][water] = 300  # B8A dark over water
            img[8][~water] = 3000
            label = water.astype(np.int16)
            label[:4, :] = -1  # a strip of no-data
            _write_tif(root / "S2Hand" / f"{chip}_S2Hand.tif", img.astype(np.int16))
            _write_tif(root / "LabelHand" / f"{chip}_LabelHand.tif", label[None])
            ids.append(chip)
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
    return root
