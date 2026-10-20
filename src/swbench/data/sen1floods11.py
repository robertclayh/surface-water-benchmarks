"""Sen1Floods11 hand-labelled Sentinel-2 chips.

Expected layout (what ``scripts/download_sen1floods11.py`` produces)::

    root/S2Hand/<chip>_S2Hand.tif        13 bands, 512 x 512, L1C reflectance x 1e4
    root/LabelHand/<chip>_LabelHand.tif  {-1: no data / cloud, 0: dry, 1: water}
    root/splits/{train,valid,test,bolivia}.txt
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from swbench.data.bands import BandSpec

IGNORE_INDEX = -1


def read_split(root: str | Path, split: str) -> list[str]:
    text = (Path(root) / "splits" / f"{split}.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_chip(root: str | Path, chip: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image [13, H, W] float32, label [H, W] int64)``; NaNs are kept."""
    import rasterio

    root = Path(root)
    with rasterio.open(root / "S2Hand" / f"{chip}_S2Hand.tif") as src:
        image = src.read().astype(np.float32)
    with rasterio.open(root / "LabelHand" / f"{chip}_LabelHand.tif") as src:
        label = src.read(1).astype(np.int64)
    label[(label != 0) & (label != 1)] = IGNORE_INDEX
    return image, label


def normalize(image: np.ndarray, spec: BandSpec) -> np.ndarray:
    """Select ``spec``'s bands from a 13-band chip and standardise them.

    Missing values become 0, i.e. the band mean.
    """
    x = image[spec.indices]
    mean = np.asarray(spec.mean, np.float32)[:, None, None]
    std = np.asarray(spec.std, np.float32)[:, None, None]
    x = (x - mean) / std
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def label_budget_subset(chips: Sequence[str], n: int | None, seed: int) -> list[str]:
    """The first ``n`` chips of a seeded permutation of the sorted chip list.

    Subsets are nested for a fixed seed (the N=8 chips are the first 8 of the
    N=16 chips), so a larger budget only ever adds labels, and every arm gets
    the same chips for a given (N, seed). ``None`` or ``n >= len`` means all.
    """
    chips = sorted(chips)
    if n is None or n >= len(chips):
        return chips
    order = np.random.default_rng(seed).permutation(len(chips))
    return [chips[i] for i in order[:n]]


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker distinct, reproducible global RNG state."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


class Sen1Floods11(Dataset):
    """Training crops (``crop_size`` set) or full chips (``crop_size=None``).

    With ``samples_per_epoch`` the dataset cycles over its chips that many
    times in total, so one pass of the loader is the whole fixed step budget
    regardless of how many chips the label budget allows. Crop positions and
    flips come from an RNG keyed on ``(seed, index)``: they do not depend on
    the number of workers.
    """

    def __init__(
        self,
        root: str | Path,
        chips: Sequence[str],
        spec: BandSpec,
        crop_size: int | None = None,
        augment: bool = False,
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        if not chips:
            raise ValueError("no chips")
        self.root, self.chips, self.spec = Path(root), list(chips), spec
        self.crop_size, self.augment = crop_size, augment
        self.samples_per_epoch, self.seed = samples_per_epoch, seed

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.chips)

    def __getitem__(self, index: int) -> dict:
        chip = self.chips[index % len(self.chips)]
        image, label = load_chip(self.root, chip)
        x = normalize(image, self.spec)
        rng = np.random.default_rng((self.seed, index))

        if self.crop_size:
            h, w = label.shape
            c = self.crop_size
            i = int(rng.integers(0, h - c + 1)) if h > c else 0
            j = int(rng.integers(0, w - c + 1)) if w > c else 0
            x, label = x[:, i : i + c, j : j + c], label[i : i + c, j : j + c]
        if self.augment:
            k = int(rng.integers(4))
            x, label = np.rot90(x, k, axes=(1, 2)), np.rot90(label, k)
            if rng.random() < 0.5:
                x, label = x[:, :, ::-1], label[:, ::-1]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "mask": torch.from_numpy(np.ascontiguousarray(label)),
            "chip": chip,
        }
