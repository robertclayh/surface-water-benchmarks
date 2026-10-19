"""Sliding-window inference and split evaluation."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from swbench.eval.metrics import SegmentationMeter


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


@torch.no_grad()
def sliding_window_probs(
    model: nn.Module, image: torch.Tensor, tile: int = 224, overlap: int = 32, batch_size: int = 16
) -> torch.Tensor:
    """Water probability ``[B, H, W]`` for full chips ``image [B, C, H, W]``.

    Every model sees the 224 px tiles it was trained on, so differences
    between arms are not confounded by a train/test resolution mismatch.
    Overlapping tiles are averaged in probability space.
    """
    b, _, h, w = image.shape
    stride = tile - overlap
    windows = [(i, j) for i in _starts(h, tile, stride) for j in _starts(w, tile, stride)]
    probs = image.new_zeros(b, h, w)
    weight = image.new_zeros(1, h, w)
    th, tw = min(tile, h), min(tile, w)
    for k in range(0, len(windows), max(1, batch_size // b)):
        chunk = windows[k : k + max(1, batch_size // b)]
        crops = torch.cat([image[:, :, i : i + th, j : j + tw] for i, j in chunk])
        p = model(crops).float().softmax(1)[:, 1]
        for n, (i, j) in enumerate(chunk):
            probs[:, i : i + th, j : j + tw] += p[n * b : (n + 1) * b]
            weight[:, i : i + th, j : j + tw] += 1
    return probs / weight


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tile: int = 224,
    overlap: int = 32,
    threshold: float = 0.5,
    amp: bool = False,
) -> SegmentationMeter:
    model.eval()
    meter = SegmentationMeter()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            probs = sliding_window_probs(model, x, tile, overlap)
        pred = (probs > threshold).long().cpu()
        for chip, p, y in zip(batch["chip"], pred, batch["mask"], strict=True):
            meter.update(chip, p, y)
    return meter
