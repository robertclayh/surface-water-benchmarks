"""Task-specific U-Net baseline trained from scratch.

A plain 4-level U-Net (Ronneberger et al., 2015) with BatchNorm. It is
deliberately conventional: the question is whether a foundation model beats
the model a hydrologist would train by default, not whether it beats a weak
straw man. Width is configurable so the baseline can be scaled down when
labels are scarce.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        base_width: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        widths = [base_width * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, widths[0])
        self.down = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(widths[i], widths[i + 1], dropout if i == depth - 1 else 0))
            for i in range(depth)
        )
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(DoubleConv(widths[i] * 2, widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        self.stride = 2**depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            h = up(h)
            h = dec(torch.cat([h, skips.pop()], dim=1))
        return self.head(h)
