import torch
from torch import nn

from swbench.eval.inference import _starts, sliding_window_probs


def test_window_starts_cover_chip():
    assert _starts(512, 224, 192) == [0, 192, 288]
    assert _starts(224, 224, 192) == [0]
    assert _starts(100, 224, 192) == [0]


class PixelEcho(nn.Module):
    """Logit for water = first channel, so the probability map is known."""

    def forward(self, x):
        return torch.cat([torch.zeros_like(x[:, :1]), x[:, :1]], 1)


def test_sliding_window_matches_full_image():
    x = torch.randn(2, 3, 512, 512)
    probs = sliding_window_probs(PixelEcho(), x, tile=224, overlap=32)
    assert torch.allclose(probs, torch.sigmoid(x[:, 0]), atol=1e-5)
