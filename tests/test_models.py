import importlib.util

import pytest
import torch

from swbench.models.factory import build_model, count_parameters
from swbench.models.unet import UNet
from swbench.models.vit import (
    MultiLevelUpsamplingDecoder,
    SimpleViTEncoder,
    ViTSegmenter,
    default_out_indices,
)

HAS_TT = importlib.util.find_spec("terratorch") is not None


def test_unet_shape():
    m = UNet(6, base_width=8)
    assert m(torch.randn(2, 6, 64, 64)).shape == (2, 2, 64, 64)


def test_out_indices():
    assert default_out_indices(24) == [5, 11, 17, 23]
    assert default_out_indices(12) == [2, 5, 8, 11]
    assert default_out_indices(2) == [0, 1]


@pytest.mark.parametrize("patch", [8, 16])
def test_decoder_restores_resolution(patch):
    d = MultiLevelUpsamplingDecoder(32, 2, patch, channels=16)
    feats = [torch.randn(1, 32, 4, 4)] * 2
    assert d(feats).shape == (1, 2, 4 * patch, 4 * patch)


def test_frozen_segmenter_trains_only_decoder():
    enc = SimpleViTEncoder(6, embed_dim=32, depth=2, heads=2, patch_size=16)
    m = ViTSegmenter(enc, decoder_channels=16, frozen=True)
    counts = count_parameters(m)
    assert counts["params_trainable"] == sum(p.numel() for p in m.decoder.parameters())
    m.train()
    assert not m.encoder.training and m.decoder.training
    out = m(torch.randn(2, 6, 64, 64))
    out.mean().backward()
    assert all(p.grad is None for p in m.encoder.parameters())


def test_segmenter_pads_non_multiple_inputs():
    enc = SimpleViTEncoder(6, embed_dim=32, depth=2, heads=2, patch_size=16)
    m = ViTSegmenter(enc, decoder_channels=16)
    assert m(torch.randn(1, 6, 70, 50)).shape == (1, 2, 70, 50)


def test_factory_validates():
    with pytest.raises(ValueError):
        build_model({"arch": "unet", "protocol": "finetune", "bands": "s1f11_6"})
    with pytest.raises(ValueError):
        build_model({"arch": "prithvi", "protocol": "frozen", "bands": "clay10", "pretrained": False})
    m = build_model({"arch": "unet", "bands": "s1f11_13", "kwargs": {"base_width": 8}})
    assert m.inc[0].in_channels == 13


@pytest.mark.skipif(not HAS_TT, reason="terratorch not installed")
@pytest.mark.parametrize("arch,bands,c,grid", [("prithvi", "prithvi6", 6, 14), ("clay", "clay10", 10, 28)])
def test_foundation_model_architectures(arch, bands, c, grid):
    """Real Prithvi-EO-2.0-300M / Clay v1.5-large graphs with random weights."""
    m = build_model({"arch": arch, "protocol": "frozen", "bands": bands, "pretrained": False})
    x = torch.randn(1, c, 224, 224)
    with torch.no_grad():
        feats = m.encoder(x)
        y = m(x)
    assert [f.shape for f in feats] == [torch.Size([1, 1024, grid, grid])] * 4
    assert y.shape == (1, 2, 224, 224)
    assert 290e6 < count_parameters(m.encoder)["params_total"] < 330e6
