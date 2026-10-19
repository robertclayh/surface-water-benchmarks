"""Build a benchmark arm from its config block."""

from __future__ import annotations

from typing import Any

from torch import nn

from swbench.data.bands import get_band_spec
from swbench.models.unet import UNet
from swbench.models.vit import ClayEncoder, PrithviEncoder, SimpleViTEncoder, ViTSegmenter

PROTOCOLS = ("scratch", "frozen", "finetune")


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """``cfg`` is the ``model`` section of a run config.

    Keys: ``arch`` (unet | prithvi | clay | simple_vit), ``protocol``
    (scratch | frozen | finetune), ``bands`` (a BandSpec name) and an optional
    ``kwargs`` dict forwarded to the architecture.
    """
    arch, protocol = cfg["arch"], cfg.get("protocol", "scratch")
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    spec = get_band_spec(cfg["bands"])
    kwargs = dict(cfg.get("kwargs") or {})
    num_classes = cfg.get("num_classes", 2)
    decoder_channels = kwargs.pop("decoder_channels", 256)

    if arch == "unet":
        if protocol != "scratch":
            raise ValueError("the U-Net baseline has no pretrained weights; use protocol: scratch")
        return UNet(len(spec), num_classes, **kwargs)

    # ``pretrained: false`` with a frozen/finetune protocol builds the same
    # graph with random weights (used for cost profiling and offline tests).
    pretrained = cfg.get("pretrained", protocol != "scratch")
    if arch == "prithvi":
        if spec.name != "prithvi6":
            raise ValueError("Prithvi expects bands: prithvi6")
        encoder = PrithviEncoder(pretrained=pretrained, **kwargs)
    elif arch == "clay":
        if spec.name != "clay10":
            raise ValueError("Clay expects bands: clay10")
        encoder = ClayEncoder(pretrained=pretrained, spec=spec, **kwargs)
    elif arch == "simple_vit":
        encoder = SimpleViTEncoder(len(spec), **kwargs)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return ViTSegmenter(encoder, num_classes, decoder_channels, frozen=protocol == "frozen")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
