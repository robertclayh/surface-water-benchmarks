"""Foundation-model encoders and the shared segmentation decoder.

Both foundation models are ViTs that emit one token per patch. Each adapter
below exposes the same contract so the decoder, the freezing logic and the
cost profiler treat them identically:

* ``forward(x) -> list[Tensor]``: feature maps ``[B, D, H/p, W/p]`` for the
  transformer blocks listed in ``out_indices``;
* ``embed_dim``, ``patch_size``, ``depth`` attributes.

Pretrained weights are loaded through IBM/NASA's TerraTorch, which ships the
reference Prithvi-EO-2.0 ViT and a copy of the Clay v1.5 encoder, so the
architectures here are the published ones rather than re-implementations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from swbench.data.bands import CLAY_10, BandSpec


def default_out_indices(depth: int, n: int = 4) -> list[int]:
    """Evenly spaced blocks ending at the last one, e.g. 24 -> [5, 11, 17, 23]."""
    n = min(n, depth)
    step = depth // n
    return [step * (i + 1) - 1 for i in range(n)]


def _require_terratorch() -> None:
    try:
        import terratorch  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Foundation-model encoders need TerraTorch: pip install -e '.[fm]'"
        ) from e


class PrithviEncoder(nn.Module):
    """NASA/IBM Prithvi-EO-2.0 (300M, no temporal/location embedding).

    Single-date input: Sen1Floods11 has one S2 acquisition per chip, so the
    model runs with ``num_frames=1``. The 3D patch embedding of the checkpoint
    has a temporal kernel of 1, so no weights are discarded.
    """

    def __init__(
        self,
        variant: str = "prithvi_eo_v2_300",
        pretrained: bool = True,
        ckpt_path: str | None = None,
        out_indices: Sequence[int] | None = None,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        _require_terratorch()
        from terratorch.models.backbones import prithvi_vit

        build = getattr(prithvi_vit, variant)
        kwargs = {"num_frames": 1, "img_size": img_size}
        if ckpt_path:
            kwargs["ckpt_path"] = ckpt_path
        # bands=None: our six bands are exactly the pretrained bands, in order.
        self.vit = build(pretrained=pretrained, bands=None, **kwargs)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[-1]
        self.depth = len(self.vit.blocks)
        self.out_indices = list(out_indices) if out_indices else default_out_indices(self.depth)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.vit.forward_features(x)
        feats = [feats[i] for i in self.out_indices]
        return self.vit.prepare_features_for_image_model(feats)


class ClayEncoder(nn.Module):
    """Clay v1.5 (large, ~300M encoder) with its wavelength-conditioned patch embed.

    Clay generates its patch-embedding kernel from the band wavelengths, so it
    receives the 10 Sentinel-2 L2A bands it was trained on. Time and lat/lon
    metadata are zeroed, which is how Clay's own downstream examples run the
    encoder when that metadata is not used.
    """

    CONFIGS = {
        "large": dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0),
        "base": dict(dim=768, depth=12, heads=12, dim_head=64, mlp_ratio=4.0),
    }

    def __init__(
        self,
        size: str = "large",
        pretrained: bool = True,
        ckpt_path: str | None = None,
        out_indices: Sequence[int] | None = None,
        spec: BandSpec = CLAY_10,
        gsd: float = 10.0,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        _require_terratorch()
        from terratorch.models.backbones.clay_v15.model import Encoder

        cfg = self.CONFIGS[size]
        self.encoder = Encoder(mask_ratio=0.0, patch_size=patch_size, shuffle=False, **cfg)
        if pretrained:
            self._load(ckpt_path)
        self.embed_dim = cfg["dim"]
        self.patch_size = patch_size
        self.depth = cfg["depth"]
        self.out_indices = list(out_indices) if out_indices else default_out_indices(self.depth)
        self.register_buffer("waves", torch.tensor(spec.wavelengths, dtype=torch.float32), persistent=False)
        self.register_buffer("gsd", torch.tensor(gsd, dtype=torch.float32), persistent=False)

    def _load(self, ckpt_path: str | None) -> None:
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(repo_id="made-with-clay/Clay", filename="v1.5/clay-v1.5.ckpt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        prefix = "model.encoder."
        state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Clay checkpoint is missing encoder weights: {missing[:5]} ...")
        del unexpected  # decoder / teacher heads are expected and ignored

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        enc = self.encoder
        b, _, h, w = x.shape
        patches, _ = enc.to_patch_embed(x, self.waves)
        zeros = x.new_zeros(b, 4)
        patches = enc.add_encodings(patches, zeros, zeros, self.gsd)
        cls = enc.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)

        gh, gw = h // self.patch_size, w // self.patch_size
        wanted = set(self.out_indices)
        out = []
        layers = enc.transformer.layers
        for i, (attn, ff) in enumerate(layers):
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens
            if i in wanted:
                t = enc.transformer.norm(tokens) if i == len(layers) - 1 else tokens
                out.append(t[:, 1:].transpose(1, 2).reshape(b, self.embed_dim, gh, gw))
        return out


class SimpleViTEncoder(nn.Module):
    """Small randomly initialised ViT with the same contract as the adapters.

    Used for CPU smoke tests and for the "same architecture, no pretraining"
    ablation; it is not one of the benchmarked foundation models.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        patch_size: int = 16,
        out_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            embed_dim, heads, embed_dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(**_layer_kwargs(layer)) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim, self.patch_size, self.depth = embed_dim, patch_size, depth
        self.out_indices = list(out_indices) if out_indices else default_out_indices(depth)

    def _pos(self, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        d = self.embed_dim // 4
        omega = 1.0 / (10000 ** (torch.arange(d, device=device) / d))
        y, x = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        y = y.flatten()[:, None] * omega
        x = x.flatten()[:, None] * omega
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)[None]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        t = self.patch_embed(x)
        b, d, gh, gw = t.shape
        t = t.flatten(2).transpose(1, 2) + self._pos(gh, gw, x.device)
        out = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.out_indices:
                f = self.norm(t) if i == self.depth - 1 else t
                out.append(f.transpose(1, 2).reshape(b, d, gh, gw))
        return out


def _layer_kwargs(layer: nn.TransformerEncoderLayer) -> dict:
    return dict(
        d_model=layer.self_attn.embed_dim,
        nhead=layer.self_attn.num_heads,
        dim_feedforward=layer.linear1.out_features,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MultiLevelUpsamplingDecoder(nn.Module):
    """Fuse several ViT layers, then upsample x2 per stage back to pixel scale.

    The same decoder (and the same hyper-parameters) is used for every ViT
    encoder so that differences between Prithvi and Clay come from the
    encoder. Its size is reported separately in the cost accounting because
    in the frozen protocol it is the only part that is trained.
    """

    def __init__(self, embed_dim: int, num_levels: int, patch_size: int, channels: int = 256, num_classes: int = 2):
        super().__init__()
        if patch_size & (patch_size - 1):
            raise ValueError("patch size must be a power of two")
        self.proj = nn.ModuleList(nn.Conv2d(embed_dim, channels, 1) for _ in range(num_levels))
        self.fuse = ConvBNReLU(channels * num_levels, channels)
        stages = []
        c = channels
        for _ in range(int(math.log2(patch_size))):
            stages.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                        ConvBNReLU(c, max(c // 2, 32))))
            c = max(c // 2, 32)
        self.up = nn.Sequential(*stages)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([p(f) for p, f in zip(self.proj, feats, strict=True)], dim=1)
        return self.head(self.up(self.fuse(x)))


class ViTSegmenter(nn.Module):
    """Encoder + decoder with the two fine-tuning protocols.

    ``frozen``: encoder weights fixed and run in eval mode under ``no_grad``
    (a linear-probe-style evaluation of the embeddings themselves).
    ``finetune``: encoder trained end to end, usually with a smaller LR
    (layer-wise decay handled by the optimiser builder).
    """

    def __init__(self, encoder: nn.Module, num_classes: int = 2, decoder_channels: int = 256, frozen: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = MultiLevelUpsamplingDecoder(
            encoder.embed_dim, len(encoder.out_indices), encoder.patch_size, decoder_channels, num_classes
        )
        self.frozen = frozen
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.stride = encoder.patch_size

    def train(self, mode: bool = True) -> ViTSegmenter:
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = self.stride
        ph, pw = (-h) % p, (-w) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        if self.frozen:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        logits = self.decoder(feats)
        return logits[..., :h, :w]
