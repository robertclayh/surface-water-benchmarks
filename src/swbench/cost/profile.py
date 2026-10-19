"""Static and dynamic compute cost of a model.

Everything here is measured, not looked up: parameter counts from the
module, FLOPs from ``torch.utils.flop_counter`` (matmul/conv FLOPs, i.e. 2 x
MACs), latency and memory from timed runs on the actual device.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from swbench.models.factory import count_parameters

# A 224 x 224 tile at Sentinel-2's 10 m GSD.
TILE_KM2 = (224 * 10 / 1000) ** 2


def _flops(fn) -> int:
    # The fused CPU attention kernels are not registered with FlopCounterMode
    # and would silently count as zero; the math backend decomposes attention
    # into counted matmuls with identical FLOPs.
    counter = FlopCounterMode(display=False)
    with sdpa_kernel(SDPBackend.MATH), counter:
        fn()
    return counter.get_total_flops()


def forward_flops(model: nn.Module, in_channels: int, size: int = 224, device: str | torch.device = "cpu") -> int:
    x = torch.randn(1, in_channels, size, size, device=device)
    model.eval()
    with torch.no_grad():
        return _flops(lambda: model(x))


def train_step_flops(model: nn.Module, in_channels: int, size: int = 224, device: str | torch.device = "cpu") -> int:
    """Forward + backward FLOPs of one training sample.

    For a frozen encoder the forward runs under ``no_grad`` and no backward
    flows through it, so this is much smaller than 3 x forward; that is the
    main computational argument for the frozen protocol.
    """
    x = torch.randn(2, in_channels, size, size, device=device)  # BatchNorm needs >1 sample
    model.train()

    def step():
        out = model(x)
        out.float().mean().backward()

    flops = _flops(step)
    model.zero_grad(set_to_none=True)
    return flops // 2


@torch.no_grad()
def inference_benchmark(
    model: nn.Module,
    in_channels: int,
    device: torch.device,
    batch_size: int = 16,
    size: int = 224,
    warmup: int = 3,
    iters: int = 10,
    amp: bool = True,
) -> dict[str, float]:
    model.eval().to(device)
    x = torch.randn(batch_size, in_channels, size, size, device=device)
    use_amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    out = {
        "infer_ms_per_tile": 1000 * dt / batch_size,
        "infer_tiles_per_s": batch_size / dt,
        "infer_km2_per_s": batch_size / dt * TILE_KM2,
    }
    if device.type == "cuda":
        out["infer_peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return out


def static_profile(model: nn.Module, in_channels: int, device: str | torch.device = "cpu") -> dict[str, Any]:
    prof: dict[str, Any] = dict(count_parameters(model))
    prof["gflops_fwd_per_tile"] = forward_flops(model, in_channels, device=device) / 1e9
    prof["gflops_train_per_tile"] = train_step_flops(model, in_channels, device=device) / 1e9
    return prof
