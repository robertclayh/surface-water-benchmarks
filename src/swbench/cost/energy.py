"""GPU energy metering.

Uses NVML's cumulative energy counter when the GPU exposes it (Volta and
newer), otherwise integrates sampled board power. Without an NVIDIA GPU or
``nvidia-ml-py`` it records nothing, and the report falls back to an
explicitly labelled TDP-based *estimate* instead of inventing a measurement.
"""

from __future__ import annotations

import threading
import time


class GPUEnergyMeter:
    def __init__(self, device_index: int = 0, interval_s: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self.joules: float | None = None
        self.method = "unavailable"
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._nvml = None

    def __enter__(self) -> GPUEnergyMeter:
        self._t0 = time.perf_counter()
        if self._nvml is None:
            return self
        try:
            self._e0 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)  # mJ
            self.method = "nvml_energy_counter"
        except Exception:
            self._e0 = None
            self.method = "nvml_power_sampling"
            self._samples: list[tuple[float, float]] = []
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            self._samples.append((time.perf_counter(), watts))
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self._nvml is None:
            return
        if self.method == "nvml_energy_counter":
            self.joules = (self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) - self._e0) / 1000.0
        else:
            self._stop.set()
            self._thread.join()
            s = self._samples
            self.joules = sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(s, s[1:], strict=False)) if len(s) > 1 else None

    @property
    def kwh(self) -> float | None:
        return None if self.joules is None else self.joules / 3.6e6


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"
