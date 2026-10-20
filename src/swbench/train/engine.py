"""One benchmark run: train an arm on a label budget, select on val, test.

Protocol choices that matter for a fair label-efficiency comparison:

* **Fixed optimiser-step budget** independent of N. Training "for E epochs"
  would give the N=8 run 30x fewer updates than the N=252 run and confound
  label scarcity with under-training.
* **Model selection on the validation split** (best water IoU over periodic
  evaluations), identical for every arm.
* **Nested label subsets** per seed (see ``label_budget_subset``).
* **Same loss, augmentation, crop size, schedule and tiling** for every arm;
  only the learning rates differ, and those are set per arm on full-data
  validation (see docs/PROTOCOL.md).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from swbench.cost.energy import GPUEnergyMeter, gpu_name
from swbench.data.bands import get_band_spec
from swbench.data.sen1floods11 import (
    IGNORE_INDEX,
    Sen1Floods11,
    label_budget_subset,
    read_split,
    worker_init_fn,
)
from swbench.eval.inference import evaluate
from swbench.models.factory import build_model, count_parameters


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    valid = (target != IGNORE_INDEX).float()
    p = logits.float().softmax(1)[:, 1] * valid
    t = (target == 1).float() * valid
    inter = (p * t).sum()
    return 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    if (target == IGNORE_INDEX).all():
        # A crop of pure no-data: keep the graph alive but contribute nothing.
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits.float(), target, ignore_index=IGNORE_INDEX)
    return ce + dice_weight * dice_loss(logits, target) if dice_weight else ce


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW with a separate (smaller) LR for pretrained encoder weights."""
    lr = cfg["lr"]
    enc_lr = cfg.get("encoder_lr", lr)
    wd = cfg.get("weight_decay", 0.01)
    enc, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (enc if name.startswith("encoder.") else rest).append(p)
    groups = [{"params": rest, "lr": lr}]
    if enc:
        groups.append({"params": enc, "lr": enc_lr})
    return torch.optim.AdamW(groups, weight_decay=wd)


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _loader(ds: Sen1Floods11, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=worker_init_fn,
        persistent_workers=workers > 0,
    )


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot for model selection; a frozen encoder never changes, so skip it."""
    skip = "encoder." if getattr(model, "frozen", False) else None
    return {
        k: v.detach().to("cpu", copy=True)
        for k, v in model.state_dict().items()
        if skip is None or not k.startswith(skip)
    }


def run(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Train + evaluate one (arm, label budget, seed) cell. Returns the result row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    data_cfg, train_cfg, eval_cfg = cfg["data"], cfg["train"], cfg.get("eval", {})
    seed = int(cfg.get("seed", 0))
    budget = cfg.get("label_budget")
    seed_everything(seed)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    spec = get_band_spec(cfg["model"]["bands"])
    root = data_cfg["root"]
    train_chips = label_budget_subset(read_split(root, "train"), budget, seed)
    val_chips = label_budget_subset(read_split(root, "valid"), data_cfg.get("val_chips"), seed)

    steps = int(train_cfg["steps"])
    bs = int(train_cfg["batch_size"])
    workers = int(data_cfg.get("workers", 4))
    train_ds = Sen1Floods11(
        root, train_chips, spec, crop_size=data_cfg.get("crop_size", 224), augment=True,
        samples_per_epoch=steps * bs, seed=seed,
    )
    train_loader = _loader(train_ds, bs, shuffle=True, workers=workers)
    eval_bs = int(eval_cfg.get("batch_size", 2))
    val_loader = _loader(Sen1Floods11(root, val_chips, spec), eval_bs, False, workers)

    model = build_model(cfg["model"]).to(device)
    opt = build_optimizer(model, train_cfg)
    warmup = int(train_cfg.get("warmup_steps", min(200, steps // 10)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, steps, warmup))
    dice_w = float(train_cfg.get("dice_weight", 1.0))
    eval_every = int(train_cfg.get("eval_every", max(1, steps // 10)))
    tile, overlap = int(eval_cfg.get("tile", 224)), int(eval_cfg.get("overlap", 32))

    best = {"iou_water": -1.0, "step": 0, "state": None}
    history = []
    train_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step = 0
    meter = GPUEnergyMeter(device.index or 0) if device.type == "cuda" else None
    t_start = time.perf_counter()
    if meter:
        meter.__enter__()
    model.train()
    t_seg = time.perf_counter()
    for batch in train_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = segmentation_loss(logits, y, dice_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
        opt.step()
        sched.step()
        step += 1

        if step % eval_every == 0 or step == steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            train_seconds += time.perf_counter() - t_seg
            val = evaluate(model, val_loader, device, tile, overlap, amp=amp).compute()
            history.append({"step": step, "loss": float(loss.item()), **{f"val_{k}": v for k, v in val.items()}})
            if val["iou_water"] > best["iou_water"]:
                best = {
                    "iou_water": val["iou_water"],
                    "step": step,
                    "state": _trainable_state(model),
                }
            model.train()
            t_seg = time.perf_counter()
        if step >= steps:
            break
    if meter:
        meter.__exit__(None, None, None)
    wall_seconds = time.perf_counter() - t_start

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    model.load_state_dict(best["state"], strict=not getattr(model, "frozen", False))

    row: dict[str, Any] = {
        "arm": cfg["name"],
        "arch": cfg["model"]["arch"],
        "protocol": cfg["model"].get("protocol", "scratch"),
        "label_budget": len(train_chips),
        "seed": seed,
        "steps": steps,
        "batch_size": bs,
        "best_step": best["step"],
        "train_seconds": train_seconds,
        "run_wall_seconds": wall_seconds,
        "train_peak_mem_gb": peak_mem,
        "run_energy_kwh": meter.kwh if meter else None,
        "energy_method": meter.method if meter else "unavailable",
        "device": gpu_name() if device.type == "cuda" else device.type,
        **count_parameters(model),
    }
    per_chip = {}
    for split in eval_cfg.get("splits", ["valid", "test", "bolivia"]):
        chips = val_chips if split == "valid" else read_split(root, split)
        if eval_cfg.get("max_chips"):  # smoke tests only; never set for reported results
            chips = chips[: int(eval_cfg["max_chips"])]
        loader = _loader(Sen1Floods11(root, chips, spec), eval_bs, False, workers)
        t0 = time.perf_counter()
        m = evaluate(model, loader, device, tile, overlap, amp=amp)
        row[f"{split}_infer_seconds"] = time.perf_counter() - t0
        for k, v in m.compute().items():
            row[f"{split}_{k}"] = v
        lo, hi = m.bootstrap("iou_water", n=int(eval_cfg.get("bootstrap", 1000)), seed=seed)
        row[f"{split}_iou_water_lo"], row[f"{split}_iou_water_hi"] = lo, hi
        per_chip[split] = {c: k.tolist() for c, k in zip(m.chips, m.counts, strict=True)}

    (out_dir / "result.json").write_text(json.dumps(row, indent=2, default=float))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=float))
    (out_dir / "per_chip_counts.json").write_text(json.dumps(per_chip))
    (out_dir / "train_chips.txt").write_text("\n".join(train_chips) + "\n")
    if cfg.get("save_checkpoint", False):
        torch.save(best["state"], out_dir / "best.pt")
    return row
