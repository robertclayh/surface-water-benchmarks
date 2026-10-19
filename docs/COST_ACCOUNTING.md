# Cost accounting

A foundation model has to buy its accuracy gain with compute. This document
explains what we measure, how, and how the report decides whether a gain is
worth it. Every number below is **measured**; nothing is copied from a model
card.

## What is measured

| column | how | where |
|---|---|---|
| `params_total`, `params_trainable` | counted from the built module | `count_parameters` |
| `gflops_fwd_per_tile` | `torch.utils.flop_counter` on one 224 × 224 tile (2 FLOPs per MAC) | `cost/profile.py` |
| `gflops_train_per_tile` | same, forward + backward, per sample | `cost/profile.py` |
| `train_seconds` | wall-clock of the 2,000 training steps, **excluding** periodic validation, CUDA-synchronised | `train/engine.py` |
| `run_wall_seconds` | the whole run including validation | `train/engine.py` |
| `train_peak_mem_gb` | `torch.cuda.max_memory_allocated` over training | `train/engine.py` |
| `run_energy_kwh` | NVML board energy counter (or integrated power samples); `None` if unavailable | `cost/energy.py` |
| `infer_ms_per_tile`, `infer_km2_per_s`, `infer_peak_mem_gb` | timed bf16 batched inference on the profiling GPU | `swbench-cost` |
| `{split}_infer_seconds` | wall-clock of sliding-window evaluation of a split | `train/engine.py` |
| `train_pflops_per_run` | `steps × batch × gflops_train_per_tile` | report |

Two details that change the numbers materially:

* **Attention FLOPs.** PyTorch's fused CPU attention kernels are invisible to
  `FlopCounterMode`; we count under the math SDPA backend, which decomposes
  attention into counted matmuls (tested in `tests/test_cost.py`).
* **Frozen ≠ free.** A frozen encoder still runs its full forward pass at
  every training step (augmentation changes the input), so frozen training
  costs `forward(encoder) + forward/backward(decoder)`, not just the decoder.

Energy is board energy of one GPU only (no CPU, DRAM, cooling). When NVML is
unavailable the column is left empty; the report never substitutes a
TDP × time estimate for a measurement.

## Static costs (measured)

Measured with `swbench-cost configs/arms/*.yaml --skip-latency`
(`results/static_profile.csv`). These depend only on the architecture, not
on the GPU, the data or the training outcome. Per-run training FLOPs assume
the protocol's 2,000 steps × batch 8 = 16,000 tile-passes. Inference per km²
uses the evaluation tiling (9 tiles of 224 px with 32 px overlap per
512 × 512 chip of 26.2 km²).

| arm | params | trainable | GFLOPs fwd / tile | GFLOPs train / tile | PFLOPs / training run | GFLOPs / km² inference | × `unet13` (infer / train) |
|---|---|---|---|---|---|---|---|
| `unet13` | 7.8M | 7.8M | 18.8 | 56.0 | 0.90 | 6 | 1.0× / 1.0× |
| `unet6` | 7.8M | 7.8M | 18.6 | 55.5 | 0.89 | 6 | 1.0× / 1.0× |
| `prithvi_frozen` | 307.7M | 3.8M | 127.1 | 134.0 | 2.14 | 44 | 6.8× / 2.4× |
| `prithvi_finetune` | 307.7M | 307.7M | 127.1 | 380.6 | 6.09 | 44 | 6.8× / 6.8× |
| `clay_frozen` | 315.2M | 3.8M | 547.0 | 567.0 | 9.07 | 188 | 29.1× / 10.1× |
| `clay_finetune` | 315.2M | 315.2M | 547.0 | 1639.4 | 26.23 | 188 | 29.1× / 29.3× |

What this already tells us, before any training:

* **Inference is where the overhead is permanent.** Training is paid once;
  inference is paid for every square kilometre, every scene, forever. Prithvi
  costs ~7× and Clay ~29× the U-Net per km². For continental-scale or
  operational near-real-time mapping this, not training, dominates.
* **Clay is ~4.3× Prithvi** at the same parameter count because its 8 px
  patches give 784 tokens per tile versus Prithvi's 196, and attention and
  MLP cost scale with token count.
* **Frozen encoders cut training cost by ~3×** relative to fine-tuning
  (no backward through the encoder), but not inference cost.

FLOPs are a proxy. Measured GPU seconds (`train_seconds`, `infer_km2_per_s`)
also reflect memory bandwidth and kernel efficiency and are the numbers the
report uses for verdicts.

## Deciding whether the overhead is justified

For each GFM arm, label budget N and test split, the report computes the
**matched-seed IoU gain** over the baseline (`unet13` by default) with a
bootstrap 95% CI over seeds, and classifies it:

* **justified**: the entire CI is above `min_gain` (default **+0.02 water
  IoU**, i.e. two IoU points; revisit it once the seed-to-seed spread of
  your own runs is known);
* **not justified**: the entire CI is below `min_gain` (gain too small to
  matter, or a loss);
* **inconclusive**: the CI straddles `min_gain`; more seeds are needed before
  claiming either way.

Next to each verdict the report prints `extra_train_seconds` and
`extra_train_seconds_per_iou_point`, so a reader with a different exchange
rate between accuracy and compute can draw their own line, and the
**label multiplier**: how many labelled chips the baseline needs to match the
GFM at N (e.g. ×4 at N=16 means the U-Net needs ~64 chips). When labelling a
chip costs an analyst an hour and a GPU-hour costs a few dollars, the label
multiplier is often the number that decides.

`min_gain` is a CLI argument (`swbench-report --min-gain 0.01`); the verdict
table is regenerated from data, never written by hand.
