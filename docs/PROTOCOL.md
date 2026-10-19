# Fine-tuning and evaluation protocol

This document is the contract every benchmark arm follows. If a number in
`results/RESULTS.md` was produced any other way, it is not comparable.

## Question

For surface-water extent mapping from a single Sentinel-2 scene, how much
accuracy does a pretrained geospatial foundation model (GFM) add over a
task-specific U-Net **as a function of how many labelled chips you have**,
and what does that accuracy cost in compute?

## Data

* **Sen1Floods11, hand-labelled subset** (Bonafilia et al., CVPR-W 2020):
  446 Sentinel-2 L1C chips of 512 × 512 px at 10 m, labels
  `{-1: no data/cloud, 0: dry, 1: water}`.
* Official splits: **train 252**, **valid 89**, **test 90** (in-distribution:
  same 10 flood events) and **Bolivia 15** (a flood event and country never
  seen in training). Bolivia is our proxy for an unlabelled, data-scarce
  basin; with only 15 chips its confidence intervals are wide and are always
  reported.
* `scripts/download_sen1floods11.py` fetches exactly these files from the
  public `gs://sen1floods11` bucket over HTTPS.

## Arms

| arm | encoder | init | trained params | input bands | normalisation |
|---|---|---|---|---|---|
| `unet13` | U-Net (7.8M) | random | all | all 13 S2 bands | Sen1Floods11 train stats |
| `unet6` | U-Net (7.8M) | random | all | Prithvi's 6 bands | Sen1Floods11 train stats |
| `prithvi_frozen` | Prithvi-EO-2.0-300M | pretrained | decoder only (3.8M) | B2 B3 B4 B8A B11 B12 | Prithvi pretraining stats |
| `prithvi_finetune` | Prithvi-EO-2.0-300M | pretrained | all (308M) | as above | as above |
| `clay_frozen` | Clay v1.5 large | pretrained | decoder only (3.8M) | 10 L2A bands (no B1, B9, B10) | Clay `sentinel-2-l2a` stats |
| `clay_finetune` | Clay v1.5 large | pretrained | all (315M) | as above | as above |

`unet13` is the baseline the report compares against: it is the model a
practitioner would actually train. `unet6` sees exactly Prithvi's bands, so
`prithvi_* − unet6` isolates the effect of pretraining from the effect of
having more spectral bands.

Each GFM gets the bands and normalisation it was pretrained with; the U-Nets
get statistics computed on the training split only
(`scripts/compute_band_stats.py`).

### Decoder

Both GFMs use the same `MultiLevelUpsamplingDecoder`: features from blocks
5, 11, 17, 23 (of 24) are projected to 256 channels, concatenated, fused by a
3 × 3 conv, then upsampled ×2 per stage (4 stages for Prithvi's 16 px
patches, 3 for Clay's 8 px patches) to full resolution. ~3.8M parameters.
It is intentionally light so that the frozen protocol measures the
embeddings, not a large decoder.

### Protocols

* **frozen**: encoder weights fixed, encoder in eval mode under `no_grad`.
  Only the decoder is optimised.
* **finetune**: everything optimised; the encoder at `encoder_lr = 2e-5`,
  decoder at `lr = 1e-3`.
* **scratch** (U-Nets): everything optimised at `lr = 1e-3`.

## Label budgets

`N ∈ {8, 16, 32, 64, 128, 252}` training chips, 5 seeds each. For a given
seed the subsets are **nested**: the 8 chips at N=8 are the first 8 of the
16 at N=16, and so on (`label_budget_subset`). Every arm sees **the same
chips** for a given (N, seed), which is what makes matched-seed differences
meaningful.

## Training

Identical for every arm (`configs/base.yaml`):

* **2,000 optimiser steps, batch 8, regardless of N.** Epoch-based training
  would give N=8 runs ~30× fewer updates than N=252 runs and confound
  label scarcity with under-training.
* Random 224 × 224 crops (the GFMs' native input size), random 90° rotations
  and horizontal flips.
* Loss: cross-entropy + soft Dice on the water class; no-data pixels ignored.
* AdamW, weight decay 0.05, 100 warm-up steps then cosine decay, gradient
  clipping at 1.0, bf16 autocast on GPU.
* Validation every 200 steps on the valid split; **the checkpoint with the
  best validation water IoU is the one that is tested.**

### Learning rates

The LRs in `configs/arms/*.yaml` are starting defaults (common values for
U-Nets and for ViT-L fine-tuning). Before the main sweep, tune them per arm
family on the **full training set and the validation split only** (never
test or Bolivia), e.g.

```bash
for lr in 3e-4 1e-3 3e-3; do swbench-train configs/arms/unet13.yaml --out runs_lr/lr$lr train.lr=$lr eval.splits=[valid]; done
for elr in 1e-5 2e-5 5e-5; do swbench-train configs/arms/prithvi_finetune.yaml --out runs_lr/elr$elr train.encoder_lr=$elr eval.splits=[valid]; done
```

and commit the chosen values to the arm configs. LRs are then **not**
re-tuned per label budget: low-N results reflect a practitioner using
sensible defaults, not an oracle-tuned model, and this applies to every arm
equally.

### Validation set size

By default the full 89-chip valid split is used for model selection at every
label budget. In a truly scarce setting you would not have 89 labelled
validation chips; set `data.val_chips` to subsample it (nested by seed) to
study that regime. This is reported as a limitation, not hidden.

## Evaluation

* Full 512 × 512 chips, **sliding-window inference with 224 px tiles and
  32 px overlap** for every arm (so no model is evaluated at a resolution it
  was not trained at), probabilities averaged in overlaps, threshold 0.5.
* **Metrics are pooled over all valid pixels of a split** (Sen1Floods11
  convention): water IoU (headline), F1, precision, recall, dry IoU, mIoU,
  accuracy.
* **Uncertainty**: a chip-level bootstrap (1,000 resamples) for each run, and
  across-seed spread for each (arm, N). Comparisons between arms use
  **matched-seed differences** with a bootstrap CI over seeds.

## Reproducing

```bash
pip install -e '.[fm,energy]'
python scripts/download_sen1floods11.py --out data/sen1floods11
swbench-sweep configs/sweep.yaml --out runs --shard 0/4   # one shard per GPU
swbench-cost configs/arms/*.yaml --out results/profiles.csv
swbench-report --runs runs --out results --profiles results/profiles.csv
```

Each run directory contains `config.yaml` (the resolved config),
`train_chips.txt`, `history.json`, `per_chip_counts.json` and `result.json`,
so any number in the report can be traced to its run.

## Known limitations

* **Sensor/processing mismatch.** Sen1Floods11 is Sentinel-2 **L1C** (top of
  atmosphere). Prithvi-EO-2.0 was pretrained on HLS surface reflectance and
  Clay on L2A. Both GFMs therefore see a mild domain shift that the U-Nets
  (normalised on this data) do not. This is realistic for anyone reusing
  Sen1Floods11, but it is a handicap for the GFMs and is noted next to every
  result.
* **Single date.** Prithvi's temporal and Clay's time/location embeddings are
  unused (Clay's metadata vector is zeroed).
* **One decoder design.** A heavier decoder (UPerNet) might favour
  fine-tuning further; we chose a light one to keep the frozen protocol a
  fair test of the embeddings.
* **Bolivia has 15 chips** from a single event. It tests transfer to one
  unseen basin, not to "data-scarce basins" in general.
