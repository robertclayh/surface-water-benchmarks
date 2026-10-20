#!/bin/bash
# GPU inference latency, throughput and memory for every arm -> results/profiles.csv.
# Submitted by `slurm/submit.sh cost`. Must use the same GPU type as the sweep.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
swb_activate
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible to PyTorch; refusing to run on CPU')"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
swbench-cost configs/arms/{unet13,unet6,prithvi_frozen,prithvi_finetune,clay_frozen,clay_finetune}.yaml \
    --out results/profiles.csv
