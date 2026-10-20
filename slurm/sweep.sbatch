#!/bin/bash
# One (arm, N, seed) cell of configs/sweep.yaml per array task.
# Submitted by `slurm/submit.sh sweep`; the array index selects the cell.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
swb_activate

arm="$1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
swbench-sweep configs/sweep.yaml \
    "data.root=$SWB_DATA" "data.workers=$((SLURM_CPUS_PER_TASK - 1))" \
    --arms "$arm" --out "$SWB_RUNS" \
    --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
