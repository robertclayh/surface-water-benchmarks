#!/bin/bash
# One line of slurm/lr_grid.txt per array task: full training set, valid split only.
# Submitted by `slurm/submit.sh lr`; summarise with scripts/summarize_lr.py.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
swb_activate
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible to PyTorch; refusing to run on CPU')"

line=$(grep -vE '^\s*(#|$)' slurm/lr_grid.txt | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r tag arm overrides <<<"$line"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "lr grid: $tag $arm $overrides"
# shellcheck disable=SC2086
swbench-train "configs/arms/$arm.yaml" \
    "data.root=$SWB_DATA" "data.workers=$((SLURM_CPUS_PER_TASK - 1))" "eval.splits=[valid]" \
    label_budget=null seed=0 $overrides \
    --out "$SWB_LR_RUNS/$tag"
