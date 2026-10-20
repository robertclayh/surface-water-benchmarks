#!/bin/bash
# Submit benchmark stages to Slurm. Run from anywhere on a login node.
#
#     slurm/submit.sh check              # pre-flight: GPU visible, weights load, tests pass
#     slurm/submit.sh lr [range]         # learning-rate grid (slurm/lr_grid.txt) or tasks e.g. 15-18
#     slurm/submit.sh sweep [arm ...]    # one array job per arm, one task per (N, seed) cell
#     slurm/submit.sh cost               # GPU latency / throughput / memory per arm
#
# Resubmitting `sweep` is safe: finished cells (result.json present) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
if [[ "$SWB_ACCOUNT" == CHANGE_ME ]]; then
    echo "set SWB_ACCOUNT in slurm/env.sh (or export it) first" >&2
    exit 1
fi
mkdir -p "$SWB_SCRATCH/logs"
logs="$SWB_SCRATCH/logs"
# shellcheck disable=SC2206
common=(--account="$SWB_ACCOUNT" $SWB_GPU_FLAGS)

case "${1:-}" in
    check)
        sbatch "${common[@]}" --job-name=swb-check --output="$logs/%x-%j.out" slurm/check.sbatch
        ;;
    lr)
        n=$(grep -cvE '^\s*(#|$)' slurm/lr_grid.txt)
        sbatch "${common[@]}" --job-name=swb-lr --array="${2:-0-$((n - 1))}" \
            --output="$logs/%x-%A_%a.out" slurm/lr_tune.sbatch
        ;;
    sweep)
        shift
        arms=("$@")
        if [[ ${#arms[@]} -eq 0 ]]; then
            arms=(unet13 unet6 prithvi_frozen prithvi_finetune clay_frozen clay_finetune)
        fi
        cells=$("$SWB_VENV/bin/python" -c "import yaml; s = yaml.safe_load(open('configs/sweep.yaml')); print(len(s['label_budgets']) * len(s['seeds']))")
        for arm in "${arms[@]}"; do
            [[ -f "configs/arms/$arm.yaml" ]] || { echo "unknown arm $arm" >&2; exit 1; }
            sbatch "${common[@]}" --job-name="swb-$arm" --array="0-$((cells - 1))" \
                --output="$logs/%x-%A_%a.out" slurm/sweep.sbatch "$arm"
        done
        ;;
    cost)
        sbatch "${common[@]}" --job-name=swb-cost --output="$logs/%x-%j.out" slurm/cost.sbatch
        ;;
    *)
        sed -n '2,9p' "$0"
        exit 1
        ;;
esac
