# Cluster settings shared by every script in slurm/. Edit the defaults below
# (or export the variables before running) for your allocation.
#
# Defaults target UVA Rivanna/Afton: the `gpu` partition, one A100 80GB per
# job. Every arm must run on the same GPU type, otherwise the train-time and
# energy columns of the cost accounting are not comparable.

export SWB_ACCOUNT="${SWB_ACCOUNT:-CHANGE_ME}"             # your Slurm allocation (sbatch -A)
export SWB_GPU_FLAGS="${SWB_GPU_FLAGS:---partition=gpu --gres=gpu:a100:1 --constraint=a100_80gb}"
# Pinned: the venv is built on this Python, so a changed module default would break it.
export SWB_MODULES="${SWB_MODULES:-miniforge/24.11.3-py3.12}"

export SWB_SCRATCH="${SWB_SCRATCH:-/scratch/$USER/swbench}"
export SWB_DATA="${SWB_DATA:-$SWB_SCRATCH/data/sen1floods11}"
export SWB_RUNS="${SWB_RUNS:-$SWB_SCRATCH/runs}"
export SWB_LR_RUNS="${SWB_LR_RUNS:-$SWB_SCRATCH/runs_lr}"
export SWB_VENV="${SWB_VENV:-$HOME/.venvs/swbench}"
export HF_HOME="${HF_HOME:-$SWB_SCRATCH/hf}"              # pretrained weights cache

swb_activate() {
    module purge >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    module load $SWB_MODULES
    # shellcheck disable=SC1091
    source "$SWB_VENV/bin/activate"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # Weights are prefetched by slurm/setup.sh; never download inside a job.
        export HF_HUB_OFFLINE=1
    fi
}
