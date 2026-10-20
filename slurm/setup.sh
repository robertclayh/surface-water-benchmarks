#!/bin/bash
# One-time setup on a login node: environment, data, pretrained weights.
#
#     bash slurm/setup.sh
#     slurm/submit.sh check     # then verify GPU, weights and tests on a GPU node
#
# Login nodes have internet access but little memory, so this only downloads;
# anything that builds a model runs in the check job, offline.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh

module purge >/dev/null 2>&1 || true
# shellcheck disable=SC2086
module load $SWB_MODULES
if [[ ! -x "$SWB_VENV/bin/python" ]]; then
    python -m venv "$SWB_VENV"
fi
source "$SWB_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[fm,energy,dev]'

mkdir -p "$SWB_SCRATCH/logs" "$HF_HOME"
python scripts/download_sen1floods11.py --out "$SWB_DATA"

# The same files the Prithvi (TerraTorch) and Clay loaders request, into the
# shared HF_HOME cache.
python - <<'EOF'
from huggingface_hub import hf_hub_download

for repo, filename in (
    ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "Prithvi_EO_V2_300M.pt"),
    ("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt"),
):
    print("cached", hf_hub_download(repo_id=repo, filename=filename))
EOF

echo "setup done: data in $SWB_DATA, weights in $HF_HOME, env in $SWB_VENV"
echo "next: slurm/submit.sh check"
