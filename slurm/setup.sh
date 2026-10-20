#!/bin/bash
# One-time setup on a login node: environment, data, pretrained weights.
#
#     bash slurm/setup.sh
#
# Login nodes have internet access; jobs then run with HF_HUB_OFFLINE=1.
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

# Instantiating the pretrained arms downloads and caches their checkpoints.
python - <<'EOF'
from swbench.models.factory import build_model

for arch, bands in (("prithvi", "prithvi6"), ("clay", "clay10")):
    m = build_model({"arch": arch, "bands": bands, "protocol": "frozen"})
    print(f"{arch}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters, weights cached")
EOF

python -m pytest -q
echo "setup done: data in $SWB_DATA, weights in $HF_HOME, env in $SWB_VENV"
