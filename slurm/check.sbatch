#!/bin/bash
# Pre-flight on a GPU node: PyTorch sees the GPU, both pretrained models load
# from the offline cache and run a bf16 forward pass, and the test suite passes.
# Submitted by `slurm/submit.sh check`.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
swb_activate

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'EOF'
import torch

from swbench.models.factory import build_model

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU visible: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU: the driver is likely too old for this CUDA build")
dev = torch.device("cuda")
for arch, bands, c in (("prithvi", "prithvi6", 6), ("clay", "clay10", 10)):
    m = build_model({"arch": arch, "bands": bands, "protocol": "frozen"}).to(dev).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(torch.randn(2, c, 224, 224, device=dev))
    print(f"{arch}: pretrained weights loaded, {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params, "
          f"output {tuple(out.shape)}")
    del m
EOF
python -m pytest -q
echo "check passed"
