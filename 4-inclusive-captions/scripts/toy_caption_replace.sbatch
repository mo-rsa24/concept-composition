#!/bin/bash
#SBATCH --job-name=toy-replace
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/concept-composition/4-inclusive-captions/data/toy-relabelling-ladder/%x-%j.out
#
# The toy relabelling experiment with replacing, at the adding design's shares, three seeds each,
# plus the reference model's weights at 1,000 replaced. Small outputs (JSON, npz, two .pth files);
# they land in the repo checkout and come back with scripts/cluster.sh pull.
#
#   scripts/cluster.sh run "sbatch --nodelist=<healthy node> 4-inclusive-captions/scripts/toy_caption_replace.sbatch"
set -uo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=$REPO/4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose
COUNTS=${COUNTS:-1,5,20,49,182,462,750,1000}

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-}  $(date -Is)  repo $REPO"
[ -x "$PY" ] || { echo "ABORT: co3 python missing at $PY" >&2; exit 5; }
mkdir -p "$OUT" "$REPO/4-inclusive-captions/data/toy-relabelling-ladder"
used=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$used" ] && [ "$used" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${used:-unknown}%" >&2; exit 3; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || { echo "ABORT: nvidia-smi failed" >&2; exit 4; }
"$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); x = torch.ones(1000, 1000, device='cuda'); print('matmul', (x @ x).sum().item()); sys.exit(0 if ok else 1)" \
  || { echo "ABORT: no working GPU under co3" >&2; exit 4; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO" || exit 7
"$PY" 4-inclusive-captions/scripts/toy_caption_replace.py --counts "$COUNTS" --tag -sweep --save-weights 1000:20261022 \
  || { echo "ABORT: sweep failed" >&2; exit 9; }
echo "=== done $(date -Is)"
