#!/bin/bash
#SBATCH --job-name=toy-corr
#SBATCH --partition=bigbatch
#SBATCH --time=0-03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/concept-composition/4-inclusive-captions/data/toy-relabelling-ladder/%x-%j.out
#
# The toy corrector result in the replacing design at one or more counts (toy_replace_corrector.py).
#   scripts/cluster.sh sh "COUNTS=0 TAG=-k0 sbatch --exclude=<faulted> 4-inclusive-captions/scripts/toy_replace_corrector.sbatch"
set -uo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
COUNTS=${COUNTS:?set COUNTS, e.g. 0,100}
TAG=${TAG:?set TAG, e.g. -k0}
OUT=$REPO/4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-}  $(date -Is)  counts $COUNTS"
[ -x "$PY" ] || { echo "ABORT: co3 python missing" >&2; exit 5; }
mkdir -p "$OUT"
used=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$used" ] && [ "$used" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${used:-unknown}%" >&2; exit 3; }
"$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda', ok); x = torch.ones(1000, 1000, device='cuda'); print('matmul', (x @ x).sum().item()); sys.exit(0 if ok else 1)" \
  || { echo "ABORT: no working GPU under co3" >&2; exit 4; }
cd "$REPO" || exit 7
"$PY" 4-inclusive-captions/scripts/toy_replace_corrector.py --counts "$COUNTS" --tag="$TAG" || { echo "ABORT: run failed" >&2; exit 9; }
echo "=== done $(date -Is)"
