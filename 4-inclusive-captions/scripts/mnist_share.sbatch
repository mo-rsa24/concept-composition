#!/bin/bash
#SBATCH --job-name=mnist-share
#SBATCH --partition=bigbatch
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/mnist-share/logs/%x-%j.out
#
# Plan 03 of scope 11, the MNIST share run: train one share model, then draw its eight rows.
# Checkpoints and samples go to /datasets only. Scoring runs on the laptop, where the judge lives.
#
#   scripts/cluster.sh run "SHARE=0.1 SEED=1 sbatch --nodelist=<healthy node> 4-inclusive-captions/scripts/mnist_share.sbatch"
set -uo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SHARE=${SHARE:?set SHARE, e.g. 0.1}
SEED=${SEED:?set SEED, 1 to 3}
GUIDE=${GUIDE:-3}
SAMPLES=samples; [ "$GUIDE" = 3 ] || SAMPLES=samples-g$GUIDE
OUT=/datasets/mmolefe/poe_repair_min/outputs/mnist-share
export MNIST_RAW=/datasets/mmolefe/raw/mnist
M=poe_repair/experiments/inclusive_captions/mnist

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-}  $(date -Is)  share $SHARE seed $SEED  repo $REPO"
[ -x "$PY" ] || { echo "ABORT: co3 python missing at $PY" >&2; exit 5; }
for f in train-images-idx3-ubyte train-labels-idx1-ubyte; do
  [ -s "$MNIST_RAW/$f" ] || { echo "ABORT: $MNIST_RAW/$f missing" >&2; exit 6; }
done
mkdir -p "$OUT/checkpoints" "$OUT/$SAMPLES" "$OUT/logs"
used=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$used" ] && [ "$used" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${used:-unknown}%" >&2; exit 3; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || { echo "ABORT: nvidia-smi failed" >&2; exit 4; }
"$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); x = torch.ones(1000, 1000, device='cuda'); print('matmul', (x @ x).sum().item()); sys.exit(0 if ok else 1)" \
  || { echo "ABORT: no working GPU under co3" >&2; exit 4; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO" || exit 7
t0=$(date +%s)
"$PY" -I $M/train.py --share "$SHARE" --seed "$SEED" --steps 6000 --width 48 --out "$OUT/checkpoints" \
  || { echo "ABORT: training failed" >&2; exit 9; }
t1=$(date +%s)
"$PY" -I $M/sample.py --ckpt "$OUT/checkpoints/share-${SHARE}-seed-${SEED}.pt" --out "$OUT/$SAMPLES" --guide "$GUIDE" \
  || { echo "ABORT: drawing failed" >&2; exit 10; }
t2=$(date +%s)
echo "=== done $(date -Is)  training $((t1 - t0)) s  drawing $((t2 - t1)) s"
