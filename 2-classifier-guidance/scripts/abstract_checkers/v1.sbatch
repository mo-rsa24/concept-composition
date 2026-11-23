#!/bin/bash
#SBATCH --job-name=l8-v1
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-8/logs/%x-%j.log
# Step V1. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> v1.sbatch <ladder|full>
# ladder: 1% data built twice, and the zero-push check on one pair at 2 seeds.
# full: the whole data built twice, and the zero-push check on all 20 pairs at 8 seeds.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-8
OUT=$ROOT/v1
MN=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
MODEL=$MN/models/share-0.0/unet.pt
RAW=$MN/mnist-raw
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/2-classifier-guidance/scripts/abstract_checkers
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -f "$S/build_env.py" ] && [ -f "$S/zero_push.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
[ -f "$MODEL" ] && [ -f "$RAW/train-images-idx3-ubyte" ] || { echo "ABORT: model or MNIST missing"; exit 3; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
[ "$USED" -lt 1024 ] && [ "$APPS" -eq 0 ] || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=0
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) python=$PY snap=$SNAP stage=$STAGE"

case "$STAGE" in
  ladder)
    "$PY" "$S/build_env.py" --mnist "$RAW" --out "$OUT/ladder/data-1pct" --scale 0.01 --twice || exit 10
    "$PY" "$S/zero_push.py" --model "$MODEL" --out "$OUT/ladder/zero-push.json" --pairs 5-6 --seeds 2
    rc=$? ;;
  full)
    "$PY" "$S/build_env.py" --mnist "$RAW" --out "$OUT/data" --twice --sheet "$OUT/canvas-kinds.png" || exit 10
    "$PY" "$S/zero_push.py" --model "$MODEL" --out "$OUT/zero-push.json" --wandb-name loop8-v1-zero-push
    rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
