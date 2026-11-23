#!/bin/bash
#SBATCH --job-name=l9-w4
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-9/logs/%x-%j.log
# Step W4. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> w4.sbatch <ladder|final>
# ladder: smoke test, overfit 16 canvases, and 1% of the data. final: trains on the full data and reads unseen_test once.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-9
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/2-classifier-guidance/scripts/noise_text_critic
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -f "$S/common.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
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

FULL=/datasets/mmolefe/poe_repair_min/loop-8/v1/data
SMALL=/datasets/mmolefe/poe_repair_min/loop-8/v1/ladder/data-1pct
OUT=$ROOT/w4
[ -f "$FULL/dataset.npz" ] && [ -f "$SMALL/dataset.npz" ] || { echo "ABORT: digit data missing"; exit 3; }
run() { "$PY" "$S/w4_train.py" "$@"; }
case "$STAGE" in
  ladder)
    run smoke --data "$SMALL" --out "$OUT/ladder" || exit 10
    run overfit --data "$SMALL" --out "$OUT/ladder" || exit 11
    run small --data "$SMALL" --out "$OUT/ladder"
    rc=$? ;;
  final)
    run final --data "$FULL" --out "$OUT/full" --ladder "$OUT/ladder" --wandb-name loop9-w4-final
    rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
