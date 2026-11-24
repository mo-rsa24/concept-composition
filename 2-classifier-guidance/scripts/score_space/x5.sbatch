#!/bin/bash
#SBATCH --job-name=l10-x5
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-10/logs/%x-%j.log
# Loop 10, step X5 (artifacts/notes/progress/score-space/04-design.md, run group 1). Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> x5.sbatch <ladder|full>
# ladder: smoke test, positive control at timestep 100, overfit 16 canvases, 1% of the data. full: the verdict read.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-10
OUT=$ROOT/x5
MODEL=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.0/unet.pt
DATA=/datasets/mmolefe/poe_repair_min/loop-8/v1/data
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/2-classifier-guidance/scripts/score_space
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -f "$S/x5_signal.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
[ -f "$MODEL" ] && [ -f "$DATA/dataset.npz" ] || { echo "ABORT: model or data missing"; exit 3; }
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

run() { "$PY" "$S/x5_signal.py" "$@" --model "$MODEL" --data "$DATA" --out "$OUT"; }
case "$STAGE" in
  ladder)
    run smoke || exit 10
    run control || exit 11
    run overfit || exit 12
    run small
    rc=$? ;;
  full)
    for f in smoke control overfit; do
      "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/$f.json'))['pass'] else 1)" \
        || { echo "ABORT: ladder rung $f did not pass"; exit 13; }
    done
    run full --wandb-name loop10-x5-full
    rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
