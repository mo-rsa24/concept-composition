#!/bin/bash
#SBATCH --job-name=l8-v3
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-8/logs/%x-%j.log
# Step V3. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> v3.sbatch <smoke|tune|final|stages>
# smoke: pair 0 and 1 at 4 seeds, every row, and the zero-push check. tune: picks the push length on seen pairs.
# final: the unseen pairs, once. stages: every in-between output of the count checker for the miniature page.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-8
MN=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
MODEL=$MN/models/share-0.0/unet.pt
JUDGE=$MN/judge-v3/judge.pt
HEAD=$ROOT/v2/count-final/count-head-map.pt
OUT=$ROOT/v3/test-1
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/2-classifier-guidance/scripts/abstract_checkers
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -f "$S/v3_compose.py" ] && [ -f "$S/explain_stages.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
for f in "$MODEL" "$JUDGE" "$HEAD"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 3; }; done
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

run() { "$PY" "$S/v3_compose.py" "$@" --model "$MODEL" --head "$HEAD" --judge "$JUDGE" --out "$OUT"; }
case "$STAGE" in
  smoke) run smoke; rc=$? ;;
  tune)  run tune; rc=$? ;;
  final) run final --wandb-name loop8-v3-test1-final; rc=$? ;;
  stages)
    "$PY" "$S/explain_stages.py" --model "$MODEL" --head "$HEAD" --examples "$ROOT/v2/count-final/examples" \
      --out "$ROOT/v3/stages/count-stages.npz"
    rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
