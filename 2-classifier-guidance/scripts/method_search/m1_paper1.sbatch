#!/bin/bash
#SBATCH --job-name=l6-m1-p1
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M1, Paper 1. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m1_paper1.sbatch <ladder|full>
# ladder: smoke test, overfit, 1% of the data and the zero-push check. full: the 35,000-sample pool, both checkers, four rows.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
OUT=$ROOT/m1/paper-1
MN=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
A=$MN/two-experts/models/e04-either/unet.pt
B=$MN/two-experts/models/e59-either/unet.pt
JUDGE=$MN/judge-v3/judge.pt
: "${SNAP:?SNAP must name the code snapshot}"
CODE=$SNAP/repo/2-classifier-guidance/scripts
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -f "$CODE/method_search/paper1_rows.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
for f in "$A" "$B" "$JUDGE"; do [ -f "$f" ] || { echo "ABORT: missing $f"; exit 3; }; done
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
[ "$USED" -lt 1024 ] && [ "$APPS" -eq 0 ] || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=0 PC_NOCROSS=1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) python=$PY snap=$SNAP stage=$STAGE"

S=$CODE/method_search
PC=$S/train_checker.py
train_both() {  # <pool> <out> <steps>
  "$PY" "$PC" train --kind pair --mnist "$1" --out "$2/pair" --steps "$3" || return 1
  "$PY" "$PC" train --kind separate --mnist "$1" --out "$2/separate" --steps "$3" || return 1
}

case "$STAGE" in
  ladder)
    L=$OUT/ladder; mkdir -p "$L"
    "$PY" "$S/expert_pool.py" --a "$A" --b "$B" --train 30 --held 5 --out "$L/pool-1pct" || exit 10
    "$PY" "$S/checker_overfit.py" --pool "$L/pool-1pct" --out "$L/overfit.json" || exit 11
    train_both "$L/pool-1pct" "$L/checkers-1pct" 400 || exit 12
    "$PY" "$S/paper1_rows.py" --a "$A" --b "$B" --pair-head "$L/checkers-1pct/pair/head.pt" \
      --sep-head "$L/checkers-1pct/separate/head.pt" --judge "$JUDGE" --out "$L/rows-smoke" --pairs 4-7 --seeds 8 || exit 13
    rc=0 ;;
  full)
    F=$OUT/full; mkdir -p "$F"
    [ -f "$F/pool/pool.json" ] || "$PY" "$S/expert_pool.py" --a "$A" --b "$B" --train 3000 --held 500 --out "$F/pool" || exit 10
    [ -f "$F/checkers/separate/head.pt" ] || train_both "$F/pool" "$F/checkers" 4000 || exit 12
    "$PY" "$S/paper1_rows.py" --a "$A" --b "$B" --pair-head "$F/checkers/pair/head.pt" \
      --sep-head "$F/checkers/separate/head.pt" --judge "$JUDGE" --out "$F/rows" --wandb-name loop6-m1-paper1-rows
    rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
