#!/bin/bash
#SBATCH --job-name=l6-m11-vote
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster84,mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82,mscluster48,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M11, Paper 1 on digits: the experts' vote against a coin-flip side, with and without the corrector, and with no held side,
# beside paste. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m11_vote.sbatch <smoke|full>
# smoke: digits 4 and 7 at 8 seeds, every arm, with the held-side control. full: five tuning pairs at seeds 192 to 255.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
OUT=$ROOT/m11
CK=$ROOT/m1/paper-1/full/checkers
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

[ -d "$ROOT" ] && [ -f "$CODE/method_search/m11_vote.py" ] && [ -f "$CK/pair/head.pt" ] && [ -f "$CK/separate/head.pt" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
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
COMMON=(--a "$A" --b "$B" --pair-head "$CK/pair/head.pt" --sep-head "$CK/separate/head.pt" --judge "$JUDGE")
case "$STAGE" in
  smoke) "$PY" "$S/m11_vote.py" "${COMMON[@]}" --out "$OUT/smoke" --pairs 4-7 --seeds 8; rc=$? ;;
  full)  "$PY" "$S/m11_vote.py" "${COMMON[@]}" --out "$OUT/full" --wandb-name loop6-m11-vote; rc=$? ;;
  shared-floor) # the shared-canvas experiment again over the "floor" base, after the first run's "blank" base ran off to full ink
         "$PY" "$S/m11_vote.py" "${COMMON[@]}" --out "$OUT/shared-floor" --arms shared --shared-base floor --wandb-name loop6-m11-shared-floor; rc=$? ;;
  diag-bare)      "$PY" "$S/m11_vote.py" "${COMMON[@]}" --out "$OUT/diagnosis/bare" --arms shared --shared-base floor --shared-no-corrector --shared-no-pushes; rc=$? ;;
  diag-corrector) "$PY" "$S/m11_vote.py" "${COMMON[@]}" --out "$OUT/diagnosis/corrector-only" --arms shared --shared-base floor --shared-no-pushes; rc=$? ;;
  diag-pushes)    "$PY" "$S/m11_vote.py" "${COMMON[@]}" --out "$OUT/diagnosis/pushes-only" --arms shared --shared-base floor --shared-no-corrector; rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
