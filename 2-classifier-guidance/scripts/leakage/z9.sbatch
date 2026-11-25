#!/bin/bash
#SBATCH --job-name=l12-z9
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster48,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-12/logs/%x-%j.log
# Loop 12 step Z9. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> z9.sbatch <first|experts|plain>
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-12
OUT=$ROOT/z9
MNIST=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/mnist-raw
CO3=/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f/upstream/co3
HUB=/datasets/mmolefe/poe_repair_min/loop-6/code/f0410a14-f2bdb6d/hub/scripts/reproductions
: "${SNAP:?SNAP must name the code snapshot}"
REPO=$SNAP/repo
SPLIT=$REPO/artifacts/notes/progress/leakage/09-dcfm-colored-mnist/split.json
S=$REPO/2-classifier-guidance/scripts/leakage
SPLITS=$REPO/2-classifier-guidance/scripts/checker_choice/splits.json
BOXES=$REPO/artifacts/notes/progress/leakage/03-one-fix/boxes.json
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
[ -d "$ROOT" ] && [ -d "$CO3" ] && [ -f "$S/z9_train.py" ] && [ -f "$S/z9_sample.py" ] && [ -f "$MNIST/train-images-idx3-ubyte" ] && [ -f "$SPLIT" ] \
  || { echo "ABORT: snapshot, M1 renders or $ROOT not visible on $(hostname -s)"; exit 3; }
[ -f "$REPO/scripts/shared/disallowed_subjects.py" ] || { echo "ABORT: subject guard missing from the snapshot"; exit 8; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
DEV=${GPU:-0}   # the approved card on a shared-device run, else the card Slurm gave
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $DEV | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
# the guard board question 103 approved, scripts/launch_sdlora.sh lines 33 to 40: under 100 MiB, or under 4096 MiB at 0% utilisation
if [ "$USED" -lt 100 ]; then :;
elif [ "$USED" -lt 4096 ] && [ "$UTIL" -eq 0 ]; then :;
else echo "ABORT: device $DEV has ${USED} MiB in use at ${UTIL}% utilisation, refusing to share"; exit 4; fi
echo "guard reading: device $DEV ${USED} MiB at ${UTIL}% utilisation"
export CUDA_VISIBLE_DEVICES=$DEV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $DEV) device=$DEV pid=$$ python=$PY snap=$SNAP stage=$STAGE"
export PYTHONPATH="$HUB:$CO3:$REPO/scripts/shared:$REPO"
mkdir -p "$OUT"
case "$STAGE" in
  first)   "$PY" "$S/z9_train.py" judge --mnist "$MNIST" --split "$SPLIT" --out "$OUT/judge" && \
           "$PY" "$S/z9_train.py" expert --mnist "$MNIST" --split "$SPLIT" --holder 0 --out "$OUT/experts" --time 200; rc=$? ;;
  experts) "$PY" "$S/z9_train.py" expert --mnist "$MNIST" --split "$SPLIT" --holder 0 --out "$OUT/experts" --steps "${STEPS:-40000}" && \
           "$PY" "$S/z9_train.py" expert --mnist "$MNIST" --split "$SPLIT" --holder 1 --out "$OUT/experts" --steps "${STEPS:-40000}"; rc=$? ;;
  plain)   PYTHONPATH="$ROOT/pylib:$PYTHONPATH" TORCH_HOME=$ROOT/torchhome "$PY" "$S/z9_sample.py" plain --mnist "$MNIST" --split "$SPLIT" \
             --experts "$OUT/experts" --judge "$OUT/judge/judge.pt" --out "$OUT"; rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
