#!/bin/bash
#SBATCH --job-name=l6-m9-r1
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster50,mscluster51,mscluster71,mscluster82,mscluster48
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M9, rung 1. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m9_rung1.sbatch <smoke|full|boot-smoke|boot-full>
# smoke: both stages on seed 9. full: the reference process, then the regional process, on all 8 seeds.
# boot-smoke and boot-full: the four region rows with bootstrapping on, on seed 9 or on all 8 seeds.
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/repo/2-classifier-guidance/scripts/method_search/m9_rung1.py
BOXES=$ROOT/m9/rung1-input/boxes.json
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
[ -f "$S" ] && [ -f "$BOXES" ] || { echo "ABORT: snapshot or boxes missing on $(hostname -s)"; exit 3; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
[ "$USED" -lt 1024 ] && [ "$APPS" -eq 0 ] || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SNAP/repo:$SNAP/repo/scripts/shared"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) python=$PY snap=$SNAP stage=$STAGE"
case "$STAGE" in
  smoke)      OUT=$ROOT/m9/smoke; SEEDS="--seeds 9"; PARTS="refs regional" ;;
  full)       OUT=$ROOT/m9/full;  SEEDS="";          PARTS="refs regional" ;;
  boot-smoke) OUT=$ROOT/m9/smoke; SEEDS="--seeds 9"; PARTS="bootstrap" ;;
  boot-full)  OUT=$ROOT/m9/full;  SEEDS="";          PARTS="bootstrap" ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
for PART in $PARTS; do
  "$PY" "$S" "$PART" --boxes "$BOXES" --out "$OUT" $SEEDS || { echo "[$(date -u +%H:%M:%S)] DONE part=$PART exit=11"; exit 11; }
done
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=0"
