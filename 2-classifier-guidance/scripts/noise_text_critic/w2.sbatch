#!/bin/bash
#SBATCH --job-name=l9-w2
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-9/logs/%x-%j.log
# Step W2. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> w2.sbatch <prepare-smoke|prepare|read-smoke|read|probe-smoke|probe|fair-smoke|fair>
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

case "$STAGE" in
  prepare-smoke) "$PY" "$S/w2_prepare.py" --smoke; rc=$? ;;
  prepare)       "$PY" "$S/w2_prepare.py"; rc=$? ;;
  read-smoke)    "$PY" "$S/w2_read.py" --smoke; rc=$? ;;
  read)          "$PY" "$S/w2_read.py"; rc=$? ;;
  probe-smoke)   "$PY" "$S/w2b_probe.py" extract --smoke --out "$ROOT/w2b/smoke" && "$PY" "$S/w2b_probe.py" fit --out "$ROOT/w2b/smoke"; rc=$? ;;
  probe)         "$PY" "$S/w2b_probe.py" extract --out "$ROOT/w2b/full" && "$PY" "$S/w2b_probe.py" fit --out "$ROOT/w2b/full"; rc=$? ;;
  fair-smoke)    "$PY" "$S/w2c_fair.py" --smoke; rc=$? ;;
  fair)          "$PY" "$S/w2c_fair.py"; rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
