#!/bin/bash
#SBATCH --job-name=l6-m1-p2
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82,mscluster48,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M1, Paper 2. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m1_paper2.sbatch <check|smoke|render> [shard num_shards]
set -uo pipefail
STAGE=${1:?stage}; SHARD=${2:-0}; NUM=${3:-1}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
OUT=$ROOT/m1/paper-2
HEAD=/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2/readout_pair/readout-multipair/head_step_002000.pt
CO3=/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f/upstream/co3
: "${SNAP:?SNAP must name the code snapshot}"
REPO=$SNAP/repo
HUB=$SNAP/hub/scripts/reproductions
SPLITS=$REPO/2-classifier-guidance/scripts/checker_choice/splits.json
S=$REPO/2-classifier-guidance/scripts/method_search
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -d "$CO3" ] && [ -f "$S/paper2_rows.py" ] && [ -f "$HUB/co3_common.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
[ -f "$HEAD" ] && [ -f "$SPLITS" ] || { echo "ABORT: missing checker or splits"; exit 3; }
[ -f "$REPO/scripts/shared/disallowed_subjects.py" ] || { echo "ABORT: subject guard missing from the snapshot"; exit 8; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
[ "$USED" -lt 1024 ] && [ "$APPS" -eq 0 ] || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) python=$PY snap=$SNAP stage=$STAGE shard=$SHARD/$NUM"
export PYTHONPATH="$HUB:$CO3:$REPO/scripts/shared:$REPO"
mkdir -p "$OUT"
case "$STAGE" in
  check)  "$PY" "$S/paper2_rows.py" check --head "$HEAD" --splits "$SPLITS" --out "$OUT/controls"; rc=$? ;;
  smoke)  "$PY" "$S/paper2_rows.py" render --head "$HEAD" --splits "$SPLITS" --out "$OUT/smoke" --pairs-limit 1 --seeds-limit 1; rc=$? ;;
  render) "$PY" "$S/paper2_rows.py" render --head "$HEAD" --splits "$SPLITS" --out "$OUT/full" --shard "$SHARD" --num-shards "$NUM"; rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
