#!/bin/bash
#SBATCH --job-name=l6-m7-train
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --exclude=mscluster107
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M7: train one unconditional edm2-img64-xs model and sample one batch from it. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m7_train.sbatch <rung> <set> <images seen> [batch, default 2048]
set -uo pipefail
RUNG=${1:?rung}; SET=${2:?set}; NIMG=${3:?images seen}; BATCH=${4:-2048}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
E=/home-mscluster/mmolefe/Playground/PhD/projective-composition/reference/edm2
DATA=$ROOT/m7/full/data/$SET.zip
OUT=$ROOT/m7/$RUNG/models/$SET
: "${SNAP:?SNAP must name the code snapshot}"
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
[ -f "$E/train_edm2.py" ] && [ -f "$DATA" ] && [ -x "$PY" ] || { echo "ABORT: EDM2, $DATA or python missing on $(hostname -s)"; exit 3; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
read -r USED UTIL TOTAL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu,memory.total --format=csv,noheader,nounits -i 0 | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
[ "$USED" -lt 1024 ] && [ "$APPS" -eq 0 ] || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
BATCH_GPU=128
[ "$TOTAL" -ge 40000 ] && BATCH_GPU=256
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) memory=${TOTAL}MiB batch_gpu=$BATCH_GPU python=$PY snap=$SNAP rung=$RUNG set=$SET nimg=$NIMG"
mkdir -p "$OUT"
cd "$E" || exit 3
# EDM2 needs every interval to be a whole number of batches.
[ $((NIMG % BATCH)) -eq 0 ] || { echo "ABORT: $NIMG is not a multiple of the $BATCH-image batch"; exit 2; }
[ "$BATCH_GPU" -gt "$BATCH" ] && BATCH_GPU=$BATCH
STATUS=$((NIMG / 8 / BATCH * BATCH)); [ "$STATUS" -lt "$BATCH" ] && STATUS=$BATCH
M7_NIMG="$NIMG" "$PY" "$SNAP/repo/2-classifier-guidance/scripts/method_search/edm2_scaled_ramp.py" --outdir="$OUT" --data="$DATA" --cond=False --preset=edm2-img64-xs --duration="$NIMG" --batch="$BATCH" --batch-gpu="$BATCH_GPU" \
  --status="$STATUS" --snapshot="$NIMG" --checkpoint="$NIMG" --seed=20261124
rc=$?
echo "[$(date -u +%H:%M:%S)] TRAIN exit=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
WANDB__SERVICE_WAIT=300 "$PY" "$SNAP/repo/2-classifier-guidance/scripts/method_search/edm2_log_wandb.py" "$OUT" "m7-edm2-$RUNG-$SET" \
  || echo "W&B logging failed; the stats stay in $OUT/stats.jsonl"
NET=$(ls -t "$OUT"/network-snapshot-*.pkl 2>/dev/null | head -1)
[ -n "$NET" ] || { echo "ABORT: no network snapshot in $OUT"; exit 7; }
"$PY" generate_images.py --net="$NET" --outdir="$OUT/samples" --seeds=0-15 --steps=32 --batch=16
rc=$?
echo "[$(date -u +%H:%M:%S)] DONE rung=$RUNG set=$SET net=$NET sample_exit=$rc"
exit $rc
