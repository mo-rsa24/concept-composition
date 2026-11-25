#!/bin/bash
#SBATCH --job-name=l6-m7-sample
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclude=mscluster106,mscluster107,mscluster108,mscluster109,mscluster50,mscluster51,mscluster71,mscluster82,mscluster48
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M7: draw 1,024 samples from one trained model with EDM2's 32-step Heun sampler and no guidance.
#   sbatch --export=ALL,SNAP=<snapshot> m7_sample.sbatch <rung> <set> <first seed>
# Each model draws from its own seeds: position 1 from 10000, position 2 from 20000, position 3 from 30000, the empty scene from 40000.
set -uo pipefail
RUNG=${1:?rung}; SET=${2:?set}; FIRST=${3:?first seed}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
E=/home-mscluster/mmolefe/Playground/PhD/projective-composition/reference/edm2
MODELS=$ROOT/m7/$RUNG/models/$SET
OUT=$ROOT/m7/$RUNG/samples/$SET
: "${SNAP:?SNAP must name the code snapshot}"
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
NET=$(ls "$MODELS"/network-snapshot-*-0.100.pkl 2>/dev/null | sort | tail -1)
[ -n "$NET" ] && [ -x "$PY" ] || { echo "ABORT: no trained network in $MODELS or no python"; exit 3; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
[ "$USED" -lt 1024 ] && [ "$APPS" -eq 0 ] || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=0
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) net=$NET seeds=$FIRST-$((FIRST + 1023)) snap=$SNAP"
mkdir -p "$OUT"
cd "$E" || exit 3
"$PY" generate_images.py --net="$NET" --outdir="$OUT" --seeds="$FIRST-$((FIRST + 1023))" --steps=32 --batch=256
rc=$?
COUNT=$(ls "$OUT"/*.png 2>/dev/null | wc -l)
echo "[$(date -u +%H:%M:%S)] DONE rung=$RUNG set=$SET exit=$rc samples=$COUNT"
[ "$rc" -eq 0 ] && [ "$COUNT" -eq 1024 ] || exit 9
