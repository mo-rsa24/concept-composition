#!/bin/bash
#SBATCH --job-name=mnist_unet
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/unet-%j.log
# Plan 24 (route step 7b): time or train one canvas U-Net. Writes to the scratch folder because
# /datasets is down (environment/storage.md, "When /datasets is down").
# Usage: sbatch [--nodelist=<node>] scripts/mnist/canvas_unet.sbatch <share> [canvas_unet.py train flags]
#   timing: sbatch ... canvas_unet.sbatch 0.1 --time 200
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
SHARE=$1; shift
cd "$REPO"
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
[ -f "$ROOT/mnist-raw/train-images-idx3-ubyte" ] || { echo "MNIST raw files missing under $ROOT/mnist-raw"; exit 2; }
USED=$(df --output=pcent "$ROOT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $ROOT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) disk=${USED}% commit=$(git rev-parse --short HEAD) share=$SHARE args=$*"
"$PY" 2-classifier-guidance/scripts/mnist/canvas_unet.py train --mnist "$ROOT/mnist-raw" --share "$SHARE" \
    --out "$ROOT/models" "$@"
echo "[$(date -u +%H:%M:%S)] done"
