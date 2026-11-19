#!/bin/bash
#SBATCH --job-name=two_experts_eval
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/two-experts-eval-%j.log
# Plan 32: each side-mixed expert alone (bar 58 of 64 per digit), then the side-mixed plain product under each base.
# The plain-product counts are rendered here and left unread until the fixed-side control has picked the base.
# Usage: sbatch scripts/mnist/two_experts_eval.sbatch <alone|cells>
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/two-experts; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
nvidia-smi -L | grep -q GPU || { echo "GPU guard"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD)"
a() { CUDA_VISIBLE_DEVICES=$1 $PY $S alone --expert $M/$2/unet.pt --digits $3 --name $2 --out $OUT --judge $J; }
c() { CUDA_VISIBLE_DEVICES=$1 $PY $S cells --a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --push none --base $2 --tau ${3:-0} --name side-none-$2${3:-} --out $OUT --judge $J; }
( a 0 e04-either 0-4 && c 0 avgnull ) > $ROOT/logs/two-experts-eval-c0.log 2>&1 &
( a 1 e59-either 5-9 && c 1 floor 0.3 && c 1 floor 1.0 ) > $ROOT/logs/two-experts-eval-c1.log 2>&1 &
wait
echo "[$(date -u +%H:%M:%S)] done"
