#!/bin/bash
#SBATCH --job-name=two_experts_fixed
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/two-experts-eval-fixed-%j.log
# Plan 32: each fixed-side expert alone (bar 58 of 64 per digit), then the fixed-side plain product under each base.
# The fixed-side control picks the base (review file, "The base"). Runs on card 1; the retrain holds card 0.
# Usage: sbatch scripts/mnist/two_experts_eval_fixed.sbatch   (biggpu allows one job per user: with another job running,
#   run it on mscluster85 instead: CUDA_DEV=0 nohup bash <this file> > <log> 2>&1 &)
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/two-experts; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-1}
nvidia-smi -L | grep -q GPU || { echo "GPU guard"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD)"
for k in e04-left:0-4 e59-right:5-9; do
  $PY $S alone --expert $M/${k%%:*}/unet.pt --digits ${k##*:} --name ${k%%:*} --out $OUT --judge $J
done
c() { $PY $S cells --a $M/e04-left/unet.pt --b $M/e59-right/unet.pt --push none --base $1 --tau ${2:-0} --name fixed-none-$1${2:-} --out $OUT --judge $J; }
c avgnull; c floor 0.3; c floor 1.0
echo "[$(date -u +%H:%M:%S)] done"
