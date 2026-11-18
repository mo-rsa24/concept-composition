#!/bin/bash
#SBATCH --job-name=mnist_onpolicy
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/onpolicy-%j.log
# Plan 24 loop 2 round 9: render a checker's push on its own training pairs and add them to the draws.
# Usage: sbatch scripts/mnist/onpolicy.sbatch <head> <s> <draws in name> <draws out name> <seed0>
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
cd "$REPO"
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) args=$*"
"$PY" 2-classifier-guidance/scripts/mnist/onpolicy.py --models "$ROOT/models" --checkers "$ROOT/checkers" \
    --judge3 "$ROOT/judge-v3/judge.pt" --head "$1" --s "$2" --draws "$ROOT/checkers/$3" --out "$ROOT/checkers/$4" --seed0 "$5" --per-pair "${6:-12}"
echo "[$(date -u +%H:%M:%S)] done"
