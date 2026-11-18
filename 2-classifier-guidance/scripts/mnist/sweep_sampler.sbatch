#!/bin/bash
#SBATCH --job-name=mnist_sampler
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/sampler-%j.log
# Plan 24 loop 2 rounds 6 and 7. Usage: sbatch scripts/mnist/sweep_sampler.sbatch <out name> <head> <setting> ...
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
NAME=$1; HEAD=$2; shift 2
cd "$REPO"
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) head=$HEAD settings=$*"
"$PY" 2-classifier-guidance/scripts/mnist/sweep_sampler.py --models "$ROOT/models" --checkers "$ROOT/checkers" \
    --judge3 "$ROOT/judge-v3/judge.pt" --head "$HEAD" ${PAIRS:+--pairs "$PAIRS"} --out "$ROOT/five-versions/$NAME" "$@"
echo "[$(date -u +%H:%M:%S)] done"
