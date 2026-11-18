#!/bin/bash
#SBATCH --job-name=mnist_checker
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/checker-%j.log
# Plan 24 task 1.5 and 1.7: the checker's draws, or one or more checker trainings run in turn.
# Usage: sbatch [--nodelist=<node>] [--dependency=...] scripts/mnist/checker.sbatch draws
#        sbatch ... scripts/mnist/checker.sbatch train "<name> <checker.py train flags>" ["<name> <flags>" ...]
#   e.g. checker.sbatch train "pair-full --kind pair" "separate --kind separate"
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
MODE=$1; shift
cd "$REPO"
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
USED=$(df --output=pcent "$ROOT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $ROOT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) mode=$MODE"
C=2-classifier-guidance/scripts/mnist/checker.py
if [ "$MODE" = draws ]; then
  "$PY" $C draws --models "$ROOT/models" --judge "$ROOT/judge/judge.pt" --out "$ROOT/checkers/draws.npz"
else
  for run in "$@"; do
    name=${run%% *}; flags=${run#* }
    echo "[$(date -u +%H:%M:%S)] training $name: $flags"
    "$PY" $C train --models "$ROOT/models" --draws "$ROOT/checkers/draws.npz" --mnist "$ROOT/mnist-raw" \
        --out "$ROOT/checkers/$name" $flags
  done
fi
echo "[$(date -u +%H:%M:%S)] done"
