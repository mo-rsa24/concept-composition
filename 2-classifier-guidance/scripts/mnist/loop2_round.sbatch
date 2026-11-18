#!/bin/bash
#SBATCH --job-name=mnist_loop2
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/loop2-%j.log
# Plan 24 loop 2, one round: train the checker without the three tuning pairs, render those pairs at
# s = 3 and 10, and count them with the loop-1 judge.
# Usage: sbatch scripts/mnist/loop2_round.sbatch <round name> [extra checker.py train flags]
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
NAME=$1; shift
DEV="0,6 1,2 6,9"
cd "$REPO"
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
USED=$(df --output=pcent "$ROOT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $ROOT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
M=2-classifier-guidance/scripts/mnist
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) round=$NAME flags=$*"
"$PY" $M/checker.py train --kind pair --models "$ROOT/models" --draws "$ROOT/checkers/draws-v3.npz" --mnist "$ROOT/mnist-raw" \
    --out "$ROOT/checkers/loop2-$NAME" --exclude-pairs "$DEV" "$@"
"$PY" $M/five_versions.py --models "$ROOT/models" --checkers "$ROOT/checkers" --judge "$ROOT/judge/judge.pt" \
    --out "$ROOT/five-versions/loop2-$NAME" --pairs "$DEV" --pair-only --s-list 3,10 --pair-head "loop2-$NAME"
"$PY" $M/count_v3.py --judge "$ROOT/judge-v3/judge.pt" canvases "$ROOT/five-versions/loop2-$NAME"
echo "[$(date -u +%H:%M:%S)] done"
