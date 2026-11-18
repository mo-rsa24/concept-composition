#!/bin/bash
#SBATCH --job-name=mnist_five
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/five-%j.log
# Plan 24 task 1.6 (and the cap sweep, task 1.7): the five-version sheet on seeds 0 to 63.
# Usage: sbatch [--nodelist=<node>] [--dependency=...] scripts/mnist/five_versions.sbatch <out name> [five_versions.py flags]
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
NAME=$1; shift
cd "$REPO"
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
USED=$(df --output=pcent "$ROOT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $ROOT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) name=$NAME args=$*"
"$PY" 2-classifier-guidance/scripts/mnist/five_versions.py --models "$ROOT/models" --checkers "$ROOT/checkers" \
    --judge "$ROOT/judge/judge.pt" --out "$ROOT/five-versions/$NAME" "$@"
echo "[$(date -u +%H:%M:%S)] done"
