#!/bin/bash
#SBATCH --job-name=cg_canary
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/concept-composition/logs/cg_canary_%j.log
# Route step 5: does the repaired classifier-guided product sampler, at strength 0, render cat x dog
# byte for byte like plain product? Writes canary.json under OUT.
# Usage: sbatch [--nodelist=<node>] 2-classifier-guidance/scripts/presence_head_canary.sbatch [seed]
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
OUT=/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary
SEED=${1:-9}
cd "$REPO"

[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
mkdir -p "$OUT"
USED=$(df --output=pcent "$OUT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $OUT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) disk=${USED}% commit=$(git rev-parse --short HEAD) seed=$SEED"

$PY 2-classifier-guidance/scripts/presence_head_canary.py --seed "$SEED"
echo "[$(date -u +%H:%M:%S)] done"
