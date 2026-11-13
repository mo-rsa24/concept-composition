#!/bin/bash
#SBATCH --job-name=r11-three
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster68,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83,mscluster88
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/%x-%j.out
#
# Slot R11, three concepts: four kinds of triple (columns) by joint prompt, plain product and ours
# (rows) at the shared seed 10, then the sheet. One bigbatch RTX 3090, twelve tiles.
#
#   sbatch 1-lora-correction/scripts/review_three_concepts.sbatch      (from the cluster checkout)
set -uo pipefail
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/concept-composition}"
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)  repo $REPO"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted ($UTIL)" >&2; exit 4;; esac
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 6)" || { echo "ABORT: torch sees no CUDA" >&2; exit 6; }
echo "=== disk ${USE}%  gpu ${USED} MiB ${UTIL}%  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO" || exit 7
export POE_REPO="$REPO"
"$PY" 1-lora-correction/scripts/review_three_concepts.py --render; R=$?
"$PY" 1-lora-correction/scripts/review_three_concepts.py --sheet; S=$?
echo "=== done $(date -Is)  render exit $R  sheet exit $S  tiles $(find "$OUT/three_concepts" -name '*.png' ! -name 'three_concepts.png' | wc -l) of 12"
exit $(( R || S ))
