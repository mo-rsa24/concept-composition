#!/bin/bash
#SBATCH --job-name=tome
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster68,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83,mscluster88
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/%x-%j.out
#
# ToMe on SDXL from its official code (scripts/review_tome_cell.py) for Table 1 (R1) and Figure 5, on one
# bigbatch RTX 3090: its pinned torch 2.0.1 has no Blackwell kernels. CELLS names a file of
# 'group|concept a|concept b|joiner|seed' lines; SHARD=k/n takes every n-th line from k.
#
#   sbatch --export=ALL,CELLS=<file>,SHARD=0/4 <copy>/scripts/review_tome.sbatch
set -uo pipefail
HERE="${HERE:-/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers/bench13/scripts}"
PY=/home-mscluster/mmolefe/miniforge3/envs/tome/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
SHARD="${SHARD:-0/1}"
: "${CELLS:?set CELLS to a cells file}"

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  cells $CELLS  shard $SHARD  start $(date -Is)"
[ -d "$OUT" ] || { echo "ABORT: $OUT not visible from this node" >&2; exit 3; }
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted ($UTIL)" >&2; exit 4;; esac
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 6)" || { echo "ABORT: torch sees no CUDA" >&2; exit 6; }
echo "=== disk ${USE}%  gpu ${USED} MiB ${UTIL}%  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

K=${SHARD%/*}; N=${SHARD#*/}
MINE="$OUT/launchers/bench13/$(basename "$CELLS" .txt)_${K}of${N}.txt"
awk -v k="$K" -v n="$N" '(NR-1) % n == k' "$CELLS" > "$MINE"
( while sleep 60; do nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | sed "s/^/mem $(date +%T) /"; done ) &
MON=$!
"$PY" "$HERE/review_tome_cell.py" --setting default --cells "$MINE"; R=$?
kill $MON 2>/dev/null
echo "=== done $(date -Is)  exit $R  tiles $(find "$OUT/tome_default" -name tome.png | wc -l)"
exit $R
