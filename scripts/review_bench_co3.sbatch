#!/bin/bash
#SBATCH --job-name=r14-co3
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster68,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83,mscluster88
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/%x-%j.out
#
# Slot R14, compute job 13: CO3 at its published default on GenEval's two-object prompts, seeds 9 to 12,
# from the shared starting latent (1-lora-correction/scripts/review_baseline_cell.py). One bigbatch RTX 3090; SHARD=k/n
# takes every n-th cell from k, so two jobs split the 360 cells.
#
#   sbatch --export=ALL,SHARD=0/2 <copy>/scripts/review_bench_co3.sbatch
#
# The cell list comes from scripts/review_bench_sets.py beside this script; the CO3 code and the package
# come from the cluster checkout.
set -uo pipefail
HERE="${HERE:-/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers/bench13/scripts}"
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
SHARD="${SHARD:-0/1}"

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  shard $SHARD  start $(date -Is)  repo $REPO  here $HERE"
[ -d "$OUT" ] || { echo "ABORT: $OUT not visible from this node" >&2; exit 3; }
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted ($UTIL)" >&2; exit 4;; esac
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 6)" || { echo "ABORT: torch sees no CUDA" >&2; exit 6; }
echo "=== disk ${USE}%  gpu ${USED} MiB ${UTIL}%  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO" || exit 7
export POE_REPO="$REPO" REVIEW_BASELINE_OUT="$OUT"
K=${SHARD%/*}; N=${SHARD#*/}
CELLS="$OUT/launchers/bench13/co3_cells_${K}of${N}.txt"
"$PY" "$HERE/review_bench_sets.py" --co3-cells geneval_two | awk -v k="$K" -v n="$N" '(NR-1) % n == k' > "$CELLS"
echo "=== $(wc -l < "$CELLS") cells in $CELLS"
"$PY" 1-lora-correction/scripts/review_baseline_cell.py --method co3 --setting default --cells "$CELLS"; R=$?
echo "=== done $(date -Is)  exit $R  tiles $(find "$OUT/co3_default/geneval_two" -name co3.png | wc -l) of 360"
exit $R
