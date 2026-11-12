#!/bin/bash
# Compute job 13, slot R14 (report/paper-review/prze-10-compute.md), on one card by
# the shared-device path; SHARD=k/n renders every n-th prompt from k in every step, so n cards split it. In order: the joint prompt on GenEval's two-object and counting prompts, then
# PoE and ours (the adapter of record on all 50 steps) on the two-object prompts, then the joint prompt on
# T2I-CompBench's spatial and complex validation prompts (10 images each). CO3 on the two-object
# prompts runs separately on bigbatch (scripts/review_bench_co3.sbatch).
#
#   timeout 30 ssh mscluster112 'GPU=0 SHARD=0/3 nohup bash <copy>/scripts/review_bench_renders.sh \
#       > /datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/bench-geneval-112.log 2>&1 < /dev/null & disown'
#
# Prompts, seeds and dropped prompts: scripts/review_bench_sets.py. A tile that exists is skipped, so a
# relaunch resumes. The new scripts and the benchmark prompt files are read from beside this script
# (<copy>/scripts, <copy>/data/prompt-sets), so a copy on /datasets runs against the cluster
# checkout's package and renderers without that checkout being synced, as wave 1 did.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
CKPT=$(awk '/^checkpoint:/ {print $2}' "$REPO/artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml")
TAG=$(awk '/^candidate:/ {print $2}' "$REPO/artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml")
: "${GPU:?set GPU to the device index}"
SHARD="${SHARD:-0/1}"; K=${SHARD%/*}; N=${SHARD#*/}
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO" REVIEW_BASELINE_OUT="$OUT"
mkdir -p "$OUT/logs" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }

echo "=== bench-geneval  here $HERE  node $(hostname -s)  device $GPU  shard $SHARD  pid $$  python $PY  repo $REPO  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 2; }
cd "$REPO" || exit 7
echo "=== disk ${USE}%  adapter $TAG $CKPT"

mapfile -t ALL < <("$PY" "$HERE/review_bench_sets.py" --pairs geneval_two)
[ "${#ALL[@]}" -eq 99 ] || { echo "ABORT: ${#ALL[@]} two-object pairs, not 99" >&2; exit 2; }
PAIRS=(); for i in "${!ALL[@]}"; do [ $(( i % N )) -eq "$K" ] && PAIRS+=("${ALL[$i]}"); done
echo "=== shard $SHARD: ${#PAIRS[@]} of 99 two-object pairs for PoE and ours"
SEEDS="9 10 11 12"

echo "=== joint geneval_two $(date -Is)";   "$PY" "$HERE/review_bench_joint.py" --set geneval_two   --shard "$SHARD" || echo "joint geneval_two FAILED"
echo "=== joint geneval_count $(date -Is)"; "$PY" "$HERE/review_bench_joint.py" --set geneval_count --shard "$SHARD" || echo "joint geneval_count FAILED"
echo "=== poe geneval_two $(date -Is)"
CONCEPT_PAIR_OUT="$OUT/bench_poe/geneval_two" "$PY" scripts/concept_pair_sheet.py --column poe --pairs "${PAIRS[@]}" --seeds $SEEDS \
  || echo "poe geneval_two FAILED"
echo "=== ours geneval_two $(date -Is)"
FIGURE_CANDIDATES_OUT="$OUT/bench_ours/geneval_two" "$PY" scripts/figure_candidates.py --column ours --pairs "${PAIRS[@]}" \
  --seeds $SEEDS --checkpoint "$CKPT" --window 50 --tag "$TAG" || echo "ours geneval_two FAILED"
for SET in t2i_spatial t2i_complex; do
  echo "=== joint $SET $(date -Is)"; "$PY" "$HERE/review_bench_joint.py" --set $SET --shard "$SHARD" || echo "joint $SET FAILED"
done
echo "=== done $(date -Is)  t2i joint $(find "$OUT/bench_joint/t2i_spatial" "$OUT/bench_joint/t2i_complex" -name joint.png | wc -l) of 5970  joint $(find "$OUT/bench_joint/geneval_two" "$OUT/bench_joint/geneval_count" -name joint.png | wc -l) of 716" \
     " poe $(find "$OUT/bench_poe/geneval_two" -name poe.png | wc -l) of 396  ours $(find "$OUT/bench_ours/geneval_two" -name 'ours_*.png' | wc -l) of 396"
