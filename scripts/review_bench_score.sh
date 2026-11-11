#!/bin/bash
# Compute job 13, slot R14: score the renders with each benchmark's own evaluator once they exist, then
# write the table. GenEval first (1,756 images: joint 676, PoE 360, ours 360, CO3 360), then
# T2I-CompBench spatial (2,640 joint) and complex (2,960 joint). Runs on one non-Blackwell card, since
# the geneval (torch 1.12) and t2icomp (torch 2.0.1) envs have no Blackwell kernels.
#
#   timeout 30 ssh mscluster108 'GPU=1 nohup bash <copy>/scripts/review_bench_score.sh > <OUT>/logs/bench13-score.log 2>&1 < /dev/null & disown'
#
# Writes <OUT>/bench_scores/{geneval_images.csv, t2i_spatial_images.csv, t2i_complex_images.csv,
# standard_benchmarks.csv}. standard_benchmarks.csv is written after GenEval with the T2I-CompBench
# cells empty, and again once both T2I-CompBench sets are scored.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
S="$OUT/bench_scores"
GE=/datasets/mmolefe/envs/geneval/bin/python
T2=/datasets/mmolefe/envs/t2icomp/bin/python
: "${GPU:?set GPU to the device index}"
export CUDA_VISIBLE_DEVICES="$GPU" REVIEW_BASELINE_OUT="$OUT"
mkdir -p "$S" || exit 2
echo "=== bench-score  node $(hostname -s)  device $GPU  pid $$  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }

count() { find "$@" -name '*.png' 2>/dev/null | wc -l; }
wait_for() {  # wait_for <label> <target> <dirs...>: until the count reaches the target, 16 h at most
  local label=$1 target=$2; shift 2
  for _ in $(seq 1 960); do
    local n; n=$(count "$@"); [ "$n" -ge "$target" ] && { echo "=== $label: $n of $target $(date -Is)"; return 0; }
    sleep 60
  done
  echo "ABORT: $label reached $(count "$@") of $target in 16 h" >&2; return 1
}
card_free() {  # never share a card: wait until this device holds under 1 GB
  for _ in $(seq 1 240); do
    read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; return 1;; esac
    [ "$USED" -lt 1024 ] && return 0
    sleep 60
  done
  echo "ABORT: device $GPU stayed busy for 4 h" >&2; return 1
}

wait_for "GenEval renders" 1756 "$OUT/bench_joint/geneval_two" "$OUT/bench_joint/geneval_count" \
  "$OUT/bench_poe/geneval_two" "$OUT/bench_ours/geneval_two" "$OUT/co3_default/geneval_two" || exit 5
card_free || exit 4
rm -rf "$S/geneval_work"
"$GE" "$HERE/review_bench_geneval.py" --work "$S/geneval_work" --out "$S/geneval_images.csv" || { echo "GenEval FAILED"; exit 6; }
python3 "$HERE/review_bench_table.py" --geneval "$S/geneval_images.csv" --out "$S/standard_benchmarks.csv"

for SET in t2i_spatial t2i_complex; do
  N=$([ $SET = t2i_spatial ] && echo 2640 || echo 2960)
  wait_for "$SET renders" $N "$OUT/bench_joint/$SET" || exit 5
  card_free || exit 4
  rm -rf "$S/${SET}_work"
  "$T2" "$HERE/review_bench_t2i.py" --set $SET --work "$S/${SET}_work" --out "$S/${SET}_images.csv" || echo "$SET FAILED"
done
python3 "$HERE/review_bench_table.py" --geneval "$S/geneval_images.csv" \
  --t2i $(ls "$S"/t2i_*_images.csv 2>/dev/null) --out "$S/standard_benchmarks.csv"
echo "=== done $(date -Is) -> $S/standard_benchmarks.csv"
