#!/bin/bash
# Slot R14 (Table 2): score each named set with its benchmark's own evaluator as soon as all four
# methods (joint, PoE, ours, CO3) hold every image, whichever set completes first, then rebuild the table. GenEval sets go through
# review_bench_geneval.py (geneval env), T2I-CompBench sets through review_bench_t2i.py (t2icomp env).
# Runs on one card that must be empty; never shares one.
#
#   ssh <node> 'GPU=0 setsid nohup bash <copy>/scripts/review_bench_score.sh geneval_two geneval_colorattr \
#       > <OUT>/logs/table2-score-<node>.log 2>&1 < /dev/null &'
#
# Writes <OUT>/bench_scores/<set>_images.csv (one row per image) and, after every set,
# <OUT>/bench_scores/standard_benchmarks.csv from every <set>_images.csv present.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
S="$OUT/bench_scores"
GE=/datasets/mmolefe/envs/geneval/bin/python
T2=/datasets/mmolefe/envs/t2icomp/bin/python
: "${GPU:?set GPU to the device index}"
[ $# -ge 1 ] || { echo "usage: GPU=n $0 <set> [<set> ...]" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU" REVIEW_BASELINE_OUT="$OUT"
mkdir -p "$S" || exit 2
echo "=== table2-score  node $(hostname -s)  device $GPU  pid $$  sets $*  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }

expected() { "$GE" -c "import sys; sys.path.insert(0, '$HERE'); from review_bench_sets import SETS, load; print(len(load('$1')[0]) * len(SETS['$1']['seeds']))"; }
counts() {  # non-empty renders per method, "joint poe ours co3"
  local s=$1
  echo "$(find "$OUT/bench_joint/$s" -name joint.png -size +0 2>/dev/null | wc -l)" \
       "$(find "$OUT/bench_poe/$s" -name poe.png -size +0 2>/dev/null | wc -l)" \
       "$(find "$OUT/bench_ours/$s" -name 'ours_*_w50.png' -size +0 2>/dev/null | wc -l)" \
       "$(find "$OUT/co3_default/$s" -name co3.png -size +0 2>/dev/null | wc -l)"
}
complete() {  # 0 when every method holds the expected count of non-empty renders
  local s=$1 n c; n=$(expected "$s"); c=$(counts "$s")
  if [ "$(echo $c | tr ' ' '\n' | awk -v n=$n '$1 >= n' | wc -l)" -eq 4 ]; then
    echo "=== $s complete: joint poe ours co3 = $c of $n each $(date -Is)"; return 0
  fi
  [ "${TICK:-0}" -eq 0 ] && echo "... $s: joint poe ours co3 = $c of $n $(date -Is)"
  return 1
}
card_free() {  # never share a card: this device must hold under 1 GB
  for _ in $(seq 1 60); do
    read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; return 1;; esac
    [ "$USED" -lt 1024 ] && return 0
    echo "... device $GPU holds $USED MiB; waiting"; sleep 60
  done
  echo "ABORT: device $GPU stayed busy for 1 h" >&2; return 1
}

score() {
  local SET=$1
  card_free || exit 4
  rm -rf "$S/${SET}_work"
  case "$SET" in
    geneval_*) "$GE" "$HERE/review_bench_geneval.py" --set "$SET" --work "$S/${SET}_work" --out "$S/${SET}_images.csv" ;;
    t2i_*) "$T2" "$HERE/review_bench_t2i.py" --set "$SET" --work "$S/${SET}_work" --out "$S/${SET}_images.csv" ;;
  esac || { echo "$SET FAILED $(date -Is)"; return 1; }
  IMAGES=$(ls "$S"/{geneval_two,geneval_colorattr,geneval_position,t2i_color,t2i_shape,t2i_texture}_images.csv 2>/dev/null)
  "$GE" "$HERE/review_bench_table.py" --images $IMAGES --out "$S/standard_benchmarks.csv"
  echo "=== $SET scored $(date -Is)"
}

# Score whichever pending set completes first, in the order given; poll every minute for 8 h at most.
PENDING=("$@")
for i in $(seq 0 479); do
  TICK=$((i % 10)) LEFT=()
  for SET in "${PENDING[@]}"; do
    if complete "$SET"; then score "$SET"; else LEFT+=("$SET"); fi
  done
  PENDING=("${LEFT[@]}")
  [ ${#PENDING[@]} -eq 0 ] && break
  sleep 60
done
[ ${#PENDING[@]} -gt 0 ] && echo "ABORT: still incomplete after 8 h: ${PENDING[*]}" >&2
echo "=== done $(date -Is) -> $S/standard_benchmarks.csv"
