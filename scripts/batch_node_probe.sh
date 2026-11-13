#!/bin/bash
# Measure, on one batch-partition node, what each Table 2 renderer costs on its card: seconds per image
# and peak GPU memory, for the joint prompt, PoE, ours and CO3. Everything is written to a scratch tree
# (never the real output tree), with the real seed latents linked in so CO3 and the joint renderer read
# the same starting noise as the real runs.
#
#   timeout 30 ssh mscluster131 'PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python nohup bash \
#     /datasets/mmolefe/poe_repair_min/outputs/paper_review/claims/bin/batch_node_probe.sh \
#     > /datasets/mmolefe/poe_repair_min/outputs/paper_review/scratch_3060/probe-131.log 2>&1 < /dev/null & disown'
#
# Per method the log ends with a line "PROBE <method> images N wall S s peak_mib M exit R".
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
BENCH="${BENCH:-$ROOT/launchers/bench13/scripts}"
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
CO3_REPO="${CO3_REPO:-$ROOT/code_2ac55e56}"
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
GPU="${GPU:-0}"
METHODS="${METHODS:-joint poe ours co3}"
SCR="$ROOT/scratch_3060/$(hostname -s)-$(basename "$(dirname "$(dirname "$PY")")")"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== probe  node $(hostname -s)  device $GPU  pid $$  python $PY  scratch $SCR  $(date -Is)"
mkdir -p "$SCR" || { echo "ABORT: cannot create $SCR (is /datasets visible?)" >&2; exit 3; }
touch "$SCR/.write_test" && rm "$SCR/.write_test" && echo "=== /datasets writable from $(hostname -s)"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader -i "$GPU"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.ones(1000,1000,device='cuda'); print('matmul sum', float((x@x).sum()))" \
  || { echo "ABORT: torch cannot use the card" >&2; exit 4; }
[ -e "$SCR/latents" ] || ln -s "$ROOT/latents" "$SCR/latents"
nproc; free -g | head -2

CKPT=$(awk '/^checkpoint:/ {print $2}' "$REPO/artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml")
TAG=$(awk '/^candidate:/ {print $2}' "$REPO/artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml")
PAIR="a photo of a bench|a sports ball"
printf 'geneval_two|a photo of a bench|a sports ball|and|9\ngeneval_two|a photo of a bench|a sports ball|and|10\n' > "$SCR/co3_cells.txt"

run() {  # run <method> <image glob> <command...>: time it and sample the card's memory every half second
  local m=$1 glob=$2; shift 2
  local mem="$SCR/$m.mem"; : > "$mem"
  ( while :; do nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits >> "$mem"; sleep 0.5; done ) &
  local sampler=$!
  local t0; t0=$(date +%s)
  echo "=== $m start $(date -Is)"
  "$@"; local r=$?
  local t1; t1=$(date +%s)
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  local n; n=$(find "$SCR" -path "*$glob" -newer "$SCR/co3_cells.txt" 2>/dev/null | wc -l)
  echo "PROBE $m images $n wall $((t1 - t0)) s peak_mib $(sort -n "$mem" | tail -1) exit $r"
}

cd "$REPO" || exit 7
for m in $METHODS; do
  case $m in
    joint) run joint "bench_joint/*/joint.png" env POE_REPO="$REPO" REVIEW_BASELINE_OUT="$SCR" \
             "$PY" "$BENCH/review_bench_joint.py" --set geneval_two --shard 0/99 ;;
    poe)   run poe "bench_poe/*/poe.png" env POE_REPO="$REPO" CONCEPT_PAIR_OUT="$SCR/bench_poe/geneval_two" \
             "$PY" 3-understanding-compositionality/scripts/concept_pair_sheet.py --column poe --pairs "$PAIR" --seeds 9 10 ;;
    ours)  run ours "bench_ours/*/ours_*.png" env POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$SCR/bench_ours/geneval_two" \
             "$PY" 1-lora-correction/scripts/figure_candidates.py --column ours --pairs "$PAIR" --seeds 9 10 \
             --checkpoint "$CKPT" --window 50 --tag "$TAG" ;;
    co3)   ( cd "$CO3_REPO" && run co3 "co3_default/*/co3.png" env POE_REPO="$CO3_REPO" REVIEW_BASELINE_OUT="$SCR" \
             "$PY" 1-lora-correction/scripts/review_baseline_cell.py --method co3 --setting default --cells "$SCR/co3_cells.txt" ) ;;
  esac
done
echo "=== probe done $(date -Is)"
