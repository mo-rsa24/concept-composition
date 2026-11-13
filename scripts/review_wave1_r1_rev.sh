#!/bin/bash
# The R1 launcher run backwards: the same jobs as review_wave1_r1.sh with every prompt list reversed,
# so a second card can work from the other end beside a forward run. Each render script skips a tile
# that exists, so the two meet in the middle; a tile both reach at once is written twice, identically.
#
# Wave 1's three R1 jobs for the ICLR 56173 review response (1-lora-correction/report/paper-review/prze-10-compute.md,
# jobs 4 to 6): the 276 Attend-and-Excite prompts at seeds 9 to 12, under protocol rules 2 and 4.
#
#   ROLE=r1-joint  the joint prompt as written: "{a} and {b}" for 240 prompts, "{a} with {b}" for 36   1,104 tiles
#   ROLE=r1-poe    the plain product of the two concepts, no adapter                                   1,104 tiles
#   ROLE=r1-ours   the product with the adapter of record on all 50 steps                              1,104 tiles
#
#   ssh mscluster108 'ROLE=r1-joint GPU=0 PY=<co3 python> nohup bash <dir>/review_wave1_r1.sh > <OUT>/logs/<name>.log 2>&1 < /dev/null & disown'
#
# The render scripts and the prompt list are read from beside this script, so a copy of all four on
# /datasets runs against the cluster checkout's package without that checkout being synced. The 45
# prompts naming a frog or a mouse run under the pair ledger's 2026-11-10 decision (--allow-ask-first).
# A tile that exists is skipped, so a relaunch resumes.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
CKPT=/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt
TAG=v58-03-self_s030000
SEEDS="9 10 11 12"
: "${ROLE:?set ROLE to r1-joint, r1-poe or r1-ours}" "${GPU:?set GPU to the device index}"
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO"
mkdir -p "$OUT/logs" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }

echo "=== $ROLE reversed  node $(hostname -s)  device $GPU  pid $$  python $PY  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
cd "$REPO"

LIST="$HERE/review_benchmark_pairs.txt"
mapfile -t AND  < <(grep -v '^#' "$LIST" | awk -F'|' '$3=="and"  {print $1"|"$2}' | tac)
mapfile -t WITH < <(grep -v '^#' "$LIST" | awk -F'|' '$3=="with" {print $1"|"$2}' | tac)
[ "${#AND[@]}" -eq 240 ] && [ "${#WITH[@]}" -eq 36 ] || { echo "ABORT: ${#AND[@]} and-prompts, ${#WITH[@]} with-prompts, not 240 and 36" >&2; exit 2; }

case "$ROLE" in
  r1-joint)
    # The forward run does the "with" prompts first, so this one starts at the end of the "and" prompts.
    CONCEPT_PAIR_OUT="$OUT/r1_joint" "$PY" "$HERE/concept_pair_sheet.py" --column mono --allow-ask-first --pairs "${AND[@]}"  --seeds $SEEDS || echo "ROLE r1-joint and FAILED"
    CONCEPT_PAIR_OUT="$OUT/r1_joint" "$PY" "$HERE/concept_pair_sheet.py" --column mono --allow-ask-first --joint-template "{a} with {b}" \
      --pairs "${WITH[@]}" --seeds $SEEDS || echo "ROLE r1-joint with FAILED" ;;
  r1-poe)
    CONCEPT_PAIR_OUT="$OUT/r1_poe" "$PY" "$HERE/concept_pair_sheet.py" --column poe --allow-ask-first --pairs "${WITH[@]}" "${AND[@]}" --seeds $SEEDS \
      || echo "ROLE r1-poe FAILED" ;;
  r1-ours)
    [ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 2; }
    FIGURE_CANDIDATES_OUT="$OUT/r1_ours" "$PY" "$HERE/figure_candidates.py" --column ours --allow-ask-first --pairs "${WITH[@]}" "${AND[@]}" \
      --seeds $SEEDS --checkpoint "$CKPT" --window 50 --tag "$TAG" || echo "ROLE r1-ours FAILED" ;;
  *) echo "ABORT: unknown ROLE $ROLE" >&2; exit 2 ;;
esac
echo "=== done $ROLE $(date -Is)"
