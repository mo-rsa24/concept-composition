#!/bin/bash
# Route step 4 of the ICLR 56173 review response: the census and the showcase rule's renders,
# under 1-lora-correction/report/paper-review/protocol.md. Launched straight on a Blackwell node with nohup, because
# biggpu allows one Slurm job per user and that slot is held.
#
#   ROLE=census-a  joint prompt and plain product for census pairs 1 to 30, seeds 9 to 16
#   ROLE=census-b  the same for census pairs 31 to 51 and the two showcase pairs, then the showcase
#                  rule: each of the 9 candidate adapters over all 50 steps on the two showcase pairs
#
#   ssh mscluster111 'ROLE=census-a GPU=0 nohup bash <repo>/scripts/review_step4_renders.sh > <OUT>/logs/census-a.log 2>&1 &'
#
# The plain product comes from concept_pair_sheet.py, which attaches no adapter at all, so no adapter
# result exists on any evaluation pair. Every render starts from the shared seed latent. A tile that
# exists is skipped, so a relaunch resumes.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
OUT=/datasets/mmolefe/poe_repair_min/outputs/review_step4
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
CK=/datasets/mmolefe/poe_repair_min/outputs/showcase
SEEDS="9 10 11 12 13 14 15 16"
: "${ROLE:?set ROLE to census-a or census-b}" "${GPU:?set GPU to the device index}"
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO" CONCEPT_PAIR_OUT="$OUT/census" FIGURE_CANDIDATES_OUT="$OUT/showcase"
mkdir -p "$OUT/census" "$OUT/showcase" "$OUT/logs" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }

echo "=== $ROLE  node $(hostname -s)  device $GPU  pid $$  commit $(git -C "$REPO" rev-parse --short HEAD)  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
cd "$REPO"

mapfile -t ALL < <(grep -v '^#' scripts/review_step4_census_pairs.txt)
case "$ROLE" in
  census-a) PAIRS=("${ALL[@]:0:30}") ;;
  census-b) PAIRS=("${ALL[@]:30}" "a cat|a dog" "an elephant|a penguin") ;;
  *) echo "ABORT: unknown ROLE $ROLE" >&2; exit 2 ;;
esac
echo "pairs: ${#PAIRS[@]}"

for col in mono poe; do
  echo "--- $col  $(date +%H:%M:%S)"
  "$PY" scripts/shared/concept_pair_sheet.py --column "$col" --pairs "${PAIRS[@]}" --seeds $SEEDS || echo "COLUMN $col FAILED"
done

if [ "$ROLE" = census-b ]; then
  for c in cohort3/v58-03-self/checkpoints/lora_step_025000.pt cohort3/v58-03-self/checkpoints/lora_step_027500.pt \
           cohort3/v58-03-self/checkpoints/lora_step_030000.pt cohort3/v58-03-self/checkpoints/lora_step_050000.pt \
           cohort3/v58-05-self-contrast/checkpoints/lora_step_015000.pt cohort2/v57-11b-x0-self-decay/checkpoints/lora_step_050000.pt \
           cohort3/v58-01-plain/checkpoints/lora_step_047500.pt cohort3/v58-02-contrast/checkpoints/lora_step_027500.pt \
           cohort4/v58f-05-self-contrast/checkpoints/lora_step_025000.pt; do
    run=$(echo "$c" | cut -d/ -f2); step=$(basename "$c" .pt | sed 's/lora_step_//')
    echo "--- showcase $run $step  $(date +%H:%M:%S)"
    "$PY" 1-lora-correction/scripts/figure_candidates.py --column ours --pairs "a cat|a dog" "an elephant|a penguin" --seeds $SEEDS \
      --checkpoint "$CK/$c" --window 50 --tag "${run}_s${step}" || echo "SHOWCASE $run $step FAILED"
  done
fi
echo "=== done $ROLE $(date -Is)"
