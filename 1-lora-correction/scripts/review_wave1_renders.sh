#!/bin/bash
# Wave 1 of the ICLR 56173 review response (1-lora-correction/report/paper-review/prze-10-compute.md, the jobs table),
# under 1-lora-correction/report/paper-review/protocol.md. Launched straight on a node with nohup over SSH, because
# biggpu allows one Slurm job per user and that slot is held.
#
#   ROLES is a space-separated list, run in order in one process:
#     fig5   Figure 5's missing row, lighthouse and stormy sea at seed 10: joint, PoE, ours     3 tiles
#     early  ours with the adapter on the first 10 and the first 25 steps, the product alone
#            after, on the 53 held-out pairs at seeds 9 to 16                                   848 tiles
#     r3     ours with the adapter on all 50 steps, the same pairs and seeds                   424 tiles
#
#   ssh mscluster112 'ROLES="fig5 early r3" GPU=0 nohup bash <dir>/review_wave1_renders.sh > <OUT>/logs/112.log 2>&1 < /dev/null & disown'
#
# The adapter is the one adapter_of_record.yaml names. Every render starts from the shared seed
# latent. A tile that exists is skipped, so a relaunch resumes. The pair list is read from beside
# this script, so a copy of both on /datasets runs without the cluster checkout being synced.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
CKPT=/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt
TAG=v58-03-self_s030000
SEEDS="9 10 11 12 13 14 15 16"
: "${ROLES:?set ROLES, e.g. \"fig5 early r3\"}" "${GPU:?set GPU to the device index}"
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO"
mkdir -p "$OUT/logs" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }

echo "=== $ROLES  node $(hostname -s)  device $GPU  pid $$  commit $(git -C "$REPO" rev-parse --short HEAD)  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
cd "$REPO"

mapfile -t HELD < <(grep -v '^#' "$HERE/review_held_out_pairs.txt")
[ "${#HELD[@]}" -eq 53 ] || { echo "ABORT: ${#HELD[@]} held-out pairs, not 53" >&2; exit 2; }

for role in $ROLES; do
  echo "--- $role  $(date +%H:%M:%S)"
  case "$role" in
    fig5)
      CONCEPT_PAIR_OUT="$OUT/figure5" "$PY" scripts/shared/concept_pair_sheet.py --column mono --pairs "a lighthouse|a stormy sea" --seeds 10 || echo "ROLE fig5 mono FAILED"
      CONCEPT_PAIR_OUT="$OUT/figure5" "$PY" scripts/shared/concept_pair_sheet.py --column poe  --pairs "a lighthouse|a stormy sea" --seeds 10 || echo "ROLE fig5 poe FAILED"
      FIGURE_CANDIDATES_OUT="$OUT/figure5" "$PY" 1-lora-correction/scripts/figure_candidates.py --column ours --pairs "a lighthouse|a stormy sea" --seeds 10 \
        --checkpoint "$CKPT" --window 50 --tag "$TAG" || echo "ROLE fig5 ours FAILED" ;;
    early)
      FIGURE_CANDIDATES_OUT="$OUT/early_steps" "$PY" 1-lora-correction/scripts/figure_candidates.py --column ours --pairs "${HELD[@]}" --seeds $SEEDS \
        --checkpoint "$CKPT" --window 10 25 --tag "$TAG" || echo "ROLE early FAILED" ;;
    r3)
      FIGURE_CANDIDATES_OUT="$OUT/r3_ours" "$PY" 1-lora-correction/scripts/figure_candidates.py --column ours --pairs "${HELD[@]}" --seeds $SEEDS \
        --checkpoint "$CKPT" --window 50 --tag "$TAG" || echo "ROLE r3 FAILED" ;;
    *) echo "ABORT: unknown role $role" >&2; exit 2 ;;
  esac
done
echo "=== done $ROLES $(date -Is)"
