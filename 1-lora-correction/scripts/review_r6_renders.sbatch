#!/bin/bash
#SBATCH --job-name=prze2-r6
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster68,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/%x-%j.out
#
# PRZE-2 piece 5: appendix figure R6 on one bigbatch RTX 3090. Sixteen tiles (four pairs at seed 10:
# joint, joint + Langevin, PoE, PoE + Langevin at c = 3, k = 20), four CO3 tiles from the same
# starting latent, then the five-column sheet and its sidecar. The plan is
# 1-lora-correction/plans/prove-the-langevin-corrector-for-the-six-column-figure.md.
#
#   sbatch --export=ALL,REPO=<checkout or worktree> <that checkout>/scripts/review_r6_renders.sbatch
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
[ -f "$OUT/latents/seed10.pt" ] || { echo "ABORT: $OUT/latents/seed10.pt missing (1-lora-correction/scripts/review_seed_latents.py)" >&2; exit 5; }
echo "=== disk ${USE}%  gpu ${USED} MiB ${UTIL}%  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO" || exit 7
export POE_REPO="$REPO"
"$PY" 1-lora-correction/scripts/review_r6_renders.py --render; R=$?
"$PY" 1-lora-correction/scripts/review_baseline_cell.py --method co3 --setting default --group r6 --seeds 10 \
  --pairs "a cat|a dog" "an elephant|a penguin" "a tent|a snowy mountain" "a lighthouse|a stormy sea"; C=$?
"$PY" 1-lora-correction/scripts/review_r6_renders.py --sheet; S=$?
echo "=== done $(date -Is)  render exit $R  co3 exit $C  sheet exit $S  tiles $(find "$OUT/r6_correctors" "$OUT/co3_default/r6" -name '*.png' 2>/dev/null | grep -c seed10)"
exit $(( R || C || S ))
