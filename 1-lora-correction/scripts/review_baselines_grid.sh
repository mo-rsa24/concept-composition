#!/usr/bin/env bash
# Tuning grid renders for the ICLR 56173 review response (1-lora-correction/plans/the-baselines-the-reviewers-asked-for.md, task 3): the four tuning
# pairs at seeds 9 to 12, one folder per setting, under protocol rule 3's grids.
#
#   METHOD=co3 SETTINGS="l0.5_ts6 l1.0_ts6" sbatch --partition=bigbatch --nodes=1 --time=03:00:00 \
#     --exclude=<faulted list> --output=<OUT>/logs/grid_%x_%j.log <launchers>/review_baselines_grid.sh
#
# Setting names: co3 l<λ>_ts<timesteps corrected>; superdiff s<steps>_<clamp on|off>. A tile that exists is
# skipped, so a relaunch resumes.
set -uo pipefail
# sbatch runs a copy of this file from the Slurm spool, so the scripts it calls are named by absolute path.
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
: "${METHOD:?set METHOD to co3 or superdiff}" "${SETTINGS:?set SETTINGS to a space-separated list}"
PAIRS=("a lion|a meerkat" "a camel|a pelican" "a donkey|a duck" "a chicken|a rabbit")
SEEDS="9 10 11 12"

echo "=== grid $METHOD [$SETTINGS]  node $(hostname -s)  device $GPU  pid $$  job ${SLURM_JOB_ID:-none}  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
[ -f "$OUT/latents/seed12.pt" ] || { echo "ABORT: no saved latents under $OUT/latents" >&2; exit 5; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader

knobs() {  # setting name -> JSON overrides of the method's published default
  case "$METHOD" in
    co3) local l=${1#l}; l=${l%%_*}; local ts=${1##*_ts}
         echo "{\"lmda\": $l, \"num_ts_to_correct\": $ts}" ;;
    superdiff) local s=${1#s}; s=${s%%_*}; local c=${1##*_}
         [ "$c" = on ] && c=true || c=false
         echo "{\"steps\": $s, \"kappa_clamp\": $c}" ;;
  esac
}

for S in $SETTINGS; do
  K=$(knobs "$S")
  echo "--- $METHOD $S  $K  $(date +%H:%M:%S)"
  "$PY" "$HERE/review_baseline_cell.py" --method "$METHOD" --setting "$S" --knobs "$K" \
    --pairs "${PAIRS[@]}" --seeds $SEEDS --group tuning || echo "!!! $METHOD $S FAILED"
  echo "    tiles $(ls "$OUT/${METHOD}_$S"/*/seed*/"$METHOD".png 2>/dev/null | wc -l) of 16"
done
echo "=== done $(date -Is)"
