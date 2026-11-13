#!/usr/bin/env bash
# CO3 on three training pairs at seeds 1 to 3, for appendix A's joint-prompt-beside-CO3 figure.
# The same sample_co3.py call and settings as scripts/showcase/run_co3_targets.sh (CO3 vendored
# unmodified under composition/debottam_co3), run with the co3 environment on a bigbatch RTX 3090.
#
#   sbatch --partition=bigbatch ... --wrap 'bash review_co3_selection.sh'   (one bigbatch RTX 3090)
#
# Writes OUT/<pair>/algo-co3-hybrid_/<joint prompt>/seed-<n>/*.png, 9 pictures in all.
set -uo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
CO3=$REPO/composition/debottam_co3
# co3 (with xformers) on the bigbatch RTX 3090s, where every earlier CO3 render ran; co3_bw has no
# xformers, and CO3 switches it on, so the Blackwell fails at model load (2026-11-11).
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review/co3_selection
SEEDLIST="[1,2,3]"
GPU=0
PAIRS=("a_horse__x__a_pig|a horse|a pig|a horse and a pig"
       "a_lion__x__a_meerkat|a lion|a meerkat|a lion and a meerkat"
       "a_donkey__x__a_duck|a donkey|a duck|a donkey and a duck")

echo "=== node $(hostname)  start $(date -Is)  PID $$"
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
mkdir -p "$OUT"
USEPCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "$USEPCT" -ge 90 ] && { echo "ABORT: /datasets is ${USEPCT}% full" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu $GPU faulted" >&2; exit 4;; esac
[ "$USED" -gt 1024 ] && { echo "ABORT: gpu $GPU has ${USED} MiB in use" >&2; exit 5; }
CUDA_VISIBLE_DEVICES=$GPU "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 6)" \
  || { echo "ABORT: torch sees no CUDA device under $PY" >&2; exit 6; }
echo "=== disk ${USEPCT}%  gpu ${USED} MiB ${UTIL}%"

cd "$CO3" || exit 7
OK=0; FAIL=0
for e in "${PAIRS[@]}"; do
  IFS='|' read -r SLUG CA CB PJ <<< "$e"
  echo "--- $SLUG  seeds $SEEDLIST  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" sample_co3.py \
    --guidance_scale 0.8 --n_timesteps 50 \
    --prompt "${CA}+${CB}+${PJ}" --prompt_orig "$PJ" \
    --output_path "$OUT/$SLUG/" --output_path_all "$OUT/$SLUG/" \
    --sd_version xl --resolution_h 1024 --resolution_w 1024 \
    --seeds "$SEEDLIST" --negative_prompt '' \
    --num_ts_to_correct 6 --num_latent_corrector_steps 5 --num_resampling_steps 3 \
    --corrector_algo co3-hybrid --modulate_comp_weights True --beta 0.9 --lmda 0.8
  if [ $? -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); echo "!!! failed: $SLUG"; fi
done
echo "=== done $(date -Is)  pairs ok $OK, failed $FAIL, pictures $(find "$OUT" -name '*.png' | wc -l)"
