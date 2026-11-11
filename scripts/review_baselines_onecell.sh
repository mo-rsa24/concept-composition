#!/usr/bin/env bash
# One tuning cell per baseline for the ICLR 56173 review response (plans/prze-review-baselines.md, task 1):
# lion and meerkat at seed 9, SDXL, 1024 by 1024, each method at its published default.
#
#   sbatch --partition=bigbatch --nodes=1 --time=01:00:00 --exclude=<faulted list> \
#     --output=<OUT>/logs/baselines_onecell_%j.log <launchers>/review_baselines_onecell_v2.sh
#
# The python scripts are read from beside this file, so a copy of all four on /datasets runs against the cluster
# checkout's package without that checkout being synced. Order: the seed latents, CO3, SuperDiff, then ToMe if its
# environment finished building (ToMe backpropagates through the U-Net, so it is the likeliest to run out of memory).
set -uo pipefail
# sbatch runs a copy of this file from the Slurm spool, so the scripts beside it are named by absolute path.
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
CO3PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
TOMEPY=/home-mscluster/mmolefe/miniforge3/envs/tome/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
PAIR="a lion|a meerkat"
SEED=9

echo "=== baselines one-cell  node $(hostname -s)  device $GPU  pid $$  job ${SLURM_JOB_ID:-none}  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$CO3PY" ] || { echo "ABORT: python not found at $CO3PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$CO3PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader

[ -f "$OUT/latents/seed16.pt" ] || "$CO3PY" "$HERE/review_seed_latents.py" || { echo "ABORT: seed latents" >&2; exit 5; }

peak() { nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader; }
for M in co3 superdiff; do
  echo "--- $M  $(date +%H:%M:%S)"
  ( while sleep 20; do echo "    mem $(peak)"; done ) & W=$!
  "$CO3PY" "$HERE/review_baseline_cell.py" --method "$M" --setting default --pairs "$PAIR" --seeds $SEED \
    || echo "!!! $M FAILED"
  kill "$W" 2>/dev/null
done

if [ -x "$TOMEPY" ] && grep -q ENV_OK "$OUT/logs/tome_env_build.log" 2>/dev/null; then
  echo "--- tome  $(date +%H:%M:%S)"
  ( while sleep 20; do echo "    mem $(peak)"; done ) & W=$!
  "$TOMEPY" "$HERE/review_tome_cell.py" --setting default --pairs "$PAIR" --seeds $SEED || echo "!!! tome FAILED"
  kill "$W" 2>/dev/null
else
  echo "--- tome skipped: its environment has not finished building"
fi
echo "=== done $(date -Is)"
