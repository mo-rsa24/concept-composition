#!/bin/bash
#SBATCH --job-name=gd_reproduce
#SBATCH --partition=biggpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/concept-composition/logs/gd_reproduce_%j.log
# Route step 4: classifier guidance on the released 256 px unconditional ImageNet model.
# Usage: sbatch [--nodelist=<node>] scripts/...sbatch identity
#        sbatch [--nodelist=<node>] scripts/...sbatch guide "<classes>" "<scales>"
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
UPSTREAM=/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion
WEIGHTS=/datasets/mmolefe/poe_repair_min/weights/guided-diffusion
OUT=/datasets/mmolefe/poe_repair_min/outputs/guided_diffusion_reproduce
RESULTS=$REPO/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights
MODE=${1:?identity or guide}
cd "$REPO"

[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
mkdir -p "$OUT" "$RESULTS"
USED=$(df --output=pcent "$OUT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $OUT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
for f in 256x256_diffusion_uncond.pt 256x256_classifier.pt; do
  [ -s "$WEIGHTS/$f" ] || { echo "weights guard: $WEIGHTS/$f missing"; exit 5; }
done
echo "[$(date -u +%H:%M:%S)] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) disk=${USED}%"

if [ "$MODE" = identity ]; then
  $PY 2-classifier-guidance/scripts/guided_diffusion_reproduce.py --repo "$UPSTREAM" --weights "$WEIGHTS" \
      --out "$OUT/identity" --results "$RESULTS/identity" --mode identity --n_seeds 4
else
  $PY 2-classifier-guidance/scripts/guided_diffusion_reproduce.py --repo "$UPSTREAM" --weights "$WEIGHTS" \
      --out "$OUT/guide" --results "$RESULTS/guide" --mode guide --classes ${2:?classes} --scales ${3:?scales} --n_seeds 16
fi
echo "[$(date -u +%H:%M:%S)] done"
