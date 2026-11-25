#!/bin/bash
# Step M10, each expert alone: 256 samples per count from the final 0.100 snapshot, 32 Heun steps, seeds 0 to 255, on this pod's 4090.
# Usage: COLOUR=r|b NET=<pkl> bash experts_alone.sh
set -uo pipefail
source /workspace/co3/bin/activate
C=${COLOUR:?}; NET=${NET:?}; O=/workspace/m10/experts-alone
for K in 0 1 2; do
  (cd /workspace/m10/edm2 && python generate_images.py --net="$NET" --outdir="$O/$C-count-$((K + 1))" --seeds=0-255 --class=$K --steps=32 --batch=128) \
    >> "$O/$C.log" 2>&1 || { echo "FAILED $C $K"; exit 1; }
done
echo "DONE $C $(date -u +%H:%M:%S)"
