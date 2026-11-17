#!/usr/bin/env bash
# Plan 20, piece 4: wait for piece 3's chain to finish, then draw the released model's sheet on GPU 0.
# Usage on the node: PY=<python> nohup bash piece4_256.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
D=/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos
: "${PY:?set PY}"
echo "piece 4 waiter $(date -Is) on $(hostname) PID $$"
until grep -q "PIECE3 DONE" $D/logs/piece3_256.log; do
  grep -q "failed" $D/logs/piece3_256.log && { echo "piece 3 failed, not drawing"; exit 1; }
  sleep 60
done
echo "piece 3 done, drawing $(date -Is)"
GPU=0 bash $D/code/launch_gpu.sh $D/code/sheet_256.py --separate $D/classifiers/separate_256.pt \
  --pair $D/classifiers/pair_256.pt --out $D/sheets/released_256 > $D/logs/sheet_256.log 2>&1 \
  || { echo "sheet failed"; exit 1; }
echo "PIECE4 DONE $(date -Is)"
