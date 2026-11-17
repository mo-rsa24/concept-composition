#!/usr/bin/env bash
# Plan 20, piece 5, for one 64 px model: wait for its training to print DONE, then on GPU 0 of the same
# node draw 512 plain-product draws, train its pair-aware classifier against them, and draw its sheet.
# Usage on the training's node: ARM=<no_both|all> PY=<python> nohup bash piece5_64.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
D=/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos
C=$D/code; K=$D/classifiers
: "${ARM:?set ARM}"; : "${PY:?set PY}"
M=$D/runs/train_64_$ARM/ema_0.9999_050000.pt
echo "piece 5 waiter for $ARM $(date -Is) on $(hostname) PID $$"
until grep -q "^DONE" $D/logs/train_64_$ARM.log; do
  pgrep -f "train_64.py --arm $ARM" > /dev/null || { echo "training $ARM not running and no DONE"; exit 1; }
  sleep 120
done
until [ -f $K/separate_64.pt ]; do sleep 60; done
sleep 30
echo "training done, drawing $(date -Is)"
GPU=0 bash $C/launch_gpu.sh $C/sheet_64.py draw --model $M --out $D/draws/product_64_$ARM.npz \
  > $D/logs/draws_64_$ARM.log 2>&1 || { echo "draws failed"; exit 1; }
GPU=0 bash $C/launch_gpu.sh $C/classifiers.py train-pair --size 64 --ckpt $K/pair_64_$ARM.pt \
  --fakes $D/draws/product_64_$ARM.npz > $D/logs/pair_64_$ARM.log 2>&1 || { echo "pair-aware training failed"; exit 1; }
GPU=0 bash $C/launch_gpu.sh $C/sheet_64.py sheet --model $M --separate $K/separate_64.pt --pair $K/pair_64_$ARM.pt \
  --out $D/sheets/model_64_$ARM --sheet-name sheet-64-$ARM.png > $D/logs/sheet_64_$ARM.log 2>&1 || { echo "sheet failed"; exit 1; }
echo "PIECE5 DONE $ARM $(date -Is)"
