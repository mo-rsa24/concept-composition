#!/usr/bin/env bash
# Plan 20, piece 3, on one two-card node: the separate-label classifier, the product's draws on both
# cards, then the pair-aware classifier. Each step goes through launch_gpu.sh's guard.
# Usage on the node: PY=<python> nohup bash piece3_256.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
D=/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos
C=$D/code; K=$D/classifiers; mkdir -p "$K" "$D/draws" "$D/logs"
: "${PY:?set PY}"
echo "piece 3 chain $(date -Is) on $(hostname) PID $$"

GPU=0 bash $C/launch_gpu.sh $C/classifiers.py train-separate --size 256 --ckpt $K/separate_256.pt \
  > $D/logs/separate_256.log 2>&1 || { echo "separate-label training failed"; exit 1; }
echo "separate done $(date -Is)"

GPU=0 bash $C/launch_gpu.sh $C/classifiers.py draw --separate $K/separate_256.pt --out $D/draws/product_256_a.npz \
  --n 256 --batch 8 --seed0 1000 > $D/logs/draws_256_a.log 2>&1 &
pa=$!
GPU=1 bash $C/launch_gpu.sh $C/classifiers.py draw --separate $K/separate_256.pt --out $D/draws/product_256_b.npz \
  --n 256 --batch 8 --seed0 2000 > $D/logs/draws_256_b.log 2>&1 &
pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
[ $ra -eq 0 ] && [ $rb -eq 0 ] || { echo "draws failed: $ra $rb"; exit 1; }
echo "draws done $(date -Is)"

GPU=0 bash $C/launch_gpu.sh $C/classifiers.py train-pair --size 256 --ckpt $K/pair_256.pt \
  --fakes $D/draws/product_256_a.npz $D/draws/product_256_b.npz > $D/logs/pair_256.log 2>&1 \
  || { echo "pair-aware training failed"; exit 1; }
echo "PIECE3 DONE $(date -Is)"
