#!/usr/bin/env bash
# Plan 20, pieces 3 and 4, on one two-card node: the product's draws on both cards (the released model
# pushed by the released classifier's cat and dog classes at scale 10), a 16-draw preview, the pair-aware
# classifier against those draws and the single-animal images, then the released model's sheet.
# Each step goes through launch_gpu.sh's guard.
# Usage on the node: PY=<python> nohup bash piece3_256.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
D=/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos
C=$D/code; K=$D/classifiers; mkdir -p "$K" "$D/draws" "$D/logs" "$D/figures"
: "${PY:?set PY}"
echo "pieces 3 and 4 chain $(date -Is) on $(hostname) PID $$"

GPU=0 bash $C/launch_gpu.sh $C/classifiers.py draw --push imagenet --out $D/draws/product_256_imagenet_a.npz \
  --n 256 --batch 8 --seed0 1000 > $D/logs/draws_256_imagenet_a.log 2>&1 &
pa=$!
GPU=1 bash $C/launch_gpu.sh $C/classifiers.py draw --push imagenet --out $D/draws/product_256_imagenet_b.npz \
  --n 256 --batch 8 --seed0 2000 > $D/logs/draws_256_imagenet_b.log 2>&1 &
pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
[ $ra -eq 0 ] && [ $rb -eq 0 ] || { echo "draws failed: $ra $rb"; exit 1; }
echo "draws done $(date -Is)"

"$PY" -I -c "
import numpy as np; from PIL import Image
u=np.load('$D/draws/product_256_imagenet_a.npz')['u8'][:16]
g=Image.new('RGB',(4*260+4,4*260+4),'white')
for i,x in enumerate(u): g.paste(Image.fromarray(x.transpose(1,2,0)),(4+(i%4)*260,4+(i//4)*260))
g.save('$D/figures/preview-product-256-imagenet-draws.png')"

GPU=0 bash $C/launch_gpu.sh $C/classifiers.py train-pair --size 256 --ckpt $K/pair_256.pt \
  --fakes $D/draws/product_256_imagenet_a.npz $D/draws/product_256_imagenet_b.npz > $D/logs/pair_256.log 2>&1 \
  || { echo "pair-aware training failed"; exit 1; }
echo "pair-aware done $(date -Is)"

GPU=0 bash $C/launch_gpu.sh $C/sheet_256.py --pair $K/pair_256.pt --out $D/sheets/released_256 \
  > $D/logs/sheet_256.log 2>&1 || { echo "sheet failed"; exit 1; }
echo "PIECES 3 AND 4 DONE $(date -Is)"
