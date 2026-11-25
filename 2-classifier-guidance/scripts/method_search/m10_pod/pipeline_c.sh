#!/bin/bash
# Step M10 on red's pod, part C (board question 116): the checker that reads the noisy picture, its zero-push check, then its row.
set -uo pipefail
source /workspace/co3/bin/activate
cd /workspace/m10
step() { echo "[$(date -u +%H:%M:%S)] $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
step python scripts/m10_compose.py train-checker --reader raw --out-name checker-raw
step python scripts/m10_compose.py row --row zero --reader raw --out-name checker-raw --block 0 --limit 16
for B in 0 1 2 3; do step python scripts/m10_compose.py row --row checker --reader raw --out-name checker-raw --block $B; done
echo "[$(date -u +%H:%M:%S)] PIPELINE C DONE"
