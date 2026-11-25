#!/bin/bash
# Step M10 on red's pod, part A in order: blue's experts-alone samples and paste sources, the plain product's four blocks, the checker's training.
set -uo pipefail
source /workspace/co3/bin/activate
cd /workspace/m10
step() { echo "[$(date -u +%H:%M:%S)] $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
step env COLOUR=b NET=/workspace/m10/models/b-full/network-snapshot-0002000-0.100.pkl bash experts_alone.sh
step python scripts/m10_compose.py sources --colour b --seeds 0-1023
step python scripts/m10_compose.py sources --colour b --seeds 100000-100511
while ! grep -q 'EMPTY SOURCES DONE' sources-empty.log; do grep -q Traceback sources-empty.log && { echo "FAILED: empty sources"; exit 1; }; sleep 20; done
for B in 0 1 2 3; do step python scripts/m10_compose.py row --row plain --block $B; done
step python scripts/m10_compose.py train-checker
echo "[$(date -u +%H:%M:%S)] PIPELINE A DONE"
