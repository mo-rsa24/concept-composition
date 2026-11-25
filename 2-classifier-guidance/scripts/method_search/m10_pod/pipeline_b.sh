#!/bin/bash
# Step M10 on red's pod, part B: the mask paste's four blocks (CPU) and the corrector row's four blocks (GPU).
set -uo pipefail
source /workspace/co3/bin/activate
cd /workspace/m10
step() { echo "[$(date -u +%H:%M:%S)] $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
for B in 0 1 2 3; do step python scripts/m10_compose.py paste --block $B; done
for B in 0 1 2 3; do step python scripts/m10_compose.py row --row corrector --block $B; done
echo "[$(date -u +%H:%M:%S)] PIPELINE B DONE"
