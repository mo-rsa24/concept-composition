#!/bin/bash
# Step M10 diagnosis after the checker missed its bar: the same training, 1,000 updates, at four noise windows from the push window down.
source /workspace/co3/bin/activate
cd /workspace/m10
for W in 3.8,10.5 1.2,3.8 0.3,1.2 0.002,0.3; do
  python scripts/m10_compose.py train-checker --sigma-window $W --updates 1000 --out-name checker-diag/sigma-$W > /dev/null 2>&1
  echo "$W $(cat checker-diag/sigma-$W/checker.json | tr -d '\n ')"
done
echo DIAG DONE
