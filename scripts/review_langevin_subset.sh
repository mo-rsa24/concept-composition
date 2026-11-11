#!/bin/bash
# PoE + Langevin on protocol rule 6's labelled subset (288 cells), one shard per card, launched straight on
# a node with nohup. Runs scripts/review_langevin_subset.py against the step 8 worktree, the code that drew
# Figure 5's PoE + Langevin tiles, so the subset and the figure come from one corrector.
#
#   ssh mscluster111 'SHARD=0 SHARDS=2 GPU=0 nohup bash <dir>/review_langevin_subset.sh > <OUT>/logs/<name>.log 2>&1 < /dev/null & disown'
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/datasets/mmolefe/poe_repair_min/worktrees/prze2}"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
: "${SHARD:?set SHARD}" "${SHARDS:?set SHARDS}" "${GPU:?set GPU to the device index}"
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO"
mkdir -p "$OUT/logs" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }

echo "=== langevin subset shard $SHARD of $SHARDS  node $(hostname -s)  device $GPU  pid $$  repo $REPO at $(git -C "$REPO" rev-parse --short HEAD)  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
cd "$REPO"
"$PY" "$HERE/review_langevin_subset.py" --shard "$SHARD" --shards "$SHARDS" || echo "SHARD $SHARD FAILED"
echo "=== done shard $SHARD $(date -Is)"
