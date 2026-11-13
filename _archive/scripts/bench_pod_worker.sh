#!/bin/bash
# Laptop-side driver that makes one RunPod pod a worker on Table 2's chunk ledger (scripts/bench_claims.py
# on the cluster): claim a chunk on the cluster (mkdir over ssh), render it on the pod over ssh, copy the
# pod's new images to the laptop and on to the cluster output tree every SYNC_S seconds while it runs (so
# the claim never looks stale), mark the chunk done on the cluster, repeat. One copy runs per pod.
#
# One copy per card on the pod (POD_GPU), e.g. for the 4-card pod, on the laptop:
#   for g in 0 1 2 3; do POD_SSH="root@185.216.22.81 -p 36792 -i ~/.ssh/id_ed25519" POD_NAME=pod1-g$g POD_GPU=$g \
#     METHODS=joint nohup bash scripts/bench_pod_worker.sh > ~/bench_pod_stage/pod1-g$g.log 2>&1 & done
# Start the first copy alone and let its preflight pass before starting the rest.
#
# What the pod must hold before the first run (checked by the preflight below, which refuses otherwise):
#   - the code bundle at POD_REPO (default /workspace/poe): the poe_repair package and scripts/, plus the
#     bench scripts (review_bench_joint.py, review_bench_sets.py, showcase/, data/prompt-sets/) at POD_BENCH;
#   - SDXL from Hugging Face in the pod's HF cache;
#   - the seed latents copied from the cluster's $ROOT/latents into $POD_OUT/latents, AND the cached
#     starting noise that init_latents_for_cell reads for a_cat__x__a_dog. That function falls back to a
#     latent drawn from the seed number when its cache is missing, with no error, so the preflight compares
#     what it returns against $POD_OUT/latents for every seed and stops on any difference;
#   - for ours, the adapter checkpoint at POD_CKPT (the cluster's adapter_of_record.yaml names it);
#   - for co3, composition/debottam_co3 inside POD_REPO.
set -uo pipefail
: "${POD_SSH:?ssh destination and options for the pod, e.g. 'root@1.2.3.4 -p 22022'}"
: "${POD_NAME:?a short name for this pod, used in the claim owner and the staging folder}"
METHODS="${METHODS:-joint}"
POD_REPO="${POD_REPO:-/workspace/poe}"
POD_BENCH="${POD_BENCH:-$POD_REPO/scripts}"
POD_OUT="${POD_OUT:-/workspace/out}"
POD_PY="${POD_PY:-python}"
POD_GPU="${POD_GPU:-0}"
POD_CKPT="${POD_CKPT:-/workspace/ckpt/adapter.pt}"
OURS_TAG="${OURS_TAG:-v58-03-self_s030000}"
CLUSTER="${CLUSTER:-mscluster107}"
ROOT="${ROOT:-/datasets/mmolefe/poe_repair_min/outputs/paper_review}"
CPY="${CPY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
LEDGER="BENCH=$ROOT/launchers/bench13/scripts OURS_TAG=$OURS_TAG $CPY $ROOT/claims/bin/bench_claims.py"
STAGE="${STAGE:-$HOME/bench_pod_stage/$POD_NAME}"
SYNC_S="${SYNC_S:-120}"
OWNER="pod:$POD_NAME:$(hostname -s):$$"
DIRS="bench_joint bench_poe bench_ours co3_default"
log() { echo "[$(date -u +%FT%TZ)] $OWNER $*"; }
pod() { ssh -o BatchMode=yes -o ConnectTimeout=20 $POD_SSH "$@"; }
cl() { timeout 120 ssh -o BatchMode=yes $CLUSTER "$@"; }
mkdir -p "$STAGE"

sync_once() {  # pod -> laptop -> cluster, png and json only, never overwriting a tile already on the cluster
  local d
  for d in $DIRS; do
    rsync -a --include='*/' --include='*.png' --include='*.json' --exclude='*' \
      -e "ssh -o BatchMode=yes $(echo "$POD_SSH" | cut -s -d' ' -f2-)" \
      "$(echo "$POD_SSH" | cut -d' ' -f1):$POD_OUT/$d/" "$STAGE/$d/" 2>/dev/null || true
    [ -d "$STAGE/$d" ] && rsync -a --ignore-existing "$STAGE/$d/" "$CLUSTER:$ROOT/$d/"
  done
}

log "preflight on the pod"
pod "cd $POD_REPO && CUDA_VISIBLE_DEVICES=$POD_GPU POE_REPO=$POD_REPO REVIEW_BASELINE_OUT=$POD_OUT $POD_PY - <<'EOF'
import sys, torch
from pathlib import Path
sys.path.insert(0, '.')
from poe_repair.composers._helpers import init_latents_for_cell
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.run import make_ctx
assert torch.cuda.is_available(), 'no CUDA on the pod'
print('card', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GB')
ctx = make_ctx()
for seed in [9, 10, 11, 12, 13, 14, 15, 16, 59, 60]:
    saved = Path('$POD_OUT') / 'latents' / f'seed{seed:02d}.pt'
    init, _ = init_latents_for_cell(cell_from_slug('a_cat__x__a_dog', seed), ctx)
    ok = saved.exists() and torch.equal(init.cpu().half(), torch.load(saved, map_location='cpu').half().reshape(init.shape))
    print('seed', seed, 'latent matches the cluster copy' if ok else 'LATENT MISMATCH OR MISSING')
    if not ok:
        sys.exit(9)
EOF" || { log "ABORT: preflight failed on the pod (see above)"; exit 2; }

while :; do
  line=$(cl "$LEDGER claim --owner $OWNER --methods $METHODS") || { log "nothing left for $METHODS"; exit 0; }
  read -r ID METHOD SET K N <<< "$line"
  log "claimed $ID ($METHOD $SET shard $K/$N)"
  t0=$(date +%s)
  case "$METHOD" in
    joint) CMD="POE_REPO=$POD_REPO REVIEW_BASELINE_OUT=$POD_OUT $POD_PY $POD_BENCH/review_bench_joint.py --set $SET --shard $K/$N" ;;
    poe|ours)
      cl "$LEDGER pairs $ID" > "$STAGE/$ID.pairs" || { log "cannot read pairs for $ID"; exit 3; }
      PAIRS=$(sed "s/'/'\\\\''/g; s/^/'/; s/\$/'/" "$STAGE/$ID.pairs" | tr '\n' ' ')
      if [ "$METHOD" = poe ]; then
        CMD="POE_REPO=$POD_REPO CONCEPT_PAIR_OUT=$POD_OUT/bench_poe/$SET $POD_PY scripts/concept_pair_sheet.py --column poe --pairs $PAIRS --seeds 9 10 11 12"
      else
        CMD="POE_REPO=$POD_REPO FIGURE_CANDIDATES_OUT=$POD_OUT/bench_ours/$SET $POD_PY scripts/figure_candidates.py --column ours --pairs $PAIRS --seeds 9 10 11 12 --checkpoint $POD_CKPT --window 50 --tag $OURS_TAG"
      fi ;;
    co3)
      cl "$LEDGER cells $ID" > "$STAGE/$ID.cells" || { log "cannot read cells for $ID"; exit 3; }
      pod "mkdir -p $POD_OUT/cells && cat > $POD_OUT/cells/$ID.txt" < "$STAGE/$ID.cells"
      CMD="POE_REPO=$POD_REPO REVIEW_BASELINE_OUT=$POD_OUT $POD_PY scripts/review_baseline_cell.py --method co3 --setting default --cells $POD_OUT/cells/$ID.txt" ;;
    *) log "unknown method $METHOD"; cl "$LEDGER release $ID badmethod"; exit 3 ;;
  esac
  pod "cd $POD_REPO && CUDA_VISIBLE_DEVICES=$POD_GPU $CMD" > "$STAGE/$ID.log" 2>&1 &
  RPID=$!
  while kill -0 "$RPID" 2>/dev/null; do
    sleep "$SYNC_S"; sync_once
  done
  wait "$RPID"; r=$?
  sync_once
  if cl "$LEDGER finish $ID"; then
    log "done $ID in $(( $(date +%s) - t0 )) s (pod exit $r)"
  else
    log "FAILED $ID (pod exit $r), claim released; tail of $STAGE/$ID.log:"; tail -n 5 "$STAGE/$ID.log"
    exit 4
  fi
done
