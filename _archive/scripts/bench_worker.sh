#!/bin/bash
# One card's worker for Table 2's renders (compute job 13): claim a chunk from the ledger
# (scripts/bench_claims.py), render it, mark it done, repeat until nothing is left that this card may take.
# Launched over SSH outside Slurm (setsid nohup), so it yields: before each chunk, and every 20 s during
# one, it reads the card, and if any process other than its own renderer is on it (or memory it cannot
# account for), it kills its renderer by pid, gives the claim back and exits.
#
#   timeout 30 ssh mscluster131 'GPU=0 METHODS=joint,poe,ours setsid nohup bash \
#     /datasets/mmolefe/poe_repair_min/outputs/paper_review/claims/bin/bench_worker.sh \
#     > /datasets/mmolefe/poe_repair_min/outputs/paper_review/claims/logs/worker-mscluster131-0.log 2>&1 < /dev/null & disown'
#
# METHODS is the allowlist this card's class passed in the node probe (scripts/batch_node_probe.sh).
# Exit codes: 0 nothing left, 3 disk or filesystem, 4 card not usable or taken, 5 yielded mid-chunk,
# 6 three chunks failed in a row.
set -uo pipefail
ROOT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
BIN="${BIN:-$ROOT/claims/bin}"
BENCH="${BENCH:-$ROOT/launchers/bench13/scripts}"
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/concept-composition}"
CO3_REPO="${CO3_REPO:-$ROOT/code_2ac55e56}"
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python}"
: "${GPU:?set GPU to the device index}"
: "${METHODS:?set METHODS, e.g. joint,poe,ours}"
MAX_CHUNKS="${MAX_CHUNKS:-1000000}"
HOST=$(hostname -s)
OWNER="$HOST:gpu$GPU:$$"
export CUDA_VISIBLE_DEVICES="$GPU" BENCH REVIEW_BASELINE_OUT="$ROOT"
CKPT=$(awk '/^checkpoint:/ {print $2}' "$REPO/artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml")
TAG=$(awk '/^candidate:/ {print $2}' "$REPO/artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml")
export OURS_TAG="$TAG"
LEDGER=("$PY" "$BIN/bench_claims.py")
log() { echo "[$(date -u +%FT%TZ)] $OWNER $*"; }

log "start  methods $METHODS  python $PY  repo $REPO  adapter $TAG"

foreign_on_card() {  # prints a reason and returns 0 if anything but pid $1 (may be empty) uses the card
  local mine=${1:-} line pid mem own=0 total util
  while IFS=, read -r pid mem; do
    pid=${pid// /}; mem=${mem// /}
    [ -z "$pid" ] && continue
    if [ "$pid" = "$mine" ]; then own=$mem; else echo "foreign pid $pid using ${mem} MiB"; return 0; fi
  done < <(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  read -r total util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  case "$util" in ''|*[!0-9]*) echo "card faulted (utilisation '$util')"; return 0;; esac
  # memory nobody listed (another user's process hidden from us) beyond what our own renderer holds
  [ $(( total - own )) -gt 1024 ] && { echo "card holds $total MiB, $own MiB of it ours"; return 0; }
  return 1
}

guards() {
  local use why
  [ -d "$ROOT" ] || { log "ABORT: $ROOT not visible"; exit 3; }
  use=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
  [ -n "$use" ] && [ "$use" -lt 90 ] || { log "ABORT: $ROOT is ${use:-unknown}% full"; exit 3; }
  if why=$(foreign_on_card ""); then log "YIELD before claiming: $why"; exit 4; fi
  if command -v squeue >/dev/null && [ -n "$(squeue -h -w "$HOST" -t R 2>/dev/null | head -1)" ]; then
    log "YIELD: Slurm now runs a job on $HOST: $(squeue -h -w "$HOST" -t R -o '%i %u %j' | head -3 | tr '\n' ';')"; exit 4
  fi
}

"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { log "ABORT: torch sees no CUDA"; exit 4; }
fails=0; chunks=0
while [ "$chunks" -lt "$MAX_CHUNKS" ]; do
  guards
  line=$("${LEDGER[@]}" claim --owner "$OWNER" --methods "$METHODS") || { log "nothing left for $METHODS"; exit 0; }
  read -r ID METHOD SET K N <<< "$line"
  log "claimed $ID ($METHOD $SET shard $K/$N)"
  t0=$(date +%s)
  case "$METHOD" in
    joint) env POE_REPO="$REPO" "$PY" "$BENCH/review_bench_joint.py" --set "$SET" --shard "$K/$N" & ;;
    poe)   mapfile -t PAIRS < <("${LEDGER[@]}" pairs "$ID")
           ( cd "$REPO" && exec env POE_REPO="$REPO" CONCEPT_PAIR_OUT="$ROOT/bench_poe/$SET" \
               "$PY" scripts/concept_pair_sheet.py --column poe --pairs "${PAIRS[@]}" --seeds 9 10 11 12 ) & ;;
    ours)  mapfile -t PAIRS < <("${LEDGER[@]}" pairs "$ID")
           ( cd "$REPO" && exec env POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$ROOT/bench_ours/$SET" \
               "$PY" scripts/figure_candidates.py --column ours --pairs "${PAIRS[@]}" --seeds 9 10 11 12 \
               --checkpoint "$CKPT" --window 50 --tag "$TAG" ) & ;;
    co3)   CELLS="$ROOT/claims/cells/$ID.txt"; mkdir -p "$ROOT/claims/cells"; "${LEDGER[@]}" cells "$ID" > "$CELLS"
           ( cd "$CO3_REPO" && exec env POE_REPO="$CO3_REPO" \
               "$PY" scripts/review_baseline_cell.py --method co3 --setting default --cells "$CELLS" ) & ;;
    *) log "unknown method $METHOD"; "${LEDGER[@]}" release "$ID" badmethod; exit 6 ;;
  esac
  RPID=$!
  log "renderer pid $RPID"
  while kill -0 "$RPID" 2>/dev/null; do
    sleep 20
    kill -0 "$RPID" 2>/dev/null || break
    if why=$(foreign_on_card "$RPID"); then
      log "YIELD mid-chunk: $why; killing renderer $RPID and releasing $ID"
      kill "$RPID"; sleep 10; kill -9 "$RPID" 2>/dev/null
      "${LEDGER[@]}" release "$ID" yield
      exit 5
    fi
  done
  wait "$RPID"; r=$?
  if "${LEDGER[@]}" finish "$ID"; then
    log "done $ID in $(( $(date +%s) - t0 )) s (renderer exit $r)"; fails=0; chunks=$((chunks + 1))
  else
    fails=$((fails + 1)); log "FAILED $ID after $(( $(date +%s) - t0 )) s (renderer exit $r), released; $fails in a row"
    [ "$fails" -ge 3 ] && { log "three chunks failed in a row, stopping"; exit 6; }
  fi
done
log "stopped after $chunks chunks (MAX_CHUNKS)"
