#!/bin/bash
# Loop 12 step Z8 on an approved shared device (board question 103): the screening pass, its identity check first, one after the other,
# on one card of a node held by another user's Slurm job. Launch over SSH with absolute paths:
#   ssh mscluster107 'SNAP=<snapshot> GPU=1 HOST_JOB=<the other user's job> nohup bash <snapshot>/repo/2-classifier-guidance/scripts/leakage/z8_shared_card.sh \
#     > /datasets/mmolefe/poe_repair_min/loop-12/logs/z8-shared-card.log 2>&1 < /dev/null & disown'
# Rules (environment/hpc/execution-protocol.md step 3, and the owner's watchdog): the card must carry no foreign process over 1 GB at the start of
# each run (the guard in z8.sbatch); every 30 seconds the watchdog reads the card and stops our run, logging why, once a process not ours passes
# 1 GB there; the approval ends when the host job does. Copied from Loop 6's m5_shared_card.sh.
set -uo pipefail
: "${SNAP:?SNAP must name the code snapshot}"; : "${GPU:?GPU must name the approved card}"
D=$SNAP/repo/2-classifier-guidance/scripts/leakage
STEPS=${STEPS:-"screen"}
STATUS=/datasets/mmolefe/poe_repair_min/loop-12/z8/shared-card-status.txt
mkdir -p /datasets/mmolefe/poe_repair_min/loop-12/z8
LIMIT_MIB=1024
HOST_JOB=${HOST_JOB:-}
host_alive() { [ -z "$HOST_JOB" ] || [ -n "$(squeue -j "$HOST_JOB" -h -t R 2>/dev/null)" ]; }
echo "[$(date -u +%H:%M:%S)] shared-device run node=$(hostname -s) device=$GPU wrapper_pid=$$ snap=$SNAP host_job=$HOST_JOB"

foreign_over_limit() {  # prints "pid MiB" for any process on the card outside our process group above the limit
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU" | tr -d ',' | while read -r pid mib; do
    [ -z "$pid" ] && continue
    pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$pg" = "$1" ] && continue
    [ "${mib:-0}" -gt "$LIMIT_MIB" ] && echo "$pid $mib"
  done
}

for step in $STEPS; do
  host_alive || { echo "[$(date -u +%H:%M:%S)] not starting $step: host job $HOST_JOB has ended, so the approval no longer applies"; exit 10; }
  setsid bash "$D/z8.sbatch" "$step" &
  child=$!
  sleep 2
  pgid=$(ps -o pgid= -p "$child" | tr -d ' ')
  echo "[$(date -u +%H:%M:%S)] started $step pid=$child pgid=$pgid"
  while kill -0 "$child" 2>/dev/null; do
    if ! host_alive; then
      echo "[$(date -u +%H:%M:%S)] STOPPED $step: host job $HOST_JOB has ended, so the approval no longer applies"
      kill -TERM -- "-$pgid"; sleep 20; kill -KILL -- "-$pgid" 2>/dev/null
      echo "stopped $step at $(date -u +%FT%TZ): host job $HOST_JOB ended" >> "$STATUS"; exit 10
    fi
    bad=$(foreign_over_limit "$pgid")
    if [ -n "$bad" ]; then
      echo "[$(date -u +%H:%M:%S)] STOPPED $step: a process not ours holds more than $LIMIT_MIB MiB on device $GPU: $bad"
      kill -TERM -- "-$pgid"; sleep 20; kill -KILL -- "-$pgid" 2>/dev/null
      echo "stopped $step at $(date -u +%FT%TZ): foreign process $bad" >> "$STATUS"; exit 9
    fi
    sleep 30
  done
  wait "$child"; rc=$?
  echo "[$(date -u +%H:%M:%S)] finished $step exit=$rc"
  echo "$step exit=$rc at $(date -u +%FT%TZ)" >> "$STATUS"
done
echo "[$(date -u +%H:%M:%S)] DONE shared-device run"
