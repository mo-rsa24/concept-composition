#!/bin/bash
# Step M5 on an approved shared device (board question 90): the overfit and 1% rungs for both lists, one after another, on one card of a
# node held by another user's Slurm job. Launch over SSH with absolute paths:
#   ssh mscluster107 'SNAP=<snapshot> GPU=1 HOST_JOB=<the other user's job> nohup bash <snapshot>/repo/2-classifier-guidance/scripts/method_search/m5_shared_card.sh \
#     > /datasets/mmolefe/poe_repair_min/loop-6/logs/m5-shared-card.log 2>&1 < /dev/null & disown'
# Rules (environment/hpc/execution-protocol.md step 3, and the owner's watchdog): the card must carry no foreign process at the start of
# each run; every 30 seconds the watchdog reads the card and stops our run, logging why, once a process not ours passes 1 GB there.
set -uo pipefail
: "${SNAP:?SNAP must name the code snapshot}"; : "${GPU:?GPU must name the approved card}"
D=$SNAP/repo/2-classifier-guidance/scripts/method_search
STEPS=${STEPS:-"train:overfit:current train:tiny:current train:overfit:balanced train:tiny:balanced"}   # "read:reread" reruns the CO3 read
STATUS=/datasets/mmolefe/poe_repair_min/loop-6/m5/shared-card-status.txt
LIMIT_MIB=1024
HOST_JOB=${HOST_JOB:-}   # the other user's Slurm job holding the node; the approval ends when it does
host_alive() { [ -z "$HOST_JOB" ] || [ -n "$(squeue -j "$HOST_JOB" -h -t R 2>/dev/null)" ]; }
echo "[$(date -u +%H:%M:%S)] shared-device run node=$(hostname -s) device=$GPU wrapper_pid=$$ snap=$SNAP"

foreign_over_limit() {  # prints "pid MiB" for any process on the card outside our process group above the limit
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU" | tr -d ',' | while read -r pid mib; do
    [ -z "$pid" ] && continue
    pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$pg" = "$1" ] && continue
    [ "${mib:-0}" -gt "$LIMIT_MIB" ] && echo "$pid $mib"
  done
}

export SHARED=1
failed=""
for item in $STEPS; do
  IFS=: read -r kind stage list <<< "$item"; set -- "$stage" "${list:-}"; step="$stage $list"
  JOB=$D/m5_$kind.sbatch
  # a failed overfit stops that list's 1% rung, as the Slurm dependency would
  if [ "$1" = tiny ] && [ "$failed" = "$2" ]; then echo "[$(date -u +%H:%M:%S)] skipped tiny $2: its overfit failed"; continue; fi
  host_alive || { echo "[$(date -u +%H:%M:%S)] not starting $step: host job $HOST_JOB has ended, so the approval no longer applies"; exit 10; }
  # the guard at the start of m5_train.sbatch refuses a card holding any process or more than 1 GB
  setsid bash "$JOB" "$1" ${2:+"$2"} &
  child=$!
  sleep 2
  pgid=$(ps -o pgid= -p "$child" | tr -d ' ')
  echo "[$(date -u +%H:%M:%S)] started $1 $2 pid=$child pgid=$pgid"
  while kill -0 "$child" 2>/dev/null; do
    if ! host_alive; then
      echo "[$(date -u +%H:%M:%S)] STOPPED $step: host job $HOST_JOB has ended, so the approval no longer applies"
      kill -TERM -- "-$pgid"; sleep 20; kill -KILL -- "-$pgid" 2>/dev/null
      echo "stopped $step at $(date -u +%FT%TZ): host job $HOST_JOB ended" >> "$STATUS"
      exit 10
    fi
    bad=$(foreign_over_limit "$pgid")
    if [ -n "$bad" ]; then
      echo "[$(date -u +%H:%M:%S)] STOPPED $1 $2: a process not ours holds more than $LIMIT_MIB MiB on device $GPU: $bad"
      kill -TERM -- "-$pgid"; sleep 20; kill -KILL -- "-$pgid" 2>/dev/null
      echo "stopped $1 $2 at $(date -u +%FT%TZ): foreign process $bad" >> "$STATUS"
      exit 9
    fi
    sleep 30
  done
  wait "$child"; rc=$?
  echo "[$(date -u +%H:%M:%S)] finished $1 $2 exit=$rc"
  echo "$1 $2 exit=$rc at $(date -u +%FT%TZ)" >> "$STATUS"
  if [ "$1" = overfit ] && [ "$rc" -ne 0 ]; then failed=$2; fi
done
echo "[$(date -u +%H:%M:%S)] DONE shared-device run"
