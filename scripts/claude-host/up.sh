#!/bin/bash
# From the laptop: make sure one Claude host job is running on the cluster's batch partition, and copy the host files to
# /home-mscluster/mmolefe/claude-host/ first. Safe to run any time: it submits only when no host job is queued or running.
#   scripts/claude-host/up.sh          copy the files, then submit if none is running
#   scripts/claude-host/up.sh status   show the host jobs and the lock
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
H=/home-mscluster/mmolefe/claude-host
LOGIN=${CLUSTER_LOGIN:-mscluster}
ssh_() { timeout 90 ssh -o ConnectTimeout=20 "$LOGIN" "$@"; }

if [ "${1:-}" = status ]; then
  ssh_ "squeue -u mmolefe -h -n claude-host -o '%i %T %M %L %N'; echo lock: \$(cat $H/lock 2>/dev/null); tail -3 \$(ls -t $H/logs/host-*.log 2>/dev/null | head -1) 2>/dev/null"
  exit
fi

ssh_ "mkdir -p $H/logs $H/prompts $H/started" || { echo "cluster not answering"; exit 1; }
timeout 120 scp -q -o ConnectTimeout=20 "$HERE/host.sbatch" "$HERE/sessions.txt" "$LOGIN:$H/" || { echo "copy failed"; exit 1; }
timeout 120 scp -q -o ConnectTimeout=20 "$HERE"/prompts/*.txt "$LOGIN:$H/prompts/" || { echo "prompt copy failed"; exit 1; }
ssh_ "if [ -n \"\$(squeue -u mmolefe -h -n claude-host -o %i)\" ]; then echo 'a host job is already queued or running:'; squeue -u mmolefe -h -n claude-host -o '%i %T %N'; else sbatch $H/host.sbatch; fi"
