#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=1-00:00:00
#SBATCH --job-name=claude-host
#SBATCH --output=/home-mscluster/mmolefe/claude-host/logs/host-%j.log
#SBATCH --signal=B:USR1@7200
#SBATCH --exclude=mscluster171
#
# Keeps this project's Claude Code sessions alive on the cluster, one batch node at a time.
#
# What it does, in order:
#   1. Waits until no other host job holds the lock (/home-mscluster/mmolefe/claude-host/lock). If one does, it writes its own job
#      id to the takeover file, which tells that host to stop its sessions and release the lock.
#   2. Takes the lock and starts one tmux window per line of sessions.txt: a Claude session with that name, Remote Control on.
#      A session that has run before is resumed by name; a new one starts from its prompt file.
#   3. Two hours before its 1-day limit Slurm sends USR1, and the job submits its successor. When the successor starts it asks
#      for the takeover, and this job sends /exit to every session, waits, and releases the lock.
# The conversation history lives in ~/.claude on the shared home filesystem, so a session resumed on the new node carries on.
#
# Start or restart (after a cluster outage, or the first time): scripts/claude-host/up.sh from the laptop.
# Watch: tmux -L claude-host attach, on the node squeue names for job claude-host.
# Permission rules for the sessions are the owner's to set (in the cluster's ~/.claude settings, or CLAUDE_ARGS in host.env);
# this script passes nothing about permissions itself.
set -u
H=${CLAUDE_HOST_DIR:-/home-mscluster/mmolefe/claude-host}   # home, not /datasets: some nodes cannot see /datasets inside a job
REPO=${REPO:-/home-mscluster/mmolefe/Playground/PhD/concept-composition}
CLAUDE=${CLAUDE:-/home-mscluster/mmolefe/.local/bin/claude}
CLAUDE_ARGS=""
[ -f "$H/host.env" ] && . "$H/host.env"          # may set CLAUDE_ARGS, or TEST_COMMAND for a dry run without Claude
LOCK=$H/lock
TAKEOVER=$H/takeover
SOCK=claude-host
ME=${SLURM_JOB_ID:?run under Slurm}
log() { echo "[$(date -u '+%F %T')] $*"; }

NEXT=""
submit_next() {
  NEXT=$(sbatch --parsable "$H/host.sbatch" 2>/dev/null)
  log "submitted the next host job: ${NEXT:-none (sbatch failed)}"
}
on_usr1() { log "USR1: time to hand over"; HANDOVER=1; submit_next; }
HANDOVER=0
trap on_usr1 USR1

log "host $ME on $(hostname -s), repo $REPO"
mkdir -p "$H/started" "$H/logs"

# 1. Wait for the lock.
while [ -s "$LOCK" ] && [ "$(cat "$LOCK")" != "$ME" ]; do
  prev=$(cat "$LOCK")
  if [ -z "$(squeue -h -j "$prev" -o %T 2>/dev/null)" ]; then
    log "lock held by $prev, which is no longer running: taking it"
    break
  fi
  echo "$ME" > "$TAKEOVER"
  log "waiting for host $prev to hand over"
  sleep 20
done
echo "$ME" > "$LOCK"
log "lock taken"

# 2. Start the sessions.
first=1
while IFS='|' read -r name prompt; do
  case "$name" in ''|\#*) continue ;; esac
  if [ -n "${TEST_COMMAND:-}" ]; then
    cmd="$TEST_COMMAND $name"
  elif [ -f "$H/started/$name" ]; then
    cmd="$CLAUDE $CLAUDE_ARGS --resume $name --remote-control $name"
  else
    cmd="$CLAUDE $CLAUDE_ARGS --name $name --remote-control $name \"\$(cat $H/prompts/$prompt)\""
    touch "$H/started/$name"
  fi
  if [ $first = 1 ]; then
    tmux -L $SOCK new-session -d -s host -n "$name" -c "$REPO" "$cmd; echo session $name ended; sleep infinity"
    first=0
  else
    tmux -L $SOCK new-window -t host -n "$name" -c "$REPO" "$cmd; echo session $name ended; sleep infinity"
  fi
  log "started $name"
done < "$H/sessions.txt"

# 3. Run until a successor asks for the takeover.
while true; do
  sleep 30 & wait $!
  # A successor that failed or vanished before taking over is replaced, so one bad node cannot break the chain.
  if [ $HANDOVER = 1 ] && [ "$(cat "$LOCK")" = "$ME" ] && { [ -z "$NEXT" ] || [ -z "$(squeue -h -j "$NEXT" -o %T 2>/dev/null)" ]; }; then
    log "next host ${NEXT:-none} is not queued or running: submitting again"
    sleep 60; submit_next
  fi
  if [ -s "$TAKEOVER" ] && [ "$(cat "$TAKEOVER")" != "$ME" ]; then
    log "takeover asked by $(cat "$TAKEOVER"): stopping sessions"
    for w in $(tmux -L $SOCK list-windows -t host -F '#W' 2>/dev/null); do
      tmux -L $SOCK send-keys -t "host:$w" Escape
      tmux -L $SOCK send-keys -t "host:$w" "/exit" Enter
    done
    sleep 60
    tmux -L $SOCK kill-server 2>/dev/null
    [ "$(cat "$LOCK")" = "$ME" ] && : > "$LOCK"
    log "lock released; host $ME done"
    exit 0
  fi
done
