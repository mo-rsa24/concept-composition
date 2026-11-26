#!/usr/bin/env bash
# Start a background Claude loop the moment a progress-board step turns green. It costs no tokens while it waits.
#
#   scripts/start_when_step_done.sh <step id> <session name> <prompt file>
#
# Run it in a tmux session so it outlives this terminal. It starts the loop once, does nothing if a session with that
# name already exists, and waits while `claude agents` cannot answer or background sessions cannot write to this checkout.
set -uo pipefail

STEP=$1
NAME=$2
PROMPT=$3
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
[ -f "$PROMPT" ] || { echo "no prompt file at $PROMPT" >&2; exit 1; }

state() {
  python3 -c "import json,sys; print(next(s['state'] for s in json.load(open('artifacts/notes/progress/board.json'))['steps'] if s['id']==sys.argv[1]))" "$STEP"
}

shared_writes() {
  python3 -c "import json; print(json.load(open('.claude/settings.json')).get('worktree', {}).get('bgIsolation', ''))"
}

WAIT="${WAIT_SECONDS:-300}"
echo "$(date '+%F %T') waiting for step $STEP to finish before starting: $NAME"
while [ "$(state 2>/dev/null)" != "done" ]; do
  sleep "$WAIT"
done

until [ "$(shared_writes 2>/dev/null)" = "none" ]; do
  echo "$(date '+%F %T') waiting: background sessions cannot write to this checkout yet"
  sleep "$WAIT"
done

# Exit 0: a session with this name exists. Exit 1: none does. Anything else: the listing could not be read.
exists() {
  local listing
  listing="$(claude agents --json --all 2>/dev/null)" || return 2
  printf '%s' "$listing" | python3 -c "
import json, sys
try:
    rows = json.load(sys.stdin)
    found = any(r.get('name') == sys.argv[1] for r in rows)
except Exception:
    sys.exit(2)
sys.exit(0 if found else 1)
" "$NAME"
}
while :; do
  exists
  found=$?
  [ "$found" -le 1 ] && break
  echo "$(date '+%F %T') waiting: claude agents did not answer"
  sleep "$WAIT"
done
if [ "$found" -eq 0 ]; then
  echo "$(date '+%F %T') $NAME already exists, so nothing was started"
  exit 0
fi

echo "$(date '+%F %T') step $STEP is done, starting: $NAME"
OUTPUT="$(claude --bg --permission-mode auto --model claude-opus-5-5 --effort high --name "$NAME" \
  "/loop Follow $PROMPT to its end. Read artifacts/notes/progress/prompts/shared-rules.md on every tick." 2>&1)"
echo "$OUTPUT"
ID="$(printf '%s' "$OUTPUT" | grep -oE 'backgrounded · [0-9a-f]{8}' | grep -oE '[0-9a-f]{8}$' | head -1)"
[ -n "$ID" ] || { echo "$(date '+%F %T') could not read the new session id" >&2; exit 1; }

# A background session can register and never start. It has started once its transcript exists.
started() { ls "$HOME"/.claude/projects/*/"$ID"*.jsonl >/dev/null 2>&1; }
for attempt in 1 2 3; do
  for _ in $(seq 1 "${START_CHECKS:-24}"); do
    started && { echo "$(date '+%F %T') $NAME is running as $ID"; exit 0; }
    sleep "${START_SECONDS:-5}"
  done
  echo "$(date '+%F %T') $ID has not started, restarting it (attempt $attempt of 3)"
  claude respawn "$ID"
done
started && { echo "$(date '+%F %T') $NAME is running as $ID"; exit 0; }
echo "$(date '+%F %T') $NAME did not start after 3 restarts" >&2
exit 1
