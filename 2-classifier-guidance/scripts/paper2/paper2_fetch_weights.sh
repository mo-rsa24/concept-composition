#!/bin/bash
# paper2_fetch_weights.sh: setup plan task 6, the model weights into HF_HOME on /datasets. CPU work.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_fetch_weights.sh            # fetch, then check offline
#   bash 2-classifier-guidance/scripts/paper2/paper2_fetch_weights.sh --check-only
#
# Only after the owner's yes (setup plan instruction 5.4). Run from the cluster checkout on the held node.
# Fetches SDXL base 1.0 (fp16 files only), its fp16-fix VAE, DINOv2 large, Qwen3-VL-8B-Instruct and
# GroundingDINO tiny, about 26 GB, into /datasets/mmolefe/.cache/huggingface; prints for each whether it
# was cached and its size; checks each config opens with HF_HUB_OFFLINE=1. Passes when every config opens
# and the home folder's caches grew by no more than 1 GB (a download that missed HF_HOME lands there).
# Writes outputs/paper2/paper2-fetch-weights/sizes.txt in the checkout and a copy beside the log.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
cd "$P2_REPO" || exit 7
p2_log paper2-fetch-weights
p2_disk_guard "$P2_HF_HOME"
p2_disk_guard "$P2_ROOT/paper2-fetch-weights"
p2_python
export HF_HOME="$P2_HF_HOME"
HOME_CACHES=("$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.cache/pip")
home_bytes() { du -scb "${HOME_CACHES[@]}" 2>/dev/null | tail -1 | cut -f1; }
BEFORE=$(home_bytes)
"$PY" "$P2_SCRIPTS/paper2_fetch_weights.py" --out outputs/paper2/paper2-fetch-weights/sizes.txt "$@"
rc=$?
AFTER=$(home_bytes)
GROWTH=$(( ${AFTER:-0} - ${BEFORE:-0} ))
LINE="home caches (${HOME_CACHES[*]}) grew by $GROWTH bytes"
echo "$LINE" | tee -a outputs/paper2/paper2-fetch-weights/sizes.txt
cp outputs/paper2/paper2-fetch-weights/sizes.txt "$P2_ROOT/paper2-fetch-weights/sizes.txt"
if [ "$GROWTH" -gt 1000000000 ]; then echo "FAIL: the home folder's caches grew by more than 1 GB"; exit 3; fi
[ $rc -eq 0 ] && echo "PASS: every config opens offline" || echo "FAIL: a config does not open offline (exit $rc)"
exit $rc
