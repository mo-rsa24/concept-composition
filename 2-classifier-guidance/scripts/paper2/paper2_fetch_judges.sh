#!/bin/bash
# paper2_fetch_judges.sh: species judge step 0.14's fetch, the two judges' models the setup fetch (task 6) leaves out.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_fetch_judges.sh            # fetch, then check offline
#   bash 2-classifier-guidance/scripts/paper2/paper2_fetch_judges.sh --check-only
#
# CPU work, on the held node, from the cluster checkout. Fetches openai/clip-vit-large-patch14 (the CLIP crop judge and
# the old judge) and timm/convnext_large.fb_in22k_ft_in1k (the ImageNet crop judge) into
# /datasets/mmolefe/.cache/huggingface, with the same guards as paper2_fetch_weights.sh: the disk guard on /datasets,
# HF_HOME there, and a stop if the home folder's caches grow past 1 GB. Prints for each whether it was cached, its size,
# and whether it opens offline (CLIP through transformers, the classifier through timm.create_model). Writes
# outputs/paper2/paper2-fetch-weights/sizes-judges.txt in the checkout and a copy beside the log.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
cd "$P2_REPO" || exit 7
p2_log paper2-fetch-judges
p2_disk_guard "$P2_HF_HOME"
p2_disk_guard "$P2_ROOT/paper2-fetch-weights"
p2_python
export HF_HOME="$P2_HF_HOME"
OUT=outputs/paper2/paper2-fetch-weights/sizes-judges.txt
HOME_CACHES=("$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.cache/pip")
home_bytes() { du -scb "${HOME_CACHES[@]}" 2>/dev/null | tail -1 | cut -f1; }
BEFORE=$(home_bytes)
"$PY" "$P2_SCRIPTS/paper2_fetch_weights.py" --set judges --out "$OUT" "$@"
rc=$?
GROWTH=$(( $(home_bytes) - ${BEFORE:-0} ))
echo "home caches (${HOME_CACHES[*]}) grew by $GROWTH bytes" | tee -a "$OUT"
cp "$OUT" "$P2_ROOT/paper2-fetch-weights/sizes-judges.txt"
if [ "$GROWTH" -gt 1000000000 ]; then echo "FAIL: the home folder's caches grew by more than 1 GB"; exit 3; fi
[ $rc -eq 0 ] && echo "PASS: both configs open offline" || echo "FAIL: a config does not open offline (exit $rc)"
exit $rc
