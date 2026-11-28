#!/usr/bin/env bash
# Paper 2's miniature toy rung on mscluster110 (the owner's held Blackwell node). Run from the cluster checkout:
#
#   cd /home-mscluster/mmolefe/Playground/PhD/concept-composition
#   nohup bash 2-classifier-guidance/scripts/paper2/toy/run_on_mscluster110.sh stage1 > /datasets/mmolefe/poe_repair_min/toys/paper2/run-stage1.log 2>&1 &
#
# Before the first run, copy the laptop's training canvases and the 6,000-update base there (through mscluster):
#   scp outputs/paper2-toy/train_canvases.npz outputs/paper2-toy/base-6000-updates.pt mscluster:/datasets/mmolefe/poe_repair_min/toys/paper2/out/
#
# Parts, run one at a time; each later part starts only after the README records that the one before passed:
#   stage1   train the 16,000-update base, read stage 1 on fresh seeds (the gate), again on the original seeds, and sort the misses
#   dials    train the four dials
#   stage2 | stage3 | corr | stage4 | stage5
# Logs: /datasets/mmolefe/poe_repair_min/toys/paper2/<part>.log. Results: .../toys/paper2/results/. Checkpoints: .../toys/paper2/out/.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
D=/datasets/mmolefe/poe_repair_min/toys/paper2
S=2-classifier-guidance/scripts/paper2/toy/dials_toy.py
export PAPER2_TOY_OUT=$D/out PAPER2_TOY_RESULTS=$D/results PYTHONFAULTHANDLER=1
mkdir -p "$PAPER2_TOY_OUT" "$PAPER2_TOY_RESULTS"

# refuse to start while another user's process holds the GPU
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    u=$(ps -o user= -p "$p" || true)
    if [ -n "$u" ] && [ "$u" != "$USER" ]; then echo "GPU process $p belongs to $u; not starting"; exit 1; fi
done
[ -f "$PAPER2_TOY_OUT/train_canvases.npz" ] || { echo "missing $PAPER2_TOY_OUT/train_canvases.npz (copy it from the laptop first)"; exit 1; }

step() { local name=$1; shift; echo "$(date -Is) start $name"; "$PY" -u "$S" "$@" > "$D/$name.log" 2>&1; echo "$(date -Is) done $name"; }

case "${1:-}" in
    stage1)
        step base base --updates 16000
        step stage1-fresh stage1 --test-seed0 21024 --tune-seed0 10512 --key stage1
        step stage1-second stage1 --key stage1_second_reading_seeds_20000 --suffix -second-reading-seeds-20000
        step misses-fresh misses --test-seed0 21024 ;;
    base3)
        # the owner's one more base (2026-11-28, "try a wider base model"): width 128, self-attention at 8 by 8 and 4 by 4, 40,000
        # updates; the bar stays R1_BASE_COUNT_MIN 0.95, read on new fresh seeds 22,048 to 23,071 (tuning 11,024 to 11,535).
        # The 6,000- and 16,000-update bases stay in out/ under their own names.
        step base3 base --updates 40000 --width 128 --attn
        "$0" base3-read ;;
    base3-read)
        # the read alone, on the base3 already in out/base.pt (a suffix starting with a dash needs the --suffix= form)
        step stage1-base3 stage1 --test-seed0 22048 --tune-seed0 11024 --key stage1_base3_fresh_seeds_22048 --suffix=-base3-seeds-22048
        step misses-base3 misses --test-seed0 22048 --key stage1_misses_base3 --suffix=-base3 ;;
    dials) step dials dials --kept-per 1024 ;;
    stage2|stage3|corr|stage4|stage5) step "$1" "$1" ;;
    *) echo "usage: $0 stage1|base3|base3-read|dials|stage2|stage3|corr|stage4|stage5"; exit 2 ;;
esac
