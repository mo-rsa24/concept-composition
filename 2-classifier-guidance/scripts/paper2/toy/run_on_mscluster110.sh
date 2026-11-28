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
#   co3-base | co3-read | co3-ladder   the toy CO3 base (toy_co3.py), its stage 1, its dial stages
#   ref-ladder                         the same dial stages on the width-128 conditional base, the reference row
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
    co3-base)
        # the toy CO3 base (toy_co3.py): one expert network on partial conditions, width 128 with attention, 40,000 updates like
        # base3, saved as out/co3/co3-base.pt; then its stage-1 read
        echo "$(date -Is) start co3-base"
        PAPER2_TOY_WORK=$D/out/co3 "$PY" -u 2-classifier-guidance/scripts/paper2/toy/toy_co3.py train --updates 40000 > "$D/co3-base.log" 2>&1
        echo "$(date -Is) done co3-base"
        "$0" co3-read ;;
    co3-read)
        # stage 1 on the composed sampler, bars R1_BASE_COUNT_MIN and R1_BASE_LOOKALIKE_MAX unchanged, on the fresh seeds fixed in
        # toy_co3.py before this read: test 23,072 to 24,095, tuning 11,536 to 12,047
        export PAPER2_TOY_BASE_KIND=co3 PAPER2_TOY_BASE=$D/out/co3/co3-base.pt PAPER2_TOY_WORK=$D/out/co3 PAPER2_TOY_KEY_PREFIX=co3_ PAPER2_TOY_PICS=co3
        step co3-stage1 stage1 --test-seed0 23072 --tune-seed0 11536 --key stage1_fresh_seeds_23072 --suffix=-co3-seeds-23072
        step co3-misses misses --test-seed0 23072 --key stage1_misses_seeds_23072 --suffix=-co3-seeds-23072 ;;
    co3-ladder)
        # the dial stages on the composed base: keys co3_dials, co3_stage2 ... co3_stage5, files in out/co3/, pictures in results/co3/
        export PAPER2_TOY_BASE_KIND=co3 PAPER2_TOY_BASE=$D/out/co3/co3-base.pt PAPER2_TOY_WORK=$D/out/co3 PAPER2_TOY_KEY_PREFIX=co3_ \
               PAPER2_TOY_STAGE1_KEY=co3_stage1_fresh_seeds_23072 PAPER2_TOY_PICS=co3
        step co3-dials dials --kept-per 1024
        for s in stage2 stage3 corr stage4 stage5; do step "co3-$s" "$s"; done ;;
    ref-ladder)
        # the reference row: the same dial stages on the passing width-128 conditional base (out/base.pt, read on 22,048 to 23,071);
        # keys ref_base3_dials ... ref_base3_stage5, files in out/ref-base3/, pictures in results/ref-base3/
        export PAPER2_TOY_BASE=$D/out/base.pt PAPER2_TOY_WORK=$D/out/ref-base3 PAPER2_TOY_KEY_PREFIX=ref_base3_ \
               PAPER2_TOY_STAGE1_KEY=stage1_base3_fresh_seeds_22048 PAPER2_TOY_PICS=ref-base3
        step ref-base3-dials dials --kept-per 1024
        for s in stage2 stage3 corr stage4 stage5; do step "ref-base3-$s" "$s"; done ;;
    *) echo "usage: $0 stage1|base3|base3-read|dials|stage2|stage3|corr|stage4|stage5|co3-base|co3-read|co3-ladder|ref-ladder"; exit 2 ;;
esac
