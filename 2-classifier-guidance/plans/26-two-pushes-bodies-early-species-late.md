# 🧪 Two pushes, bodies early and species late: does a second push turn two bodies into a cat and a dog

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/26-two-pushes-bodies-early-species-late.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, beside [plan 25](25-the-readout-checker-on-poes-own-states.md), after the rung 2 checker loop in
[the checker loop file](../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md). Its pass marks are in
[its review file](review/26-two-pushes-bodies-early-species-late.md).

## Table of contents

- [Quick context](#quick-context)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

**Why this plan exists**

The readout checker's push turns PoE's one fused animal into two bodies on most held-out seeds, but rarely
into a clean cat and dog: the usual result is two cats, a cat and a kitten, or a hybrid beside a cat. How many
bodies is settled in the first 10 steps; which animal each body becomes is settled later (on lion × meerkat the
species still changed between steps 25 and 50). The checker answers only "clean pair?", so its push makes two
bodies and leaves the species to PoE, whose cat expert mostly wins.

**What it does**

A second checker asks "is there a clear cat?" and "is there a clear dog?" and pushes from step 10 on, raising
whichever animal is missing. It is the fine-tuned CLIP, the checker that reads half-drawn guesses best from
step 10 (0.72 at step 10, 0.82 at step 15 on the clean-pair question) and whose push changes faces and fur, the
change a cat-into-dog needs. The readout checker keeps the first 10 steps.

## Goal

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

A species checker reading both animals at 0.80 by step 15, and a four-row push sheet on seeds 9 to 16 that says
whether bodies-then-species gives more clean cat-and-dog pictures than the bodies push alone. Questions Q1 and
Q2 in [the review file](review/26-two-pushes-bodies-early-species-late.md).

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- mscluster111 and mscluster112 are Blackwell cards (96 GB) driven only by `co3_bw`
  ([the node table](../../environment/hpc/nodes.md)).
- Outputs go under `POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch` while `/datasets` is down
  ([storage](../../environment/storage.md)).
- Jobs start with `nohup` on the node, outside Slurm; logs are under `outputs/classifier_guidance_ladder/rung2/logs/`.
- A folder created on one node can look missing on another for minutes; `ls` its parent on that node first.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 1. 🚀 Train the species checker, then draw the four rows

- [x] **1.1 Code.** `finetune_species.py`, `clip_push.TwoStageHead`, `render_results.py --species --split`;
  checkpoints with one output load unchanged.
- [x] **1.2 Train the species checker.** `rung2/species.sh` on mscluster112. The log prints each output's
  balanced accuracy per step before training and after each epoch, and the Q1 verdict at step 15.
- [ ] **1.3 Draw the bodies row.** `render_results.py --readout --head readout_pair/readout-catdog-identity/head_step_002000.pt
  --window 0 10 --seeds 9 10 11 12 13 14 15 16 --out-name result-sheet-catdog-bodies`.
- [x] **1.4 Draw the species row.** `render_results.py --clip finetune/species-clip/best.pt --window 10 50 ...
  --out-name result-sheet-catdog-species`.
- [x] **1.5 Draw the combined row.** `render_results.py --readout --head <iteration 6> --species finetune/species-clip/best.pt
  --split 10 ... --out-name result-sheet-catdog-both`.
- [ ] **1.6 Compare.** One overview, seeds by rows, the four rows' tiles side by side, sent with draft calls from
  full-size tiles.

▶ Next: [2. Call the sheets](#2--call-the-sheets)

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow.

### 2. 👁️ Call the sheets

◀ Needs: [1.6 Compare](#1--train-the-species-checker-then-draw-the-four-rows)

- [ ] **2.1 Call every full-size tile** on the ladder single < hybrid < two bodies or heads < two clear, and
  write the "two clear" counts per row and strength into the review file's Q2.
- [ ] **2.2 Count the turned pairs:** tiles where the bodies row shows two of the same animal and the combined
  row, same seed and strength, shows a cat and a dog.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents)

Under `outputs/classifier_guidance_ladder/rung2/` on the scratch mount: `finetune/species-clip/best.pt`,
`probe/finetune_species.json`, and the three sheets `result-sheet-catdog-bodies`, `-species`, `-both`. Verdicts go
into the review file and a section of [the checker loop file](../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).
