# 🧪 The readout checker on PoE's own early states: can it read steps 0 to 10, and does its push then make a clean cat and dog

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/25-the-readout-checker-on-poes-own-states.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which follows the rung 2 checker loop recorded in
[the checker loop file](../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md). Its pass marks are in
[its review file](review/25-the-readout-checker-on-poes-own-states.md).

## Table of contents

- [Quick context](#quick-context)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Error Matrix](#error-matrix)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

**Why this plan exists**

Cat × dog fuses or separates within sampler steps 0 to 10. Two kinds of checker exist. The readout checker
reads SDXL's own features and its push redraws the layout: iteration 6 gave two or more animals on 5 of 8
held-out seeds, with two near-clean pairs. The CLIP checker fine-tuned on finished pictures reads them at
0.900, but its push, which passes through the decoded picture, left the layout as plain PoE on 7 or 8 of 8
seeds. The readout checker reads steps 0 and 5 at only 0.63 and 0.74, because it learnt from finished
pictures with noise added, never from the sampler's own early states.

**What it does**

It draws 1,200 cat × dog seeds twice (plain PoE and the joint prompt), saving the early states. The CLIP checker
judges each finished picture, and every state of the run inherits that verdict. The readout checker then trains on
those states beside iteration 4's data. This is discriminator guidance's training set, a sampler's own states,
with labels by content in place of by source, so the joint-against-PoE style difference that broke the
rung 2 discriminator cannot carry the label.

**Considerations**

The clean examples come almost all from joint runs, because plain PoE ends clean on almost no seeds. Joint runs
that finish fused are rejects too, and the reading check uses joint runs only, so the prompt alone cannot pass
it. The rank-32 adapter's runs would add clean endings from the PoE side, but its weights are on `/datasets`,
which is down. The identity push also turned drawn seeds into photographs; the review file reads style beside
the count.

## Goal

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

A readout checker that reads the held-out joint runs' states at steps 0 to 10 at 0.75 or more, and a push sheet
on seeds 9 to 16 judged against plain PoE, the random push and the identity push. The questions and pass marks
are Q1 and Q2 in [the review file](review/25-the-readout-checker-on-poes-own-states.md).

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- mscluster111 and mscluster112 are Blackwell cards (96 GB) that only the `co3_bw` environment can drive;
  `co3` falls back to the CPU in silence ([the node table](../../environment/hpc/nodes.md)).
- Outputs go under `POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch` while `/datasets` returns
  input/output errors ([storage](../../environment/storage.md)).
- Jobs start with `nohup` on the node over SSH, outside Slurm, so the queue does not show them; their logs are
  under `outputs/classifier_guidance_ladder/rung2/logs/` ([the launch protocol](../../environment/hpc/execution-protocol.md)).
- The mkdir claim queue can skip blocks on NFS: after a drawing run, list `trajectories/claims/*` without a
  `done` file, remove them and run again.
- A plain PoE render takes about 5.8 s and a joint render about 5.1 s on a Blackwell.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 1. 🚀 Draw, label, train, push

- [x] **1.1 Draw the runs.** `scripts/rung2/render_trajectories.py` on mscluster111 and 112, seeds 2001 to 3200,
  through `rung2/trajectories.sh`. Done when both logs print `TRAJECTORIES DONE`, the claims without `done` are
  rerun, and `trajectories/states/` holds 2,400 files.
- [ ] **1.2 Label them.** `scripts/rung2/label_trajectories.py`. Prints the count per split, mode and label;
  writes `trajectories/labels.json` and `trajectories/spot-check.png`. Stop and show the spot-check sheet.
- [ ] **1.3 Score iteration 4 on the held-out runs.** `train_readout_pair.py --manifest training_set_catdog_more23.json
  --aug --pair-prompt --traj-labels <labels.json> --score readout-catdog-more3-long/head_step_002000.pt`; the
  baseline for Q1, in `score-trajectories.json`.
- [ ] **1.4 Train.** Iteration 4's settings (`--pair-prompt --aug --early-frac 0.5 --accum 4 --steps 2000`) plus
  `--traj-labels <labels.json> --traj-frac 0.5`, run `readout-catdog-traj`. Every 500 steps the log prints the
  reading check beside the 220-picture evaluation.
- [ ] **1.5 Push sheet.** `render_results.py --readout --head readout_pair/readout-catdog-traj/head_step_002000.pt
  --seeds 9 10 11 12 13 14 15 16 --out-name result-sheet-catdog-traj`, then a comparison overview beside plain
  PoE, the random push and the identity push. Stop and send it with draft calls from full-size tiles.

▶ Next: [2. Judge the sheet](#2--judge-the-sheet)

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow.

### 2. 👁️ Judge the sheet

◀ Needs: [1.5 Push sheet](#1--draw-label-train-push)

- [ ] **2.1 Check the labels.** Open `trajectories/spot-check.png`: rows 1 and 2 are runs called clean, rows 3 and
  4 runs called reject. A row-1 or row-2 picture that is not a clean cat and dog is a labelling error; more than 3 of
  16 means the margin goes up before training.
- [ ] **2.2 Call the push sheet.** For each of seeds 9 to 16 at each strength, place the full-size tile on the
  ladder single < hybrid < two bodies or heads < two clear. Write the counts into the review file's Q2.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Under `outputs/classifier_guidance_ladder/rung2/` on the scratch mount: `trajectories/` (states, pictures,
labels.json, spot-check.png), `readout_pair/readout-catdog-traj/` (checkpoints, check.json with the reading check),
`result-sheet-catdog-traj/`. The verdicts go into the review file and a row of
[the checker loop file](../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

## Error Matrix

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents)

| Symptom | Cause | Fix |
| --- | --- | --- |
| fewer than 2,400 state files after both logs print `TRAJECTORIES DONE` | an NFS mkdir claim was skipped | remove the claims without `done` and run `trajectories.sh` again |
| the reading check sits at 0.50 at every step | the head was scored on a joint-only set it cannot separate, or labels inverted | look at `spot-check.png` and the held-out label counts in `split.json` |
| the push turns drawings into photographs | the training data leans to photographs | note it on the call; the review file reads style beside the count |
