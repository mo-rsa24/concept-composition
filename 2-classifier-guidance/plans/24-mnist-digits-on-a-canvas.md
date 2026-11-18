# 🧪 MNIST digits on a canvas: does a pair-aware checker's push turn fused digits into two, and how good must the checker be

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/24-mnist-digits-on-a-canvas.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which is the route's [step 7b](route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be)
and the board's row [A2b](../report/classifier-guided-poe/a2b-mnist-digits-on-a-canvas.md) in
[the classifier-guided PoE board](../report/classifier-guided-poe/board.md). Its bar is
[the A2b review file](review/a2b-mnist-digits-on-a-canvas.md).

## Table of contents

- [Quick context](#quick-context)
- [Considerations](#considerations)
- [The claim under test](#the-claim-under-test)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Error Matrix](#error-matrix)
- [Orchestration](#orchestration)
- [Next step](#next-step)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**Why this plan exists**

Rung 2 on SDXL cat × dog stalled on the checker: its best held-out [balanced accuracy](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#balanced-accuracy-the-hit-rate-on-each-class-averaged) was 0.655,
CLIP on finished pictures reached 0.771, and every training picture cost a render and a label. On
MNIST the labels are exact and free, so checkers can be made weaker on purpose, and the compose
count against checker accuracy says whether a cat × dog checker near 0.655 could steer at all.
Step 10's next move (more cat × dog data, or a checker that judges each animal) waits on that curve.

**What it does**

Three class-conditional pixel U-Nets train on 32 × 64 canvases of real MNIST digits, at two-digit
shares 0, 0.01 and 0.1, with five digit pairs never shown together. The plain product of two digit
experts is read first. Where it fuses, a pair-aware checker on the frozen U-Net's decoder features
pushes the product, beside a separate-label control<sub>[1](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/02-the-product-as-two-classifiers.md#23-pushing-harder-with-the-same-two-classifiers "Classifier guidance by hand, chapter 2, section 2.3: more of the same two separate-label pushes on the product base, where the fused animal's share rises and the pair's falls"), [2](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/04-the-separate-label-push-built.md "Classifier guidance by hand, plan 04: the separate-label push built and measured on the four-picture toy at strengths 0, 1, 3 and 10")</sub>, a random push and best-of-N. Then the checker
is made weaker by less data and by flipped labels, and the cap is swept. The full design, the
checker's recipe table and the columns are in route step 7b and are not repeated here.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**Where it runs**

On `bigbatch` RTX 3090 cards, pinned to a node that passed a two-minute probe job, at most 6 jobs at
once. The 200-step timing runs on the laptop's RTX 4070 and on one probed 3090 before the three
U-Nets launch.

**Where outputs go**

`/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/` while `/datasets` returns
`Input/output error`; the reader's choice. Models are a few MB and canvases kilobytes, so the
disk guard reads `/home-mscluster`, which once hit 100%. The laptop's own runs write to
`data/mnist-canvas/` (gitignored).

**Who calls the sheets**

Claude calls the judge's 64 tiles, the plain-product sheets and the five-version sheet by eye at
full resolution, and the walk continues on those calls; the reader can veto any call afterwards.
Until the reader checks the 64 tiles, the judge gate reads "validated against Claude's calls".

**Seeds**

Seeds 0 to 63 are evaluation only. Every draw a checker trains on comes from seed 1,000 up, and
the trainer refuses a lower seed and prints the range it read.

**Commits**

The walk commits and pushes on `exec/classifier-guidance-ladder`, because `scripts/cluster.sh run`
needs a committed, pushed tree for the cluster to fast-forward to.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

Where the plain product fuses two digits, a pair-aware checker's push makes it draw two clear,
right digits on seen and held-out pairs, more than a random push, a separate-label push and
best-of-N at matched compute; and that push still composes at some checker accuracy below 0.771.

## Goal

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar** is [the A2b review file](review/a2b-mnist-digits-on-a-canvas.md)'s, read on share 0.01
if more than one share fuses, unchanged. The plain-product rule there decides whether the checker is
built at all; the 0.771 rule there decides step 10's move.

**Time box:** sittings M1 to M3, about 6 hours of the reader's time. If M3 ends without the
five-version sheet, the walk stops and looks at why.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- [The nodes](../../environment/hpc/nodes.md#partitions): `bigbatch` RTX 3090 24 GB; idle nodes are
  often dead cards, so probe and pin; 6 jobs per user.
- [Storage](../../environment/storage.md): large outputs on `/datasets`; the scratch folder above is
  the reader's exception while `/datasets` is down.
- The laptop: `/home/molef/micromamba/envs/sdxl-local/bin/python` (torch with CUDA, no scipy) on an
  RTX 4070 Laptop GPU with 8 GB; MNIST's raw files at
  `/home/molef/PhD/replications/original-superdiff/data/MNIST/raw`.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Check each sheet](#2-check-each-sheet)** walked.

- [ ] **0.1** `/verify-plan 2-classifier-guidance/plans/24-mnist-digits-on-a-canvas.md`, then `/xref-pillar` on it, at the walk's close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build, train and render

◀ **Needs: nothing first.**

▶ **Next: [2. Check each sheet](#2-check-each-sheet)**

- [ ] **1.1** The canvas U-Net and its trainer, `scripts/mnist/canvas_unet.py` (new), reusing
  `Canvases` and `HELD_OUT` from `scripts/mnist/canvas_sheet.py`; time 200 training steps on the
  laptop and on one probed 3090, write each share's projected time, and stop for the reader's yes.
- [ ] **1.2** Train the three U-Nets at shares 0, 0.01 and 0.1 on `bigbatch`.
- [ ] **1.3** The judge with a fused test, `scripts/mnist/judge.py` (new), reusing `blobs`,
  `crop28` and `judge_net` from `scripts/mnist/judge_sheet.py`; a 64-tile sheet of pasted canvases,
  Claude's calls, and the judge's agreement with them.
- [ ] **1.4** Plain product on 7 & 8 and the five held-out pairs, seeds 0 to 63, all three shares,
  and the support check at share 0.1; one sheet per share; the plain-product rule applied.
- [ ] **1.5** The pair-aware checker and the separate-label heads on the frozen decoder features,
  `scripts/mnist/checker.py` (new), with the toy's weighted loss from
  `3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_classifier_push.py`,
  on draws from seed 1,000 up.
- [ ] **1.6** The five-version sheet on seeds 0 to 63 for 7 & 8 and the five held-out pairs, with
  the push in `poe_repair/methods/_classifier_guided.py`'s form and cap; every guided column
  byte-identical to plain product at s = 0.
- [ ] **1.7** Sweep 1 (1%, 5%, 25% of the data; 10% and 20% flipped labels) and sweep 2 (cap 0.1,
  0.3, 1.0), one knob per run; the accuracy-against-compose figure with 0.655 and 0.771 marked.
- [ ] **1.8** The reading against the review file, the card, A2b's page, the board redrawn with
  `--step 7b`, and the route's new rows.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Check each sheet

◀ **Needs: [1. Build, train and render](#1--build-train-and-render)**

- [ ] **2.1** Look over Claude's calls on the judge's 64 tiles; veto any you disagree with.
- [ ] **2.2** Look over the plain-product sheet per share against Claude's calls.
- [ ] **2.3** Look over the five-version sheet's calls at full resolution.
- [ ] **2.4** Read the sweep figure, and say yes or no to the step 10 edit the 0.771 rule proposes.

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the walk's close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Three U-Net checkpoints, the judge, the checkers, the plain-product, judge and five-version sheets,
the sweep figure with its sidecar, all under the scratch folder; sheets, calls and figures filed
under `2-classifier-guidance/artifacts/results/how-good-must-the-checker-be-on-mnist/` with its card.

## Error Matrix

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Symptom | Cause | Fix |
|---|---|---|
| `Input/output error` on `/datasets` | the mount is down | write to the scratch folder |
| a job dies in under a second on `bigbatch` | a dead card on an idle node | probe idle nodes with a two-minute job, pin a healthy one |
| out of memory on the laptop's 8 GB | batch too large | halve the batch and say so |
| a share's projected time passes 2 hours | the U-Net is too wide | stop with the number and propose fewer channels |
| the judge disagrees on more than 6 of 64 | blob cutting or the fused test | show the misses, try one fix, then fall back to eye calls |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

The computable stretches are chained in Slurm with `--dependency=afterok`, so they keep running
when the laptop is off: the three U-Nets, then plain product and the support check, then the
checkers, then the five-version sheet, then the sweeps. A self-paced `/loop` in the walk's session
reads the queue and the outputs, makes the calls and applies the written rules; it lives in the
laptop's Claude Code session and stops when that session does. The walk map is
`2-classifier-guidance/artifacts/drips/a2b-mnist-digits-on-a-canvas/EXECUTE_WALK.md`.

## Next step

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents)

Route step 10, carrying the move the 0.771 rule picked.
