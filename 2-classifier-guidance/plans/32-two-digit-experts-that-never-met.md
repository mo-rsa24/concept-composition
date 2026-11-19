# 🧪 Two digit experts that never met: does the product draw one digit from each, and do the checkers' pushes make it reliable

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/32-two-digit-experts-that-never-met.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, a follow-on to the route's
[step 7b](route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be).
Its bars are in [its review file](review/32-two-digit-experts-that-never-met.md).

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

Step 7b composed two digits inside one U-Net that had seen every digit, where both checkers' pushes
gave two clear, right digits on 285 of 320 fresh held-out seeds. A product of experts is best
justified when the experts were trained apart and are composed at sampling time. This plan asks
whether the 7b pushes still work there. The reasoning, the words and the bars are in
[the review file](review/32-two-digit-experts-that-never-met.md#why-this-is-asked).

**What it does**

Trains a 0–4 expert and a 5–9 expert, each on one digit per canvas in the left or right half, and
a fixed-side pair of experts as the control. Composes each pair with a blank-canvas base. Reads
the plain product, a random push, both 7b pushes retrained to read pixels, and a push from two
one-half ink checkers that never saw two digits on a canvas.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**Where it runs**

On `bigbatch` RTX 3090 cards, with the dead-card exclude list plan 24 used, at most 6 jobs at once.
The four experts train together.

**Where outputs go**

`/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/two-experts/`, as plan 24 did while
`/datasets` returned `Input/output error`; the disk guard reads `/home-mscluster`. Check
`/datasets` first and use it if it is back.

**Who calls the sheets**

Claude calls each expert-alone sheet and the headline column of every cell by eye at full
resolution; the reader can veto any call.

**Seeds**

Seeds 0 to 63 are evaluation only. Every render a checker trains on comes from seed 1,000 up, and
the trainer refuses a lower seed.

**Commits**

On `exec/classifier-guidance-ladder`, since `scripts/cluster.sh run` needs a pushed tree. The
cluster checkout is fast-forwarded by hand, as in plan 24, because other sessions leave
uncommitted edits there.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

Composed with a blank base, a 0–4 expert and a 5–9 expert that never saw each other's digits give
two clear, right digits on different halves on some seeds but not reliably; step 7b's two pushes,
retrained to read pixels, make it reliable; and the fixed-side experts compose with no push.

## Goal

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar** is [the review file](review/32-two-digit-experts-that-never-met.md#the-bars-written-before-the-run)'s,
unchanged after the run.

**Time box:** one day of wall time, about 3 hours of the reader's. If the fixed-side control fails
its bar, the walk stops and looks at the base and the sampler before anything else.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- [The nodes](../../environment/hpc/nodes.md#partitions): `bigbatch` RTX 3090 24 GB; probe or
  exclude dead cards; 6 jobs per user.
- [Storage](../../environment/storage.md): large outputs on `/datasets`; the scratch folder while
  it is down ("When `/datasets` is down").
- The laptop: `/home/molef/micromamba/envs/sdxl-local/bin/python` (torch with CUDA); MNIST's raw
  files at `/home/molef/PhD/replications/original-superdiff/data/MNIST/raw`.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Check each sheet](#2-check-each-sheet)** walked.

- [ ] **0.1** `/verify-plan 2-classifier-guidance/plans/32-two-digit-experts-that-never-met.md`, then `/xref-pillar` on it, at the walk's close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build, train and render

◀ **Needs: nothing first.**

▶ **Next: [2. Check each sheet](#2-check-each-sheet)**

- [ ] **1.1** Half-canvas digits: `Canvases.one_half(d, side)` in `scripts/mnist/canvas_sheet.py`,
  using `two()`'s per-digit layout; a 16-tile sheet of each side to check it.
- [ ] **1.2** Expert training switches in `scripts/mnist/canvas_unet.py`: which digits (`0-4` or
  `5-9`), which side (`either`, `left`, `right`), no two-digit canvases; the 7b recipe otherwise.
- [ ] **1.3** Train the four experts on `bigbatch` together.
- [ ] **1.4** The two-expert sampler with the closed-form blank base, `scripts/mnist/two_experts.py`
  (new): each expert alone on 64 seeds per digit, then plain product and the random push for the
  five pairs on both expert pairs; the identity check that s = 0 gives byte-identical canvases to
  plain product.
- [ ] **1.5** The three pixel checkers, `scripts/mnist/pixel_checker.py` (new): pair-aware with two
  on-policy passes on its own pushed renders, separate-label, and one-half ink; each one's held-out
  accuracy printed.
- [ ] **1.6** The guided cells: both pushes and the one-half ink push on the side-mixed experts, both
  pushes on the fixed-side experts.
- [ ] **1.7** One sheet per cell, the judge's counts, Claude's eye calls of the headline column,
  the reading against the review file, and the results card.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Check each sheet

◀ **Needs: [1. Build, train and render](#1--build-train-and-render)**

- [ ] **2.1** Open each expert-alone sheet: one digit per tile, on the left or right half, never
  the other expert's digits. Veto any of Claude's calls you disagree with.
- [ ] **2.2** Open the fixed-side plain-product sheet: one digit in each half on almost every tile.
  If many tiles are fused or empty, say so before anything else is read.
- [ ] **2.3** Open the side-mixed cells side by side (none, random, both pushes, one-half ink) and
  look at the same seed across them; veto any headline call.

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the walk's close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Four expert checkpoints, three checkers, one sheet per cell with counts and calls, under the
output folder; sheets and figures filed under
`2-classifier-guidance/artifacts/results/do-two-experts-that-never-met-compose/` with its card.

## Error Matrix

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Symptom | Cause | Fix |
|---|---|---|
| `Input/output error` on `/datasets` | the mount is down | write to the scratch folder |
| a job dies in under a second on `bigbatch` | a dead card | add the node to the exclude list and resubmit |
| an expert alone draws the other range's digits | the digit filter did not apply | print the label counts the trainer saw; retrain |
| fixed-side plain product fails its bar | the blank base's sign or scale, or the sampler | check ε_blank against a blank canvas noised by hand at three steps |
| the pushes wreck strokes | too strong at eta 1.0 | report it; do not tune, s and the cap are fixed by 7b |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

The four experts train in parallel; the checkers and cells are chained after them with
`--dependency=afterok`, so they keep running when the laptop is off. A self-paced `/loop` reads the
queue and the outputs, makes the calls, and applies the review file's bars.

## Next step

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents)

If the both-pushes bar is met, anywhere placement (the review file's first open item). If only the
fixed-side control composes, the result goes to the paper as the condition separately trained
experts need: a shared background.
