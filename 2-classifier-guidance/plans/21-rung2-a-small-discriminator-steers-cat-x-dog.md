# 🧪 Rung 2: a small pair-aware discriminator steers SDXL's cat × dog product on held-out seeds

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which is the route's step 10 and the board's row A3q in
[the classifier-guided PoE board](../report/classifier-guided-poe/board.md). Only its first reading
lets rung 3 start.

## Table of contents

- [Quick context](#quick-context)
- [Considerations](#considerations)
- [The claim under test](#the-claim-under-test)
- [Why this plan exists](#why-this-plan-exists)
- [What happens](#what-happens)
- [Description](#description)
- [Purpose](#purpose)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Success and failure outcomes](#success-and-failure-outcomes)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Figure Catalog](#figure-catalog)
- [Error Matrix](#error-matrix)
- [Orchestration](#orchestration)
- [References](#references)
- [Next step](#next-step)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What this is about**

SDXL's product of "a photo of a cat" and "a photo of a dog" draws one fused animal: best-of-16
composed 0 of 128 particles. A small discriminator is trained to tell "a cat and a dog apart" from
the product's own renders and from a lone cat or a lone dog, on about a dozen renders checked by
eye from seeds 1 to 8. Its log-odds gradient is then added to the product at every step, and the
renders are judged on seeds 9 to 12, which no training render came from.

**What it is compared against**

Plain product from the same starting noise; best-of-N plain renders with the same discriminator
picking one, at the same GPU time as one guided render; and a random push whose size matches the
real push at every step. Only how the discriminator is used changes between columns.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**The card**

`mscluster107` GPU 0 under `co3` (Quadro RTX 8000), with `mscluster108` GPU 1 as the fallback.
On 2026-11-17 at 08:45 UTC plan 20 held `mscluster111` and `112` (its 64 px trainings, then its
piece 5 waiters) and `mscluster107` GPU 1. Every column renders on the one card the timing ran on.

**The discriminator's body**

SDXL's own down and middle blocks, frozen, read the noisy latent on the empty prompt; plan 18's
attention-pool head (about 9.3 million weights) is the only trained part and gives one logit. With
41 training tiles, training plan 18's 11 million adapter weights on the trunk as well would let the
discriminator memorise its 17 positives. The check holds out seeds 7 and 8, so it measures reading
new pictures rather than fitting the training ones.

**The compute, measured on `mscluster107` GPU 0**

A plain product render takes 27.5 s and a guided one 42.2 s (medians of 4, 50 steps, 1024 px),
so a guided render costs 1.53 plain ones and peaks at 15.7 GB. The rung needs about 2.1
GPU-hours on that card, run back to back, so about 2.2 hours of wall clock plus the reader's calls
between pieces:

| Piece | Renders or steps | GPU time |
|---|---|---|
| training set, seeds 1 to 8 | 48 renders at ≤ 27.5 s | 22 min |
| the discriminator | about 1,500 steps | about 65 min (inferred from plan 18's `dog-01`, 750 steps per 32 min with its probes) |
| result sheet, seeds 1 to 4 and 9 to 12 | 42 plain-cost renders (plain, best-of-N's second draw, the random push at 3 strengths, the strength-0 check) and 24 guided | 36 min |

Fallback: `mscluster108` GPU 1, also a Quadro RTX 8000 under `co3`, re-timed there before any
column renders, since every column of the sheet renders on one card.

**Best-of-N's pick**

N = 2: the guided render's time over the plain render's, 1.53, rounded. Best-of-N therefore gets
about 30% more GPU time than one guided render (55 s against 42 s), which favours the picker and
makes the support reading harder to reach. The two draws are the seed's own starting noise, which
is the plain product's tile, and one more draw from fresh noise. The discriminator's log-odds on
each draw's final latent at the last sampler timestep picks one.

**The training set**

Positives are only tiles the reader calls one cat and one dog, apart and intact. Joint-prompt
renders that fail are in neither class. Negatives are the product's renders from seeds 1 to 8 and
renders of a lone cat and a lone dog. Every tile that enters training is an intact photo holding
exactly the animals its role names, so a scene where one concept repeats (a pattern of many cats)
is left out. No render from seeds 9 to 16 enters training.

**The calls**

Each tile is called three ways, as the
[rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md#words-this-file-uses)
defines: the animals (one single, one fused, two separate), the picture (intact or damaged), and
the species.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

A pair-aware discriminator's push makes SDXL's cat × dog product draw two separate, intact animals
on seeds it never trained beside, more often than best-of-N with the same discriminator, and its
direction is what does it.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#what-happens) ➡️

The exact toy showed the pair-aware push repairs the product's sampler where the target composes.
This is the cheapest real test on SDXL: one pair, eight seeds, a few checked renders. A pass
starts rung 3; any other reading names the one change worth trying before more compute is spent.

## What happens

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description) ➡️

Five pieces, one per round, each ending on a test: the timing, the training set, the
discriminator, the result renders, then the calls and the reading.

## Description

Navigation: ⬅️ [Previous](#what-happens) | 📋 [TOC](#table-of-contents) | [Next](#purpose) ➡️

Code in `2-classifier-guidance/scripts/rung2/`, run from the cluster checkout. Every large output
under `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/`. Card and
sheets under `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/`.

## Purpose

Navigation: ⬅️ [Previous](#description) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

Settle the board's row A3q, and with it whether rung 3 starts.

## Goal

Navigation: ⬅️ [Previous](#purpose) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar** is the seven readings in
[the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md#the-question-written-before-the-run),
written 2026-11-17 before any render, at strengths 10, 30 and 100. Done when one of them is printed
with its counts, judged by the reader from full-resolution tiles.

**Time box:** 3 sittings, about 4.5 hours of the reader's time, plus the render and training wall
clock.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#success-and-failure-outcomes) ➡️

- [The shared-device path](../../environment/hpc/execution-protocol.md): per-device `nvidia-smi`,
  `torch.cuda.is_available()` asserted inside the script, absolute paths in the SSH launch line.
- [The nodes](../../environment/hpc/nodes.md#partitions): `co3` on `mscluster107` and `108`;
  `co3` on a Blackwell runs on the CPU in silence.
- [Storage](../../environment/storage.md): large outputs on `/datasets` only.
- The fp16 combine differs from fp32 by about 3 grey levels over 50 steps, so the strength-0 check
  compares against plain product rendered in the same loop.

## Success and failure outcomes

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

| Outcome | What it means | What follows |
|---|---|---|
| reading 1, support | the push generalises across seeds and its direction does it | rung 3, the route's step 10b |
| readings 2 or 3 | the discriminator picks, or any push this size separates the animals | the choice the review file names for each |
| readings 4 or 7 | wrecked, or too weak to read | the rerun the review file names, before any verdict |
| readings 5 or 6 | memorised, or a null | more checked renders, or stop before scaling |

## Tasks

Navigation: ⬅️ [Previous](#success-and-failure-outcomes) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Test each piece](#2-test-each-piece)** walked, since this plan was written by the walk and is checked at its close.

- [ ] **0.1** `/verify-plan 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md`, then `/xref-pillar` on it, at the walk's close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build the five pieces

◀ **Needs: nothing first.** This group starts the walk; [group 0](#0--check-this-plan-before-working-from-it) runs at its close.

▶ **Next: [2. Test each piece](#2-test-each-piece)**

- [x] **1.1** The timing: on `mscluster107` GPU 0 under `co3`, assert torch sees the card, render 4
  plain and 4 guided products (seeds 1 to 4, strength 30, plan 18's `dog-01` step 3,000 classifier
  as a stand-in head with the same trunk), time each, and write GPU-hours, wall clock, card,
  fallback and N into [Considerations](#considerations). `scripts/rung2/time_renders.py`.
  ✓ verified 2026-11-17: Quadro RTX 8000, torch 2.5.1; plain 27.2 to 27.7 s, guided 42.1 to
  42.3 s, ratio 1.53, N = 2; the cap fired on 2 to 6% of steps at strength 30 ·
  `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/compute/timing.json`
- [x] **1.2** The training set: on seeds 1 to 8, the joint prompt, "a cat on the left and a dog on
  the right", "a cat and a dog sitting apart", a lone cat, a lone dog, and the plain product; one
  sheet. If fewer than 10 positives pass the reader's calls, more seed 5 to 8 renders with the two
  wordings.
  ✓ verified 2026-11-17: 48 renders on `mscluster107` GPU 0, 127 to 130 s per seed; final latents
  saved beside each tile · `rung2/training_set/`, `training-set/training-set-sheet.png`
- [x] **1.3** The discriminator: SDXL's UNet frozen and read on the empty prompt up to its middle
  block, with plan 18's attention-pool head trained alone on the 41 tiles of
  `scripts/rung2/training_set.json`, noised the same way; its push is the log-odds gradient. First
  a check run holding out seeds 7 and 8 (trained on seeds 1 to 6: 10 positives, 18 negatives;
  checked on 7 positives, 6 negatives): balanced accuracy at least 0.75 on at least 10 of the first
  20 sampler steps, 4 noise draws per tile. Then the final head on all eight seeds.
  `scripts/rung2/train_discriminator.py`, `scripts/rung2/discriminator.py`.
  ✓ verified 2026-11-17: held-out check passed, 13 of 20 sampler steps (steps 7 to 19 at 0.80 to
  1.00; steps 0 to 6 at 0.59 to 0.73); final head on all eight seeds, fit 20 of 20 ·
  `rung2/discriminator/check-holdout-7-8/gate.json`, `rung2/discriminator/final-all-seeds/head_step_001500.pt`,
  `discriminator/held-out-accuracy-per-step.png`
- [ ] **1.4** The result renders: a random-push mode in the product sampler that leaves strength 0
  untouched; the best-of-N pick; strength 0 against plain product on seed 1; then seeds 1 to 4 and
  9 to 12 with plain product, best-of-N, guided at 10, 30 and 100, and the random push at the same
  three, on one card; one sheet.
- [ ] **1.5** The calls as a CSV, the reading printed with its counts, the card, A3q's lines
  ticked, the route rows, the board redrawn with `--step 10`.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Test each piece

◀ **Needs: [1. Build the five pieces](#1--build-the-five-pieces)**

- [x] **2.1** Read the timing block in [Considerations](#considerations) and say yes or change it.
  You should see seconds per plain and per guided render on one named card, and an N.
  ✓ confirmed by hand 2026-11-17: "yes", taking the product of "a cat" and "a dog", the wording
  step 5's strength-0 check used.
- [x] **2.2** Call every training-set tile at full resolution, the three calls of
  [instruction 16 of the route](route-classifier-guided-poe-from-toy-to-animals.md#16--call-each-rungs-sheets-at-full-resolution-in-steps-10-10c-and-10e).
  ✓ confirmed 2026-11-17: the reader accepted Claude's calls on the 48 tiles ("agree"), then
  delegated the calls on 25 further fresh-noise tiles with veto after; 73 tiles, 17 positives,
  24 intact negatives · `training-set/calls.csv`, `training-set/problem-seeds.csv`
- [ ] **2.3** Read the discriminator's balanced accuracy per sampler step against task 1.3's rule.
- [ ] **2.4** Call every result tile at full resolution, the same way.
- [ ] **2.5** Read the printed reading and its counts against
  [the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md).

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the walk's close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

`timing.json`; the training-set renders and their calls; the discriminator checkpoint and its
accuracy per step; the result tiles with per-step push logs; the per-tile calls as a CSV; the
card; the review file answered.

## Figure Catalog

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

| Figure | What it shows |
|---|---|
| `training-set-sheet.png` | seeds 1 to 8 (rows) by the three pair wordings, lone cat, lone dog and plain product (columns) |
| `result-sheet.png` | seeds 1 to 4 beside 9 to 12 (rows) by plain product, best-of-N, guided and random push at 10, 30 and 100 (columns) |

## Error Matrix

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Symptom | Cause | Fix |
|---|---|---|
| a GPU job at 0 MiB on its card | `co3` on a Blackwell | `co3_bw` on `mscluster110` to `112` |
| the run dies before `nohup` starts | a relative path in the SSH launch line | `cd` to the cluster checkout first |
| out of memory in the discriminator's backward pass | the trunk's activations | gradient checkpointing on the trunk |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

The walk map is `2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/EXECUTE_WALK.md`.

## References

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [the route, step 10](route-classifier-guided-poe-from-toy-to-animals.md#10--rung-2-a-small-discriminator-steers-cat--dog-on-held-out-seeds)
- [A3q's page](../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md)
- [the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md)
- [plan 18, the dog classifier](18-a-dog-classifier-trained-the-papers-way.md)

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

The route's step 10b plans rung 3 if reading 1 holds; otherwise the review file's next step for
the reading that held.
