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
- [Engagement Instructions](#engagement-instructions)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What this is about**

SDXL's product of "a photo of a cat" and "a photo of a dog" draws one fused animal: best-of-16
composed 0 of 128 particles. A small discriminator is trained to tell "a cat and a dog apart" from
the product's own renders and from a lone cat or a lone dog, on about a dozen renders checked by
eye from seeds 1 to 8. Its [log-odds gradient](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/03-the-discriminator-and-the-density-ratio.md#32-its-log-odds-slope-is-a-difference-of-scores) is then added to the product at every step, and the
renders are judged on seeds 9 to 12, which no training render came from.

**What it is compared against**

Plain product from the same starting noise; best-of-N plain renders with the same discriminator
picking one, at the same GPU time as one guided render; and a random push whose size matches the
real push at every step. Only how the discriminator is used changes between columns.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**The card**

One Quadro RTX 8000 under `co3`. Every column renders on the one card the timing ran on. Which
node and GPU each run used is in
[the review file's Runs table](review/a3q-cat-x-dog-quick-discriminator-test.md#runs).

**The discriminator's body**

SDXL's own down and middle blocks, frozen, read the noisy latent on the empty prompt; plan 18's
attention-pool head (about 9.3 million weights) is the only trained part and gives one logit. With
41 training tiles, training plan 18's 11 million adapter weights on the trunk as well would let the
discriminator memorise its 17 positives. The check holds out seeds 7 and 8, so it measures reading
new pictures rather than fitting the training ones.

**The compute, measured on that card**

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

The exact toy showed the pair-aware push<sub>[1](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/04-the-dropped-term.md#44-the-pair-aware-push "Classifier guidance by hand, chapter 4, section 4.4: the pair-aware push, a discriminator's log-odds trained on joint-caption draws against the product's own draws, applied with a weight on the product"), [2](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/08-the-pair-aware-push-built.md "Classifier guidance by hand, plan 08: the pair-aware push built and measured on the four-picture toy at weights 0, 0.5, 1 and 2")</sub> repairs the product's sampler where the target composes.
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

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**

### 1. 🔧 Build the five pieces

◀ **Needs: nothing first.** This group starts the walk; [group 0](#0--check-this-plan-before-working-from-it) runs at its close.

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

  <a href="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/training-set/training-set-sheet.png"><img src="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/training-set/training-set-sheet.png" width="140" alt="Seeds 1 to 8 down the rows by six prompts across ('a cat and a dog', 'a cat on the left and a dog on the right', 'a cat and a dog sitting apart', 'a cat', 'a dog', and the plain product of the two): the three pair prompts mostly show a cat and a dog together, some as two dogs or two cats or line drawings, the lone prompts one animal each, and the product column one fused cat-dog animal per seed"></a>

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

  <a href="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-per-step.png"><img src="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-per-step.png" width="200" alt="Balanced accuracy on y from 0.5 to 1.0 against sampler step 0 to 19 of 50 on x, for held-out seeds 7 and 8 (7 positives, 6 negatives, 4 noise draws each): steps 0 to 6 sit at 0.59 to 0.73 in a shaded band under a dashed line at 0.75, steps 7 to 19 at 0.80 to 1.00, and a dotted line at 0.5 is labelled coin flip"></a>

- [x] **1.4** The result renders: a random-push mode in the product sampler that leaves strength 0
  untouched; the best-of-N pick; strength 0 against plain product on seed 1; then seeds 1 to 4 and
  9 to 12 with plain product, best-of-N, guided at 10, 30 and 100, and the random push at the same
  three, on one card; one sheet.
  ✓ verified 2026-11-17: strength 0 identical to plain product on seed 1; the random push's norm
  over the guided push's 1.0000 at every step of all 24 random renders; the cap fired on 2 to 42%
  of steps at s = 10, 80 to 98% at 30, 100% at 100; 64 tiles on `mscluster107` GPU 0 ·
  `rung2/results/result.json`, `result-sheet/result-sheet.png`

  <a href="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet/result-sheet.png"><img src="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet/result-sheet.png" width="140" alt="Seeds 1 to 4 (training) and 9 to 12 (held out) down the rows by eight columns across: plain product, best of 2 picked by the discriminator, guided at s = 10, 30 and 100, and a random push of the same size at s = 10, 30 and 100. The plain product shows one fused cat-dog animal per seed; the guided columns change the picture, to a black-and-white dog on seed 1, two drawn animals side by side on seed 4, line drawings on seed 11 with a car at s = 100, and patchy cut-out textures at s = 30 and 100; the random-push columns stay close to the plain product"></a>

  - [ ] **1.4.1** The cap-0.3 rerun: seeds 1 to 4 and 9 to 12 at s = 100 with the push capped at
    0.3 of the product prediction's norm, the rerun [the review file's reading 7](review/a3q-cat-x-dog-quick-discriminator-test.md#the-question-written-before-the-run)
    names for a push the cap clipped. At s = 100 the cap fired on 100% of steps (task 1.4).
    `scripts/rung2/render_results.py` with the cap set. Not run: it needs the first
    discriminator's head, which is on `/datasets` only, and `/datasets` has returned
    Input/output error since about 13:10 UTC on 2026-11-17.
  - [x] **1.4.2** The v2 discriminator: task 1.3's head and held-out check, retrained with the
    pair-prompt renders that failed added as negatives (`scripts/rung2/training_set_v2.json`: 64
    tiles, 17 positives, 47 negatives), the reader's choice after the diagnostics under
    [`diagnostics/`](../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/diagnostics/). Run from `/home-mscluster/mmolefe/poe_repair_min_scratch`
    while `/datasets` is down.
    ✓ verified 2026-11-17: the held-out check failed, 0 of 20 sampler steps at balanced accuracy
    0.75 or more on seeds 7 and 8 (0.53 to 0.63 at every step), so no push was rendered;
    re-rendered latents matched the originals on 73 of 73 pictures ·
    `discriminator/held-out-accuracy-v1-against-v2.png`

    <a href="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-v1-against-v2.png"><img src="../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-v1-against-v2.png" width="200" alt="Held-out balanced accuracy on y from 0.5 to 1.0 against sampler step 0 to 19 of 50 on x, seeds 7 and 8: v1 (negatives were lone animals and plain products) rises from about 0.6 to 0.80 to 1.00 from step 7, passing 13 of 20 steps; v2 (negatives also include hybrids and wrong pairs from the pair prompts) stays between 0.53 and 0.63 at every step, under the dashed bar at 0.75, failing 0 of 20"></a>

- [ ] **1.5** The calls as a CSV, the reading printed with its counts, the card, A3q's lines
  ticked, the route rows, the board redrawn with `--step 10`.

▶ **Next: [2. Test each piece](#2-test-each-piece)**

### Close out. 🔄 Record what this plan taught

◀ **Needs: [group 0](#0--check-this-plan-before-working-from-it)** done at the walk's close, and
every group above attempted, including any that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md`
  - Done when: each failure has an entry in `environment/known-failures.md`, or there were none.
- [ ] **Bring the tree current** with what ran.
  - Paste: `/sync-plan-tree 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md — <one line on the reading that held>`
  - Done when: this plan's ticks, the route's rows and the board's row A3q match what ran.

▶ **Next: [the next step](#next-step).**

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

#### Pending: to be generated from prompts

Nothing pending. The scope's [illustrated map](diagram-prompts.md) holds four prompts, all
rendered; none is drawn for this plan alone.

#### Generated during plan execution

All in `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/`.

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| [`training-set/training-set-sheet.png`](../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/training-set/training-set-sheet.png) | — | the training set's renders | task 1.2 | ✅ on disk, worked by task 1.2 and instruction 2.2 | seeds 1 to 8 (rows) by the three pair wordings, lone cat, lone dog and plain product (columns) |
| [`discriminator/held-out-accuracy-per-step.png`](../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-per-step.png) | — | whether the discriminator reads new pictures | task 1.3 | ✅ on disk, worked by task 1.3; instruction 2.3 reads it | balanced accuracy on y from 0.5 to 1.0 against sampler step 0 to 19 of 50 on x, held-out seeds 7 and 8 (7 positives, 6 negatives, 4 noise draws each), with the 0.75 bar dashed |
| [`discriminator/held-out-accuracy-v1-against-v2.png`](../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-v1-against-v2.png) | — | the same check for the first discriminator and the v2 retrain | task 1.4.2 | ✅ on disk, worked by task 1.4.2 | the same axes, one line per discriminator |
| [`result-sheet/result-sheet.png`](../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet/result-sheet.png) | — | the result renders | task 1.4 | ✅ on disk, worked by task 1.4; instruction 2.4 calls its tiles | seeds 1 to 4 beside 9 to 12 (rows) by plain product, best-of-N, guided and random push at 10, 30 and 100 (columns) |

#### Organization workflow

1. Each task saves its figure into the results folder above.
2. Each gets a card entry in that folder's `README.md`, per `ARTIFACT_TREE_FORMAT.md`; the folder
   has no `README.md` yet, and task 1.5 writes the card.
3. The Status column names the file once it is on disk.
4. The card and [the review file](review/a3q-cat-x-dog-quick-discriminator-test.md) link the result sheet.

## Error Matrix

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

#### From global catalog

No entry in `~/.claude/GLOBAL_ERROR_CATALOG.md` matches this plan's failures.

#### From project catalog

| Symptom | Cause | Fix | Entry |
|---|---|---|---|
| the run dies before `nohup` starts | a relative path in the SSH launch line | `cd` to the cluster checkout first | [poe-launch-001](../../environment/known-failures.md#entry-id-poe-launch-001) |

#### Named for this plan, not yet catalogued

The close out's `/ingest-error-pattern` moves any of these that fires into the project catalog.

| Symptom | Cause | Fix |
|---|---|---|
| a GPU job at 0 MiB on its card | `co3` on a Blackwell | `co3_bw` on `mscluster110` to `112` |
| out of memory in the discriminator's backward pass | the trunk's activations | gradient checkpointing on the trunk |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

The walk map is `2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/EXECUTE_WALK.md`.

| Step | Command | Triggered by | Outcome |
|---|---|---|---|
| Walk the plan | `/drip-execute-plan 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md` | the recommended prompt | the open tasks run in order, one per round |
| Check the plan | `/verify-plan 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md` | task 0.1, at the walk's close | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md` | task 0.1, at the walk's close | terms already explained elsewhere linked |
| Capture failures | `/ingest-error-pattern --from-run-log 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md` | the close out | failures entered in `environment/known-failures.md` |
| Bring the tree current | `/sync-plan-tree 2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md` | the close out | ticks, the route's rows and the board match what ran |

## References

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [the route, step 10](route-classifier-guided-poe-from-toy-to-animals.md#10--rung-2-a-small-discriminator-steers-cat--dog-on-held-out-seeds)
- [A3q's page](../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md)
- [the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md)
- [plan 18, the dog classifier](18-a-dog-classifier-trained-the-papers-way.md)

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents) | [Next](#engagement-instructions) ➡️

The route's step 10b plans rung 3 if reading 1 holds; otherwise the review file's next step for
the reading that held.

## Engagement Instructions

Navigation: ⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

How to check the finished plan, from the laptop checkout's root. The first two checks can be run
now.

**The discriminator's held-out check** (task 1.3):

```bash
scripts/cluster.sh sh "cat /datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/discriminator/check-holdout-7-8/gate.json"
```

Expected: 13 of 20 sampler steps at balanced accuracy 0.75 or more. This needs `/datasets` to read.

**The result renders and their checks** (task 1.4):

```bash
python3 -c "import json; d=json.load(open('2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet/result.json')); print(d['card'], d['seeds'], d['strengths'], d['best_of_n'], d['canary'])"
```

Expected: `Quadro RTX 8000 [1, 2, 3, 4, 9, 10, 11, 12] [10, 30, 100] 2 {'seed': 1, 'strength_0_identical_to_plain': True}`.

**The calls and the reading** (task 1.5, instructions 2.4 and 2.5): the per-tile calls CSV in the
results folder carries the reader's calls on all 64 tiles, not Claude's drafts, and
[the review file](review/a3q-cat-x-dog-quick-discriminator-test.md#the-question-written-before-the-run) has one of its seven readings ticked
with its counts out of 4.
