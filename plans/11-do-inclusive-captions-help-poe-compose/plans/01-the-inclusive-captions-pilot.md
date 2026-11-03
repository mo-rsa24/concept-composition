# 🧪 The inclusive-captions pilot, end to end

## Recommended prompt

```text
/drip-execute-plan plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md
```

## Position in the plan tree

[Root](../../../MASTER_PLAN.md) › [do inclusive captions help PoE compose](../MASTER_PLAN.md) ›
this plan. It is the scope's only plan.

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

**What this is about**

Two SDXL LoRAs trained on identical photographs, differing only in how the 131 photos holding
both a cat and a dog are captioned, scored on whether the product of the cat and dog conditions
draws one cat and one dog.

**What already exists**

The 491 photos, measured, audited and carded. The caption sampler, committed and checked off 200,000
draws. The instrument that counts animals and names them.

**What does not**

An image-and-caption denoising trainer. Every trainer in this repo learns from cached noise
tensors toward a correction target, with no photographs and no text encoder in the loop.

**What you will hold at the end**

One figure: the compose rate for run A beside run B on 128 shared seeds, with the criterion's
three lines beside it and exactly one of them ticked.

## Considerations

**The effect and the resolution are the same size.**

The design predicts run B's product composes more often because 16.3% of its cat-captioned
training draws hold a dog against 0% of run A's. At a rate near 0.5 on 128 seeds per run, two
standard errors of the difference are about 12.5 points, and about 7.5 if the rates come in near
0.1 instead. So a real but modest effect lands inside the band where the answer is "could not
tell", and the band's formula has to be written down before the numbers exist rather than argued
about afterwards.

**One cell makes the comparison readable.**

The blank-caption rate. It is what classifier-free guidance subtracts at sampling time and the
product subtracts twice, at coefficient -14 against -6.5 in a single-condition render. Two runs
with different blank rates are not comparable, so both share one 10% rate on every photo. A bug
that moved it by 1.6 points was already found and fixed in the caption sampler.

**The failure to watch for is averaging, not nullity.**

If the cat condition absorbs its dog-present draws as an average rather than as a second mode,
every rate goes to zero and run B's cats look worse than run A's: blurred, or carrying a second
indistinct animal shape. Read as a plain null, that retires the idea for the wrong reason, so the
criterion names it as its own outcome.

**The photos have not been looked at.**

COCO's annotations label drawings and statues as animals and miss animals that are present. 131
is small enough to audit by eye, and an unaudited pool is a pool whose effect size is unknown.

## The claim under test

A single-animal condition trained with both-animal photos under its own name keeps the other
animal inside it, and the product of two such conditions composes more often than the product of
two conditions trained on jointly-captioned photos alone.

## Why this plan exists

**The question is about training data, and every other scope here is about the model.**

Scopes 01 to 09 repair composition at inference time or with an adapter learned from cached
noise. None of them touches what the conditions were trained on. If the product's failure is
inherited from captions rather than created by the sampler, no amount of adapter work reaches it,
and this is the cheapest test of that.

**It is also the cheapest version of a larger design.**

The share sweep asks how compose rate moves as the dog-present share of the cat condition rises.
This plan trains one point of that sweep against one control. If the two do not separate, the
sweep is not worth building.

## What happens

```
491 COCO photos  ──┬──> run A: both-animal photos captioned jointly        ──> LoRA A
  200 cat-only     │
  200 dog-only     └──> run B: 40% cat, 40% dog, 10% joint, per draw       ──> LoRA B
   91 both-animal
                        both: blank 10% on every photo

LoRA A, LoRA B  ──> 128 shared seeds, product of cat and dog  ──> count and name the animals
                                                              ──> compose rate, A beside B
                                                              ──> read against the criterion
```

## Description

Piece 1 writes the criterion into the review file. Piece 2 finishes the data build: COCO captions
into the manifest, every path checked, and the both-animal photos audited by eye. Piece 3
builds the trainer against this repo's own SDXL loader and proves it on a 20-step smoke run.
Piece 4 adds the four-column sheet, including the product column, which is renderable at step 0
because combining two conditions happens at sampling time. Piece 5 trains the two runs. Piece 6
renders and scores 128 seeds each and reads the figure against piece 1.

## Purpose

Decide whether the caption assignment has a measurable effect on what a single-animal condition
covers, and whether composition follows it.

## Goal

A compose-rate figure for run A and run B on 128 shared seeds, with exactly one of the
criterion's three lines ticked against it, and the finding written to `report/`.

## Environment Facts This Plan Depends On

**The `co3` absolute python path.** `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`. A
bare `python` in a job script is a catalogued failure here.

**Large artifacts go to `/datasets` only.** Checkpoints and renders write under
`/datasets/mmolefe/poe_repair_min/`. `/home-mscluster` hit 100% once and killed checkpointing
with no error. Any disk guard must check the filesystem actually being written to.

**`biggpu` allows one job per user**, so a 2,000-step run is started with `nohup` outside Slurm
on a pinned free device and is invisible to `squeue`; it is harvested with `pgrep` instead.

**Rank-32 LoRA training needs about 25 GB**, so on a 3090 gradient checkpointing is required. The
rank here is not yet chosen, which is piece 3's own decision.

**The fp16 upcast rule.** SDXL's VAE declares `force_upcast`, so it runs in fp32 while the UNet
runs in fp16. The repo's loader already does this; a trainer that re-casts the VAE will produce
latents that do not match the sampling path.

## Success and failure outcomes

**Success.** Run B's product compose rate exceeds run A's by more than the criterion's band on
128 shared seeds, and the held-out denoising loss on unseen both-animal photos is lower under B.
The caption assignment has an effect and composition follows it.

**No support.** Run B's rate is at or below run A's, and the losses match. The captioning did not
widen the condition by an amount this experiment could detect, and the idea moves to the
from-scratch model, which has no inherited support to erode. This is not reported as a null: 128
seeds cannot tell a true zero from a true gain of 6 points that measured below zero.

**Inconclusive.** The difference falls inside the band. The seeds were not enough to resolve the
move, and the cost of resolving it is stated rather than the result being reported as either of
the two above.

**The outcome to be careful with.** All rates near zero with run B's cats visibly worse than run
A's. That is the condition averaging its two kinds of draw, not holding two modes, and the
criterion separates it from the null.

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 🧪 Write the criterion before anything runs

- [x] ✅ **1.1** Fill the review file's question section with all six parts, each carrying a number:
      the comparison, what counts as support, what counts as a null, the inconclusive band, the
      seed counts and what would surprise.
  - Output: [the review file](../review/01-does-captioning-widen-the-cat-condition.md)
  - The averaging outcome is named as its own line, separate from the null.
  - Committed before the first training step of either run, which is the whole point of the
    ordering.

▶ **Next: [instruction 7](#7--read-the-criterion-and-approve-it)**, you approve it before any
code is written.

### 2. 🔧 Finish the data build

◀ **Needs: [instruction 7](#7--read-the-criterion-and-approve-it)** done, so the criterion is
fixed and cannot be shaped by anything the build reveals.

- [x] ✅ **2.1** Read `captions_train2017.json` in the pool script and write one caption per photo
      into the manifest, so the neither-animal filler photos carry their own caption rather than
      an empty string.
  - Edits: [the pool script](../../../scripts/census/inclusive_captions_pool.py)
  - Check: every row of `manifest.json` has a non-empty `caption` field.
  - ✅ verified 2026-11-03, `dae7c3c9`: `missing_caption: 0` of 862 chosen photos.
- [x] ✅ **2.2** Open every path in the manifest and report any that does not resolve.
  - Check: every chosen photo opens.
  - ✅ verified 2026-11-03, `dae7c3c9`: `missing_file: 0` of 862 chosen photos (862 because the
    no-both run's filler photos are picked alongside the ones the other runs use).
- [x] ✅ **2.3** Re-run the caption sampler's check against the real manifest rather than
      placeholders, so the shares are measured on the photos that will actually train.
  - Command: `python3 -m poe_repair.experiments.inclusive_captions.captions --check`
  - What you want: `PASSED`, and the dog-present share of run B's cat draws at 0.169.
  - ✅ verified 2026-11-03, `dae7c3c9`: PASSED on the real manifest, share 0.169, every caption
    share within 0.002 of its design value, blank rates spread 0.001 across all four runs.

▶ **Next: [instruction 8](#8--audit-the-both-animal-photos)**, the eye audit, which only you
can do.

### 3. 🔧 Build the trainer

◀ **Needs: [instruction 8](#8--audit-the-both-animal-photos)** done, so the photo count the
trainer reads is the audited one.

- [ ] **3.1** Write the image-and-caption LoRA trainer against this repo's own SDXL loader:
      `load_sdxl_models`, `encode_prompt_sdxl`, `load_ddim_scheduler` and `decode_latents` from
      [the SDXL runtime](../../../poe_repair/_sdxl/runtime.py). Do not fetch the diffusers SDXL
      LoRA example; building against the loader keeps the training path identical to the sampling
      path every other experiment here uses and needs no network on a compute node.
  - Output: `poe_repair/experiments/inclusive_captions/trainer.py`
  - The LoRA goes on through peft 0.19.1, already installed in `co3`.
  - Disk guard checks `/datasets`, which is where the checkpoints land.
- [ ] **3.2** Run the 20-step smoke run on the three toy photos and report the loss curve and the
      checkpoint it wrote.
  - What you want: finite loss, a checkpoint on disk, exit 0, under a minute.

▶ **Next: [instruction 9](#9--run-the-smoke-test-yourself)**, you run it and see it work.

### 4. 🔬 The four-column watching sheet

◀ **Needs: [instruction 9](#9--run-the-smoke-test-yourself)** done, so the loop is known to run
before anything is built on top of it.

- [ ] **4.1** Render a sheet of eight fixed seeds down the rows and four columns across: "a photo
      of a dog", "a photo of a cat", "a photo of a cat and a dog", and the product of the cat and
      dog conditions. 512 square at 25 steps. Log it to W&B as one image per checkpoint.
  - Output: `poe_repair/experiments/inclusive_captions/sheets.py`
  - The product column is rendered at step 0 as well, before any training, because combining two
    conditions is a sampling-time operation and needs no trained weights.
  - Seeds 0 to 7, the same eight at every checkpoint and in every run.
- [ ] **4.2** Prove the sheet on the smoke run's checkpoint and report the W&B link.

▶ **Next: [instruction 10](#10--look-at-the-step-0-sheet)**, you judge whether the four columns
show what they claim to.

### 5. 🚀 Train run A and run B

◀ **Needs: [instruction 10](#10--look-at-the-step-0-sheet)** done.

- [ ] **5.1** Launch run A: joint captions on the both-animal photos, 2,000 steps, checkpoints
      and sheets at 0, 250, 500, 1,000 and 2,000.
  - W&B: `prime_lab/poe-repair-inclusive-captions`, one run, grouped with B.
  - Started with `nohup` outside Slurm on a pinned free `biggpu` device, since `biggpu` allows
    one job per user.
- [ ] **5.2** Launch run B: 40% cat, 40% dog, 10% joint, resampled per draw.
- [ ] **5.3** Harvest both: the queue, `pgrep` on the session node, the output count against what
      this plan expected, and the tail of each log.

▶ **Next: [instruction 11](#11--watch-the-sheets-fill-in-wb)**, you watch the four columns move
while the runs train.

### 6. 📊 Score 128 seeds and read it

◀ **Needs: [instruction 11](#11--watch-the-sheets-fill-in-wb)** done and both runs at step 2,000.

- [ ] **6.1** Render 128 product seeds per arm at 1,024 square, 50 steps. Composition calls are
      made on these, never on the 512-square watching sheets.
- [ ] **6.2** Score each render in three categories: one cat and one dog, a hybrid, or a single
      animal. GroundingDINO counts the animals and the crop classifier names them.
  - Report the count the classifier called uncertain, since it has never been tested on a frame
    holding two animals with overlapping boxes.
- [ ] **6.3** Draw the compose-rate figure: the rate for base SDXL, run A and run B, each with its
      interval, the three categories stacked, and the criterion's band drawn on it.
  - Output: `artifacts/results/do-inclusive-captions-help-poe-compose/` with its sidecar json and
    its card entry.

▶ **Next: [instruction 12](#12--tick-one-of-the-criterions-three-lines)**, the verdict, which is
yours.

## Instructions

For you to follow manually. Do these yourself.

### 7. 📖 Read the criterion and approve it

◀ **Needs: [task 1.1](#1--write-the-criterion-before-anything-runs)** done.

7.1 **Open [the review file](../review/01-does-captioning-widen-the-cat-condition.md)** and read
its question section.

7.2 **Check each of the six parts carries a number**, not a word. "B beats A by more than the
band" is not a criterion; "B exceeds A by more than two standard errors of the difference,
computed from the measured rates by the formula in the review file" is.

7.3 **Say which of the three lines you would be willing to publish.** If you would argue with any
of them after seeing a result, say so now, because that is the only moment the argument is free.

✅ Approved, or sent back with what to change.

▶ **Next: [task 2.1](#2--finish-the-data-build)**.

### 8. 👁️ Audit the both-animal photos

◀ **Needs: [tasks 2.1 to 2.3](#2--finish-the-data-build)** done.

8.1 **Open the nine audit pages** at
`/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/audit_page_00.png` and the eight
numbered after it, written by
[the audit render](../../../scripts/census/audit_both_animal_photos.py) at 16 tiles a page and 448
pixels a tile. The 48-tile sheets beside them are too small to name a species off, which this
project has already been caught doing.

8.2 **For each tile, decide whether it really holds one live cat and one live dog.** Reject a
drawing, a statue, a plush toy, a photo where the second animal is barely visible, and a photo
where the annotation is simply wrong.

8.3 **Write the rejected tile numbers down**, with one word each for why.

✅ Done 2026-11-03, judged by Claude at the user's request: 91 of 131 survive, 33 rejected and 7
uncertain, with the reason for each in
[the audit verdicts](../../../artifacts/results/the-inclusive-captions-photo-pool/audit-verdicts.json).
Ten rejections are a cat beside a depiction of a dog. The failure branch below fired, and the
single-animal sets dropped from 300 to 200 to restore the effect size.

What you wanted: a count of how many of the 131 survive, and the reasons the rest did not.
What failure looks like: more than about 20 rejections. That moves the dog-present share of run
B's cat draws below what 128 seeds can resolve, and the single-name caption shares have to rise
again before anything trains.

▶ **Next: [task 3.1](#3--build-the-trainer)**.

### 9. 🔨 Run the smoke test yourself

◀ **Needs: [tasks 3.1 and 3.2](#3--build-the-trainer)** done.

9.1 **On the session node**, run the smoke test:

```bash
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m \
  poe_repair.experiments.inclusive_captions.trainer --smoke
```

9.2 **Watch the loss print.** You want a finite number that does not grow monotonically over 20
steps.

9.3 **Check the checkpoint landed** under `/datasets/mmolefe/poe_repair_min/`, not under
`/home-mscluster`.

✅ What you want: exit 0, a loss under 1.0, a checkpoint on disk, inside a minute.
What failure looks like: `CUDA out of memory` means gradient checkpointing is off; a loss of
`nan` means the VAE was cast to fp16, which the fp16 upcast rule forbids.

▶ **Next: [task 4.1](#4--the-four-column-watching-sheet)**.

### 10. 👁️ Look at the step-0 sheet

◀ **Needs: [tasks 4.1 and 4.2](#4--the-four-column-watching-sheet)** done.

10.1 **Open the W&B run** and find the sheet image at step 0.

10.2 **Read the four columns against each other on one row.** Column 1 should be a dog, column 2
a cat, column 3 whatever base SDXL does with the joint prompt, and column 4 the product, which at
step 0 is plain PoE on an untrained model.

10.3 **Confirm column 4 is not a copy of column 3.** They are different computations and a sheet
where they match tile for tile means the product is not being computed.

✅ What you want: four visibly different columns on the same seed.
What failure looks like: columns 3 and 4 identical, which means the product column is rendering
the joint prompt.

▶ **Next: [tasks 5.1 to 5.3](#5--train-run-a-and-run-b)**.

### 11. 📈 Watch the sheets fill in W&B

◀ **Needs: [tasks 5.1 and 5.2](#5--train-run-a-and-run-b)** launched.

11.1 **Open the W&B group** holding run A and run B.

11.2 **At each checkpoint, compare the two runs' sheets tile for tile on the same seed.** The
question to hold while looking: does run B's cat column start showing dogs, and does its product
column start showing two animals, while run A's stays put.

11.3 **Write down the first checkpoint at which you can tell the two runs apart by eye**, or that
you cannot.

✅ What you want: a note naming the checkpoint and what you saw.
What failure looks like: run B's cat column going blurry or growing an indistinct second shape.
That is the averaging outcome, and it is a result rather than a bug.

▶ **Next: [tasks 6.1 to 6.3](#6--score-128-seeds-and-read-it)**.

### 12. ⚖️ Tick one of the criterion's three lines

◀ **Needs: [tasks 6.1 to 6.3](#6--score-128-seeds-and-read-it)** done.

12.1 **Open the compose-rate figure** and read run B's rate against run A's, with the band drawn
on it.

12.2 **Tick exactly one line in the review file**: support, null, or inconclusive. Where the
averaging outcome fired, tick that instead and say so.

12.3 **Write the verdict into the review file in your own words**, with the run ids and the bar
it was judged against. The numbers stay in the figure and its sidecar.

✅ What you want: one line ticked and a verdict nobody has to reconstruct later.

▶ **Next: the closing pass.** The finding goes to `report/`, the figure is filed with its card
entry, and `sync-plan-tree` reconciles the scope.

## Outputs

| What | Where |
|---|---|
| the criterion | [the review file](../review/01-does-captioning-widen-the-cat-condition.md) |
| the manifest with captions | `/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/manifest.json` |
| the trainer | `poe_repair/experiments/inclusive_captions/trainer.py` |
| the sheet renderer | `poe_repair/experiments/inclusive_captions/sheets.py` |
| the checkpoints | `/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions/<run>/` |
| the watching sheets | W&B, `prime_lab/poe-repair-inclusive-captions` |
| the compose-rate figure and its sidecar | `artifacts/results/do-inclusive-captions-help-poe-compose/` |
| the finding | `report/` |

## Figure Catalog

| Figure | What is on each axis, and where the numbers come from |
|---|---|
| the compose-rate figure | compose rate on y, a 0-to-1 scale, one bar per model (base SDXL, run A, run B) on x, each split into one cat and one dog, a hybrid, and a single animal, with the criterion's band drawn across it. Numbers from the sidecar json beside it, 128 seeds per arm for A and B. |
| the four-column watching sheet | eight fixed seeds down the rows, four prompts across (dog, cat, joint, product), one image per checkpoint per run, logged to W&B. Not a rate: eight seeds is a picture. |

## Error Matrix

| Symptom | What it means | What to do |
|---|---|---|
| `CUDA out of memory` during training | the LoRA rank needs more than the device has | pass gradient checkpointing; rank 32 needs about 25 GB on a 3090 |
| loss is `nan` from step 1 | the VAE was cast to fp16 | SDXL's VAE declares `force_upcast` and runs in fp32; the repo's loader already handles this |
| a bare `python` in a job script | catalogued failure here | use the `co3` absolute path |
| the run completed but wrote nothing | the disk guard checked the wrong filesystem | the guard must check `/datasets`, which is where the output goes |
| `squeue` shows nothing while the run is alive | `biggpu` allows one job per user, so the run is outside Slurm | harvest with `pgrep -af 'sweep\|train'` on the session node |
| the crop classifier calls most two-animal frames uncertain | it has never been tested on overlapping boxes | report the uncertain count; the three-category score depends on this and it is an open question |
| columns 3 and 4 of the sheet are identical | the product column is rendering the joint prompt | check the product is `eps_uncond + g * (eps_a + eps_b - 2 * eps_uncond)` |

## Orchestration

Walked with `/drip-execute-plan`. The scoring end reuses the detection scorer and the crop
classifier rather than re-implementing either. The closing pass files the figure per
`ARTIFACT_TREE_FORMAT.md`, writes the finding per `REPORT_FORMAT.md`, and runs `sync-plan-tree`
from the scope root.

## References

- [the design, checked](../../../artifacts/ideas/inclusive-captions-axis/COMPILED.md)
- [the claim-by-claim reasoning](../../../artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md)
- [the photo pool](../../../artifacts/results/the-inclusive-captions-photo-pool/README.md)
- [does the cat prompt already draw dogs](../../../artifacts/results/does-the-cat-prompt-already-draw-dogs/README.md)
- [the caption sampler](../../../poe_repair/experiments/inclusive_captions/captions.py)
- [the SDXL runtime this builds against](../../../poe_repair/_sdxl/runtime.py)
- Bradley et al., 2502.04549, for the condition under which adding two experts' scores is exact.
- CoInD, 2503.01145, which attacks the same quantity from the training side with a loss term.

## Next step

Task 1.1: write the criterion.
