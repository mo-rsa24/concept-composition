# 🧪 Experts that compose at inference: two separately trained digit experts, either side, 90 of 100 clean pairs

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/33-experts-that-compose-at-inference.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, the follow-on to [plan 32](32-two-digit-experts-that-never-met.md). Its bars are in
[its review file](review/33-experts-that-compose-at-inference.md).

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

Plan 32 found that two digit experts trained apart, each drawing its digit on either side, give a product with no mass: both
pushes reached 4 of 320, and a push or sampler cannot add support. The goal now is two experts that never met composing on
**90 of 100 clean pairs, each digit on either side, on pairs no expert has seen**. The reasoning, what Liu 2022, Bradley 2025,
Bradley 2025b and Du 2023 say, the seven experiments and every bar are in
[the review file](review/33-experts-that-compose-at-inference.md).

**What it does**

Tests, cheapest first: composing the existing experts by region at inference with no training (E0), a union base as a
control for the support argument (E1), experts trained with clutter (E2), experts with a local patch architecture after
Bradley 2025b (E2b), then a better sampler (E3), the pixel pushes (E4) and a shared latent space (E5) on whichever
composes. One setting is chosen on five tuning pairs and read once on the other 20, at fresh seeds.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**Where it runs.** `biggpu` allows one job per user, so training runs outside Slurm on `mscluster85`'s RTX 3090 and the one
card of `mscluster107` not held by another job, detached with `ssh -f` and `setsid nohup`. `bigbatch` is capped at 6 jobs and
another session may hold them. fp16 trains a 40,000-step expert in about 100 minutes on one card.

**Where outputs go.** `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/33-experts-at-inference/`; local copies under
`2-classifier-guidance/data/mnist-canvas/33-experts-at-inference/`. Results are filed to `artifacts/results/33-experts-at-inference/`.

**Who calls the sheets.** Claude, at full resolution; the reader can veto. The headline is counted by eye.

**Seeds.** Seeds 0 to 63 tune; 64 to 127 are read once at the end; every training draw of a checker comes from seed 1,000 up.

**Commits.** On `exec/classifier-guidance-ladder`; the cluster checkout is fast-forwarded by hand.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

A 0–4 expert and a 5–9 expert that were trained apart, never saw each other's digits and never saw two digits on a canvas, can
be composed at inference, with each digit on either side, so that 90% of seeds give two clear, right digits, on every pair of a
0–4 digit with a 5–9 digit.

## Goal

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar** is [the review file](review/33-experts-that-compose-at-inference.md#the-bars-written-before-the-run)'s, unchanged after
the run. **Time box:** two days of wall time. If E0 and E1 are done and both are null, the trainings of E2 and E2b are what remains;
if neither of those reaches 160 of 320, the plan reports where it stopped and what each experiment bought.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- [The nodes](../../environment/hpc/nodes.md#partitions): `biggpu` one job per user; `bigbatch` RTX 3090 24 GB, probe or exclude dead cards.
- [Storage](../../environment/storage.md): the scratch folder while `/datasets` is down.
- The laptop python: `/home/molef/micromamba/envs/sdxl-local/bin/python`; MNIST raw files at
  `/home/molef/PhD/replications/original-superdiff/data/MNIST/raw`. Fashion-MNIST, for the clutter, is not on disk and must be fetched (task 1.3).

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Check each sheet](#2-check-each-sheet)** walked.

- [ ] **0.1** `/verify-plan 2-classifier-guidance/plans/33-experts-that-compose-at-inference.md`, then `/xref-pillar` on it, at the close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build, train and render

◀ **Needs: nothing first.**

▶ **Next: [2. Check each sheet](#2-check-each-sheet)**

- [ ] **1.1** E0 and E1 first, no training: region composition in `scripts/mnist/two_experts.py` (a `--compose region` mode:
  random side per seed, full-canvas and half-canvas-crop variants) and the union base (`--base union` reading the step 7b share-0 model);
  run on the five tuning pairs with plan 32's experts; write the counts into the review file before anything else runs.
- [ ] **1.2** Read Liu 2022, Bradley 2025 (the clutter section), Bradley 2025b appendix E.3 in the paper itself and Du 2023
  section 4 and its sampler section; correct the review file's "What the papers say" wherever the reading disagrees with it.
- [ ] **1.3** E2: fetch Fashion-MNIST, add a clutter option to `canvas_sheet.py` and the trainer (`--clutter`), check a 16-tile sheet
  of each expert's training canvases, train the 0–4 and the 5–9 clutter experts, run them alone (bar 58 of 64), compose.
- [ ] **1.4** E2b: write the patch architecture's design (patch width, cell size, side conditioning, centre-cell assembly, the k = 8
  control) into the review file from Bradley 2025b appendix E.3, then train the two local experts and the control, run alone, compose.
- [ ] **1.5** E3 on the best of E0, E2 and E2b (Langevin and HMC correction with `c` and `k` as in the project's Langevin note,
  the project's twisted SMC), E4 (retrain the pixel pair-aware checker on the best product's own renders, two passes), E5 only if
  the others fall short of 160 of 320.
- [ ] **1.6** The final read: one chosen setting, the other 20 pairs, seeds 64 to 127, sheets at full resolution, the headline by eye,
  and the no-harm check on plan 32's fixed-side experts.
- [ ] **1.7** The reading against the review file, the results card, the report page and board row, and the route's new rows.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Check each sheet

◀ **Needs: [1. Build, train and render](#1--build-train-and-render)**

- [ ] **2.1** Open the clutter sheet of each expert's training canvases: one digit and 0 to 2 non-digit objects, no digit of the other
  range. Veto anything that leaks the other expert's digits.
- [ ] **2.2** Open each experiment's sheet on 4 and 7 and look at the same seeds across experiments; veto any call.
- [ ] **2.3** Open the final read's sheets, one per pair; say whether 90% by eye is what you see.

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Experts and checkpoints under the scratch folder; one sheet and count per experiment and pair; the final read's sheets; filed under
`2-classifier-guidance/artifacts/results/33-experts-at-inference/` with its card.

## Error Matrix

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Symptom | Cause | Fix |
|---|---|---|
| `QOSMaxJobsPerUserLimit` on `biggpu` | one job per user | run outside Slurm on `mscluster85` |
| an expert draws the other range's digits | the digit filter did not apply | print the label counts the trainer saw; retrain |
| clutter objects look like digits | wrong distractor source | use Fashion-MNIST items, check the sheet before training |
| canvases saturate to white | a base whose prediction diverges at low noise | use a base with spread or an expert's own no-digit prediction |
| zero correction differs from plain product | a sampler bug | stop; fix before reading any count |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

E0 and E1 are minutes; E2 and E2b are four trainings of about 100 minutes, two at a time on the two free cards. A self-paced `/loop`
reads the logs, applies the review file's bars, and writes each correction into the review file before it reruns.

## Next step

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents)

If an experiment reaches the headline, its recipe goes to the paper as the condition separately trained experts need. If none does,
the result is how far each got and what the product lacked.
