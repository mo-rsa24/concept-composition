# 🧭 Route: two papers, from saved work to submission

## Quick context

**What this is**

The one ordered route from the work already done to two submitted papers. The live state of every step is
on [the progress map](../artifacts/notes/progress/map.png), drawn from `artifacts/notes/progress/board.json`.
This file holds the reasons and the order, and the board holds the status.

**Why it exists**

Three projects ([LoRA correction](../1-lora-correction/README.md), [classifier guidance](../2-classifier-guidance/README.md), [inclusive captions](../4-inclusive-captions/README.md)) produced proofs of concept on toy
data, MNIST and SDXL. None has been scaled, and most of what they found sits in session transcripts and
unfiled artifacts. The route saves and absorbs that work, learns the current field, then scales.

## The two papers

| | Paper 1: experts that never met | Paper 2: restoring plurality in one model |
|---|---|---|
| Problem | Two diffusion models trained apart have a product of experts with no images showing both concepts, and no joint prompt exists | One model has seen both concepts, yet its product fuses them, and even CO3 misses a quarter of two-object prompts |
| Solution | Build a target where each model draws only its own region, then guide it | Train a pair-aware and separate-label checker at scale, and add its push to the product and to CO3 |
| Targets | 4 to 6 on [the scoreboard](../artifacts/notes/progress/scoreboard.png): each rival's best reported result, moved one step past it, with side-by-side win counts | 4 to 6 on the same scoreboard, set the same way |
| Evidence now | 1,200 of 1,280 by the judge on 20 unseen MNIST pairs, fixed halves, pixel space | 54 of 128 clean or near-clean on cat × dog; CO3 gets 101 |

Step C4 tests both statements against the literature. Step D4 records a go or no-go for each paper.

## The steps

**A. Save.** Protect the ICLR paper in `writing/`, then commit the changed paths in named batches.

**B and T. Absorb and tidy, in three parallel looping sessions.**

| Session | Steps | Prompt |
|---|---|---|
| 1. Records | A2, B1, T1, T2, B2 | [Session 1](../artifacts/notes/progress/prompts/session-1-records.md) |
| 2. Artifacts | B3, B4, T3, T4 | [Session 2](../artifacts/notes/progress/prompts/session-2-artifacts.md) |
| 3. Pillars and repo | B5, T5, then the join: B6, C1, C3, C4, D1 | [Session 3](../artifacts/notes/progress/prompts/session-3-pillars-and-repo.md) |
| 4. Rivals and targets | K1 to K6, and keeps every target current | [Session 4](../artifacts/notes/progress/prompts/session-4-rivals-and-targets.md) |
| 5. Reproduce and choose | D2, D3, starting when D1 is done | [Session 5](../artifacts/notes/progress/prompts/session-5-reproduce-and-choose.md) |

All three follow [the shared rules](../artifacts/notes/progress/prompts/shared-rules.md). Each chosen session
closes through [its closing prompt](../artifacts/notes/progress/prompts/closing-prompts.md).

**C. Learn the field.** The loop scouts the field and writes the benchmark spec, starting from
[the first pass over the rivals](../artifacts/notes/progress/rival-datasets-first-pass.md). Step C2 is your 11
reading sittings, in the order of [the reading plan](../artifacts/notes/progress/targets/reading-plan.md), which
holds each sitting's command and its 90-minute calendar block between 2026-11-22 and 2026-11-27:

| Sitting | Idea | Papers |
|---|---|---|
| 1 | [When a product is valid](../artifacts/notes/progress/targets/reading-plan.md#sitting-1-when-a-product-is-valid) | Projective Composition, CoInD, Composable Diffusion (Liu) |
| 2 | [Changing each model's weight while sampling](../artifacts/notes/progress/targets/reading-plan.md#sitting-2-changing-each-models-weight-while-sampling) | SuperDiff, Logical Guidance, Lift Scores |
| 3 | [Samplers and correctors](../artifacts/notes/progress/targets/reading-plan.md#sitting-3-samplers-and-correctors) | [Feynman-Kac Correctors](../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/07-feynman-kac-correctors.md#words-this-file-uses), Energy-Based MCMC (Du), CFG as a Predictor-Corrector |
| 4 | [Binding each word to its object](../artifacts/notes/progress/targets/reading-plan.md#sitting-4-binding-each-word-to-its-object) | ToMe, CoMat, Attend-and-Excite |
| 5 | [Steering the start and the early steps](../artifacts/notes/progress/targets/reading-plan.md#sitting-5-steering-the-start-and-the-early-steps) | Rectify Then Diffuse, CO3 |
| 6 | [What breaks outside the training data](../artifacts/notes/progress/targets/reading-plan.md#sitting-6-what-breaks-outside-the-training-data) | Catastrophic Compositional Generation, Local Mechanisms (Bradley) |
| 7 | [Composing by regions](../artifacts/notes/progress/targets/reading-plan.md#sitting-7-composing-by-regions) | Diffusion Domain Expansion, MultiDiffusion |
| 8 | [Keeping subjects apart through attention](../artifacts/notes/progress/targets/reading-plan.md#sitting-8-keeping-subjects-apart-through-attention) | FreeFuse, PiCo, FlipConcept |
| 9 | [Amplifying a product](../artifacts/notes/progress/targets/reading-plan.md#sitting-9-amplifying-a-product) | Compositional Sculpting, Test-Time Composition (Wang), Compositional Abilities (Okawa) |
| 10 | [What an outside planner or critic buys](../artifacts/notes/progress/targets/reading-plan.md#sitting-10-what-an-outside-planner-or-critic-buys) | RPG, CountLoop, Revisiting CFG |
| 11 | [When a sample commits](../artifacts/notes/progress/targets/reading-plan.md#sitting-11-when-a-sample-commits) | Biroli, Achilli, Albrychiewicz |

**D. Build and test.** Session 5 reproduces each rival's published number on its own protocol and chooses the
checker architecture on small data. The owner then records the go or no-go.

**E. Scale, one ladder per dataset, on `bigbatch` and `biggpu` only.**

| Rung | What runs | Checkpoint saved |
|---|---|---|
| 0. Smoke test | One batch, end to end | Every output file exists |
| 1. Overfit | 16 images | The model memorises them |
| 2. Small | 1% of the data, train and evaluate | First metric and image sheet |
| 3. Verify | Zero push equals plain, random push does nothing | Controls pass |
| 4. Tune | Parallel sweep, one setting frozen | Chosen config in git |
| 5. Grow | 10%, 30%, then 100% of the data | Metric against data size |
| 6. Full | Three seed sets, every rival | Table with intervals, image grids |

Each rung saves its config, metrics, image sheet and commit. Paper 1 runs CLEVR, the CoInD datasets, the
CIFAR-10 disjoint split and chest X-ray. Paper 2 runs GenEval, T2I-CompBench, RareBench and a held-out set of
Attend-and-Excite prompts.

**F. Write.** Tables with confidence intervals, qualitative comparison grids, and a public code release.

## The calendar

The `calendar-cadence` agent reads the board and keeps the calendar current. It places a block when its work
becomes ready, pulls a loop forward when the steps before it finish early, and ticks finished blocks.
