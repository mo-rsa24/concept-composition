# Can two experts trained on separate colours compose scenes neither saw? (CLEVR Stage 2)

## What this asks

Paper 1 composes diffusion models trained on separate data. On CLEVR with one object per expert at a fixed place, sample-and-paste is
nearly perfect: each of M7's four experts places its object in 256 of 256 samples. This test removes what pasting needs. One expert sees
only red objects and one only blue, 1 to 3 at a time, anywhere in the frame, and neither is told where to draw. A scene asks for x red and
y blue objects, all separate, and the count rises from 2 until the composition breaks. The owner chose this test on 2026-11-24, in chat, as
option A, "two experts, told what and not where, pushed from 2 objects up to where they break". Its source is Stage 2 of
[the CLEVR plan](../../targets/clevr-scaling-plan.md).

## Words this uses

**Expert.** A small diffusion model trained only on scenes of one colour. It takes a count label k, from 1 to 3, and draws k objects of its
colour anywhere.

**Cell.** One request, x red and y blue objects, with x and y each 1, 2 or 3. The 9 cells ask for 2 to 6 objects.

**Exact share.** The share of a cell's images holding exactly x red and y blue objects, all separate.

## The run

**Group.** Comparison runs. **Varied axis.** How the product of the two experts is sampled. The models, cells, seeds and scorer stay fixed.

| Row | What it does | What it separates |
|---|---|---|
| Plain product | red expert at count x, plus blue expert at count y, minus the empty-scene model, with EDM2's 32-step Heun sampler | the floor |
| Product with Loop 8's count checker | the same, plus the count checker's push, retrained on these experts' maps at the noise levels of the sampler's early steps and pushing only on those steps, as in [its digit test](../../abstract-checkers/06-does-it-compose/review.md) | Paper 1's method |
| Product with Du et al.'s MCMC corrector | the plain product with the corrector's steps between sampling steps | a better sampler, apart from a new signal |

Sample-and-paste has no row, because nothing tells it where each object goes.

## Data and models

**Renders.** The CLEVR generator and Blender setup from M7, at 320 by 240, squashed to 64 by 64, with default shapes, sizes, materials,
camera jitter and light jitter, and objects placed anywhere by the generator's own rule. Only the colour is restricted.

| Set | Pictures | Used by |
|---|---:|---|
| Red only, 1, 2 or 3 objects | 5,000 per count, 15,000 in all | the red expert |
| Blue only, 1, 2 or 3 objects | 5,000 per count, 15,000 in all | the blue expert |
| 0 to 3 red with 0 to 3 blue, every combination | 250 per combination, 4,000 in all | the scorer only, 3,200 to train and 800 held out |

**Models.** Each expert is EDM2's `edm2-img64-xs` with its count as a class label, trained for 2,000,896 images at batch 256 with the
learning-rate ramp scaled to the run, the recipe M7's overfit and 1% rungs passed. The empty-scene model is M7's, trained on the same
renderer. No model sees a scene holding both colours, and only the scorer does.

## The scorer

A small convolutional network reads a 64 by 64 picture and counts red and blue objects, each from 0 to 3. It must count both colours exactly
on at least 95% of the 800 held-out renders before any sample is scored. Merged objects lower the count, so a merge counts as a failure.
Thirty-two images per cell and row are also called by eye at full size under neutral names, and compared with the scorer.

## The bars, fixed before the first run

**Each expert alone.** At each count k, at least 90% of 256 samples hold exactly k objects of the expert's colour and none of the other.
An expert that misses makes the stage inconclusive.

**Where a row breaks.** The largest total N for which every cell with x + y = N has an exact share of at least 50%, Bradley's half-of-images
rule applied per cell.

| Result | What it takes |
|---|---|
| Support | the count checker row leads the plain product by at least 5 points of exact share, averaged over the 9 cells, and by more than 1.96 standard errors over cells, and leads the corrector row by the same |
| Corrector explains it | the corrector row matches the checker row within 1.96 standard errors |
| Null | no lead over the plain product |
| Inconclusive | an expert misses 90%, the scorer misses 95%, or the eye calls disagree with the scorer on more than 1 image in 10 |

**Past 6 objects.** If a row holds every cell up to 6 objects, a third expert sees only green, and cells up to 9 objects run on the same bars.
It is costed below and starts only with the owner's yes.

## Cost and what it buys

The renders run on CPU. Blender 2.78c, which the CLEVR generator needs, ships no GPU code for the cluster's RTX 3090 and Blackwell cards and
tries to build it with NVIDIA's compiler, which the cluster does not have, so every GPU frame fails (job 94448).

Rates are M7's measurements: 5.46 seconds per single-object render on one CPU node, with more objects budgeted at 7 seconds, and 2.25
seconds per thousand images seen on one RTX PRO 6000 Blackwell.

| Part | CPU node-hours | GPU-hours |
|---|---:|---:|
| Renders, 30,000 for the experts and 4,000 for the scorer | about 66 | none |
| Two experts at 2,000,896 images each, with start-up | none | about 2.6 |
| Count checker retrained on the experts' samples | none | about 1 |
| Sampling 3 rows by 9 cells by 1,024 images, three models per step, with the corrector's extra steps | none | about 4 |
| Scorer, expert checks and pastes | about 1 | none |
| **Stage 2 in all** | **about 67** | **about 7.6** |
| A green expert and cells to 9 objects, only with the owner's yes | about 33 | about 4 |

**Buys.** Paper 1's test where pasting cannot apply, with a measured point where two separately trained experts stop composing, and
whether a checker moves that point.

## The ladder

1. **Smoke.** 8 renders per set checked by eye for colour and count, one expert trained for 65,536 images, and one batch sampled.
2. **Overfit.** One expert fits 16 pictures, and its samples hold the right count of the right colour in at least 15 of 16.
3. **1% of the data.** 50 pictures per count train each expert, and the logged loss falls.
4. **Controls.** The scorer's 95% check and each expert's 90% check.
5. **Comparison.** The three rows on the 9 cells.

## What this cannot say

Two colours make each expert's objects easy to tell apart, so this tests composition of counts and places, and leaves identity mixing to
Paper 2. CLEVR objects at 64 pixels are small and seldom touch, so occlusion is rare here. The empty-scene model and the experts share one
renderer, which a real deployment of separately trained models would not guarantee.
