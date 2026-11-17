# 🧪 Our photos, both ways: classifier guidance with the released model and with a 64 px model we train

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/20-our-photos-both-ways.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which is the route's step 8 and the board's rows A2r and A2c in
[the classifier-guided PoE board](../report/classifier-guided-poe/board.md). It runs beside the
scale-up ladder and gates nothing.

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

Step 7 showed on the exact toy that a pair-aware push<sub>[1](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/04-the-dropped-term.md#44-the-pair-aware-push "Classifier guidance by hand, chapter 4, section 4.4: the pair-aware push, a discriminator's log-odds trained on joint-caption draws against the product's own draws, applied with a weight on the product"), [2](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/08-the-pair-aware-push-built.md "Classifier guidance by hand, plan 08: the pair-aware push built and measured on the four-picture toy at weights 0, 0.5, 1 and 2")</sub> turns the product's stacked draws into
side-by-side ones, while a separate-label push<sub>[1](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/02-the-product-as-two-classifiers.md#23-pushing-harder-with-the-same-two-classifiers "Classifier guidance by hand, chapter 2, section 2.3: more of the same two separate-label pushes on the product base, where the fused animal's share rises and the pair's falls"), [2](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/04-the-separate-label-push-built.md "Classifier guidance by hand, plan 04: the separate-label push built and measured on the four-picture toy at strengths 0, 1, 3 and 10")</sub> does not. This plan runs the same two pushes with
guided-diffusion's own code on our cat, dog and cat-with-dog images, two ways.

**The released model**

The authors' 256 px unconditional model stays frozen. Their released classifier's cat classes
(281 to 285) and dog classes (151 to 268) give two separate-label pushes, which together form the
product. A pair-aware classifier is trained on our both-animal images against the product's own
finished draws, and its push is added on top.

**The model we train**

Their 64 px model is trained on our images with a cat label, a dog label, and a null label 10% of
the time, so the product of a cat expert and a dog expert can be formed. It is trained twice: once
without the both-animal images, so the product never saw a pair, and once on all of them.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**The image pool**

`/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool_v2/manifest.json`: 200 cat-only,
200 dog-only and 150 both, each checked by the inclusive-captions pilot's judge. 312 of the 550 are
SD3.5 renders (100 cat, 100 dog, 112 both) and 238 are COCO photos. The pool is read only; the pilot
is another session's live work.

**What the pair-aware classifier could learn instead of the pair**

Its real pile is three quarters SD3.5 renders and the product's draws are guided-diffusion's look,
so look alone would separate them. The single-animal images from the same pool join the negatives
beside the product's draws, which puts SD3.5 renders on both sides.

**The both-animal images in the all-images 64 px model**

Each time one is drawn it carries the cat label or the dog label at random, so both experts see pairs.

**Running their code**

`co3` has neither `mpi4py` nor `blobfile`, which their trainers import. A launcher imports their
training loop unchanged and replaces the multi-process and file-system calls with one-process
stand-ins, so `co3` stays untouched.

**The classifiers**

Each starts from the released classifier of its size with a new output layer, because 550 images is
little to learn from. The separate-label classifier has two sigmoid outputs, cat present and dog
present.

**Where the GPU work runs**

A free card is taken in this order: the Blackwells `mscluster110` to `112` under `co3_bw`, then
`mscluster106`, `107` and `108` under `co3`, through the shared-device path; only then `bigbatch`
through `sbatch`.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

On real cat and dog images, a separate-label push only re-weights the product, and a pair-aware push
carries the term the product drops, so it draws two separate animals on more seeds.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#what-happens) ➡️

The toy has a known correction and no real images. SDXL has real images and no control over the
training data. This plan sits between them: real images, and in the second arm a model whose
training data we choose, so whether the product ever saw a pair is a switch we set.

## What happens

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description) ➡️

Six pieces, one per round, each ending on a test: the image set, the two 64 px trainings, the two
256 px classifiers, the released model's sheet, the 64 px classifiers and sheets, then the close.

## Description

Navigation: ⬅️ [Previous](#what-happens) | 📋 [TOC](#table-of-contents) | [Next](#purpose) ➡️

Code in `2-classifier-guidance/scripts/classifier_guidance_photos/`, copied to
`/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/code/` before each launch.
Every large output under `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/`.
Card and sheets under
`2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/`.

## Purpose

Navigation: ⬅️ [Previous](#description) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

Settle the board's rows A2r and A2c.

## Goal

Navigation: ⬅️ [Previous](#purpose) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar, written before any classifier is trained.** In at least one arm, on seeds 1 to 16, the
pair-aware push gives two separate animals on at least 4 more of 16 seeds than the separate-label
push, with pictures no more damaged than the plain product's. Judged by the reader from
full-resolution tiles, each called single animal, hybrid, two bodies, or two clear animals. The
pair-aware push is read at weights 1 and 2.

**Time box:** 4 sittings, about 4.5 hours of the reader's time, plus the training wall clock.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#success-and-failure-outcomes) ➡️

- [The shared-device path](../../environment/hpc/execution-protocol.md): per-device `nvidia-smi`,
  a guard inside the launch script, absolute paths, scripts on `/datasets`, harvest by `pgrep`.
- [The nodes](../../environment/hpc/nodes.md#partitions): `co3_bw` on `mscluster110` to `112`,
  `co3` on `106` to `109`; the Blackwell CUDA probe before GPU work; the `bigbatch` exclude list.
- [Storage](../../environment/storage.md): large outputs on `/datasets` only.
- The released weights at `/datasets/mmolefe/poe_repair_min/weights/guided-diffusion/`; the cluster
  clone at `/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion`, commit `22e0df8`.

## Success and failure outcomes

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

| Outcome | What it means | What follows |
|---|---|---|
| the bar passes in an arm | the push carries the dropped term on real images | the card, the finding, the board's row ticked |
| the bar misses in both arms | the push does not reach real images at this size | the diagnosis ladder: the minimal-pair gap, push weights, the classifier's accuracy per step |
| the separate-label push composes as often as the pair-aware one | the both-animal images carry both labels into the experts | read against the model trained without them |

## Tasks

Navigation: ⬅️ [Previous](#success-and-failure-outcomes) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Test each piece](#2-test-each-piece)** walked, since this plan was written by the walk and is checked at its close.

- [ ] **0.1** `/verify-plan` on this file, then `/xref-pillar` on it, at the walk's close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build the six pieces

◀ **Needs: nothing first.** This group starts the walk; [group 0](#0--check-this-plan-before-working-from-it) runs at its close.

▶ **Next: [2. Test each piece](#2-test-each-piece)**

- [x] **1.1** The image set: square crops resized to 256 and 64 px under
  `classifier_guidance_photos/images/{256,64}/`, filenames carrying the label, and a `counts.json`.
  `build_photo_set.py`; a COCO photo's square slides along its long edge to cover the animals'
  boxes. ✓ verified 2026-11-17: 200 cat (100 COCO, 100 SD3.5), 200 dog (100, 100), 150 both (38,
  112); 22 of the 38 COCO both crops keep under 90% of the animals' box area, and by eye at sheet
  scale 36 of 38 show both animals · `photo-set-counts.json`, `example-one-per-label.png`,
  `coco-both-crops.png`

  <a href="../artifacts/results/does-a-pair-aware-push-compose-our-photos/example-one-per-label.png"><img src="../artifacts/results/does-a-pair-aware-push-compose-our-photos/example-one-per-label.png" width="200" alt="Four images across, 256 px above and the same image at 64 px below: a COCO tabby cat sitting, a COCO collie's head, a COCO brindle dog and black cat on a striped armchair, and an SD3.5 render of a ginger cat lying beside a brown dog"></a>
  <a href="../artifacts/results/does-a-pair-aware-push-compose-our-photos/coco-both-crops.png"><img src="../artifacts/results/does-a-pair-aware-push-compose-our-photos/coco-both-crops.png" width="200" alt="All 38 COCO both-animal crops in a grid, each titled with the share of the animals' box area it keeps, from 0.57 to 1.00; most show a cat and a dog together, and a few are photo collages or show one animal only in part"></a>
- [x] **1.2** Label dropout in the 64 px launcher, a 300-step timing run, then the two trainings
  (without both-animal images, and with all) on free cards in the order of
  [Considerations](#considerations). `train_64.py` through `launch_gpu.sh`, 50,000 steps at batch
  64, a checkpoint every 5,000. ✓ verified 2026-11-17: timing run 300 steps in 278 s on
  `mscluster111` (295.1 M parameters, loss 1.02 to 0.062); without both-animal images on
  `mscluster111` GPU 0, W&B `wubhxhse`, 400 images; all images on `mscluster112` GPU 0, W&B
  `pnwlwh0f`, 550 images; both launched 07:35 UTC at 30.6 GB and 100% utilisation
- [ ] **1.3** The 256 px pair-aware classifier: 512 product draws at 250 respaced steps, the
  released model pushed at scale 10 by the released classifier's cat and dog classes; the pair-aware
  classifier on both-animal images against those draws and the single-animal images, noised forward.
  A separate-label classifier trained on our images (`separate_256.pt`, held-out accuracy 0.93 and
  0.95 at step 0) does not steer the released model to cats or dogs at scales 10, 30 or 100
  (`push-scale-check-256.png`), so the product uses the released classifier's classes.

  <a href="../artifacts/results/does-a-pair-aware-push-compose-our-photos/push-scale-check-256.png"><img src="../artifacts/results/does-a-pair-aware-push-compose-our-photos/push-scale-check-256.png" width="200" alt="Four draws across (draw 0 to 3) by scale 10, 30 and 100 down the rows: fruit and vegetables, a dog's head, a man in a striped shirt and a bird at scale 10, turning into red and yellow food-like blobs, a jewelled texture, a blurred person and a dark animal against yellow at 30 and 100; no tile shows a cat"></a>

- [ ] **1.4** The released model's sheet on seeds 1 to 16: the product, then the product plus the
  pair-aware push at weights 1 and 2; the minimal-pair gap per step beside it.
- [ ] **1.5** When the 64 px runs land: the 64 px separate-label classifier, one pair-aware
  classifier per model, then a sheet per model with the plain product, the separate-label push, and
  the pair-aware push at weights 1 and 2; the nearest training image for each two-animal tile.
- [ ] **1.6** The calls as a CSV, the card, the finding in `report/`, the board redrawn with
  `--step 8`, and the route rows.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Test each piece

◀ **Needs: [1. Build the six pieces](#1--build-the-six-pieces)**

- [x] **2.1** Open `counts.json` and one image per label at both sizes. You should see 200, 200 and
  150, and the same picture at 256 and 64 px. ✓ verified 2026-11-17, run by Claude at the reader's
  word ("you do it"): 200 cat, 200 dog, 150 both, 550 names identical at both sizes; each 256 px
  image shrunk to 64 px differs from its 64 px file by a median 0.13 and at most 0.20 grey levels
  (of 255), against 53.96 for the same image shifted 8 px · `photo-set-counts.json`

  <a href="../artifacts/results/does-a-pair-aware-push-compose-our-photos/example-one-per-label.png"><img src="../artifacts/results/does-a-pair-aware-push-compose-our-photos/example-one-per-label.png" width="200" alt="One image per label at 256 px above and 64 px below: a cat, a dog, and two both-animal images, one a COCO photo and one an SD3.5 render; each 64 px copy is the same picture, blockier"></a>
- [ ] **2.2** Tail both training logs. You should see the loss falling and a checkpoint written.
- [ ] **2.3** Read each 256 px classifier's held-out accuracy per noise step.
- [ ] **2.4** Call the released model's tiles at full resolution, per [instruction 14 of the
  route](route-classifier-guided-poe-from-toy-to-animals.md#14--call-the-photo-sheets-at-full-resolution-in-step-8).
- [ ] **2.5** Call both 64 px models' tiles at full resolution, the same way.
- [ ] **2.6** Read the card and the finding against the bar.

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the walk's close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

The image set and `counts.json`; two 64 px training run folders with their logs; five classifier
checkpoints; one sheet per model; the per-tile calls as a CSV; the card; a finding under
`2-classifier-guidance/report/classifier-guided-poe/`.

## Figure Catalog

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

| Figure | What it shows |
|---|---|
| `sheet-released-256.png` | seeds 1 to 16 (rows) by the product, plus the pair-aware push at weight 1, at weight 2 (columns), at 256 px |
| `sheet-64-no-both.png` | the same at 64 px for the model trained without both-animal images, with the separate-label column |
| `sheet-64-all.png` | the same for the model trained on all images |

## Error Matrix

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'mpi4py'` | an upstream script run directly | run through the launcher |
| a GPU job at 0 MiB on its card | `co3` on a Blackwell | `co3_bw` on `mscluster110` to `112` |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

The walk map is `2-classifier-guidance/artifacts/drips/our-photos-both-ways/EXECUTE_WALK.md`.

## References

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [the route, step 8](route-classifier-guided-poe-from-toy-to-animals.md#8--our-photos-both-ways-the-released-model-and-a-model-we-train)
- [A1's page](../report/classifier-guided-poe/a1-exact-toy-classifier-push.md)
- [the pool card](../../4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md)
- [the step 4 card](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md)

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

The route's step 11 reads this plan's card and finding.

## Cross-references

- The mention of **our photos, both ways**, stage 3 of the learning table, in [the route for classifier-guided PoE](route-classifier-guided-poe-from-toy-to-animals.md#what-you-will-achieve).
