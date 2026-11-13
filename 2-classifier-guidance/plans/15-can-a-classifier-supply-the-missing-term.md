# 🧪 Can a classifier supply the term the product drops

## Recommended prompt

```text
/drip-execute-plan 1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/15-can-a-classifier-supply-the-missing-term.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [is the gap the samplers or the models](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md) ›
this plan. Step 78 in the root running order, after
[hypothesis 14](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/14-does-a-clip-free-model-hybridise.md), which asks whether CLIP is why the
product fuses. This plan asks whether a classifier can push it apart again.

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

Two captions compete, because every caption's probability is normalised over all the captions a
picture could have, so the product of "a photo of a cat" and "a photo of a dog" is maximised by
one animal that reads as both. Two binary presence questions do not compete: "a cat instance is
present" and "a dog instance is present" have no shared normaliser, so a frame holding two
animals can score high on both at once. This plan trains that two-output presence head and adds
its gradient to plain product sampling.

**What already exists**

The species instrument, measured and passing: zero-shot CLIP on the detector's padded crop called
cat on 64 of 64 renders at median margin 0.990, where GroundingDINO's own confidences separated
the classes by 0.054. The 91 audited both-animal photographs, and 3,335 cat-only and 2,708
dog-only after the same box-fraction floor. A FiLM conv net on noisy latents, timestep
conditioned, in [the twist head](../../poe_repair/experiments/twisted_smc/twist.py). A
photo-to-latent-to-noise training path in
[the inclusive-captions trainer](../../poe_repair/experiments/inclusive_captions/trainer.py).

**What does not**

Any image labelled "neither animal present". The labelled set has to be built, and its hybrid
negatives can only come from this repo's own product renders, which is a circularity the card
names rather than hides.

**What you will hold at the end**

One sheet: eight held-out cat-by-dog seeds down the rows, four classifier weights across, with
plain product, the damped product and the joint prompt as three reference columns rendered in the
same loop, every tile called by eye at full resolution.

## Considerations

**The classifier's own accuracy is not the gate, by decision.**

A held-out accuracy curve read before anything renders would be the careful order, and it was
dropped on purpose: the question is whether composition emerges at all, and the cheapest look at
that is to render through the guided sampler at training checkpoints and watch. Balanced accuracy
per label is still logged at every probe, so the curve exists at the end without ever having
gated a render.

**The probe costs more than the training it watches.**

The head is small and its data is cached latents, so 250 training steps take under a minute. One
probe at two seeds and two weights is four 50-step SDXL renders, about three and a half minutes
on a Quadro RTX 8000. So the GPU spends most of its wall clock rendering, and a two-hour run buys
roughly thirty probes and seven thousand training steps. That is the right trade here and it is
stated so nobody later reads the step count as slow training.

**A rising count and a ruined picture look the same in any norm.**

The Langevin corrector's norm guard reported healthy numbers while the samples were destroyed. So
every call in the final sweep is made by eye on a full-resolution tile, and no per-step norm
statistic may stand in for that. Two renders already read as a clear cat and a clear dog at
contact-sheet scale and as hybrids at full resolution, so hybrid is its own category and not a
near miss.

**The crude fix already puts two animals in frame.**

Weighting each expert at a constant 3.0 smears both subjects on 8 of 8 cells: both animals appear
and the picture is ruined. Beating plain product's 0 of 8 is therefore not the bar. The damped
product renders as its own column, in the same loop, and the new method is read against it.

**Nothing has tested whether the missing term is a gradient at all.**

A classifier can only supply a term that is the gradient of some scalar field. Whether the
correction the product drops has that form is untested in this repo, so a null here has two
readings that this plan cannot separate: the classifier is the wrong instrument, or no classifier
could work. The closing report says so.

## The claim under test

A two-output presence head on noisy latents, whose labels define a hybrid as neither animal
present, supplies a gradient that moves plain product sampling toward two separate animals, and
the move is visible as the head trains.

## Why this plan exists

**Every repair tried so far either adds another expert or searches the noise.**

Scope 06 holds four nulls: a Langevin corrector on the product score, Feynman-Kac steering on the
detector reward, zero-order search over the initial noise, and a noise-trajectory search. All of
them reweight or re-draw inside the space the two experts already agree on. A classifier gradient
is not another expert, so it can put mass where both experts say there is none, which is the one
thing none of the four could do.

**The labelling rule is where the meaning of "both" actually lives.**

No caption carries it. The product fuses because nothing in the objective says a cat and a dog
must be two separate things. Defining a hybrid as neither animal present puts that requirement
into a loss for the first time, which is why the rule sits in the source as a named constant
rather than in prose.

## What happens

```
91 both-animal photos      ──> both labels 1
3,335 cat-only             ──> cat 1, dog 0          ──┐
2,708 dog-only             ──> cat 0, dog 1            ├──> VAE ──> add noise at the
this repo's product renders ──> hybrid ──> both 0     ──┘          sampler's own timesteps
                                                                        │
                                                                        v
                                             two sigmoid outputs, FiLM conv on latents
                                             SDXL frozen, only the head trains
                                                                        │
                     every 250 steps ◀──────────────────────────────────┘
                     2 seeds x 2 weights through the guided composer
                     sheet + compose count + per-label accuracy ──> W&B
                                                                        │
                                                                        v
                     the best checkpoint ──> 8 seeds x 4 weights
                                             + plain product, damped product, joint prompt
                                             all in one loop ──> called by eye ──> the card
```

## Description

Piece 1 writes the labelling rule as named constants, mines hybrid negatives from existing
product renders, labels every image, and builds a sheet to look at. Piece 2 writes the composer
and its sampler, with a zero-weight canary that must be byte-identical to plain product. Piece 3
trains the head and probes it through that composer every 250 steps, logging the sheet, the
compose count and the per-label accuracy to W&B. Piece 4 renders the full sweep on the best
checkpoint with its three reference columns, calls every tile by eye, and writes the card.

## Purpose

Decide whether a presence classifier's gradient makes plain product composition draw two animals,
and see whether that emerges as the classifier learns.

## Goal

The weight-sweep sheet, `calls.json` with one record per tile, and a card stating the best
weight's count out of eight beside plain product's 0 of 8, the damped product's count and the
joint prompt's 8 of 8.

## Environment Facts This Plan Depends On

**The `co3` absolute python path.** `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`. A
bare `python` in a job script is a catalogued failure here.

**`co3` cannot drive a Blackwell card.** It carries torch 2.5.1+cu118 and `mscluster110`, `111`
and `112` need CUDA 12.8 or later, so `torch.cuda.is_available()` returns False and the job runs
on the CPU with no error. Pin to `mscluster106`, `108` or `109` and assert the GPU after pinning.

**One device for the whole comparison.** One uncorrected cell reads 0.153, 0.169 and 0.180 on the
RTX 3090, the Blackwell and the Quadro RTX 8000, so a sweep that crosses hardware is not a sweep.

**Large artifacts go to `/datasets` only.** `/home-mscluster` hit 100% once and killed
checkpointing silently. The disk guard must check the filesystem actually being written to.

**`biggpu` allows one job per user**, so the training run starts with `nohup` outside Slurm on a
pinned free device and is invisible to `squeue`; harvest it with `pgrep` instead.

**The fp16 upcast rule.** SDXL's VAE declares `force_upcast` and runs in fp32 while the UNet runs
in fp16. A float32 against float16 comparison of the same combine gives about three grey levels
over 50 steps, so every reference column renders in the same loop as the sweep it is read against.

**Render cost.** 51.5 s per 50-step cell measured on the Quadro RTX 8000, so the 56-tile final
sweep is about 50 minutes and one probe of four tiles is about 3.5 minutes.

## Success and failure outcomes

Proposed, for you to tick or rewrite in [the review file](review/15-can-a-classifier-supply-the-missing-term.md)
before the head trains.

**Support.** At some weight, at least 4 of the 8 held-out seeds hold two clear animals by eye at
full resolution, and the damped product's count on the same seeds is no higher. The classifier
gradient composes, and it does so without the smearing the crude fix pays.

**No support.** 0 or 1 of 8 at every weight. The gradient did not move composition. This cannot
distinguish "the classifier is the wrong instrument" from "the missing term is not a gradient",
because nothing in this repo has tested the second.

**Inconclusive.** 2 or 3 of 8 at the best weight, or a count that beats plain product but not the
damped product. Eight seeds cannot resolve the move.

**The outcome to be careful with.** The count rises while the pictures are ruined, which is what
the damped product already does and what the Langevin corrector's norm guard could not see. It is
its own line, not a null and not a support.

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 🔬 The labelled set, and the rule in the source

- [x] ✅ **1.1** Write the labelling rule as named constants and the labeller that applies them:
      both labels 1 needs two separate animal boxes, each carrying one species on its own, and a
      single box carrying both is a fused animal, labelled neither.
  - Output: [the labelling module](../../poe_repair/experiments/presence_head/labelling.py)
  - The rule is constants in source, never prose, so changing it after seeing the renders shows
    up in a diff.
  - ✅ verified 2026-11-04: seven cases run, a fused animal labels (0, 0) and only two separately
    named boxes label (1, 1).
  - The first rule compared the two species against each other, so it had to pick one and named a
    confident species on 32 of 33 fused renders. Each species is now read on its own as a raw
    cosine similarity with no softmax, calibrated on 400 photographs COCO annotated: at a floor
    of 0.245 a photograph passes its own species 0.925 of the time and the other 0.070.
- [x] ✅ **1.2** Mine fused-animal negatives from existing product renders across every animal
      pair available, excluding the eight held-out cat-by-dog seeds, and report the count and
      which cells they came from.
  - The negatives come from the model being corrected. Named as a circularity in
    [the card](../artifacts/results/can-a-classifier-supply-the-missing-term/README.md).
  - ✅ verified 2026-11-04: 50 fused animals across 13 pairs, led by cat × dog and tiger × dog at
    9 each, from 446 distinct renders. 481 paths were 438 images before a second mining pass, so
    de-duplication is by file content, never by path.
- [x] ✅ **1.3** Build the labelled set and render one sheet per label group. The latent cache is
      built by the trainer, once, on first run.
  - Output: `/datasets/mmolefe/poe_repair_min/outputs/presence_head/labelled_set/`
  - ✅ verified 2026-11-04: 1,028 pictures. 200 cat-only, 200 dog-only and 91 both-animal
    photographs, 93 pictures holding a real pair, 264 holding no animal, and 50 fused animals.

▶ **Next: [instruction 5](#5--look-at-the-labelled-set)**, you check the hybrids really are
hybrids before anything trains on them.

### 2. 🔧 The composer and its zero-weight canary

◀ **Needs: [instruction 5](#5--look-at-the-labelled-set)** done.

- [x] ✅ **2.1** Write the sampler: the unconditional prediction plus the guidance-scaled sum of
      the two experts' differences plus the weighted gradient of the head's log probability for
      both labels, normalised per step against the product prediction's own norm.
  - Output: [the sampler](../../poe_repair/methods/_classifier_guided.py)
  - The gradient is taken with respect to the noisy latent the head already reads, so no decode
    through the VAE is needed.
  - ✅ verified 2026-11-04: the gradient reaches the latent and is finite on an untrained head.
- [x] ✅ **2.2** Write the composer wrapper, following
      [the Langevin composer's](../../poe_repair/composers/poe_langevin.py) shape.
  - Output: [the composer](../../poe_repair/composers/poe_classifier_guided.py)
- [x] ✅ **2.3** Prove the canary: at weight 0 the render is byte-identical to plain product on
      the toy seed, both with no head attached and with an untrained head attached, since the
      sweep passes a head.
  - Command: `python 2-classifier-guidance/scripts/presence_head_canary.py`
  - What you want: identical bytes, not a small difference.
  - ✅ verified 2026-11-04 on mscluster108: all three renders share checksum `c291b259`, largest
    absolute pixel difference 0 of 255, zero pixels differing.

▶ **Next: [instruction 6](#6--run-the-canary-yourself)**.

### 3. 🚀 Train the head and probe it while it trains

◀ **Needs: [instruction 6](#6--run-the-canary-yourself)** done, so the composer is known to leave
plain product alone before a trained head is pushed through it.

- [x] ✅ **3.1** Write the trainer: two sigmoid outputs on the twist head's FiLM conv body, SDXL
      frozen, timesteps drawn from the 50 the sampler actually visits.
  - Output: [the trainer](../scripts/train_presence_head.py) and
    [the head](../../poe_repair/experiments/presence_head/head.py), with the labelling rule
    imported, not restated.
  - Disk guard reads `/datasets`, which is where the checkpoints land.
  - The 50 fused animals are 5% of the pile, so they are oversampled to a quarter of every batch.
    That share is a constant in the source for the same reason the rule's constants are.
- [ ] **3.2** Add the probe: every 250 steps, render two held-out seeds at two weights through the
      composer, and log the sheet, the compose count and balanced accuracy per label to W&B.
  - W&B: `prime_lab/poe-repair-animals-compose`.
  - The probe is a watch, not a verdict. Four tiles cannot carry a rate.
- [ ] **3.3** Launch the run on a pinned free `biggpu` device and harvest it.

▶ **Next: [instruction 7](#7--watch-the-probe-sheets-fill)**, you watch whether two animals start
appearing.

### 4. 📊 The sweep, the calls and the card

◀ **Needs: [instruction 7](#7--watch-the-probe-sheets-fill)** done, so the checkpoint and the
weight band are chosen from what you saw.

- [ ] **4.1** Render the full sweep in one loop on one device: eight held-out seeds, four
      classifier weights, plus plain product, the damped product at a constant 3.0 on each expert,
      and the joint prompt.
- [ ] **4.2** Build the sheet and cut every tile to full resolution for calling.
- [ ] **4.3** Draw the accuracy figure from the probe numbers already logged, and write the card.
  - Output: `2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/`

▶ **Next: [instruction 8](#8--call-every-tile-by-eye)**, the calls, which are yours.

## Instructions

For you to follow manually. Do these yourself.

### 5. 👁️ Look at the labelled set

◀ **Needs: [tasks 1.1 to 1.3](#1--the-labelled-set-and-the-rule-in-the-source)** done.

5.1 **Open the hybrid sheet** at the path task 1.3 reports.

5.2 **Check the hybrids are hybrids**, meaning one animal that reads as both, and not a frame
holding two animals that the labeller mistook.

✅ What you want: every tile on the hybrid sheet is one fused animal. A tile holding two separate
animals is a mislabel, and it poisons the "neither" class.

▶ **Next: [tasks 2.1 to 2.3](#2--the-composer-and-its-zero-weight-canary)**.

### 6. 👁️ Read the canary's verdict

◀ **Needs: [task 2.3](#2--the-composer-and-its-zero-weight-canary)** done.

6.1 **Open `canary.json`** under `/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/`.

6.2 **Check the three checksums are one value** and the largest absolute pixel difference is 0.
The comparison is a checksum rather than a judgment, so the run makes it and you read it.

✅ What you want: one checksum across all three. Anything else means the composer changed the
baseline, and every later comparison would be against a moved reference.

▶ **Next: [tasks 3.1 to 3.3](#3--train-the-head-and-probe-it-while-it-trains)**.

### 7. 📈 Watch the probe sheets fill

◀ **Needs: [task 3.3](#3--train-the-head-and-probe-it-while-it-trains)** launched.

7.1 **Open W&B**, project `prime_lab/poe-repair-animals-compose`, and find the run by timestamp.

7.2 **Click Media** and step through the probe sheets in training-step order.

7.3 **Write down the first step at which you see two animals**, if you ever do, and at which
weight.

7.4 **Check the pictures are not being destroyed** as the count rises. A smeared frame with two
animal shapes in it is the outcome this plan warned about, not a success.

✅ What you want: a sequence you can read as a change over time, and a step and a weight to render
the full sweep at.

▶ **Next: [tasks 4.1 to 4.3](#4--the-sweep-the-calls-and-the-card)**.

### 8. ⚖️ Call every tile by eye

◀ **Needs: [tasks 4.1 and 4.2](#4--the-sweep-the-calls-and-the-card)** done.

8.1 **Open each full-resolution tile**, never the sheet, and call it one of three things: two
clear animals, one hybrid, or one single animal.

8.2 **Tick one line in the review file** against the proposed outcomes, or rewrite them first.

✅ What you want: the best weight's count out of eight, beside all three reference columns.

▶ **Next: the closing pass.** The figure is filed with its card entry, the finding goes to
`report/`, and `sync-plan-tree` reconciles the scope.

## Outputs

| What | Where |
|---|---|
| the labelling rule | `poe_repair/experiments/presence_head/labelling.py` |
| the labelled set and its latents | `/datasets/mmolefe/poe_repair_min/outputs/presence_head/labelled_set/` |
| the sampler | `poe_repair/methods/_classifier_guided.py` |
| the composer | `poe_repair/composers/poe_classifier_guided.py` |
| the trainer | `2-classifier-guidance/scripts/train_presence_head.py` |
| the checkpoints and probe sheets | `/datasets/mmolefe/poe_repair_min/outputs/presence_head/` |
| the probe time series | W&B, `prime_lab/poe-repair-animals-compose` |
| the sweep sheet, the calls, the accuracy figure and the card | `2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/` |
| the verdict | [the review file](review/15-can-a-classifier-supply-the-missing-term.md) |

## Figure Catalog

| Figure | What is on each axis, and where the numbers come from |
|---|---|
| `weight-sweep-sheet.png` | eight held-out cat-by-dog seeds down the rows; across, four classifier weights then three reference columns (plain product, each expert damped to a constant 3.0, the joint prompt). One tile is one 1,024 px render, all from one loop on one device. Calls are made on the full-resolution tiles, not on this sheet. |
| `accuracy-by-step.png` | balanced accuracy on a 0-to-1 scale on y, the sampling step from 0 to 49 on x, one curve per label (a cat instance is present, a dog instance is present), held-out images. Numbers from the probe rows logged to W&B, written to the sidecar beside the figure. Not a gate: logged while training, drawn at the end. |
| the probe sheet | two held-out seeds down, two classifier weights across, one image per 250 training steps, in W&B. Two seeds is a picture, never a rate. |

## Error Matrix

| Symptom | What it means | What to do |
|---|---|---|
| `torch.cuda.is_available()` is False, or a 20-step job takes minutes | the job landed on a Blackwell card `co3` cannot drive | pin to `mscluster106`, `108` or `109`; check `nvidia-smi --query-compute-apps` from another shell |
| the zero-weight render differs from plain product | the composer changed the baseline | the canary is the gate; fix it before any weight is swept |
| the compose count rises and the pictures look smeared | the gradient is destroying the sample, the way the damped product does | this is its own outcome; no norm statistic may overrule the eye |
| every probe sheet looks identical | the gradient is being normalised to nothing, or the head's outputs are saturated | print the gradient's norm before and after normalisation, and the two sigmoid values |
| the run completed but wrote nothing | the disk guard checked the wrong filesystem | the guard must read `/datasets`, where the output goes |
| `squeue` shows nothing while the run is alive | `biggpu` allows one job per user, so the run is outside Slurm | harvest with `pgrep -af 'presence_head\|train'` on the session node |
| the labeller calls two-animal frames hybrid | the species classifier has never been tested on overlapping boxes | report the count it could not call; instruction 5 exists to catch this |

## Orchestration

Walked with `/drip-execute-plan`. The head reuses
[the twist head's](../../poe_repair/experiments/twisted_smc/twist.py) FiLM body and the data
path reuses [the inclusive-captions trainer's](../../poe_repair/experiments/inclusive_captions/trainer.py)
photo-to-latent loop, rather than re-implementing either. The closing pass files the figures per
`ARTIFACT_TREE_FORMAT.md`, writes the finding per `REPORT_FORMAT.md`, and runs `sync-plan-tree`
from the scope root.

## References

- [the species instrument and its bars](../../3-understanding-compositionality/scripts/census/crop_classifier_cat_vs_dog.py)
- [how the compose scorer was validated](../../poe_repair/experiments/compose_scorer_validation/validate_detection.py)
- [the photo pool](../../4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md)
- [the twist head](../../poe_repair/experiments/twisted_smc/twist.py)
- [why a corrector alone produced nothing](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)
- [what the crude damping costs](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md)
- [the compose rate's definition](../../context/world/compose-rate.md)

## Next step

Task 1.1: the labelling rule, as constants in the source.
