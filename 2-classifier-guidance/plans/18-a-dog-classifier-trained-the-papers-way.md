# 🔧 A dog classifier trained the paper's way

## Recommended prompt

```text
/drip-execute-plan 1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/plans/tools/18-a-dog-classifier-trained-the-papers-way.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [is the gap the samplers or the models](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md) ›
this plan. Step 80 in the root running order. It follows
[plan 17](17-classifier-guidance-from-the-paper.md), whose sampler is right and whose classifier drew
no dog.

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

A noise-aware "is there a dog" classifier trained as Dhariwal and Nichol train theirs: on noisy
pictures from the same distribution the model draws, with the downsampling body trained rather
than frozen. It then guides SDXL's empty prompt with plan 17's sampler.

**What already exists**

Plan 17's sampler, following Algorithm 2 and identical to the plain render at strength 0. The
cleaned picture piles and their latent cache. A frozen-feature head that drew no dog.

**What does not**

Any training picture of what SDXL draws with no prompt. The no-prompt model draws abstract
paintings, and the frozen head never saw one, so its answer on them is close to a guess.

**What you will hold at the end**

Plan 17's six-column sheet, redrawn with this classifier.

## Considerations

**The data has to cover where guidance starts.**

The paper's classifier and its model share one distribution. Here the push starts from abstract
paintings, so about 300 no-prompt renders join the "no dog" pile.

**The body is trained through small adapters.**

The paper trains its whole downsampling trunk. LoRA adapters inside SDXL's down and middle blocks
train that trunk without fitting a 1.2-billion-parameter copy of it in memory.

**One question first.**

"Is there a dog" alone. Cat, and then both, only once this draws a clean dog.

## The claim under test

A dog classifier trained on noisy pictures that include the no-prompt model's own output, with its
body trained, guides the empty prompt to a clean dog.

## Why this plan exists

**Plan 17 isolated the classifier as the failing part.**

Its sampler reproduces the plain render exactly and its pictures stay clean at every strength, so
the missing dog is the classifier's.

## What happens

```
~300 no-prompt renders  ──> no dog  ─┐
COCO photos without dogs ─> no dog   ├─> noise at the sampler's timesteps ─> SDXL down + mid, with
COCO photos with dogs   ──> dog     ─┤                                       adapters, + attention
cache renders           ──> either  ─┘                                       pool ─> "dog?"
                                       ─> check before steering ─> plan 17's sheet, redrawn
```

## Description

Piece 1 reads what the frozen head believes about the no-prompt pictures, then builds the data.
Piece 2 trains the classifier with the check before steering. Piece 3 redraws the dog sheet. Piece
4, only if piece 3 draws a clean dog, repeats it for cat and then both.

## Purpose

Establish whether a classifier trained the paper's way can guide this model to one class.

## Goal

The redrawn dog sheet with a person's call on at least 3 of 4 seeds.

## Environment Facts This Plan Depends On

**The `co3` absolute python path.** `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.

**`co3` cannot drive a Blackwell card**, so the run pins a Quadro RTX 8000 on `mscluster108`.

**peft 0.19.1 is installed in `co3`**, which is what the adapters use.

**Large artifacts to `/datasets` only**, with the disk guard on the filesystem written to.

## Success and failure outcomes

**Support.** A clean dog on at least 3 of 4 seeds at some strength.

**No support.** No dog at any strength while the check before steering passed. Then a classifier
that sees noisy pictures well still does not steer this model, which is the stronger negative.

**The check fails.** The classifier cannot tell dog from no dog at the noisy steps, and the sheet
is not drawn.

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 🔬 What the frozen head believes, then the data

- [x] ✅ **1.1** Read the frozen head's "dog present" probability on the four no-prompt pictures,
      the four "a dog" pictures and the four most-pushed pictures, at six sampler steps.
  - Output: [the belief sheet](../artifacts/results/does-classifier-guidance-draw-a-dog/what-the-helper-believes.png),
    numbers in `/datasets/mmolefe/poe_repair_min/outputs/dog_classifier/what_the_head_believes.json`
  - ✅ verified 2026-11-04: at sampler step 20 the head calls 2 of 4 clear dog photos no dog (0.00)
    and all 4 pushed pictures dog (0.77 to 0.93), none of which holds a dog; the no-prompt paintings
    it mostly calls no dog. It was taught about 3,000 single smooshed animals as no and about 500
    single animals as yes, so one clean animal reads as no. The dog-only data therefore takes single
    dogs as yes and holds no smooshes; those return at the both stage.
- [x] ✅ **1.2** Render about 300 no-prompt pictures at 25 steps on new seeds, gather dog and non-dog
      COCO photos, and sheet a sample of each pile.
  - Output: [the pile builder](../scripts/dog_classifier_piles.py) and
    [the no-prompt renderer](../scripts/render_no_prompt_pool.py)
  - ✅ verified 2026-11-04: 300 no-prompt renders, seeds 1000 to 1299, the last 78 spread over six
    `batch` nodes; 6,644 pictures, 2,844 with a dog (2,711 COCO dog photos) and 3,800 without.

▶ **Next: [instruction 5](#5--check-the-piles)**.

### 2. 🚀 Train the classifier

◀ **Needs: [instruction 5](#5--check-the-piles)** done.

- [x] ~~**2.1** LoRA adapters in SDXL's down and middle blocks plus an attention-pool head, one
      sigmoid for dog, trained at the sampler's timesteps, with probe pictures while it trains.~~
      Run `dog-01`, 3,000 steps, W&B `z2k1bdmi`; checkpoints and probes under
      `/datasets/mmolefe/poe_repair_min/outputs/dog_classifier/dog-01/`.
- [x] ~~**2.2** The check before steering: balanced accuracy at least 0.75 on at least 10 of the first
      20 sampler steps, written to `gate.json` whether it passes or not.~~ **Failed: 7 of 20.**
      Balanced accuracy rises from 0.62 at the noisiest step to 0.82 at step 19, passing only at
      steps 13 to 19.

▶ **Next: [task 3.1](#3--the-dog-sheet)** if it passed.

### 3. 📊 The dog sheet

◀ **Needs: [task 2.2](#2--train-the-classifier)** passed.

- [x] ~~**3.1** Plan 17's six columns on the same four seeds, with this classifier.~~ Drawn past the
      failed check at the user's choice, at fixed strengths 10, 30, 100 and 300 with the step-3,000
      checkpoint; strength 0 matched the empty prompt byte for byte on all four seeds. Sheet:
      `2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-sheet/dog-sheet.png`.

▶ **Next: [instruction 6](#6--call-the-dog-sheet)**.

### 4. 🚀 Then cat, then both

◀ **Needs: [instruction 6](#6--call-the-dog-sheet)** passed.

- [ ] **4.1** The same classifier for cat, then the inclusive both, on the cat × dog product.

## Instructions

For you to follow manually. Do these yourself.

### 5. 👁️ Check the piles

◀ **Needs: [tasks 1.1 and 1.2](#1--what-the-frozen-head-believes-then-the-data)** done.

5.1 **Open each pile's sample sheet** and say whether the no-dog pile holds any dog.

✅ What you want: no dog in the no-dog pile, a dog in every dog picture.

✅ checked 2026-11-04 by Claude, delegated by the user ("you decide"), open to veto: the dog sheet
shows a dog in 23 of 24, the one miss a COCO photo of two men whose annotated dog is not visible;
the no-dog sheet shows no dog in 24 of 24.

▶ **Next: [task 2.1](#2--train-the-classifier)**.

### 6. ⚖️ Call the dog sheet

◀ **Needs: [task 3.1](#3--the-dog-sheet)** done.

6.1 **Call each pushed picture** a clean dog, a dog forced in with the picture damaged, or no dog.

✅ What you want: a clean dog on at least 3 of 4 seeds.

❌ called 2026-11-05 by Claude from full-resolution tiles, delegated by the user, open to veto:
**clean dog on 2 of 4**, bar not met. Seed 9: a clean painted dog among flowers at strength 30.
Seed 10: a clean shaggy dog in a cap at strength 100. Seed 12: a dog's head on a sofa at strength 100
whose body melts into a white blanket (dog forced in, picture damaged). Seed 11: a café with no dog
at 10 to 100, and at 300 a faint dog in a wrecked street (dog forced in, picture damaged). Every
seed shows a dog at some strength, so the classifier steers; the failing seeds are where it sees
least, at the noisiest steps.

The same push from a nearer start, at the user's choice ("nearby", 2026-11-05), called by Claude
from full-resolution tiles. From "an animal": a clean grey dog on seed 10 at strength 10, a clean
drawn dog on seed 9 at 10, a clean wolf-like canine on seed 11 at 30, and on seed 12 a dog whose
body fuses with a second white body at 100. From "a cat": a cat on every seed at every strength up
to 100, and a broken cat at 300. The prompt's ordinary guidance outweighs a push of 2 to 3% of the
prediction, which matters for the cat × dog product, where two prompts push back. Sheets under
`2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/dog-01-from-an-animal/` and `.../dog-01-from-a-cat/`.

The cat × dog product pushed toward dog present with the same classifier ("product", 2026-11-05),
called by Claude from the sheet, strength 0 identical to the plain product on all four seeds. Two
separate animals in 0 of 16 pushed tiles. The push turns the fused animal into one cat on seeds 10,
11 and 12 at strength 30 to 300, and leaves seed 9 a fused animal until the picture breaks at 300.
Pushing toward dog made the product more cat. Sheet:
`2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/product-pushed-to-dog/sheet.png`.

## Outputs

| What | Where |
|---|---|
| the data and its sheets | `/datasets/mmolefe/poe_repair_min/outputs/dog_classifier/` |
| the redrawn sheet | `2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/` |

## Figure Catalog

| Figure | What is on each axis, and where the numbers come from |
|---|---|
| the belief table | rows: the four no-prompt and four "a dog" pictures; columns: clean and five sampler steps; each cell the frozen head's "dog present" probability, 0 to 1 |
| the redrawn dog sheet | four seeds down; no prompt, "a dog" under ordinary guidance, and four classifier strengths across |

## Error Matrix

| Symptom | What it means | What to do |
|---|---|---|
| the frozen head says dog on the no-prompt pictures | it was never shown them | the no-prompt renders in the no-dog pile are the fix |
| out of memory while training | the adapters' activations through the down blocks | turn on gradient checkpointing |
| the push shrinks to nothing under guidance | the classifier is satisfied without a dog | read its probability along the guided run |

## Orchestration

Walked with `/drip-execute-plan --request`. Reuses plan 17's sampler and script, the cleaned piles
and the strong head's feature hook.

## References

- Dhariwal and Nichol, arXiv 2105.05233, section 4 and Algorithm 2
- [plan 17](17-classifier-guidance-from-the-paper.md)

## Next step

Task 1.1: what the frozen head believes.
