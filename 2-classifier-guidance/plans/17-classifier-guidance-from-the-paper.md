# 🔧 Classifier guidance, reproduced from the paper before it is trusted

## Recommended prompt

```text
/drip-execute-plan 1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/plans/tools/17-classifier-guidance-from-the-paper.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [is the gap the samplers or the models](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md) ›
this plan. Step 79 in the root running order. It gates the guidance half of
[hypothesis 15](15-can-a-classifier-supply-the-missing-term.md): no result from
classifier guidance on cat × dog is read until this plan shows guidance producing one clean class.

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

Classifier guidance as Dhariwal and Nichol define it: a diffusion model trained once and left alone,
a classifier trained afterwards on noisy pictures, and the classifier's gradient added to the
model's prediction at every sampling step. The test is the paper's own case: SDXL with no prompt,
pushed toward "dog", should draw a clean dog.

**What already exists**

A presence head trained on SDXL's frozen middle block over 5,210 noisy pictures, with a "dog
present" answer among its 182 species. Its balanced accuracy on dog rises from 0.57 at sampler step
0 to 0.71 at step 19.

**What does not**

A guided sampler that follows the paper. The one used on 2026-11-04 rescaled every push to the full
norm of the noise prediction, about 265 per step against a raw gradient around 0.006, so the push
was roughly 44,000 times what the classifier asked for.

**What you will hold at the end**

One sheet: four seeds down, and across them the no-prompt picture, the pushed picture at four
strengths, and "a dog" with ordinary guidance.

## Considerations

**The two ends are the whole reading.**

The no-prompt picture is what guidance starts from and "a dog" under ordinary guidance is the look
to match. A pushed picture is read against both, so both are rendered in the same loop.

**The strength has a scale set by the paper, not by us.**

An unconditional model needed a strength around 10 before samples matched the class, but that
number belongs to the paper's classifier and its gradient size. This head's gradient is about 0.006
against a noise prediction near 265, so a strength of 30 would move the picture by under a tenth of
a percent. The formula stays the paper's; the four strengths are chosen from the gradient measured
on the empty-prompt run, so that the median push is 1, 3, 10 and 30 percent of the prediction.

**Every step is pushed.**

The paper guides all steps. The head is weakest at the noisiest steps, and the paper's own scaling
already shrinks the push there through the noise factor.

## The claim under test

Adding the presence head's gradient for "dog", in the paper's form, to SDXL's unconditional
prediction at every step draws a clean dog.

## Why this plan exists

**Nothing guided so far can be read.**

Every guided render on 2026-11-04, for the weak head, the strong head and CLIP, used the oversized
push. Their failures say nothing about classifier guidance until guidance is shown to work on the
plainest case.

## What happens

```
seed ──> no prompt, plain                     the starting point
     ──> no prompt + s × push toward "dog"    s = 1, 3, 10, 30
     ──> "a dog", ordinary guidance 7.5       the look to match
```

## Description

Piece 1 renders the two ends on four seeds. Piece 2 replaces the rescaled push with Algorithm 2's
form and checks that strength 0 reproduces the no-prompt picture byte for byte. Piece 3 renders the
pushed pictures. Piece 4, only if piece 3 draws a clean dog, applies the same push to the cat × dog
product.

## Purpose

Establish whether classifier guidance works on this model with this head before any composition
result built on it is believed.

## Goal

The sheet, with a person's call on whether any pushed picture is a clean dog on at least 3 of 4
seeds.

## Environment Facts This Plan Depends On

**The `co3` absolute python path.** `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.

**`co3` cannot drive a Blackwell card**, so the run pins a Quadro RTX 8000 on `mscluster108` and
asserts the GPU after pinning.

**One device for the whole sheet**, since the same cell reads differently across cards.

**Large artifacts to `/datasets` only**, with the disk guard on the filesystem written to.

## Success and failure outcomes

**Support.** On at least 3 of 4 seeds, a pushed picture is a clean dog photo a person would put
beside "a dog" under ordinary guidance.

**No support.** No pushed picture holds a recognisable dog at any strength. Then the head, not the
sampler, is the next suspect, and a classifier trained in the paper's style is owed.

**The outcome to be careful with.** A dog appears only at strengths that wreck the picture. That is
a dog drawn by force, and it is its own line.

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 🔧 The two ends

- [x] ✅ **1.1** Render four held-out seeds two ways: the empty prompt alone, and "a dog" under
      ordinary guidance at 7.5, 50 DDIM steps, one loop, one device.
  - Output: `2-classifier-guidance/scripts/classifier_guidance_from_the_paper.py`, sheet under
    `2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/`
  - ✅ verified 2026-11-04 on mscluster108: eight renders, seeds 9 to 12, `two-ends.png` written.

▶ **Next: [instruction 5](#5--look-at-the-two-ends)**.

### 2. 🔧 The paper's push

◀ **Needs: [instruction 5](#5--look-at-the-two-ends)** done.

- [x] ✅ **2.1** Replace the rescaled push with epsilon minus s times sqrt(1 - alpha_bar_t) times the
      gradient of log p(dog | x_t), on every step.
  - Output: `run_cfg_classifier_guided` in [the guided samplers](../../poe_repair/methods/_classifier_guided.py)
- [x] ✅ **2.2** Check strength 0 reproduces the empty-prompt render byte for byte.
  - ✅ verified 2026-11-04 on mscluster108: identical checksums on all four seeds.

▶ **Next: [task 3.1](#3--a-dog-from-nothing)**.

### 3. 🔧 A dog from nothing

◀ **Needs: [tasks 2.1 and 2.2](#2--the-papers-push)** done.

- [x] ✅ **3.1** Render the four seeds at four strengths beside the two ends, and record the raw
      gradient size and the applied push at every step.
  - Strengths 1.76, 5.29, 17.6 and 52.9, chosen so the median push on the unguided run would be
    1, 3, 10 and 30 percent of the prediction. Under guidance the push came out smaller, a median
    of 0.2 to 1.8 percent, because the head grows more confident as the picture is pushed and its
    gradient shrinks. Per-step numbers in `push.json` beside the sheet.
  - ✅ verified 2026-11-04: 16 renders, `dog-sheet.png` written.

▶ **Next: [instruction 6](#6--call-the-dog-sheet)**.

### 4. 🚀 Back to cat and dog

◀ **Needs: [instruction 6](#6--call-the-dog-sheet)** passed.

- [ ] **4.1** Apply the same push, both cat and dog, to the cat × dog product, against plain
      product and the joint prompt.

▶ **Next: [hypothesis 15](15-can-a-classifier-supply-the-missing-term.md)**.

## Instructions

For you to follow manually. Do these yourself.

### 5. 👁️ Look at the two ends

◀ **Needs: [task 1.1](#1--the-two-ends)** done.

5.1 **Open the two-ends sheet** and say what the empty prompt draws on each seed.

✅ What you want: "a dog" is a clean dog on every seed, so it can serve as the look to match.

✅ confirmed by hand 2026-11-04: the empty prompt is "just an organized mess", abstract paintings
with no dog on any seed; "a dog" is a dog on all four.

▶ **Next: [task 2.1](#2--the-papers-push)**.

### 6. ⚖️ Call the dog sheet

◀ **Needs: [task 3.1](#3--a-dog-from-nothing)** done.

6.1 **Call each pushed picture** a clean dog, a dog drawn by force with the picture damaged, or no
dog.

✅ What you want: a clean dog on at least 3 of 4 seeds at some strength.

▶ **Next: [task 4.1](#4--back-to-cat-and-dog)** if it passed; otherwise the closing pass.

## Outputs

| What | Where |
|---|---|
| the script | `2-classifier-guidance/scripts/classifier_guidance_from_the_paper.py` |
| the sheets | `2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/` |
| the head it uses | `/datasets/mmolefe/poe_repair_min/outputs/strong_presence_head/strong-01/checkpoints/head_step_004000.pt` |

## Figure Catalog

| Figure | What is on each axis, and where the numbers come from |
|---|---|
| the two-ends sheet | four held-out seeds down; across, the empty prompt alone and "a dog" at ordinary guidance 7.5. One tile is one 1,024 px render. |
| the dog sheet | four seeds down; across, the empty prompt, the push toward "dog" at strengths 1, 3, 10, 30, and "a dog". The per-step gradient size and applied push go to the sidecar json beside it. |

## Error Matrix

| Symptom | What it means | What to do |
|---|---|---|
| strength 0 differs from the empty-prompt render | the guided sampler moved the baseline | fix before any strength is read |
| colour blow-out at every strength | the push is still too large | print the applied push against the prediction's norm; it should be a small fraction |
| no change at any strength | the push is too small or the gradient is zero | print the raw gradient size per step |

## Orchestration

Walked with `/drip-execute-plan --request`. Reuses the strong head and its sampler wiring. The
closing pass files the sheets per `ARTIFACT_TREE_FORMAT.md`.

## References

- Dhariwal and Nichol, "Diffusion Models Beat GANs on Image Synthesis", arXiv 2105.05233,
  section 4.2 and Algorithm 2
- [the strong head](../../poe_repair/experiments/presence_head/strong_head.py)

## Next step

Task 1.1: the two ends.
