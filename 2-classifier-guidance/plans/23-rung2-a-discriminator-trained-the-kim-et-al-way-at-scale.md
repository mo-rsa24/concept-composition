# 🧪 Rung 2, third discriminator: Kim et al.'s joint-against-product discriminator, at scale

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/23-rung2-a-discriminator-trained-the-kim-et-al-way-at-scale.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which follows [plan 21](21-rung2-a-small-discriminator-steers-cat-x-dog.md) and
[plan 22](22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md) on the board's row A3q.

## Table of contents

- [Quick context](#quick-context)
- [Considerations](#considerations)
- [The claim under test](#the-claim-under-test)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Next step](#next-step)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**Why this plan exists**

Every discriminator trained on hand-called renders (17, then 81 positives) failed to tell two
separate things from a hybrid on pairs it never saw: held-out balanced accuracy 0.53 to 0.69 at every
early step. Kim et al. (arXiv 2211.17091) train their discriminator on tens of thousands of
examples, real data against the model's own samples, and need no labels. Here the real set is the
joint prompt's drawing and the fake set is the product's drawing, from the same noise; the
discriminator's log-odds then estimates the density ratio between them, which is the term the
product drops.

**What it does**

About 1,000 joint-prompt and 1,000 product drawings over the 87 two-things pairs of plan 22, seeds 1
to 8 and 17 to 20, each saving its own half-drawn states; a discriminator trained on those states;
the same held-out check as plan 22, on the hand-called tiles, before any push.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

The caveats are those of the Kim et al. read map
(`artifacts/drips/read-refining-generative-process-discriminator/READ_MAP.md`, "Caveats for PoE,
carried forward"); this plan meets them as follows.

**The fake set at each noise level (C1).** Fakes are the product sampler's own states at each
saved step, not finished pictures noised forward; reals are the joint sampler's own states.

**Style (C3).** Both sets come from the same SDXL, the same noise and the same pairs, so the only
systematic difference left is the prompt. Whatever the discriminator learns beyond composition shows
up in the held-out check, whose positives and negatives both come from pair prompts.

**The real set inherits the joint prompt's failures (C4).** About half the joint drawings do not
show both things (plan 22's calls), so the real set is noisy. That is the paper's setting (it
learns a ratio, not a label); the check measures whether composition survives it.

**Signal at high noise (C5) and a converged loss proving nothing (C9).** Steps are sampled evenly
over the 50, and the discriminator is judged only by the held-out check, never by its loss.

**Where outputs go.** `/home-mscluster/mmolefe/poe_repair_min_scratch` (via `POE_REPAIR_MOUNT`)
while `/datasets` is down.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

A discriminator trained without labels to tell the joint prompt's drawing from the product's, on
about 2,000 drawings, ranks hand-called two-separate-things tiles above hybrids and wrong pairs on
pairs it never saw.

## Goal

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar**, written before any drawing: on plan 22's held-out tiles (7 whole pairs plus cat × dog
seeds 7 and 8; 20 positives, 23 negatives) with those pairs left out of training, balanced accuracy
at least 0.75 on at least 10 of the first 20 sampler steps. If it passes, the rung 2 sheet is
rendered with it and judged by the rung 2 review file's seven readings, unchanged. If it fails, the
finding is that neither labelled nor unlabelled discriminators of this design read the pair on SDXL
at these sizes.

**Time box:** 1 sitting of the reader's time, plus about 2.5 hours of GPU.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- [The nodes](../../environment/hpc/nodes.md#partitions): `co3_bw` on `mscluster111` and `112`.
- [Storage](../../environment/storage.md): large outputs on `/datasets`; the scratch folder is the
  reader's exception while `/datasets` is down.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Test each piece](#2-test-each-piece)** walked.

- [ ] **0.1** `/verify-plan 2-classifier-guidance/plans/23-rung2-a-discriminator-trained-the-kim-et-al-way-at-scale.md`, then `/xref-pillar` on it, at the walk's close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build the pieces

◀ **Needs: nothing first.**

▶ **Next: [2. Test each piece](#2-test-each-piece)**

- [ ] **1.1** Draw the joint prompt and the product for the 80 training pairs (the 87 two-things
  pairs less the 7 held out) on seeds 1 to 8 and 17 to 20, saving every 5th step's state, on
  `mscluster111` and `112`.
- [ ] **1.2** Train the discriminator on joint states (real) against product states (fake) at
  matching steps, head alone on frozen SDXL, then with adapters.
- [ ] **1.3** The held-out check on plan 22's held-out tiles.
- [ ] **1.4** If it passes, the rung 2 sheet with it; the calls, the reading and the card.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Test each piece

◀ **Needs: [1. Build the pieces](#1--build-the-pieces)**

- [ ] **2.1** Look at a sample sheet of joint and product drawings before training.
- [ ] **2.2** Read the held-out curve against the bar.
- [ ] **2.3** Call the rung 2 sheet's tiles at full resolution, if one was rendered.

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the walk's close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

About 2,000 drawings with their saved states; two discriminators with held-out curves; possibly a
rung 2 sheet; results under `2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/`.

## Next step

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents)

The route's step 10b if the sheet lands on reading 1; otherwise rung 2's finding is extended with
this result.
