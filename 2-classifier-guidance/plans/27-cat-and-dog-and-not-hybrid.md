# 🧪 Cat AND dog AND NOT hybrid: SDXL gives the direction, the checker gives the weight

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/27-cat-and-dog-and-not-hybrid.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, beside [plan 25](25-the-readout-checker-on-poes-own-states.md) and
[plan 26](26-two-pushes-bodies-early-species-late.md). Its pass marks are in
[its review file](review/27-cat-and-dog-and-not-hybrid.md); its results go into
[the checker loop file](../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

## Table of contents

- [Quick context](#quick-context)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

**Why this plan exists**

Every push so far follows a checker's gradient. With room for a larger push (cap 0.3) the identity push gives a
separate cat and dog on most held-out seeds, but damages the pictures. A species push ("cat present" plus "dog
present", plan 26) pulls two bodies back into one face, because a hybrid scores high on both. Logical Guidance
(Alesiani et al., arXiv 2602.05549) gives the score of NOT a condition as −p/(1 − p) times that condition's
conditional-minus-unconditional score, and its gradient-free guidance takes the direction from the diffusion
model's own prompts and only the weight from a classifier. Its AND rule is plain addition, exact only when the
two conditions are independent given the current state; that independence is what fails for cat and dog.

**What it does**

One trajectory per seed. Each step is plain PoE minus a NOT-hybrid term: SDXL's own direction for "a hybrid animal,
half cat and half dog", weighted by the readout checker's probability that the state is not heading for a clean
pair. The checker's gradient is never used, so the term cannot follow a texture trick the checker responds to.
A fixed negative prompt with no checker weight is the control.

## Goal

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

A sheet on held-out seeds 9 to 16 that answers Q1 (does the weighted NOT-hybrid term give a separate cat and dog,
with less damage than the cap-0.3 push) and Q2 (does the checker's weight matter), per
[the review file](review/27-cat-and-dog-and-not-hybrid.md).

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- mscluster111 and mscluster112 are Blackwell cards driven only by `co3_bw` ([the node table](../../environment/hpc/nodes.md)).
- Outputs go under `POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch` while `/datasets` is down
  ([storage](../../environment/storage.md)).
- Jobs start with `nohup` on the node, outside Slurm; logs are under `outputs/classifier_guidance_ladder/rung2/logs/`.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 1. 🚀 Build and draw

- [x] **1.1 Code.** `scripts/rung2/not_hybrid.py`: a four-prompt PoE loop with the NOT-hybrid term, the λ 0 check
  against plain PoE, six columns per seed, `record.json` with the weight per step and how often the guard fired.
- [ ] **1.2 Draw.** `rung2/not-hybrid.sh` on a free card. The log prints, per seed, the λ 0 check and the mean weight.
- [ ] **1.3 Compare.** One overview beside plain PoE and the identity push at cap 0.3, sent with draft calls from
  full-size tiles.

▶ Next: [2. Call the sheet](#2--call-the-sheet)

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow.

### 2. 👁️ Call the sheet

◀ Needs: [1.3 Compare](#1--build-and-draw)

- [ ] **2.1 Call every full-size tile** on the ladder single < hybrid < two bodies or heads < two clear, noting damage
  apart, and write the counts per column into the review file's Q1 and Q2.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents)

Under `outputs/classifier_guidance_ladder/rung2/not-hybrid/` on the scratch mount: `tiles/`, `record.json`,
`sheet.png`. Verdicts go into the review file and a section of the checker loop file.
