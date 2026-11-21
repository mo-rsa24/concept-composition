# 🧪 Bodies and species pushes on every step: does the sum turn two bodies into a clean cat and dog

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/31-bodies-and-species-pushes-on-every-step.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, after [plan 26](26-two-pushes-bodies-early-species-late.md) and [plan 29](29-does-the-early-push-hold-on-new-seeds.md).
Its pass marks are in [its review file](review/31-bodies-and-species-pushes-on-every-step.md).

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

The recipe (the [readout checker](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md)'s push at strength 30, cap 0.3, steps 0 to 15) separates the cat and dog on
28 of 32 new seeds, but only 14 are clean or near-clean. The usual miss is two bodies with a borrowed face or a
hidden head. Plan 26 added a species push only from step 10, after the bodies push had stopped, and the bodies
fused again. On MNIST digits the two kinds of push were added together on every step, and two clear, right digits
on unseen pairs rose from 215 to 285 of 320 ([the MNIST review file](review/a2b-mnist-digits-on-a-canvas.md#controls-and-improvements-on-the-final-setup-written-before-they-run)).

**What it does**

Adds the species push to the bodies push on all 50 steps, the sum capped at 0.3 of PoE's prediction, and reads
whether the extra clean pairs MNIST saw carry to cat × dog. Two species checkers are tried: the fine-tuned CLIP
species checker, which reads the decoded guess, and a presence-only readout checker, which reads SDXL's own
features as MNIST's separate-label checker did.

## Goal

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

One setting picked on seeds 9 to 16, then rendered once on seeds 3301 to 3332 beside plain PoE and a size-matched
random push, and called by eye from full-size tiles against Q1 in [the review file](review/31-bodies-and-species-pushes-on-every-step.md).

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- mscluster111 and mscluster112 are Blackwell cards (96 GB) driven only by `co3_bw`
  ([the node table](../../environment/hpc/nodes.md)).
- Outputs go under `POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch` while `/datasets` is down
  ([storage](../../environment/storage.md)).
- Jobs start with `nohup` on the node, outside Slurm; logs are under `outputs/classifier_guidance_ladder/rung2/logs/`.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 1. 🚀 Add the summed push, then draw seeds 9 to 16

- [x] **1.1 Code.** `render_results.py --species <checker> --species-strength S` adds the CLIP species push to the
  readout push on the same steps; `--presence-head` does the same for the presence-only readout checker.
- [x] **1.2 Identity check.** Seed 9, species strength 0, steps 0 to 15: the guided tile matches the recipe's tile
  byte for byte.
- [x] **1.3 Draw seeds 9 to 16, all 50 steps, bodies strength 30, cap 0.3.** Four settings: the bodies push alone;
  plus the CLIP species push at strength 10; at strength 30; plus the presence-only readout checker at strength 30.
  Each sheet carries plain PoE and a size-matched random push. Time per render measured from the record.
  ✓ verified 2026-11-19: the pushed tile at weight 0 differs from the recipe's only as much as two runs of the recipe
  differ from each other (a pushed render is not bit-reproducible); 28 s, 80 s and 20 s a pushed seed.

▶ Next: [2. Pick the setting](#2--pick-the-setting)

### 3. 🚀 Draw seeds 3301 to 3332 once

◀ Needs: [2. Pick the setting](#2--pick-the-setting)

- [x] **3.1 Render the picked setting** on seeds 3301 to 3332, all 50 steps, beside plain PoE and the size-matched
  random push, with the job command and time per render recorded.
- [x] **3.2 Add the result rows** to the "What we know so far" table in [the route](route-classifier-guided-poe-from-toy-to-animals.md).

▶ Next: [4. Call the new seeds](#4--call-the-new-seeds)

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow.

### 2. 👁️ Pick the setting

◀ Needs: [1.3 Draw seeds 9 to 16](#1--add-the-summed-push-then-draw-seeds-9-to-16)

- [x] **2.1 Read the seeds 9 to 16 sheet** with Claude's pick and its reason written in the review file, and say yes
  or pick another. A setting that breaks the picture on most of the 8 seeds is shown before anything else runs.
  ✓ confirmed by hand 2026-11-19: the user answered `next` to the presence pick.

▶ Next: [3. Draw seeds 3301 to 3332 once](#3--draw-seeds-3301-to-3332-once)

### 4. 👁️ Call the new seeds

◀ Needs: [3.1 Render the picked setting](#3--draw-seeds-3301-to-3332-once)

- [ ] **4.1 Check the calls.** Every full-size tile is called on the ladder one animal < fused < two bodies <
  near-clean < clean, written as a CSV beside the sheet and counted in the review file's Q1. Read the sheet beside
  plan 29's and confirm or veto the calls.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents)

Under `outputs/classifier_guidance_ladder/rung2/` on the scratch mount: the seeds 9 to 16 sheets
`result-sheet-catdog-every-step-bodies`, `-species10`, `-species30`, `-presence`, and the new-seed sheet
`result-sheet-catdog-every-step-general`, each with its calls CSV. The reading goes into the review file and the
route's "What we know so far" table.
