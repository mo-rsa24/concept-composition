# 🧪 Each animal toward its own species: does a per-animal push after step 16 clean the borrowed faces

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/28-each-animal-toward-its-own-species.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, after [plan 27](27-cat-and-dog-and-not-hybrid.md). Its pass marks are in
[its review file](review/28-each-animal-toward-its-own-species.md); results go into
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

One trajectory per seed, the readout checker pushing PoE at cap 0.3 on steps 0 to 15 and plain PoE after, gives two
separate animals on most held-out seeds in mostly natural colour, and a clean or near-clean cat and dog on 4 of 8. The
rest carry borrowed face features: after step 15 PoE paints each body with both prompts. Every push through the
decoded picture so far changed faces, fur and eyes but not layout; aimed at the whole picture it could not say which
body should be which animal.

**What it does**

Steps 0 to 15 are the early push unchanged. At step 16 the decoded guess is boxed by the animal detector and zero-shot
CLIP names the cat box and the dog box; from step 16 to 49 each box is pushed toward its own species through the
decoded guess, capped at 0.1. One trajectory per picture.

## Goal

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

A sheet on held-out seeds 9 to 16 answering Q1 (clean or near-clean on at least 6 of 8) and Q2 (how often two animals
are found at step 16), per [the review file](review/28-each-animal-toward-its-own-species.md).

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

- [x] **1.1 Code.** `clip_push.BoxSpeciesHead` (detector at the late start, zero-shot CLIP naming, per-box push),
  `TwoStageHead` resets it per picture, `render_results.py --box-species`.
- [ ] **1.2 Draw.** `rung2/boxes.sh` on a free card; the log prints, per picture, how many animals the detector found
  at step 16.
- [ ] **1.3 Compare.** Full-size images per seed beside the early push alone, sent with draft calls.

▶ Next: [2. Call the sheet](#2--call-the-sheet)

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow.

### 2. 👁️ Call the sheet

◀ Needs: [1.3 Compare](#1--build-and-draw)

- [ ] **2.1 Call every full-size tile** clean, near-clean, two bodies with borrowed features, or one animal, and write
  the counts into the review file's Q1.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents)

Under `outputs/classifier_guidance_ladder/rung2/result-sheet-catdog-boxes/` on the scratch mount: `tiles/`,
`result.json`, `late-start/`. Verdicts go into the review file and the checker loop file.
