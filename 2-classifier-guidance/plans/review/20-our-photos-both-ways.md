# 🧪 Review: does a pair-aware push draw two separate animals on our cat and dog images more often than a separate-label push?

**Nothing has been judged yet.** This file judges [plan 20](../20-our-photos-both-ways.md), run as
step 8 of [the route from the toy to the animals](../route-classifier-guided-poe-from-toy-to-animals.md),
and settles the board's rows A2r and A2c in
[the classifier-guided PoE board](../../report/classifier-guided-poe/board.md). The image set is
built, both 64 px trainings and the 256 px chain have launched, and no sheet has been called.

**This file was filed on 2026-11-17, after the trainings launched, so nothing in it is
pre-registered here.** Both 64 px trainings launched at 07:35 UTC on 2026-11-17, and the 256 px
chain first launched later that morning (commit `3d6efb86`). The bar below is copied word for word
from the design's Goal, where it has stood since the plan's first commit (`b390d027`, 07:25 UTC on
2026-11-17), ten minutes before the trainings launched. That commit is the bar's date; this file's
date is later.

- [x] 📝 bar written before the run: in the design's Goal at commit `b390d027`, copied into
      [The question written before the run](#the-question-written-before-the-run)

## Recommended prompt (when the run lands)

```
/analyze-run @2-classifier-guidance/plans/review/20-our-photos-both-ways.md — <the per-tile calls CSV from task 1.6>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [plan 20](../20-our-photos-both-ways.md) | the design: the image set, the two arms, the classifiers, the sheets |
| [the route, step 8](../route-classifier-guided-poe-from-toy-to-animals.md#8--our-photos-both-ways-the-released-model-and-a-model-we-train) | why this runs, beside the scale-up ladder, gating nothing |
| **this file** | **the runs, the bar, and later the verdict** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

**Our images**

550 images from the inclusive-captions pool: 200 cat-only, 200 dog-only and 150 holding both a cat
and a dog. 312 are SD3.5 renders and 238 are COCO photos. Each is cropped square and kept at 256 px
and at 64 px.

**The product**

The sampler adds a cat push and a dog push to the model's own prediction, so the picture is pulled
toward "cat" and toward "dog" at once. On cat and dog it tends to draw one fused animal.

**The separate-label push**

A classifier with one output per species, "a cat is present" and "a dog is present", each answered
on its own. Its gradient is the push the product is built from. In the released-model arm it is the
released ImageNet classifier's cat classes (281 to 285) and dog classes (151 to 268).

**The pair-aware push**

A classifier trained to tell our both-animal images from the product's own finished draws and the
single-animal images. Its gradient is added on top of the product at weight 1 or 2.

**The two arms**

The released-model arm keeps the authors' 256 px unconditional model frozen. The trained-model arm
trains the authors' 64 px model on our images twice: once without the both-animal images, so the
product never saw a pair, and once on all 550.

**The four calls on every tile**

Each tile is called by eye at full resolution as one of: **single animal**, **hybrid** (one animal
carrying both species), **two bodies**, or **two clear animals**.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A missed bar in both arms closes the plan and opens the diagnosis the design
names (the minimal-pair gap, push weights, the classifier's accuracy per step). Within one arm, the
plain product and the pushed columns share a model and a starting noise per seed, so only the push
differs. Between the two arms the model, its size and its training data all differ, so each arm is
read on its own and the gap between them is never read as cause and effect. Between the two 64 px
models only the training data differs: whether the both-animal images were in it.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

States are as recorded in the walk map
(`2-classifier-guidance/artifacts/drips/our-photos-both-ways/EXECUTE_WALK.md`) and the plan 20
commits. No run below has been harvested since; "running" means running when last recorded.

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| 64 px timing run, 300 steps | setup | 2026-11-17 on `mscluster111`, before 07:35 UTC; recorded at commit `a1b14ba0` | 278 s | the per-step time that sized the trainings | done: 295.1 M parameters, loss 1.02 to 0.062 |
| 64 px model without both-animal images, 400 images | builds the trained-model arm | 2026-11-17 07:35 UTC on `mscluster111` GPU 0, W&B `wubhxhse`, PID 283386, commit `a1b14ba0` | 50,000 steps at batch 64, about 12.9 h (from the timing run), at 30.6 GB | log `classifier_guidance_photos/logs/train_64_no_both.log` under `/datasets/mmolefe/poe_repair_min/outputs/` | running when last recorded |
| 64 px model on all 550 images | builds the trained-model arm | 2026-11-17 07:35 UTC on `mscluster112` GPU 0, W&B `pnwlwh0f`, PID 522361, commit `a1b14ba0` | 50,000 steps at batch 64, about 12.9 h, at 30.6 GB | log `classifier_guidance_photos/logs/train_64_all.log` | running when last recorded |
| 256 px separate-label classifier on our images (`separate_256.pt`) | builds a push | 2026-11-17 on `mscluster108`, commit `3d6efb86` | 1.36 s per iteration at batch 32, 37 GB | `classifier_guidance_photos/classifiers/` | done: held-out accuracy 0.93 (cat) and 0.95 (dog) at step 0; it does not steer the released model to cats or dogs at scales 10, 30 or 100 (`push-scale-check-256.png`), so the product does not use it |
| 64 px separate-label classifier (`separate_64.pt`) | builds a push | 2026-11-17 on `mscluster107` GPU 1, commit `e027ef07` | not recorded | `classifier_guidance_photos/classifiers/` | done: held-out accuracy 0.73 (cat) and 0.84 (dog) at step 0 (commit `a549ec96`) |
| 256 px chain: 512 product draws with the released classifier's classes, the pair-aware classifier, the released-model sheet | Tests the claim | 2026-11-17 10:19 UTC on `mscluster108` GPUs 0 and 1, PID 228025, commit `8fcd7f8a`; replaces the first chain, stopped at 10:17 UTC | about 27 s per draw per card | log `classifier_guidance_photos/logs/piece3_256.log`; the sheet `sheet-released-256.png` | running when last recorded, sheet expected about 14:05 UTC; the sheet is not in the laptop's results folder |
| 64 px chain, model without both-animal images: draws, pair-aware classifier, sheet | Tests the claim | waiting on `mscluster111` since 08:41 UTC, PID 285185 | not recorded | log `classifier_guidance_photos/logs/piece5_64_no_both.log`; the sheet `sheet-64-no-both.png` | waiting for its training to land |
| 64 px chain, model on all images: the same | Tests the claim | waiting on `mscluster112` since 08:41 UTC, PID 525583 | not recorded | log `classifier_guidance_photos/logs/piece5_64_all.log`; the sheet `sheet-64-all.png` | waiting for its training to land |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does the pair-aware push draw two separate animals on more seeds than the separate-label
      push?**

      The bar, word for word from the design's Goal:

      **The bar, written before any classifier is trained.** In at least one arm, on seeds 1 to
      16, the pair-aware push gives two separate animals on at least 4 more of 16 seeds than the
      separate-label push, with pictures no more damaged than the plain product's. Judged by the
      reader from full-resolution tiles, each called single animal, hybrid, two bodies, or two
      clear animals. The pair-aware push is read at weights 1 and 2.

      Added during the walk, after launch, and recorded in the walk map's decisions: on the 64 px
      sheets the bar is read against the stronger separate-label scale, with each scale's
      comparison reported beside it.

      Ruled on 2026-11-17, after launch and before any tile was called: a tile called **two
      bodies** counts toward "two separate animals", because the project counts co-occurrence as
      two separate creatures in the frame, not a clean nameable cat beside a dog. The strict count,
      **two clear animals** only, is reported beside it in every row of the verdict, so the bar can
      be read both ways.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Nothing. The design's Goal holds the one bar above and no further question. Its Success and
failure outcomes table says what each result leads to and is read there.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Within each sheet every column shares the seed's starting
      noise. In the released-model arm the separate-label push was changed after a check render,
      from our classifier to the released classifier's cat and dog classes, because ours did not
      steer the released model to cats or dogs (the design's task 1.3). The pair-aware classifier
      in that arm is trained against draws from the changed product.

- [ ] ⚠️ **Was the measuring tool sound?** The calls are by eye at full resolution, because two
      hybrids have read as a clear pair at sheet scale before. Of the 38 COCO both-animal crops,
      22 keep under 90% of the animals' box area, and by eye at sheet scale 36 of 38 show both
      animals, so a few both-animal training images may hold one animal only.

- [ ] ⚠️ **Did the run respect the environment?** Outputs under `/datasets`, the Blackwell
      trainings under `co3_bw`, and every column of one sheet on one card.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a pair-aware push composes our images more often than a separate-label push | which arm, the weight, the counts out of 16 for both columns, the damaged count, and that this file was filed after launch while the bar dates from the design's first commit |
| it does not | that the released-model arm's separate-label push is ImageNet's classes, not a classifier trained on our images |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| Whether the both-animal images `both_015` and `both_027` (dog only, by eye) and the collages `both_000` and `both_017` stay in the set | the reader's call, offered in the walk map | the pair-aware classifiers' positives |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Harvest the 256 px chain and call the released-model sheet's tiles, per
[instruction 2.4 of plan 20](../20-our-photos-both-ways.md#2-test-each-piece).
