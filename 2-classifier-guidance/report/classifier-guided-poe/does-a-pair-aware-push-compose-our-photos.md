# 📒 Report: does a pair-aware push compose our cat and dog photos?

**No verdict has been recorded, because the runs that would judge this stopped in a storage outage.**
The question is whether a classifier trained on our own cat and dog images can push the product of a
cat model and a dog model into drawing two separate animals, on more seeds than a push built from
separate "cat present" and "dog present" answers. [Plan 20](../../plans/20-our-photos-both-ways.md)
holds the design and [its review file](../../plans/review/20-our-photos-both-ways.md) holds the runs.

## Table of contents

- [What would count](#what-would-count)
- [How far the work got](#how-far-the-work-got)
- [The one picture shown](#the-one-picture-shown)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would count

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-far-the-work-got) ➡️

**The bar asks for at least 4 more seeds of 16 with two separate animals than the separate-label
push gives.** It must hold in at least one of two experiments, on seeds 1 to 16, with pictures no
more damaged than the plain product's. The reader judges full-resolution tiles, and the pair-aware
push is read at weights 1 and 2 `stated` ([the plan's Goal](../../plans/20-our-photos-both-ways.md#goal)).

| Experiment | Model | Separate-label push |
|---|---|---|
| released model | the authors' 256 px unconditional model, frozen | the released classifier's cat and dog classes |
| model we train | the authors' 64 px model trained on our images, once without the both-animal images and once on all 550 | a classifier trained on our images |

On the 64 px sheets the bar is read against the stronger separate-label scale, with each scale's
comparison reported beside it. That rule was added during the walk, after launch `stated`.

> [!IMPORTANT]
> The count can be read two ways, and the review file keeps both. A tile called "two bodies" counts
> toward two separate animals under the ruling of 2026-11-17, made after launch and before any tile
> was called. The strict count takes "two clear animals" only and is reported beside it.

## How far the work got

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#the-one-picture-shown) ➡️

**The image set was built and checked, and no sheet that the bar reads exists on this machine.**
States are as the walk recorded them on 2026-11-18.

| Piece | State | Mark |
|---|---|---|
| the image set | 550 images: 200 cat, 200 dog, 150 both. 312 are SD3.5 renders and 238 are COCO photos, from [the set's counts](../../artifacts/results/does-a-pair-aware-push-compose-our-photos/photo-set-counts.json) | `verified` |
| 64 px model without both-animal images | failed at step 32,600 of 50,000, W&B `wubhxhse`, last heartbeat 2026-11-17 13:15 UTC | `stated` |
| 64 px model on all images | failed at step 27,600 of 50,000, W&B `pnwlwh0f`, same heartbeat | `stated` |
| 256 px classifier trained on our images | held-out accuracy at step 0 of 0.93 for cat and 0.95 for dog, but its push did not steer the released model to cats or dogs at scales 10, 30 or 100, so the product uses the released classifier's classes | `stated` |
| 256 px pair-aware classifier and sheet | the walk says the sheet most likely never ran; its outcome was unreadable while `/datasets` was down | `stated` |
| the tile calls and the verdict | none recorded | `verified` |

## The one picture shown

Navigation: ⬅️ [Previous](#how-far-the-work-got) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

<a href="../../artifacts/results/does-a-pair-aware-push-compose-our-photos/imagenet-product-check-256.png"><img src="../../artifacts/results/does-a-pair-aware-push-compose-our-photos/imagenet-product-check-256.png" width="200" alt="Four draws side by side, each one furry animal on a dark background"></a>

**The released model pushed by the released classifier's cat and dog classes draws an animal on 4 of
4 seeds, and none shows two separate animals** `verified` (each tile opened at double size). These
are seeds 1 to 4 at scale 10. The walk called draws 2 and 4 cat-dog hybrids and draws 1 and 3 one
dog-like animal, by eye at grid scale `stated`.

The other pictures from this work stay off the page:

| File | Why it is not shown |
|---|---|
| `2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/coco-both-crops.png` | holds COCO photographs fetched from the web |
| `2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/example-one-per-label.png` | holds COCO photographs fetched from the web |
| `2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/preview-product-256-draws.png` | a tile could not be cleared against the pair ledger |
| `2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/preview-product-256-imagenet-draws.png` | a tile could not be cleared against the pair ledger |
| `2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/push-scale-check-256.png` | a tile could not be cleared against the pair ledger |

## What this cannot tell you

Navigation: ⬅️ [Previous](#the-one-picture-shown) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Whether the pair-aware push works on photos.** The only pair-aware renders on record are a 6-tile,
25-step test that the code runs `stated`. The four draws above come from the product alone, so they
show the starting point the push would act on.

**Anything from four seeds about how often the product fuses.** The hybrid calls were made by eye at
grid scale, where two hybrids have read as a clear pair before `stated` (the review file).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The product, the push and the cap are taught on
[the code walk page](which-line-computes-which-part-of-guided-sampling.md#concepts-this-uses).

**A pair-aware question against two separate questions**

Take four invented pictures and ask two ways. "Separate" asks "is a cat present?" and "is a dog
present?" on their own. "Pair-aware" asks one question: "are a cat and a dog both here, apart?"

| Picture | Cat present? | Dog present? | Both, apart? |
|---|---|---|---|
| a cat alone | yes | no | no |
| a dog alone | no | yes | no |
| one animal with a cat's ears and a dog's nose | yes | yes | no |
| a cat beside a dog | yes | yes | yes |

The third row is the failure this work targets. Two separate answers both say yes to one fused
animal, so a push built from them has no reason to pull it apart. In this project the pair-aware
classifier's positives are the 150 both-animal images, and its negatives are the product's own
finished draws and the single-animal images `stated` (the plan's task 1.3).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the image counts | [the set's counts](../../artifacts/results/does-a-pair-aware-push-compose-our-photos/photo-set-counts.json), read 2026-11-22 | `verified` |
| the bar and the two experiments | [plan 20](../../plans/20-our-photos-both-ways.md) and [its review file](../../plans/review/20-our-photos-both-ways.md) | `stated` |
| where the runs stopped | [the walk on our photos](../../artifacts/drips/our-photos-both-ways/EXECUTE_WALK.md), last updated for the outage of 2026-11-18 | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Check which checkpoints and draws survived on `/datasets`, then resume both 64 px trainings from their last checkpoints (steps 30,000 and 25,000).
- [ ] Rerun the 256 px pair-aware classifier and its sheet on the 512 draws, if they survived.
- [ ] Choose whether both-animal images `both_015` and `both_027` (dog only, by eye) and the collages `both_000` and `both_017` stay in the set. The owner decides.
- [ ] Write this grouping's card. Its folder has none.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
