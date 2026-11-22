# 📒 Report: does classifier guidance draw a dog on SDXL?

**The recorded call is bar not met: a trained dog classifier drew a clean dog on 2 of 4 seeds, and the
bar was 3 of 4.** The test starts SDXL from an empty prompt and pushes each sampling step toward "dog"
with a classifier's gradient, the way Dhariwal and Nichol guide their model.
[Plan 18](../../plans/18-a-dog-classifier-trained-the-papers-way.md) holds the design and the calls.
This folder has no card, so the plan is the record.

## Table of contents

- [What would count](#what-would-count)
- [From an empty prompt](#from-an-empty-prompt)
- [From a nearer start, and on the product](#from-a-nearer-start-and-on-the-product)
- [Pictures kept off this page](#pictures-kept-off-this-page)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would count

Navigation: 📋 [TOC](#table-of-contents) | [Next](#from-an-empty-prompt) ➡️

**The bar is a clean dog on at least 3 of 4 seeds at some strength, after a check that the classifier
can see a dog in noisy pictures.** The check asks for balanced accuracy of at least 0.75 on at least 10
of the first 20 sampler steps `stated` ([the plan's outcomes](../../plans/18-a-dog-classifier-trained-the-papers-way.md#success-and-failure-outcomes)).

| Stage | Needed | Recorded | Mark |
|---|---|---|---|
| the check before steering | 10 of 20 steps at 0.75 or above | 7 of 20, failed. Balanced accuracy rises from 0.62 at the noisiest step to 0.82 at step 19, passing at steps 13 to 19 | `stated` |
| the dog sheet | a clean dog on 3 of 4 seeds | 2 of 4, bar not met. The sheet was drawn past the failed check at the user's choice | `stated` |

## From an empty prompt

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#from-a-nearer-start-and-on-the-product) ➡️

**Every seed shows a dog at some strength, and two of them are clean.** Strength is the dial on the
push, run at 10, 30, 100 and 300. The calls were made by Claude from full-resolution tiles at the
user's request and stay open to veto `stated` (the plan's instruction 6).

| Seed | What the push drew | Call |
|---|---|---|
| 9 | a painted dog among flowers at strength 30 | clean dog |
| 10 | a shaggy dog in a cap at strength 100 | clean dog |
| 11 | a café with no dog at 10 to 100, a faint dog in a wrecked street at 300 | dog forced in, picture damaged |
| 12 | a dog's head on a sofa at 100, its body melting into a white blanket | dog forced in, picture damaged |

At strength 0 the saved tile has the same pixels as the empty-prompt tile on 4 of 4 seeds `verified`
(compared pixel by pixel on 2026-11-22).

**An earlier classifier, trained on frozen SDXL features, misread these pictures.** Halfway through
sampling (step 20) it gave the four "a dog" renders a dog probability of 0.00, 0.85, 0.00 and 0.85, and
the four most-pushed renders, none holding a dog, 0.90, 0.92, 0.77 and 0.93 `verified` (read off its
belief sheet, which stays off this page).

## From a nearer start, and on the product

Navigation: ⬅️ [Previous](#from-an-empty-prompt) | 📋 [TOC](#table-of-contents) | [Next](#pictures-kept-off-this-page) ➡️

**The same push on the cat and dog product drew two separate animals in 0 of 16 pushed tiles.** It moved
the fused animal toward one cat, although the push asks for a dog `stated` (the plan's instruction 6).

<a href="../../artifacts/results/can-a-classifier-supply-the-missing-term/product-pushed-to-dog/sheet.png"><img src="../../artifacts/results/can-a-classifier-supply-the-missing-term/product-pushed-to-dog/sheet.png" width="200" alt="Four seeds down and six columns across: the joint prompt with a cat and a dog, the plain product with one fused animal, then four pushed strengths showing one fused or cat-like animal, with colors breaking at strength 300"></a>

Seeds 9 to 12 run down the sheet. Across are the joint prompt, the plain product, and the push at
strengths 10, 30, 100 and 300. Each column title gives the push's size: 0.5%, 1.2%, 1.6% and 1.9% of
the prediction's length, the median over the four seeds of each render's median step, computed at
[lines 124 to 126 of the product script](../../scripts/classifier_guidance_on_the_product.py) `verified`.
At strength 0 the saved tile has the same pixels as the plain product on 4 of 4 seeds `verified`.

| Where the push started | What was recorded, by strength up to 300 |
|---|---|
| "an animal" | a clean grey dog on seed 10 at 10, a clean drawn dog on seed 9 at 10, a clean wolf-like canine on seed 11 at 30, and on seed 12 a dog fused with a second white body at 100 |
| "a cat" | a cat on every seed at every strength up to 100, and a broken cat at 300 |
| the cat and dog product | seed 9 stays fused until the picture breaks at 300. Seeds 10, 11 and 12 are recorded as one cat at 30 to 300 |

All three rows are `stated` (the plan's instruction 6). From "a cat", the push measured 1.8%, 2.1%,
3.2% and 2.9% of the prediction at the four strengths `verified` (the sheet's column titles).

## Pictures kept off this page

Navigation: ⬅️ [Previous](#from-a-nearer-start-and-on-the-product) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Most sheets from this work stay off the page.** The [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md)
rules which subjects may be shown. All paths start at `2-classifier-guidance/artifacts/results/does-classifier-guidance-draw-a-dog/`.

| File | Why it is not shown |
|---|---|
| `dog-01-sheet/dog-sheet.png` | a seed 9 tile could not be cleared against the pair ledger |
| `dog-01-from-an-animal/dog-sheet.png` | seed 9 tiles could not be cleared against the pair ledger |
| `dog-01-from-a-cat/dog-sheet.png` | a faint line drawing behind the cat on seed 9 could not be cleared against the pair ledger |
| `two-ends.png`, `what-the-helper-believes.png`, `dog-sheet.png` | each holds the seed 11 empty-prompt painting, which could not be cleared against the pair ledger |
| `dog-piles-sheet_dog.png`, `dog-piles-sheet_no_dog.png`, `cat-piles/sheet_cat.png`, `cat-piles/sheet_no_cat.png` | sampled from piles that include COCO photographs fetched from the web |

## Where the sources disagree

Navigation: ⬅️ [Previous](#pictures-kept-off-this-page) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts conflict, and neither is settled by code, because both rest on a call by eye.**

| What | One source | The other source |
|---|---|---|
| the product's tiles at strength 30 on seeds 10, 11 and 12 | plan 18, called from the sheet: one cat | the three full-size tiles, opened 2026-11-22, each show one animal with features of both, such as long dog-like ear fur on a cat's body `inferred` (one reading) |
| what the first classifier drew | plan 18's opening says plan 17's classifier "drew no dog" | plan 17's instruction 6, where that sheet is called, has no call written (`2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md`) |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**How often a push draws a clean dog**

Four seeds at four strengths give a picture of the behaviour and no rate.

**Whether a pair-aware classifier would compose**

This classifier asks only "is there a dog". The product result says that one question does not separate
the animals. A classifier trained on pairs is tested on [the A3q page](a3q-cat-x-dog-quick-discriminator-test.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The product, the push and the strength are taught on
[the code walk page](which-line-computes-which-part-of-guided-sampling.md#concepts-this-uses).

**Balanced accuracy**

Take 10 dog pictures and 90 others. A classifier gets 8 of the dogs right and 45 of the others right.
Plain accuracy is 53 of 100 = 0.53, which mostly reflects the larger pile. "Balanced" means each pile
gets an equal say: (8/10 + 45/90) / 2 = (0.8 + 0.5) / 2 = 0.65. Here the bar is 0.75 at each sampler
step, and the dog classifier reads 0.62 at the noisiest step and 0.82 at step 19 `stated`. The per-pile
counts are not on this machine, so the sum cannot be rerun on them. The term index has no note for
balanced accuracy.

**A push as a share of the prediction**

If the model's prediction has length 200 and the push has length 3, the push is 3 / 200 = 1.5% of the
prediction. The product sheet's column at strength 300 reads 1.9%.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the bar, the check and every call | [plan 18](../../plans/18-a-dog-classifier-trained-the-papers-way.md), run `dog-01`, W&B `z2k1bdmi` | `stated` |
| the strength-0 pixel matches | the saved tiles in the product and `dog-01-sheet` folders, compared 2026-11-22 | `verified` |
| how the pushed sheets were made | [the empty-prompt script](../../scripts/classifier_guidance_from_the_paper.py) and [the product script](../../scripts/classifier_guidance_on_the_product.py) | `stated` |
| the first classifier's design | `2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md` | `stated` |
| 84 pictures held out of the drain | the other tiles, sheets and probes in this result's folder. They repeat a picture described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Write this grouping's card. Its folder has none.
- [ ] Recall the product's strength-30 tiles at full size, and record whether seeds 10, 11 and 12 are one cat or one fused animal. The owner decides.
- [ ] Record a call for plan 17's dog sheet, whose instruction 6 has none.
- [ ] Task 4.1 of plan 18, the same classifier for cat and then both, has not run.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
