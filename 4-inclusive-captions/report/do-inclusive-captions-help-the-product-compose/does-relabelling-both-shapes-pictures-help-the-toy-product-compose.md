# 📒 Report: does relabelling both-shapes pictures help the toy product compose? Recorded verdict: "Yes, once about 9%" (results card)

**The claim**

The [results card](../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/README.md)
records: "Yes, once about 9% of each single caption's pictures hold both shapes." The toy is a small
diffusion model trained on pictures of a red circle, a blue square, or both. Relabelling files a
both-shapes picture under "a red circle" and "a blue square" as well as under the joint caption. With
none relabelled, the [product of the two captions](../../../context/world/poe-composition.md#what-poe-composition-is)
draws no [separate pair](../../../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md). With the training set held at 6,000 pictures, one training of three reaches 30
of 300 at a 9.1% share, and all three do from 23%.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. How many pictures must hold both shapes](#1-how-many-pictures-must-hold-both-shapes)
- [2. What the corrector adds](#2-what-the-corrector-adds)
- [3. Whether stronger guidance replaces relabelling](#3-whether-stronger-guidance-replaces-relabelling)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-how-many-pictures-must-hold-both-shapes) ➡️

**No single rule covers the card's "Yes".** Four checks were written before their runs, and each is
read from its own file `verified`.

| Check | Rule written before the run | Recorded result |
|---|---|---|
| smallest working count, adding pictures | a training "draws the pair" at 30 or more of 300 separate pairs while each single caption keeps 240 of 300 in its own colour; the threshold is the smallest count where all three trainings do | `threshold: null`. At 2,000 relabelled the counts are 252, 74 and 238, and the third training's single captions fall under 240 |
| size check, replacing pictures | "share drives it" if all three 50% models reach 30 of 300 | "share drives it": 119, 281, 182 |
| corrector, adding pictures | support if the median of four trainings rises by 30 or more with 5 corrector steps, with each single caption at 240 or more | "support": median 260.5 plain, 300 corrected |
| corrector, replacing pictures | with 100 or more replaced, 20 corrector steps give at least 290 of 300 in all three trainings; with none, at most 3 | no verdict field in the files. Holds at 0%, 15% and 35%; misses at 5%, 10%, 25% and 50% ([section 2](#2-what-the-corrector-adds)) |

## 1. How many pictures must hold both shapes

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-what-the-corrector-adds) ➡️

| What training saw | What the product draws |
|---|---|
| <a href="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-training-set-with-and-without-relabelling.png"><img src="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-training-set-with-and-without-relabelling.png" width="200" alt="Three captions by three relabelling levels, eight training pictures each, with both-shapes pictures outlined under the single captions at 0, 9 and 50 percent"></a><br>Rows are the caption, columns are 0, 200 and 2,000 relabelled pictures. The share of a single caption's pool holding both shapes goes 0%, 9%, 50% `verified`. | <a href="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/what-poe-draws-at-each-relabelled-count.png"><img src="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/what-poe-draws-at-each-relabelled-count.png" width="200" alt="Nine rows of relabelled counts by three trainings, four product samples per cell: purple blobs up to 50 relabelled, separate red circles and blue squares appearing from 200"></a><br>Rows are 0 to 2,000 relabelled, columns are three trainings, and each cell is four samples with its count of 300. No cell reaches 30 below 200 relabelled `verified`. |

**With the set held at 6,000 pictures, the pair first passes 30 of 300 at a 9.1% share, in one
training of three.** Each single caption keeps 2,000 pictures and some of them are swapped for
both-shapes pictures. Counts are separate red-and-blue pairs out of 300 product samples, one number per
training, from the `apart` field of the [replacing sweep](../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/toy-caption-replace-sweep.json) `verified`.

| Pictures swapped, of 2,000 | Share holding both shapes | Separate pairs of 300 |
|---|---|---|
| 1 | 0.05% | 0, 0, 0 |
| 5 | 0.25% | 0, 0, 0 |
| 20 | 1% | 0, 0, 0 |
| 49 | 2.45% | 0, 0, 2 |
| 182 | 9.1% | 13, 5, 72 |
| 462 | 23.1% | 66, 64, 146 |
| 750 | 37.5% | 239, 129, 75 |
| 1,000 | 50% | 209, 150, 84 |

## 2. What the corrector adds

Navigation: ⬅️ [Previous](#1-how-many-pictures-must-hold-both-shapes) | 📋 [TOC](#table-of-contents) | [Next](#3-whether-stronger-guidance-replaces-relabelling) ➡️

| Eight models, three ways of sampling | Five models, five ways of asking |
|---|---|
| <a href="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-corrector-with-and-without-relabelling.png"><img src="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-corrector-with-and-without-relabelling.png" width="200" alt="Eight rows of models by plain sampling, 5 and 20 corrector steps: the 0 and 50 relabelled rows stay at 0 of 300, the 200 and 2,000 rows reach 300 of 300 with 20 steps"></a><br>Adding design. With 20 corrector steps the models at 200 and 2,000 relabelled reach 300 of 300, and the 0 and 50 rows stay at 0 `verified`. | <a href="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/every-caption-with-and-without-the-corrector.png"><img src="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/every-caption-with-and-without-the-corrector.png" width="200" alt="Five rows of relabelling by five columns: each single caption, the product, the product with the corrector and the joint caption; the corrected product reads 0 then 300 of 300"></a><br>Adding design, one model per row. The plain product reads 0, 110, 103, 51, 205 of 300 and the corrected product 0, 300, 300, 300, 300 `verified`. |

**With the set held at 6,000 pictures, 20 corrector steps reach the written 290 of 300 at two of the
six relabelled shares.** Counts are separate pairs of 300, one per training, from the seven
`toy-replace-corrector-k*.json` files in the [results folder](../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/) `verified`. These files have no entry on the card.

| Pictures swapped, of 2,000 | Share | Plain product | With 20 corrector steps | Written rule |
|---|---|---|---|---|
| 0 | 0% | 0, 0, 0 | 0, 0, 0 | holds (at most 3) |
| 100 | 5% | 1, 0, 35 | 31, 0, 185 | misses |
| 200 | 10% | 175, 0, 37 | 300, 253, 195 | misses |
| 300 | 15% | 82, 73, 132 | 300, 300, 300 | holds |
| 500 | 25% | 103, 42, 95 | 300, 36, 198 | misses |
| 700 | 35% | 119, 17, 243 | 300, 299, 300 | holds |
| 1,000 | 50% | 203, 136, 84 | 296, 262, 132 | misses |

## 3. Whether stronger guidance replaces relabelling

Navigation: ⬅️ [Previous](#2-what-the-corrector-adds) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

<a href="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-caption-model-across-guidance.png"><img src="../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-caption-model-across-guidance.png" width="200" alt="Five rows for how often a pair picture is also captioned singly, by guidance 1, 2, 3, 5 and 7, four product samples per cell with a count of 300"></a>

**Stronger guidance does not rescue the model with no relabelled pictures.** Rows are how often a
both-shapes picture is also captioned with each single name (0 to 1), columns are guidance 1, 2, 3, 5
and 7, and each cell is four samples with its count of 300. The top row reads 0, 0, 0, 0 and 1 of 300
`verified` (read on the picture; counts in `toy-caption-dial-ldm-guidance.json`).

## Where the sources disagree

Navigation: ⬅️ [Previous](#3-whether-stronger-guidance-replaces-relabelling) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Five recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| what the corrector does at 9% or more | the card: it "takes every model tested at 9% or more to 300 of 300" | the replacing corrector files: of 15 models at 10% or more, 9 reach 290 or more with 20 steps, and the lowest reaches 36 |
| where composing starts | the card's headline: "once about 9%" | the threshold rule returns `null`, and at 9.1% one training of three passes 30 of 300 |
| 200 pictures swapped, same three training seed numbers | `toy-caption-replace.json`: 0, 0, 3 of 300 | `toy-replace-corrector-k200.json`, plain: 175, 0, 37 of 300. No file records why they differ |
| how many shares the replacing design covers | the card's limits section: "two shares only" | the card's first section and the sweep file: eight shares |
| the order of the three counts at 200 added | the card's figure entry: 153, 1, 0 | the threshold file and the picture: 1, 153, 0 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Three trainings per setting leave the order between neighbouring shares unreliable.** At 37.5% the
three trainings draw 239, 129 and 75 of 300, a spread of 164 (239 minus 75).

**The adding design is a mixed comparison.** Adding relabelled pictures raises the share and also grows
the training set from 6,000 to 10,000 pictures at a fixed 8,000 training steps.

**The toy says a share matters and gives no share for real photos.** A toy caption's pool is three
pictures plus pixel noise, so this cannot say what share a real "a cat" pool needs. The scorer also
misses a reddish circle beside a bluish square `stated` (the card).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**The product of two captions**

Two coffee tasters each score three cups. Taster A gives 0.5, 0.4, 0.1 and taster B gives 0.1, 0.4,
0.5. Multiply cup by cup and you get 0.05, 0.16, 0.05, so the product favours the cup both accept. Each
taster is an "expert" and multiplying them is the "product". A diffusion model does this by adding the
two captions' noise predictions, `blank + g × (a + b − 2 × blank)`. With blank 1, a 3, b 2 and guidance
g = 2, that is 1 + 2 × (3 + 2 − 2) = 7. Read more in the [product of experts note](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this)
and [the derivation from Bayes](../../../../../goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md#1--drip-it-on-the-toy).

**The share of a caption's pictures holding both**

Add 2 two-flavour sweets to a jar of 8 plain ones and the share is 2 / 10 = 20%. Swap 2 of the 8
instead and it is 2 / 8 = 25%. Here, adding 200 to 2,000 gives 200 / 2,200 = 9.1%, and swapping 200 of
2,000 gives 200 / 2,000 = 10%.

**A Langevin corrector**

Picture a marble dropped near a bowl: each small nudge moves it downhill, with a little shake so it
does not stick. A corrector makes such nudges at every noise level, here 5 or 20 of them, each of size
0.05 of that level's noise variance. The [Langevin dynamics note](../../../../../Obsidian/Mathematics/Statistics/Markov%20Chain%20Monte%20Carlo/Langevin%20Dynamics.md#where-to-learn-this)
teaches it. The learning tree has no plan that names this corrector.

**The median of four trainings**

Sort the four counts and average the middle two: for 1, 2, 8, 9 that is (2 + 8) / 2 = 5. The corrector
rule's plain counts sort to 67, 252, 269, 290, so the median is (252 + 269) / 2 = 260.5.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the recorded verdict, how the figures were made | [results card](../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/README.md), read 2026-11-21 | `stated` |
| counts in every table | the `toy-caption-*.json` and `toy-replace-corrector-k*.json` files in the [results folder](../../artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/), read 2026-11-21 | `verified` |
| the replacing corrector's written rule and code | [the replacing corrector script](../../scripts/toy_replace_corrector.py) | `stated` |
| the plan this serves | [do inclusive captions help PoE compose](../../plans/11-do-inclusive-captions-help-poe-compose/MASTER_PLAN.md) and [the toy figure route](../../plans/route-zoom-ladder-toy-relabelling-experiment.md) | `stated` |
| what the product is | [PoE composition](../../../context/world/poe-composition.md#what-poe-composition-is) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide whether the card's "Yes, once about 9%" stands, given the `null` threshold and one training of three at 9.1%.
- [ ] Decide how the card's corrector sentence reads now that the replacing design misses its written rule at four of six relabelled shares.
- [ ] Add the seven replacing corrector files to the card.
- [ ] Find why the two runs with 200 pictures swapped and the same seed numbers draw different counts.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
