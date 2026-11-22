# 📒 Report: do inclusive labels help the product compose on MNIST? ⚪ null

**The claim**

A small diffusion model learns MNIST digits on 32 × 64 canvases. A setting called the dial says how
often a two-digit canvas is also trained under each single label (0, 0.1, 0.3, 0.6 or 1). [Plan 02](../../plans/11-do-inclusive-captions-help-poe-compose/plans/02-the-mnist-stage.md#8--tick-one-verdict-line)
records the verdict as null. The dials that count are 0, 0.1 and 0.3, and their corrected 7 × 8 products
draw a mean of 0, 0.33 and 8.0 separate right pairs of 64 by eye, all under the null line of 9.75. Dial 0.6
(mean 22.0) stays out because one training drew a lone 7 on 205 of 256 against a floor of 207, and the floor
stays as written because it was set before any model drew. Dial 1 (mean 24.0) misses its floors too. The
[counts file](../../artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/counts.json)
records `NULL` from the judge.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. What the product draws at each dial](#1-what-the-product-draws-at-each-dial)
- [2. The answer under each reading of the rule](#2-the-answer-under-each-reading-of-the-rule)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-what-the-product-draws-at-each-dial) ➡️

**The bar was written into [the bar file](../../../poe_repair/experiments/inclusive_captions/mnist/bar.py)
before any model drew.** All three trainings at a setting are read together.

| Outcome | What it takes |
|---|---|
| a dial counts | "7" alone and "8" alone each draw their lone digit at or above that dial's floor, in all three trainings |
| supported | at a counting dial, the corrected 7 × 8 product draws two separate right digits on at least 128 of 256 (32 of 64 by eye) in all three trainings, while dial 0 stays at 12 or fewer (3 of 64) |
| null | no counting dial's three-training mean reaches 39 of 256 (9.75 of 64 by eye) |
| inconclusive | anything else |

> [!WARNING]
> The choice of reader changed after the data was seen. No judge reading passed on the corrected row,
> so the owner chose blind eye calls on the first 64 seeds, with the same shares as the bar. The
> [walk's decisions table](../../artifacts/drips/inclusive-captions-mnist-stage/EXECUTE_WALK.md#decisions)
> records this `stated`.

## 1. What the product draws at each dial

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-the-answer-under-each-reading-of-the-rule) ➡️

<a href="../../artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/the-leak-read.png"><img src="../../artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/the-leak-read.png" width="200" alt="Two panels against the dial. Top: percent of 256 seeds drawing two separate right digits for the 7 x 8 and 4 x 7 products, all under the 128 of 256 support line. Bottom: percent of draws where 7, 8 and 4 alone draw their digit alone, beside dashed training lines."></a>

**The corrected 7 × 8 product stays under the 128 of 256 support line at every dial.** Both panels have
the dial on x and one dot per training. The top y-axis is the percent of 256 seeds with two separate
right digits, and the grey dotted lines are the toy's counts. The bottom y-axis is the percent of draws
where a label draws its digit alone, beside its dashed training line `verified`.

Counts are of 256 seeds by the pinned judge, from the counts file, and all 120 count cells match the
30,720 rows of [the per-tile calls](../../artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/scores.csv)
`verified`. Eye counts are of the first 64 seeds, from the `final_call` column of the
[corrected-row eye calls](../../artifacts/drips/inclusive-captions-mnist-stage/eye-calls/) `verified`. A
mean is the three trainings' sum divided by 3.

| Dial | "7" alone (floor) | "8" alone (floor) | Counts? | Corrected 7 × 8, judge | Mean of 256 | Corrected 7 × 8, eye | Mean of 64 |
|---|---|---|---|---|---|---|---|
| 0 | 253, 255, 253 (244) | 256, 256, 256 (244) | yes | 0, 0, 0 | 0 | no eye calls | |
| 0.1 | 247, 253, 252 (231) | 252, 254, 251 (230) | yes | 1, 0, 0 | 0.33 | no eye calls | |
| 0.3 | 230, 234, 235 (217) | 230, 235, 241 (214) | yes | 19, 16, 11 | 15.33 | 14, 9, 1 | 8.0 |
| 0.6 | 205, 216, 215 (207) | 203, 203, 207 (203) | no | 70, 48, 39 | 52.33 | 37, 18, 11 | 22.0 |
| 1 | 185, 210, 198 (199) | 183, 186, 180 (196) | no | 89, 41, 37 | 55.67 | 35, 27, 10 | 24.0 |

The corrected 4 × 7 product, a pair never shown together in training, draws 0 of 256 at every dial in
every training `verified`.

## 2. The answer under each reading of the rule

Navigation: ⬅️ [Previous](#1-what-the-product-draws-at-each-dial) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The floor stays as written, so dial 0.6 is excluded, and both readings under that rule give null.** The two rows that admit dial 0.6 show what the other choice would have given.

| Who reads the product | Dial 0.6 | Highest counting mean | Line | That reading would give |
|---|---|---|---|---|
| the judge, 256 seeds | excluded, as the floor is written | 15.33 (dial 0.3) | 39 | null, the value in the counts file |
| the eye, 64 seeds | excluded, as the floor is written | 8.0 (dial 0.3) | 9.75 | null |
| the judge, 256 seeds | admitted | 52.33 (dial 0.6) | 39 | inconclusive: above the null line, and only 39 to 70 of the 128 needed |
| the eye, 64 seeds | admitted | 22.0 (dial 0.6) | 9.75 | inconclusive: above the null line, and 18 and 11 are under the 32 needed |

## Where the sources disagree

Navigation: ⬅️ [Previous](#2-the-answer-under-each-reading-of-the-rule) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Five recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| what reads the verdict | `score.py` computes the recorded NULL from the judge on 256 seeds | `bar.py` says "The verdict reads the corrected 7 x 8 product by eye" on 64 seeds |
| whether the judge can be trusted at dial 0.6 | plan 02 requires 58 of 64 agreement with the eye on each dial's plain-product sheet | the walk records 50 of 64 at dial 0.6, and 42 of 64 on the corrected row |
| training length | the [route](../../plans/route-inclusive-captions-pilot.md#4--ask-the-same-question-on-mnist-digits) says 30,000 steps, marked an estimate | plan 02 records 6,000 steps for all 15 models, the owner's choice |
| share of "7" pictures that also hold an 8, at the four non-zero dials | the route: 4.3%, 8.8%, 12.4%, 15.0% | plan 03: 4.2%, 8.9%, 12.4%, 14.7% |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Raising the dial changes two things at once, so this comparison is mixed.** It raises the share of
single-label pictures holding two digits and also lowers how often the joint label "7 and 8" is trained
([plan 03](../../plans/11-do-inclusive-captions-help-poe-compose/plans/03-the-mnist-share-run.md#quick-context)).
The [share run](does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md) holds the
label mix fixed.

**The eye calls have one reader.** They are blind calls made by the assistant and accepted by the owner,
on 64 of each training's 256 seeds, and dials 0 and 0.1 have none for the corrected row.

**One pair, one small network.** Only 7 × 8 and the held-out 4 × 7 were drawn, and nothing here transfers
to SDXL without its own run.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**A purity floor**

If a baker puts raisins in 2 of every 10 buns, about 8 of 10 should come out plain. Allow a margin of 1
and the floor is 7: fewer plain buns than that means raisins are spreading. Here, at dial 0.6, 85.5% of
"7" training draws hold one digit. Take off 5 points to get 80.5%, and 0.805 × 256 = 206.08 rounds up to
a floor of 207.

**A mean of three trainings against a line**

Three dice read 2, 3 and 7. Their mean is 12 / 3 = 4, which is under a line at 5. The corrected product
at dial 0.3 reads 19, 16 and 11, with mean 46 / 3 = 15.33, under the line of 39. The line is 39 / 256 =
15.2% of seeds, the same share as 9.75 / 64.

**The dial**

Give a two-digit canvas three tickets: one for "7 and 8", and one each for "7" and "8" scaled by the
dial. At dial 1 the tickets are equal, so after the 10% blank draws the canvas is trained 30% "7 and 8",
30% "7" and 30% "8". At dial 0 only the joint ticket remains.

**The product of experts**

Two tasters score three cups 0.5, 0.4, 0.1 and 0.1, 0.4, 0.5. Multiplying gives 0.05, 0.16, 0.05, so the
cup both accept wins. Here the two tasters are the labels "7" and "8", combined at guidance 3 as blank +
3 × ("7" − blank) + 3 × ("8" − blank). See the [product of experts note](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this)
and [classifier-free guidance as a base plus a scaled difference](../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md#tasks).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| judge counts and the recorded `NULL` | [counts file](../../artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/counts.json), recounted against [the per-tile calls](../../artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/scores.csv), 2026-11-21 | `verified` |
| eye counts | [eye-call files](../../artifacts/drips/inclusive-captions-mnist-stage/eye-calls/), `final_call`, recounted 2026-11-21 | `verified` |
| the floors and lines | [the bar file](../../../poe_repair/experiments/inclusive_captions/mnist/bar.py) | `verified` |
| how the verdict was computed | [the scorer](../../../poe_repair/experiments/inclusive_captions/mnist/score.py) | `verified` |
| decisions about the reader and the floor | [the MNIST stage walk](../../artifacts/drips/inclusive-captions-mnist-stage/EXECUTE_WALK.md) | `stated` |
| the plan | [plan 02, the MNIST stage](../../plans/11-do-inclusive-captions-help-poe-compose/plans/02-the-mnist-stage.md) | `stated` |
| what the product is | [PoE composition](../../../context/world/poe-composition.md#what-poe-composition-is) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Settle tasks 5.2, 5.3 and 6.1 of plan 02, which the walk reports on and the plan leaves unticked.
- [ ] Write this folder's card. It has none, so the figure's provenance sits only in the plan and the scorer.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
