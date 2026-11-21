# 📒 Report: does the share of two-digit pictures help the product compose on MNIST? Recorded verdict: "NULL" (results card), with plan 03 at 0 of 9 tasks ticked

**The claim**

The [results card](../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/README.md)
records: "No: the verdict is NULL." This run sets the share directly: at share s, a training picture
labelled with one digit shows that digit beside a second digit with probability s. At guidance 3, 10% is
the only share where "7" and "8" alone stay clean enough to count, and there the corrected 7 × 8 product
averages 23 of 256 against a null line of 39. The verdict rests on an automatic judge that has not been
checked by eye on this run, and the plan that owns it has no task ticked.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. The product at guidance 3](#1-the-product-at-guidance-3)
- [2. The same models redrawn at guidance 1](#2-the-same-models-redrawn-at-guidance-1)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-the-product-at-guidance-3) ➡️

**The rule sits in [the share bar file](../../../poe_repair/experiments/inclusive_captions/mnist/share_bar.py),
committed before any share model trained.** Counts are of 256 seeds, read by the pinned judge.

| Outcome | What it takes |
|---|---|
| a share counts | "7" alone and "8" alone each draw their lone digit on at least round(0.95 × 256 × (1 − s)) seeds, in all three trainings |
| supported | at a counting share of 10, 20, 40 or 80%, the corrected 7 × 8 product reaches 128 in all three trainings, while 0% stays at 12 or fewer |
| null | no counting share's three-training mean reaches 39 |
| inconclusive | anything else |
| the judge's reading holds | on one 64-tile sheet per share, it agrees with blind eye calls on at least 58 of 64; otherwise that share is read by eye |

The two guidance-1 readings in the same file were written after the guidance-3 NULL and before any
guidance-1 picture was drawn `stated` (the file's comment).

## 1. The product at guidance 3

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-the-same-models-redrawn-at-guidance-1) ➡️

**Only the 10% share counts, and its corrected product averages (17 + 28 + 24) / 3 = 23 of 256.** Counts
are per training, from the `verdict` block of the
[counts file](../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/counts.json)
`verified`. The card's [count-against-share figure](../../artifacts/scenes/how-the-inclusive-labels-mnist-stage-works/figures/s2-the-product-against-what-each-label-draws-alone.png) draws them.

| Share | Floor | "7" alone | "8" alone | "7" with a second digit | Corrected 7 × 8 | Plain 7 × 8 | Counts? |
|---|---|---|---|---|---|---|---|
| 0% | 243 | 253, 255, 253 | 256, 256, 256 | 0, 0, 1 | 0, 0, 0 | 0, 0, 0 | baseline |
| 10% | 219 | 247, 240, 241 | 247, 244, 242 | 6, 12, 13 | 17, 28, 24 | 10, 22, 15 | yes |
| 20% | 195 | 166, 207, 197 | 186, 196, 188 | 88, 48, 59 | 52, 56, 35 | 85, 70, 86 | no |
| 40% | 146 | 50, 42, 43 | 44, 55, 43 | 200, 213, 212 | 40, 80, 27 | 137, 147, 142 | no |
| 80% | 49 | 1, 0, 0 | 1, 1, 0 | 250, 254, 255 | 72, 81, 22 | 114, 139, 91 | no |

## 2. The same models redrawn at guidance 1

Navigation: ⬅️ [Previous](#1-the-product-at-guidance-3) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

<a href="../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/guidance-1-first-six-models.png"><img src="../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/guidance-1-first-six-models.png" width="140" alt="Thirty rows of digit tiles in two columns, guidance 3 on the left and guidance 1 on the right, each row labelled with its share, training, what was asked and its count of 256"></a>

**At guidance 1 the file records that guidance made the extra pairs, and the product composes beyond its
labels at no judged share.** The picture has guidance 3 on the left and guidance 1 on the right. Each row
is one model and one request, with its count of 256 printed above it. At 80%, training 2, the corrected
product reads 81 at guidance 3 and 206 at guidance 1 `verified` (on the picture and in the file).

"Composes beyond its labels" needs the plain product to reach 39 and twice the larger of the two single
labels' with-a-second counts, in all three trainings. Counts are from the `shares` block of the
[guidance-1 counts file](../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/counts-g1.json) `verified`.

| Share | "7" with a second (most allowed) | Larger single label × 2 | Plain 7 × 8 | Composes beyond its labels? |
|---|---|---|---|---|
| 10% | 11, 10, 13 (38) | 30, 30, 34 | 4, 8, 9 | no |
| 20% | 35, 25, 30 (64) | 72, 60, 74 | 36, 30, 43 | no |
| 40% | 82, 77, 90 (115) | 182, 154, 180 | 115, 111, 126 | no |
| 80% | 183, 185, 192 (218) | 366, 382, 390 | 189, 180, 189 | no |

## Where the sources disagree

Navigation: ⬅️ [Previous](#2-the-same-models-redrawn-at-guidance-1) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Six recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| the verdict's standing | the card and the counts file record NULL | [plan 03](../../plans/11-do-inclusive-captions-help-poe-compose/plans/03-the-mnist-share-run.md#tasks) has 0 of its 9 task and instruction boxes ticked, including the verdict line |
| the judge check the rule requires | the rule needs 58 of 64 agreement with blind eye calls | the counts file reads `judge_eye_check: pending: plan 03 task 4.2`, and the card says the judge "has not been checked by eye on this run" |
| the guidance-1 redraw | the card: "is running and will be added here" | the guidance-1 counts file holds a complete result |
| how many models were scored | the card: 15 models, 30,720 rows, 120 counts | the counts file: 27 models, and 216 count cells; `scores.csv` holds 55,296 rows. The shares 30, 50, 60 and 70% have no card entry and no row in the bar |
| which models the picture shows | its filename says "first six models" | its row labels show ten models, three rows each |
| where the guidance-1 task lives | the bar file and scorer cite "plan 03, task 5" | plan 03 lists tasks 1 to 4 and instruction 5, and none of them covers guidance 1 |

The four extra shares do not change which shares count under the same floor formula. "7" alone reads at
most 142 at 30% (floor 170), 29 at 50% (floor 122), 6 at 60% (floor 97) and 2 at 70% (floor 73)
`verified` (computed from the counts file with round(0.95 × 256 × (1 − s))).

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The verdict rests on the judge alone.** Its eye check is pending, and the same pinned judge undercounted
clean pairs on the [dial run](do-inclusive-labels-help-the-product-compose-on-mnist.md#where-the-sources-disagree).

**"With a second" does not mean "with an 8".** The judge counts a 7 beside any second digit, so "7" alone
at 20% seed 2, two 7s, is still counted `stated` (the card).

**One small model and one sampler.** One U-Net on a 200-step cosine schedule and one corrector setting
(20 Langevin steps per level). Nothing here transfers to SDXL without its own run.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**Guidance**

Take a plain recipe at 2 spoons of sugar and a "sweet" recipe at 5. The difference is 3. Guidance 1 uses
the sweet recipe as written, 2 + 1 × 3 = 5. Guidance 3 exaggerates the difference, 2 + 3 × 3 = 11. A model
does this with its blank and labelled noise predictions, which may be why "7" alone draws pairs far above
its trained share at guidance 3. Read [classifier-free guidance as a base plus a scaled difference](../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md#tasks),
or watch [CME296 lecture 4 from 1:27:42](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=5262s).

**A purity floor**

If 1 bun in 10 is meant to carry a raisin, 9 of 10 should be plain, and a 95% allowance makes the floor
8.55. Here at share 10%, 90% of a label's pictures show its digit alone, so the floor is
round(0.95 × 256 × 0.9) = round(218.88) = 219 of 256.

**Composing beyond the labels**

If each of two friends already brings a guest to 3 of 10 parties, seeing guests at 3 parties proves
nothing about the friends combining. The rule asks for at least twice that, 6. At 20%, training 1, the
single labels draw a second digit on 35 and 36 seeds, so the product must reach 2 × 36 = 72. It reads 36.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the recorded verdict and how the run was made | [results card](../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/README.md), read 2026-11-21 | `stated` |
| guidance-3 counts | [counts file](../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/counts.json), read 2026-11-21 | `verified` |
| guidance-1 counts | [guidance-1 counts file](../../artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/counts-g1.json), read 2026-11-21 | `verified` |
| the rule | [the share bar file](../../../poe_repair/experiments/inclusive_captions/mnist/share_bar.py) | `verified` |
| the plan | [plan 03, the MNIST share run](../../plans/11-do-inclusive-captions-help-poe-compose/plans/03-the-mnist-share-run.md) | `stated` |
| what the product is | [PoE composition](../../../context/world/poe-composition.md#what-poe-composition-is) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Run the judge-against-eye check, plan 03 task 4.2. Until it passes, the rule's condition for reading the judge is unmet.
- [ ] Tick plan 03's tasks and verdict line, or record why they stay open.
- [ ] Add the guidance-1 result and the shares 30, 50, 60 and 70% to the card.
- [ ] Decide what the four extra shares are for, since the bar was written for 10, 20, 40 and 80%.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
