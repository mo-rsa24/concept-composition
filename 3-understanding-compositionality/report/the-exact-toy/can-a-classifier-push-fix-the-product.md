# 📒 Report: can a classifier's push fix the product on the exact toy

**A classifier trained on what the sampler gets wrong turns stacked draws into side-by-side pairs, and the recorded verdict is pass.** At pair weight 0.01 the exact product draws a red circle beside a blue square 300 times of 300, while the sampler that adds the two experts' scores stacks them all 300 times. These experiments add a classifier's gradient to that sampler at every step, with the experts frozen. The verdict is owned by [finding A1](../../../2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md). Every count is one seed (7) of 300 draws.

## Table of contents

- [What each push draws](#what-each-push-draws)
- [What each classifier has learned](#what-each-classifier-has-learned)
- [Which direction the push points](#which-direction-the-push-points)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What each push draws

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-each-classifier-has-learned) ➡️

| One draw with and without the true correction | The discriminator in seven stages |
|---|---|
| <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/draw-0-unguided-and-pushed.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/draw-0-unguided-and-pushed.png" width="200" alt="Draw 0 of seed 7 three times: unguided and with the true correction at scale 0 are purple overlays, and at scale 1 it is a crisp circle and square."></a> | <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png" width="200" alt="Seven stages left to right: two piles of draws, noising, the network, the chance a picture is real, the gradient, where it enters the sampler, and crisp pairs of 300 at each push weight."></a> |
| The same starting noise, with the known correction switched off and on `verified` ([hook check](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/port.json)) | The real pile is 3,500 exact draws and the fake pile is 3,500 of the sampler's draws, with 500 of each held out `verified` |

**Of the two classifiers, only the one trained on the sampler's own draws meets the written pass rule.**

| What pushes the sampler | Crisp pairs of 300 | Overlaid | File |
|---|---|---|---|
| nothing | 0 | 300 | [hook check](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/port.json) |
| true correction, scale 0.5 | 112 | 188 | [hook check](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/port.json) |
| true correction, scale 1 | 300 | 0 | [hook check](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/port.json) |
| separate-label classifier, scales 1, 2, 5, 10 | 0, 0, 2, 10 | 298, 297, 295, 290 | [separate-label counts](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/separate-label-guide.json) |
| closed-form best separate-label classifier, same scales | 0, 1, 13, 34 | 294, 298, 287, 266 | [separate-label counts](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/separate-label-guide.json) |
| discriminator, push weight 0.5, 1, 2 | 16, 172, 300 | 284, 128, 0 | [discriminator counts](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/pair-aware-guide.json) |

The pass rule, written before any classifier was trained in [plan 19](../../../2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md#goal), asks for at least 150 crisp at push weight 1 with no more overlaid than unguided, and at most 15 crisp at every scale for the separate-label push `verified`.

> [!IMPORTANT]
> The separate-label half depends on which classifier it is read on. The owner chose the trained one, and the card keeps the other number beside it.

| How the rule is read | Highest crisp count, limit 15 | That reading gives |
|---|---|---|
| on the trained classifier (recorded choice) | 10 | pass |
| on the closed-form best classifier | 34 | miss |

The [earlier six-stage version](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator__2026-11-17.png) of the figure was drawn before guiding ran. [The current version's dated copy](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator__2026-11-17-2.png) is byte-identical to the figure above `verified`.

## What each classifier has learned

Navigation: ⬅️ [Previous](#what-each-push-draws) | 📋 [TOC](#table-of-contents) | [Next](#which-direction-the-push-points) ➡️

| The discriminator | The separate-label classifier | How much each prefers the pair |
|---|---|---|
| <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/what-the-pair-aware-discriminator-says.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/what-the-pair-aware-discriminator-says.png" width="200" alt="Two rows, a real pair and a stacked sampler draw, clean and at steps 100 and 150, each titled with the discriminator output."></a> | <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/what-the-separate-label-classifier-says.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/what-the-separate-label-classifier-says.png" width="200" alt="A pair, an overlay and one circle, each with the classifier output for circle and for square at step 100."></a> | <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/minimal-pair-gap-per-step.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/minimal-pair-gap-per-step.png" width="200" alt="Log-odds on the pair minus the stacked picture against noise step. The discriminator rises to about 28, the separate-label classifier sits near 4 and shuffled labels stay at 0."></a> |
| d is the chance a picture is real. One held-out draw of each kind, one noise draw per panel `verified` | At step 100 it calls the pair 1.00 and 1.00, and the overlay 0.00 for circle and 1.00 for square `verified` | x: noise step, 200 to 0. y: log-odds on the pair minus on the stacked picture. One point is one noise step, over 256 shared noise draws `stated` ([diagnosis file](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/diagnosis.json)) |

**The discriminator prefers the pair by 27.9 log-odds at step 10, against 3.6 for the separate-label classifier.** A discriminator trained on shuffled labels stays within 0.003 of 0 at every step `stated`.

## Which direction the push points

Navigation: ⬅️ [Previous](#what-each-classifier-has-learned) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| One draw at step 150 | The push's direction, step by step |
|---|---|
| <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/draw-0-push-against-true-correction-step-150.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/draw-0-push-against-true-correction-step-150.png" width="200" alt="Three panels at step 150: the noisy draw, the true correction looking like noise and the discriminator push showing a circle-shaped and a square-shaped region."></a> | <a href="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/push-cosine-per-step.png"><img src="../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/push-cosine-per-step.png" width="200" alt="Cosine of the push against reverse step. The orange line against the sampler ratio holds near 0.85 to 0.96 and the blue line against the true correction stays under 0.21. Both fall to 0 by step 80."></a> |
| Summed over colour, red raises a pixel and blue lowers it. Cosine with the true correction on this draw is 0.15 `verified` | x: reverse step. y: cosine of the push with its target. Line: median of 300 draws, band: 25th to 75th percentile, at push weight 1 `verified` ([diagnosis file](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/diagnosis.json)) |

**The push follows the sampler's own error far more closely than the true correction.** Its median cosine with the sampler's own ratio is 0.96 at step 190 and 0.85 at step 100, and with the true correction at most 0.21. The sampler's own ratio is the true noised product's score minus the score of 3,500 finished sampler draws noised forward `verified`.

**The push at pair weight 0.01 draws more crisp pairs than raising the pair weight to 0.3 with no push.** With no push, the pair weights 0.01, 0.03, 0.1, 0.2, 0.3 and 0.5 give 0, 1, 18, 53, 115 and 246 crisp (the [comparison file](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/inclusive-captions-comparison.json)). The push at 0.01 gives 172. The discriminator needed 3,500 pair pictures of its own `verified`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#which-direction-the-push-points) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two sentences on the card conflict with its diagnosis file, and each is shown here unresolved.**

| What | The card | [The diagnosis file](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/diagnosis.json) |
|---|---|---|
| the true correction's size against the summed score | half "at every step" | the median ratio stays between 0.498 and 0.511 from step 200 to step 100, then falls to 0.491 at step 60, 0.463 at 30, 0.426 at 20 and 0.309 at 10 |
| when the push stops | "from step 80 the push is 0" | the median is 0 at step 80, and the 75th percentile is still 0.0187 of the summed score there. It is 0 from step 70. The logit clip covers every draw that ends stacked by step 60 and every draw that ends crisp by step 40 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**This repairs a sampling failure.** The exact product already composes on this toy. Where the product's own target is a hybrid, as under competing captions, this result says nothing `stated` ([finding A1](../../../2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md)).

**The labels check the picture and leave the proportions open.** A draw is called crisp, overlaid, faint or single, so nothing here shows the pairs land in the exact product's positions in the right proportions `stated`.

**Every count is one seed.** One seed of 300 draws per count, and the late-step cosines against the sampler's own ratio are to be read loosely because that target is one sharp spike per draw `stated`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**Log-odds**

A taster is 80% sure a coffee is the real blend. The odds are 0.8 / 0.2 = 4, and the log-odds are ln 4 = 1.39. At 50% they are 0. Log-odds is log(d / (1 - d)), with d the chance the picture is real. The push is the slope of the log-odds with respect to each pixel. Here the gap between pair and stacked picture is 2.2 at step 180, 20.2 at step 100 and 27.9 at step 10. Read more on the [logit](../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Logit.md#where-to-learn-this). The vault and journeys have no entry for a discriminator under that name.

**Cosine between two directions**

Take arrows (3, 4) and (4, 3). Multiply matching entries and add: 12 + 12 = 24. Divide by the two lengths, 5 × 5: 24 / 25 = 0.96, nearly the same direction. Arrows (1, 0) and (0, 1) give 0. Here the push and the sampler's own ratio have a median cosine of 0.96 at step 190. Read more on [cosine similarity](../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this).

**How the push enters the sampler**

If the predicted noise is 1.0, the push weight 1, the noise scale 0.5 and the gradient 0.4, the new prediction is 1.0 - 1 × 0.5 × 0.4 = 0.8. The line is ε̃ = ε - w × √(1 - ᾱ) × g. Here w of 0.5, 1 and 2 gives 16, 172 and 300 crisp pairs.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| counts, method and caveats | [the classifier card](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/README.md), read 2026-11-21 | `stated` |
| counts per push | the JSON files linked above, read 2026-11-21 | `verified` |
| the written pass rule | [plan 19](../../../2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md#goal) | `stated` |
| the code | [the classifier push script](../../artifacts/notes/what-poe-loses-when-it-composes/toy_classifier_push.py) | `stated` |
| 6 small draws held out of the drain | the `the-side-by-side-discriminator` real row, fake row and four guided draws in [the classifier results folder](../../artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product), which are the inputs of the seven-stage figure placed above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Record the owner's own look at the two diagnosis charts. Plan 19's step 2.5 is unticked because the walk closed without an account of them, though its numbers match.
- [ ] Check whether the crisp pairs at push weight 2 land in the exact product's positions in the right proportions. No measurement exists.
- [ ] Repeat the counts on more than one seed.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
