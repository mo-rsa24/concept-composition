# 📒 Report: how the forward process dissolves an image

**The forward process mixes a picture with noise so that the picture's share of the variance falls from 1 to almost 0 while the total stays at 1.** These pictures are stills from the [Dissolving the Dog scene](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9), which runs a 100-step cosine schedule (offset δ = 0.008, β clipped at 0.999) on a 128 by 128 photo of a dog. They teach a mechanism, so no pass rule or verdict applies to them.

## Table of contents

- [The dog dissolving](#the-dog-dissolving)
- [Why the noise carries a square root](#why-the-noise-carries-a-square-root)
- [Where the timetable comes from](#where-the-timetable-comes-from)
- [From the draw to the density and the score](#from-the-draw-to-the-density-and-the-score)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The dog dissolving

Navigation: 📋 [TOC](#table-of-contents) | [Next](#why-the-noise-carries-a-square-root) ➡️

| Five steps | What each piece of the formula protects | The scene's first page |
|---|---|---|
| <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-dog-at-five-steps.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-dog-at-five-steps.png" width="200" alt="The same dog and noise at steps 0, 25, 50, 75 and 99, fading from a clear photo to pure noise, with the kept share under each."></a> | <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/what-each-piece-protects.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/what-each-piece-protects.png" width="200" alt="Three columns: all pieces on with total variance 1.0000, square roots off at step 50 with 0.5009, and the complement off at step 75 with 0.2670."></a> | <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-consequence-page-at-t50.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-consequence-page-at-t50.png" width="200" alt="The scene page at step 50 with its formula, controls, the noisy picture, the coefficient chart and the per-step noise chart."></a> |
| Kept share ᾱ: 0.9994, 0.8356, 0.4783, 0.1335 and 0.000000243 at steps 0, 25, 50, 75 and 99 `verified` | Charts: x is the step, y the multiplier on the dog and on the noise. Numbers are the scene's own `verified` | At step 50 the image is 0.6916 of the dog plus 0.7223 of noise `verified` |

**Halfway through, at step 50, the dog keeps 0.4783 of the variance and the noise holds 0.5217.** With both square roots in place the two shares add to 1.0000. Removing the square roots drops the total to 0.5009, and using the wrong noise scale drops it to 0.2670 at step 75 `verified`.

## Why the noise carries a square root

Navigation: ⬅️ [Previous](#the-dog-dissolving) | 📋 [TOC](#table-of-contents) | [Next](#where-the-timetable-comes-from) ➡️

<a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/four-derivation-steps.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/four-derivation-steps.png" width="200" alt="Four derivation frames: scaling a variable, two noise steps splitting one unit of variance, one schedule boundary by hand, and one beta read as a drop between boundaries."></a>

**Scaling a value by c scales its variance by c², so a noise share of 0.5217 needs a multiplier of 0.7223.** The composite holds four frames, each also saved at full size:

| Frame | What it works out |
|---|---|
| [why the noise carries a square root](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/why-the-noise-carries-a-square-root.png) | 300 draws scaled by 0.7223 have variance 0.522 `verified` |
| [two steps, the variances add](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/two-steps-the-variances-add.png) | with β of 0.3 and 0.5, one unit splits into dog 0.35, earlier noise 0.15 and fresh noise 0.50 `verified` |
| [one cosine boundary by hand](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/one-cosine-boundary-by-hand.png) | boundary 51 sits at 46.25° and keeps 0.4783 `verified` |
| [each β read off the curve](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/each-beta-read-off-the-curve.png) | β₅₀ = 1 - 0.4783 / 0.4938 = 0.0315 `verified` |

## Where the timetable comes from

Navigation: ⬅️ [Previous](#why-the-noise-carries-a-square-root) | 📋 [TOC](#table-of-contents) | [Next](#from-the-draw-to-the-density-and-the-score) ➡️

| A point on a quarter circle | One boundary, term by term | Where each post sits |
|---|---|---|
| <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-point-on-the-quarter-circle.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-point-on-the-quarter-circle.png" width="200" alt="Six panels building a quarter circle with a point at 46.25 degrees, its two legs, the arc and the cosine squared curve."></a> | <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/one-boundary-term-by-term.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/one-boundary-term-by-term.png" width="200" alt="Seven underbraced terms for boundary 51 ending at 0.47826, with the division by the first boundary switched on and off."></a> | <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/where-each-post-position-comes-from.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/where-each-post-position-comes-from.png" width="200" alt="Two rows: the paper line with post 0 at 0.714 degrees and post 100 at 90 degrees, then the broken cases with the division off and with 1,000 steps."></a> |
| The point (0.6915, 0.7224) is 1 from the origin, so its two legs are the dog and noise multipliers `verified` | Dividing by 0.99984 changes C₅₁ from 0.47819 to 0.47826 and leaves β₅₀ at 0.031546 `verified` | Without the division by (1 + δ), post 100 lands at 90.72° and C₁₀₀ is 0.000158 instead of 0 `verified` |

| The schedule's name, taken apart | Why the offset and the normaliser |
|---|---|
| <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-schedule-name-taken-apart.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-schedule-name-taken-apart.png" width="200" alt="Six panels: posts as bars, the cosine curve against a straight line, the offset moving post 0, and the normaliser on and off."></a> | <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/why-the-offset-and-the-normaliser.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/why-the-offset-and-the-normaliser.png" width="200" alt="The slope of cosine squared at post 0 and post 51, the first noise step against one pixel bin, and the normaliser cancelling in each beta."></a> |
| The cosine curve drops 0.067 per 15° at the ends and 0.25 in the middle, and a straight line drops 0.1667 every 15° `verified` | The first noise scale is 0.0157 at δ = 0 and 0.0251 at δ = 0.008, against one pixel bin of 0.0078 `verified` |

## From the draw to the density and the score

Navigation: ⬅️ [Previous](#where-the-timetable-comes-from) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| The one-shot draw and its density | From the density to the score |
|---|---|
| <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-one-shot-draw-and-its-density.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/the-one-shot-draw-and-its-density.png" width="200" alt="Three columns for a 2D point at step 50, step 25 and with the wrong centre, each with the density formula, a plane of draws and the exponent against the step."></a> | <a href="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/from-the-draw-to-the-density-and-the-score.png"><img src="../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/from-the-draw-to-the-density-and-the-score.png" width="200" alt="Two derivation frames: draws at one step form a Gaussian matching a histogram, and the slope of the log density points back toward the scaled clean point."></a> |
| Plane: the two coordinates of x. Lower chart: x is the step and y is the exponent of q at the drawn point, which stays at -0.7650 with the right centre and moves with the wrong one `verified` | Histogram: x is the second coordinate of the draws, y their count, with the curve N(0.3458, 0.5217) `verified` |

**Holding the clean point fixed, the draws at step t form a Gaussian, and its score points from the draw back to the scaled clean point.** At step 50 the draw minus the centre is (0.2167, -0.8668), which is 0.7223 times the noise, so the clean point cancels `verified`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#from-the-draw-to-the-density-and-the-score) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded details do not line up, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| the printed variance sum | the [scaling frame](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/why-the-noise-carries-a-square-root.png) prints 0.72² Var(X) = 0.5217 | 0.72² = 0.5184. The unrounded multiplier gives 0.7223² = 0.5217, so the printed 0.72 is a rounding of 0.7223 |
| the scene's numbering of derivation frames | the [four-frame composite](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/four-derivation-steps.png) labels them "Derive 2 of 6" and "Derive 5 of 6" | the [density and score still](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/from-the-draw-to-the-density-and-the-score.png) labels its frames "Derive 7 of 9" and "Derive 9 of 9" |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**These numbers describe the schedule on one photo.** The forward process has no learned part, so nothing here measures a trained model or composition.

**The 2D density frames use one worked point,** x₀ = (0, 0.5) with noise (0.3, -1.2), chosen for the walk `stated`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the pictures above.

**Variance shares**

Mix a signal at scale 0.6 with noise at scale 0.8. Their variances are 0.6² = 0.36 and 0.8² = 0.64, which add to 1. The rule is: a multiplier's square is that part's share of the variance. At step 50 the shares are 0.6916² = 0.4783 for the dog and 0.7223² = 0.5217 for the noise. Read more on [the forward process](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Forward%20Process.md#where-to-learn-this), or [derive it in the journey](../../../../../goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/plans/06-the-forward-process.md#1--the-forward-process-derived).

**The cosine schedule**

Stand on a quarter circle at 60°. The cosine is 0.5 and its square is 0.25, so a quarter of the picture's variance is left. The schedule is C = cos²(angle) / cos²(first angle), with the angle growing evenly by step. At boundary 51 the angle is 46.25°, cos² is 0.47819, and dividing by 0.99984 gives 0.47826. Read more on [the cosine noise schedule](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Cosine%20Noise%20Schedule.md#the-cosine-schedule-taught-in-applied-in).

**The score**

On a line with centre 2 and variance 4, the point 5 has score -(5 - 2) / 4 = -0.75: a push back toward the centre. The score is -(x - centre) / variance. At step 50, dividing (-0.2167, 0.8668) by the variance 0.5217 gives (-0.415, 1.661). Read more on [the score function](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Score%20Function.md#where-to-learn-this).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what each still shows and how it was captured | [the forward-process card](../../artifacts/results/the-exact-toy/how-the-forward-process-dissolves-an-image/README.md), read 2026-11-21 | `stated` |
| the numbers on each still | read off the pictures, 2026-11-21 | `verified` |
| the dog photo | "YellowLabradorLooking new.jpg" by Djmirko (derivative work), Wikimedia Commons, CC BY-SA 3.0, centre-cropped and resized to 128 by 128 | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Add card entries for five stills the card's table omits: `each-beta-read-off-the-curve.png`, `from-the-draw-to-the-density-and-the-score.png`, `one-cosine-boundary-by-hand.png`, `two-steps-the-variances-add.png` and `why-the-noise-carries-a-square-root.png`.
- [ ] Locate the capture scripts. The card names `scripts/scene_stills/`, and no such file exists in this checkout.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
