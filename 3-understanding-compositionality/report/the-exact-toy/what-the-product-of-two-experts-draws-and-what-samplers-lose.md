# 📒 Report: what the product of two experts draws, and what samplers lose

**On 2D points the exact product depends on how the two experts' data overlap, and at a joint weight of 0.05 every sampler draws the shared point far less often than the exact product does (0.23 to 0.31 of draws against 0.91).** Each row of the main table is one way two datasets can relate. The left half uses ideal experts written as formulas and the right half uses trained ResNets, so a sampler's error and a network's error show up separately. No pass rule or verdict was recorded for these pictures, all made on 2026-10-29.

## Table of contents

- [The main table](#the-main-table)
- [What the exact product does](#what-the-exact-product-does)
- [What the samplers lose](#what-the-samplers-lose)
- [How the experiment is built](#how-the-experiment-is-built)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The main table

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-exact-product-does) ➡️

| Five overlaps | With the ring and bar row | The ring and bar row alone |
|---|---|---|
| <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained.png" width="200" alt="Five rows by 14 columns: the two datasets, the exact product and four samplers for ideal experts, then the two learned experts, the exact product and four samplers for trained experts."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained-six-rows.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained-six-rows.png" width="200" alt="The same table with a sixth row, a ring of eight clusters times a vertical bar. The exact product is two clusters."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained_ring.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained_ring.png" width="200" alt="The ring times bar row alone across all 14 columns."></a> |
| Each panel plots 2D points on axes from about -1 to 1, and one dot is one of 1,500 draws. One data draw and one seed `stated` | In the ideal group reverse diffusion draws the whole ring and UHA draws two clusters `stated` | The learned ring has unequal cluster heights, so the trained experts' exact product is lopsided `stated` |

**In the ring row, reverse diffusion draws mass round the whole ring where the exact product holds two clusters.** The five overlaps are same coordinates, separate coordinates, disjoint supports, a joint point added to both experts, and partial overlap `verified`. This table has no total variation numbers `stated`.

## What the exact product does

Navigation: ⬅️ [Previous](#the-main-table) | 📋 [TOC](#table-of-contents) | [Next](#what-the-samplers-lose) ➡️

| Exact products only | How far the tails reach | How much of the product is the joint point |
|---|---|---|
| <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table.png" width="200" alt="Four rows of red expert, blue expert and exact product draws, each titled with its overlap integral. The disjoint row is labelled empty."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/log-overlap.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/log-overlap.png" width="200" alt="Log density of each expert and their product at mode spreads 0.075 and 0.03. The tails of the two experts overlap at both."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/clean-share-sweep.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/clean-share-sweep.png" width="200" alt="Left, a heat map of the joint point share against mode spread and joint weight with a white line at 50 percent. Right, the share against spread for three joint weights."></a> |
| Overlap integral: 2.6e-03, 2.5e-01, 0 and 5.7e-01 for the four rows `verified` | Colour is log10 density, down to 12 orders of magnitude below each peak `verified` | Left: x is mode spread, y the joint point's weight (log), colour the share of the product on the joint draw. Right: y is that share, one curve per joint weight `verified` |

**Narrow modes let a small shared point take the whole product, and wider modes lose it.** At a joint weight of 0.01 the share falls from about 1 to near 0 as the mode spread goes from 0.05 to 0.1 `verified` (read off the right panel).

## What the samplers lose

Navigation: ⬅️ [Previous](#what-the-exact-product-does) | 📋 [TOC](#table-of-contents) | [Next](#how-the-experiment-is-built) ➡️

| Three joint weights | Four samplers on each overlap |
|---|---|
| <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/sampler-run.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/sampler-run.png" width="200" alt="Three rows of joint weight by five samplers on 2D points, each panel titled with its joint share and total variation."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-with-samplers.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-with-samplers.png" width="200" alt="Four rows of overlap by seven columns: both experts, the exact product and four samplers, ideal experts with older step sizes."></a> |
| A draw counts as joint within 0.15 of the origin. Mode spread 0.075 `verified` ([sampler numbers](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/sampler-results.json)) | Ideal experts only, drawn with the older step sizes `stated` |

**At a joint weight of 0.05 the exact share is 0.91 and every sampler lands between 0.23 and 0.31.**

| Joint weight | Reading | Perfect draws | Reverse diffusion | ULA | MALA | UHA |
|---|---|---|---|---|---|---|
| none | joint share | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| none | total variation | 0.09 | 0.33 | 0.13 | 0.12 | 0.12 |
| 0.05 | joint share | 0.91 | 0.26 | 0.24 | 0.23 | 0.31 |
| 0.05 | total variation | 0.10 | 0.66 | 0.67 | 0.68 | 0.61 |
| 0.2 | joint share | 0.97 | 0.81 | 0.70 | 0.71 | 0.88 |
| 0.2 | total variation | 0.09 | 0.36 | 0.29 | 0.28 | 0.14 |

<a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-backgrounds.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-backgrounds.png" width="200" alt="Twelve rows of three overlaps by four background settings. Each shows the experts, the background, the exact product and four samplers. The cluttered background moves the exact product to one corner."></a>

**Dividing by a cluttered background makes the exact product jump to a corner where that background is faint.** Rows pair one of three overlaps with no background, a flat one, a pooled one or a cluttered one, and columns are the experts, the background, the exact product and four samplers `stated`. The card calls this set "three backgrounds", and the figure's fourth setting is no background at all `verified`.

## How the experiment is built

Navigation: ⬅️ [Previous](#what-the-samplers-lose) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| The data | One expert as noise rises | The exact product | Perfect draws |
|---|---|---|---|
| <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/data.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/data.png" width="200" alt="Two scatter plots of 800 points, red in four clusters and blue in four clusters."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/expert-levels.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/expert-levels.png" width="200" alt="The red ideal expert density at four noise levels, from four sharp clusters to one broad blob."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/exact-product.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/exact-product.png" width="200" alt="The red and blue expert densities and their normalised exact product, two bright spots."></a> | <a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/perfect-draws.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/perfect-draws.png" width="200" alt="The exact product on a grid beside 1,500 draws from it, two clusters."></a> |
| 800 points per expert `verified` | Density at noise steps -1, 40, 70 and 95 `verified` | Computed on a grid `stated` | 1,500 draws from the grid `verified` |

**The [published scene](https://claude.ai/artifact/CuxEQhwCRBiJ7ESwtdmwYw) walks through this design.** The folder holds [the interactive scene](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/scene.html), [a design page](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/product-of-experts-toy-experiment-high-level-design-explainer.html) and [its diagram](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/product-of-experts-toy-experiment-high-level-design-explainer.svg). The [scene's sampler table](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/hld/img/samplers-table.png) is byte-identical to the four-sampler table above `verified`.

<a href="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/ten-samples.png"><img src="../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/ten-samples.png" width="200" alt="Ten training pictures with their captions above the same pictures with noise removed, from the earlier circle and square toy at 8 by 16 pixels."></a>

The [ten training samples](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/ten-samples.png) come from the earlier circle and square toy, at 8 by 16 pixels with noise of size 0.1 on every pixel `verified`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#how-the-experiment-is-built) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded pictures conflict with their neighbours, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| the exact product of disjoint supports | [the exact-products table](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table.png) marks it empty, with overlap integral 0 | [the four-sampler table](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-with-samplers.png) and the main table draw a thin vertical ridge between the bars, and [the note on the experiment](../../artifacts/notes/how-the-poe-toy-experiment-works/how-the-experiment-works.md#2-the-exact-product) says disjoint supports leave a thin ridge. The first table's bars have hard edges and the others are smoothed |
| which step sizes the published scene's table used | the card labels the four-sampler table "older step sizes" | the scene's own `samplers-table.png` is that same file, so the scene shows the older step sizes |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**One data draw and one seed per cell.** Each panel is 1,500 draws from one dataset draw, so small differences between samplers are not established `stated`.

**The ideal ring is wider than its data.** The ring's own spread is 0.03, under the ideal expert's smoothing of 0.05, so its modes are about 0.058 wide in the ideal group. Every other ideal row uses data spread 0.056 for an expert total of 0.075 `stated`.

**These are 2D points.** Nothing here measures a picture model.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**The exact product on a grid**

Three cells, with expert A giving 0.5, 0.4, 0.1 and expert B giving 0.1, 0.4, 0.5. Multiply cell by cell: 0.05, 0.16, 0.05. Divide by their sum, 0.26: 0.19, 0.62, 0.19. The product is high only where both experts put mass. Here the same steps run on a 241 by 241 grid over -1.1 to 1.1. Read more on the [product of experts](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this).

**Total variation**

The exact answer puts 0.5 in each of two bins and the draws put 0.7 and 0.3. Add the gaps, 0.2 + 0.2, and halve: 0.2. Total variation is half the sum of absolute gaps between the two distributions, 0 for identical and 1 for no overlap. Here it is binned on a 40 by 40 grid, and reverse diffusion at joint weight 0.05 scores 0.66 against 0.10 for perfect draws. The vault and journeys have no entry for total variation yet.

**Joint share**

If 7 of 10 darts land within a circle round the bullseye, the share is 0.7. Joint share is the fraction of draws within 0.15 of the origin, the point both experts hold. At joint weight 0.05 it is 0.9095 for perfect draws and 0.2645 for reverse diffusion.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what each picture shows and how it was made | [the two-experts card](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/README.md), read 2026-11-21 | `stated` |
| joint shares and total variation | [sampler numbers](../../artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/sampler-results.json), read 2026-11-21 | `verified` |
| every column's definition | [the note on how the experiment works](../../artifacts/notes/how-the-poe-toy-experiment-works/how-the-experiment-works.md) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Replace or re-render `hld/preview.png`. It opens almost entirely black (brightest colour value 79 of 255), so it is owed a caption and is not shown here.
- [ ] Say which of the two HTML files in `hld/` is the published scene. They share one title and the card does not distinguish them.
- [ ] Decide whether the published scene should show the newer step sizes.
- [ ] Compute total variation for the main table, which has none.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
