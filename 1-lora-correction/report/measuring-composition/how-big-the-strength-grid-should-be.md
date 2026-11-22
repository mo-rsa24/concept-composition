# 📒 Report: how big the paper's strength grid should be

**This page records a layout choice, and no result verdict belongs to it.** The strength grid shows one
cat and dog render at rising amounts of [the correction](../what-the-correction-is/does-the-correction-cause-composition.md#the-correction-and-its-strength).
A cropped copy set at 0.62 of the text width leaves its tiles small. [The section 3 proofreading map](../../artifacts/drips/proofreading-the-paper/proofread-section-3/PROOF_MAP.md)
records a full-width replacement as accepted, at 5.50 by 4.97 inches with 1.09 inch tiles, and a move of the
grid to the appendix `stated`.

## Table of contents

- [Three sizes of the same grid](#three-sizes-of-the-same-grid)
- [What the curve under the grid counts](#what-the-curve-under-the-grid-counts)
- [Six columns show the step before](#six-columns-show-the-step-before)
- [The four-model grid](#the-four-model-grid)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Three sizes of the same grid

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-curve-under-the-grid-counts) ➡️

**Refitting the grid to the text width makes each tile 2.9 times the area of the crop** `stated` (the
[grid card](../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/README.md)).
In all three, a column is a correction strength, a row is what was added, and the pair is cat and dog at seed 9.

| The crop | Full width, grid only | Full width, with the curve |
|---|---|---|
| <a href="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-as-it-is-in-the-paper.png"><img src="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-as-it-is-in-the-paper.png" width="200" alt="Four rows by five columns of cat and dog renders, with large row labels taking up the left third."></a> | <a href="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-fitted-grid-only.png"><img src="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-fitted-grid-only.png" width="200" alt="The same four by five grid at full width with a detector count in a circle on each tile."></a> | <a href="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-fitted-with-curve.png"><img src="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-fitted-with-curve.png" width="200" alt="The same grid above a line plot of compose rate against strength. The real correction climbs to about 94 percent and three controls stay under 10 percent."></a> |
| 3.44 by 2.49 in, tiles about 0.51 in `stated` | 5.50 by 4.10 in, tiles 0.88 in `stated` | 5.50 by 5.79 in, tiles 0.88 in `stated` |

## What the curve under the grid counts

Navigation: ⬅️ [Previous](#three-sizes-of-the-same-grid) | 📋 [TOC](#table-of-contents) | [Next](#six-columns-show-the-step-before) ➡️

**The real correction takes 30 of 32 runs to two animals at full strength, and no drawn control passes 3
of 32.** On the curve, y is the [compose rate](do-the-scored-successes-contain-both-animals.md#a-compose-rate),
x is the strength λ, and one point is 32 runs (8 pairs at 4 seeds). Each cell below is the rate in
`outputs/interaction_term/dose/dose_curves.json` multiplied by 32 `verified`.

| Row drawn, runs composed of 32 | λ 0 | λ 0.25 | λ 0.5 | λ 0.75 | λ 1 |
|---|---|---|---|---|---|
| real correction | 1 | 3 | 8 | 23 | 30 |
| wrong pair | 1 | 1 | 1 | 2 | 1 |
| wrong seed | 1 | 1 | 1 | 2 | 3 |
| wrong order | 1 | 1 | 1 | 1 | 0 |

These rates were scored by the strength-sweep script, which never reads a `__cmp` comparison strip, as
[the page on the 32 images](do-the-scored-successes-contain-both-animals.md#what-the-detector-scored-on-all-32-images) shows.

## Six columns show the step before

Navigation: ⬅️ [Previous](#what-the-curve-under-the-grid-counts) | 📋 [TOC](#table-of-contents) | [Next](#the-four-model-grid) ➡️

> [!NOTE]
> The six-column grid is not shown here because a full-resolution check could not clear its wrong-seed and
> wrong-pair rows against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets. The file stays at
> `1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-six-columns.png`.

**At strengths 0.6 and 0.65 the real-correction row already holds a cat and a dog** `verified`, while the
wrong-pair, wrong-seed and wrong-order rows stay single fused animals. The eight new tiles at those two
strengths carry no detector count, because the scores were not rerun `stated` (the grid card).

## The four-model grid

Navigation: ⬅️ [Previous](#six-columns-show-the-step-before) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The same folder holds layouts of a second figure: the joint prompt and the plain product on four models.**
Each model gets two columns and each prompt pair one row. Read at sheet scale, the cat and dog row at seed 42
holds both animals under the joint prompt on SDXL and SD 3.5, a lone cat on SD 1.4 and two dogs on SD 2.1.
The plain product draws one animal on all four `verified`.

| Fitted to the text width | At seed 9 |
|---|---|
| <a href="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/four-models-fitted-to-text-width.png"><img src="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/four-models-fitted-to-text-width.png" width="200" alt="Three prompt pairs by four models, each model with a joint prompt column and a product column."></a> | <a href="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/four-models-seed-9.png"><img src="../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/four-models-seed-9.png" width="200" alt="The same three pairs and four models at seed 9, with different pictures in every tile."></a> |
| seed 42 `inferred` (its SDXL cat and dog tile matches the seed 42 tile in the main-study grid) | seed 9, from the filename `stated` |

The seed number gives each model a different starting noise, because [the four-model script](../../scripts/same_rule_four_models.py)
draws SD 1.4, SD 2.1 and SD 3.5 noise from a generator at their own shapes and SDXL borrows a cached start
(lines 15 to 17) `verified`. Compare within a model and leave comparisons across models unread.

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-four-model-grid) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts differ across sources, and each is shown here unresolved.**

| What | One source | The other source | What the files show |
|---|---|---|---|
| which grid the manuscript carries | the grid card says the 0-to-1 crop at 0.62 of the text width | three manuscript copies include three different files: `writing/iclr/iclr2027_conference.tex` line 256, `strength-grid.pdf` at full width; `writing/overleaf-iclr/iclr2027_conference.tex` line 548, the 0-to-1 crop at 0.72; `writing/restoring-plurality-in-compositional-diffusion_overleaf/iclr2027_conference.tex` line 541, `strength-grid-seven-columns.pdf` | the crop in the second copy is byte-identical to `strength-grid-shipped-4-7-inch.pdf` in this folder. No file read here says which copy was submitted |
| which controls stay low | [the review of the strength sweep](../../artifacts/plans/completed/03-does-the-correction-cause-composition/review/03-more-correction-more-composition.md) says both controls stay at or below 6% | the curve here has a control at 3 of 32 (9.4%) at full strength | the two name different controls. The review's are a random vector and another pair's correction; [the figure script](../../scripts/compose_rate_vs_correction.py) (line 51) draws wrong pair, wrong seed and wrong order, and the 9.4% is wrong seed |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The tiles show one pair at one seed and the curve counts 8 pairs at 4 seeds.** A tile cannot stand in
for the rate, and the rate cannot say what any one tile looks like.

**The count on a tile is the detector's.** What that count misses is on
[the page about the 32 images](do-the-scored-successes-contain-both-animals.md#where-the-sources-disagree).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. [The correction and its strength](../what-the-correction-is/does-the-correction-cause-composition.md#the-correction-and-its-strength)
and [a compose rate](do-the-scored-successes-contain-both-animals.md#a-compose-rate) are taught on their own pages.

### A control row

To test whether a spice improves a soup, you also add a same-sized spoon of flour. If only the spice changes
the taste, the change came from the spice and the spoonful did nothing. A control here is a push of the same
size and schedule with one thing broken. Wrong pair takes another pair's correction, wrong seed takes the same
pair's from another seed, and wrong order shuffles its steps. At full strength they compose 1, 3 and 0 of 32
runs against 30 for the real correction.

### Tile area

Doubling a square tile's side makes its area 2 x 2 = 4 times larger, so a small change in width is a larger
change in what the eye can read. Area is side times side. Here the side goes from about 0.51 to 0.88 inches,
and the card records the gain in area as 2.9 times.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| sizes and how each grid was made | [the grid card](../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/README.md) | `stated` |
| the accepted layout | [the section 3 proofreading map](../../artifacts/drips/proofreading-the-paper/proofread-section-3/PROOF_MAP.md) | `stated` |
| runs composed | `outputs/interaction_term/dose/dose_curves.json`, read 2026-11-22 | `verified` |
| the grid and curve | [the figure script](../../scripts/compose_rate_vs_correction.py) | `verified` |
| the four-model tiles | [the four-model script](../../scripts/same_rule_four_models.py) and the script diffs in the folder | `stated` |
| 9 layout variants held out of the drain | the other four-model, fitted, full-width and SVG versions in [the grid folder](../../artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be), which repeat the layouts placed above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Say which manuscript copy was submitted, and so which strength grid it carries.
- [ ] Add the four-model layouts to the grid card, which describes only the strength grid.
- [ ] Repoint the grid card's link to the proofreading map, whose folder has moved.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
