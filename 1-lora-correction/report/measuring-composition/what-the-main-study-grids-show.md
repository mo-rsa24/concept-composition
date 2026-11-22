# 📒 Report: what the main study's comparison grids show

**No verdict has been recorded for the three grids on this page.** Each grid sets one method per column
against the same request, so a reader can compare methods on the same picture. Two more grids from this
set already have findings: [the same concept named twice](../is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md)
and [where the joint prompt itself fails](../is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md).

## Table of contents

- [Published models beside the plain product](#published-models-beside-the-plain-product)
- [Three kinds of pair](#three-kinds-of-pair)
- [What the adapter does to a joint prompt](#what-the-adapter-does-to-a-joint-prompt)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Published models beside the plain product

Navigation: 📋 [TOC](#table-of-contents) | [Next](#three-kinds-of-pair) ➡️

<a href="../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/published-models-then-ours.png"><img src="../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/published-models-then-ours.png" width="200" alt="One row of seven columns for cat and dog at seed 42. Six rendered tiles, a dotted rule, and an empty tile labelled not rendered for this project's adapter."></a>

**The saved grid has one row, cat and dog at seed 42, and its adapter tile is empty** `verified` (the picture
and [its tile list](../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/published-models-then-ours.json)).
By eye, the joint prompt draws a lone cat on SD 1.4, two dogs on SD 2.1, and a cat with a dog on SD 3.5 and
SDXL. SDXL's plain product draws one fluffy animal, and CO3 draws two cats `verified`. CO3 starts from its own noise, so
its tile is not pixel-comparable with the others `stated` (the picture's subtitle).

## Three kinds of pair

Navigation: ⬅️ [Previous](#published-models-beside-the-plain-product) | 📋 [TOC](#table-of-contents) | [Next](#what-the-adapter-does-to-a-joint-prompt) ➡️

<a href="../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/the-fix-is-not-about-animals.png"><img src="../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/the-fix-is-not-about-animals.png" width="200" alt="Six rows by three columns: joint prompt, plain product, adapter. The adapter column is empty on the two animal rows and the two object rows and filled on the lighthouse and tent rows."></a>

**The adapter column is filled on 2 of the 6 rows, both an object in a place** `verified` (the picture and
[its tile list](../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/the-fix-is-not-about-animals.json)).

| Row, seed 9 | Joint prompt | Plain product | Adapter |
|---|---|---|---|
| lighthouse and stormy sea | both | both | both |
| tent and snowy mountain | an orange tent below a mountain | a mountain shaped like a tent, fused | a dome tent and a white mound below a mountain |
| cat and dog, tiger and dog, chess board and hourglass, picnic basket and watermelon | rendered | rendered | not rendered |

## What the adapter does to a joint prompt

Navigation: ⬅️ [Previous](#three-kinds-of-pair) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

`what-the-adapter-costs-on-a-joint-prompt.png` is not shown because it failed the subject check

**Running a joint prompt through the adapter changes the picture on all four rows** `verified`. Both columns
start from one noise, so a difference across a row comes from the adapter `stated` (the
[grids card](../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/README.md)).

| Row | Base model | Through the adapter |
|---|---|---|
| cat and dog | a cat and a dog | the same two, with extra small animal shapes and an object near their feet |
| tiger and dog | a tiger with a small dog | a tiger, a tiger-coloured animal with a dog's face, a dog and a small extra dog |
| chess board and hourglass | a board of pieces and no hourglass | chess pieces with an hourglass among them |
| picnic basket and watermelon | a basket, watermelon and a bottle | a different basket, with flowers and more items added |

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-the-adapter-does-to-a-joint-prompt) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three recorded statements differ from the pictures or the code, and each is shown here unresolved.**

| What | One source | The other source | What the code shows |
|---|---|---|---|
| how many pairs the published-models grid holds | the card lists cat and dog, butterfly and flower meadow, and camel and forest | the picture and its tile list hold cat and dog only | [the grid script](../../../scripts/shared/main_study_grids.py), lines 165 to 167, lists all three. It was committed at 14:50 on 2026-09-24 and the picture is dated 14:40, so the picture likely predates the three-row version `inferred` |
| what the name "the fix is not about animals" covers | the filename and the card describe three kinds of pair | the adapter tile exists only for the two object-in-a-place rows | the tile list has no adapter path for the other four rows |
| whether a change on a joint prompt is damage | the card calls a difference here "damage outside what it was trained to repair" | on the chess row the adapter's picture holds the hourglass the base render lacks | the code cannot settle this. Whether that row is damage or a gain is a reading of the picture `inferred` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**These grids carry no rate.** Each row is one seed read by eye, and neither the tile lists nor the card
hold a score. So no compose rate from this page can be affected by the comparison-strip fault.

**The three-kinds grid cannot show that the adapter carries across kinds of pair**, because its animal and
object rows have no adapter tile.

**The joint-prompt grid covers 4 cells**, chosen for the figure, so it gives no rate of damage.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. [The correction and its strength](../what-the-correction-is/does-the-correction-cause-composition.md#the-correction-and-its-strength)
is taught on the causation page.

### The same starting noise

Roll one set of dice and hand the same numbers to two cooks, each following a different recipe that uses
those numbers. Any difference in the dishes comes from the recipe. A render starts from a block of random
numbers set by its seed, so two methods on the same seed differ only by the method. Here each row of the
two main comparison grids shares one starting noise, except the CO3 tile.

### The joint prompt and the plain product

The joint prompt is one sentence naming both things, such as "a cat and a dog". The plain
[product of experts](../../../context/world/poe-composition.md#what-poe-composition-is) runs one prompt per
concept and combines their predictions at every step. If one prompt nudges a number by +2 and the other by
+3, the product adds both: +5. The adapter is a small trained add-on meant to repair that product.

### Epochs and steps

If a book has 50 pages and you read it 4 times, you have turned 4 x 50 = 200 pages. An epoch is one pass, and
a step is one page. The grid's footer says 1,000 epochs, and [the grid script](../../../scripts/shared/main_study_grids.py)
(line 25) records an epoch as 50 steps, so 1,000 x 50 = 50,000 steps, the card's step 50,000.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what each grid holds | the three pictures and their tile lists, read 2026-11-22 | `verified` |
| how the grids were built | [the grid script](../../../scripts/shared/main_study_grids.py) and [the render script](../../scripts/figure_candidates.py) | `stated` |
| the adapter and what each grid is for | [the grids card](../../artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/README.md) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Render the adapter tiles for cat and dog at seed 42 and for the four empty rows of the three-kinds grid.
- [ ] Rebuild the published-models grid so it holds the three pairs its script lists.
- [ ] Decide whether the chess row counts as damage or as a gain.

## Cross-references

- The mention of **the joint prompt and the plain product** in [the report on render-time fixes for picture quality](../is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md).
- The mention of **the joint prompt and the plain product** in [the report on how big the strength grid should be](how-big-the-strength-grid-should-be.md).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
