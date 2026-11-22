# 📒 Report: which adapter composes a cat and a dog?

**No verdict has been recorded.** The project rendered one held-out pair, a cat and a dog, with 23
trained adapters at three seeds no adapter trained on, so they could be compared from the same
starting noise. On these sheets the seed decides whether the picture is a photograph or a drawing,
and the two adapters trained on filtered pictures are the exception `verified` (by eye).

> [!WARNING]
> No picture from this result is shown here. Each sheet holds a shape that could not be cleared
> against the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md), and
> [the list below](#sheets-kept-off-this-page) names what could not be told.

## Table of contents

- [What the sheets show](#what-the-sheets-show)
- [Two adapters against the joint prompt](#two-adapters-against-the-joint-prompt)
- [Sheets kept off this page](#sheets-kept-off-this-page)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the sheets show

Navigation: 📋 [TOC](#table-of-contents) | [Next](#two-adapters-against-the-joint-prompt) ➡️

**Most adapters keep the look their seed has with no adapter, and the two trained on filtered pictures
draw a photograph in a green field on all three seeds** `verified` (by eye, matching
[the card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/README.md)).

| Seed | No adapter | Most of the 23 adapters | The two trained on filtered pictures |
|---|---|---|---|
| 9 | one fused orange and white animal on grey | studio pictures or drawings on a grey ground | a photograph in a green field |
| 11 | one pencil-sketch face | pencil or tinted drawings | a photograph in a green field |
| 12 | one fused animal in a photograph | a photograph in dry grass | a photograph in a green field |

Two adapters draw grey mush with no clear animal on seeds 9 and 11: rank 8 at step 450,000, and the
54-cell adapter with the plural-prompt loss term `verified`.

## Two adapters against the joint prompt

Navigation: ⬅️ [Previous](#what-the-sheets-show) | 📋 [TOC](#table-of-contents) | [Next](#sheets-kept-off-this-page) ➡️

**The joint-prompt render, which training aims at, is itself wrong on two of these four cells**
`verified`. Both adapters are at training step 45,000. They are `v57-09-x0-seeds` (W&B `ek4w6lbu`) and
`v57-10-x0-contrast` (W&B `boammtzo`) `stated` (the card).

| Held-out cell | Joint-prompt render | `v57-09-x0-seeds` | `v57-10-x0-contrast` |
|---|---|---|---|
| cat and dog, seed 9 | a tabby and a white dog | a tabby and a white dog | a cat and a dog held by two invented people |
| cat and dog, seed 10 | two dogs | a cat and a dog | a cat and a dog |
| elephant and penguin, seed 9 | an elephant, no penguin | fused elephant and penguin bodies | fused elephant and penguin bodies |
| elephant and penguin, seed 10 | an elephant and a penguin-like bird | an elephant, a penguin and a small extra elephant | an elephant, a penguin and a small extra elephant |

## Sheets kept off this page

Navigation: ⬅️ [Previous](#two-adapters-against-the-joint-prompt) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Eight sheets are not shown.** Paths are relative to
`1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/`.

| File | What could not be told |
|---|---|
| `every-adapter-cat-and-dog-seed-09.png` | a curved dark shape in the 182-cell tile, and small figures at the top of the 72-cell and 54-cell control tiles |
| `every-adapter-cat-and-dog-seed-11.png` | a long, thin line-drawn creature above the animals in the 23-cell tile |
| `every-adapter-cat-and-dog-seed-12.png` | a contorted tan form in the tree of the tile whose empty branch comes from the cache |
| `every-adapter-cat-and-dog-by-seed.png` | holds the same tiles as the three sheets above |
| `every-adapter-cat-and-dog-by-seed-v2.png` | holds the seed 12 tile named above |
| `how-a-render-fails-gallery.png` | a small animal held in the invented-people panel |
| `how-a-render-fails-paper.png` | the same panel |
| `x0-seeds-against-x0-contrast-step-45000.png` | winged shapes in the sky of the lower-right tile |

## Where the sources disagree

Navigation: ⬅️ [Previous](#sheets-kept-off-this-page) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The card's two counts differ from the sheets, and the code agrees with the sheets.**

| What | The card | The sheet | What the code shows |
|---|---|---|---|
| columns on the paper version of the sheet | "10 adapter columns plus the no-adapter control" | 10 columns in all: 9 adapters and the control | `GROUPS_V2`, lines 48 to 57 of [the grouping script](../../../_archive/scripts/across_adapters_by_family.py), lists the same 10 `verified` |
| how many adapters are compared | "Every adapter this project has trained" | 23 adapters on each per-seed sheet | `ROWS` in [the render script](../../../scripts/across_adapters.py) lists 25. Rows 24 and 25 fall after the sheet's 24 tiles, and the card dates row 25 to 2026-09-26, after the sheets of 2026-09-21. The later `v58` adapters are in neither `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Three seeds of one pair give no rate, so these sheets cannot rank the adapters.** Every read is by
eye, and no compose rate is quoted here. Columns differ in more than one setting, so a gap between two
of them is a mixed comparison, as [the adapter settings page](how-the-adapter-settings-change-cat-and-dog.md#what-this-cannot-tell-you)
explains for rank and training steps.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy is invented. The plain product, the joint prompt, the same starting noise and the adapter are
taught in [the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses), and rank in
[the adapter settings page](how-the-adapter-settings-change-cat-and-dog.md#concepts-this-uses).

**Reading a sheet by row and by column**

Two bakers each bake from two recipe cards:

| | Baker A | Baker B |
|---|---|---|
| card 1 | sponge | sponge |
| card 2 | biscuit | sponge |

Along a row the card is fixed, so a difference is the baker. Down a column the baker is fixed, so a
difference is the card. Here a row is one seed and a column is one adapter. The combined sheet has
3 seeds by 24 columns, which is 72 tiles.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what each sheet is and how it was made | [its card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/README.md) | `stated` |
| which adapter is in each column | `ROWS` in [the render script](../../../scripts/across_adapters.py) | `stated` |
| what the tiles show | all eight sheets, opened 2026-11-22 | `verified` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Look at the eight sheets kept off this page and decide whether any may be shown.
- [ ] Correct the card's column count for the paper version of the sheet.
- [ ] Link the comparison behind the card's statement that like-for-like training of the first-10-steps adapter made no significant difference. The card says it is not linked yet.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
