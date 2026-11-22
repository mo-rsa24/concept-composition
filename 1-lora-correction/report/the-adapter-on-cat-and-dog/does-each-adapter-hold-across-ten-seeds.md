# 📒 Report: does each adapter hold across ten seeds?

**No verdict and no rate have been recorded.** Seven grids render seven pairs at ten seeds with
fourteen adapters and three reference methods, so a method can be read across seeds instead of on
one lucky render. They are for looking at, and nothing has scored them.

## Table of contents

- [What the grids hold](#what-the-grids-hold)
- [What the three opened grids show](#what-the-three-opened-grids-show)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the grids hold

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-three-opened-grids-show) ➡️

**Each grid is 10 rows of seeds by 17 columns of methods, which is 170 renders per pair.** No picture
is embedded here, because a tile at this size cannot be cleared against the
[pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md).

| Part of a grid | What it is | Mark |
|---|---|---|
| rows | seeds 1, 2, 4 and 9 to 15 | `verified` (each grid's JSON file) |
| first three columns | joint prompt, plain product, CO3 | `verified` |
| other 14 columns | one adapter each, the list `ADAPTERS` in [the grid script](../../../scripts/seed_method_grid.py) | `verified` |
| pairs | elephant and penguin, cat and dog, tiger and dog, cat and teapot, horse and bicycle, picnic basket and watermelon, camel and forest | `stated` ([the card](../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/README.md)) |
| training | none of the seven is in the training list of the [v58 pair pool](../../../artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml) | `verified` |

> [!WARNING]
> The short summary sheet stays off this page.
> `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/five-picked-pairs.png`
> is not shown: in its camel and forest adapter tile, small dark flecks in the air cannot be told
> apart from insects at the 300 pixels the sheet holds. The seven grids stay off for the same reason.

🖼️ **Figure wanted**: the five picked pairs with the camel and forest tile checked at full size by
the owner, or replaced, then embedded here.

## What the three opened grids show

Navigation: ⬅️ [Previous](#what-the-grids-hold) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Along a row, most adapter columns keep the same layout, because they start from the same noise.**
The cat and dog, elephant and penguin, and horse and bicycle grids were opened whole, at about 105
pixels per tile. That size is enough to see the layout and too small to call composition.

| Grid | What repeats along its rows `verified` |
|---|---|
| cat and dog | adapter columns share a scene per seed; an invented person appears in many adapter columns at seeds 11 and 14 |
| elephant and penguin | the scene holds per seed across adapter columns |
| horse and bicycle | an invented man appears in many adapter columns at seed 14, and in at least one at seed 15 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#what-the-three-opened-grids-show) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**CO3's column cannot be compared pixel by pixel with the rest.** CO3 draws its own noise from the
same seed number, while every other column starts from the cached noise of `a_cat__x__a_dog` at that
seed `stated` (the card). The checkpoint also varies by adapter: two use the steps the training
notes name as good and the others use their last checkpoint. R1 renders with the two experts weighted
3.0 and 3.0, so its column differs on a second setting.

The five picked pairs are one hand-picked seed each and use the reward fine-tuned adapter K0 at
2,000 steps. They were rendered on 2026-09-26, before [the protocol](../paper-review/protocol.md)
fixed one adapter and a rule for picking images on 2026-11-10.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The plain product, the joint prompt, the same starting noise and the adapter are taught in
[the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses).

**Seen seeds and unseen seeds**

If you practise a recipe with eggs from boxes 1 to 8 and test it with boxes 9 to 16, the second test
uses eggs the practice never touched. Here seeds 1 to 8 were training seeds for other pairs, and seeds
9 to 16 were never trained on `stated` (the card). A grid's rows 1, 2 and 4 use the first kind and its
rows 9 to 15 use the second.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| how the grids were made, 2026-09-26 | [the card](../../artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-each-adapter-hold-across-ten-seeds/README.md) and [the grid script](../../../scripts/seed_method_grid.py) | `stated` |
| rows and columns | the JSON file beside each grid, read 2026-11-22 | `verified` |
| what repeats along rows | three grids opened 2026-11-22 | `verified` |
| 7 pictures held out of the drain | the other tiles and layouts in this result's folder. They repeat a picture placed above, or could not be cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Check the camel and forest tile at full size, and decide whether the summary sheet can be shown.
- [ ] Open the four grids not looked at here: tiger and dog, cat and teapot, picnic basket and watermelon, camel and forest.
- [ ] Decide whether these grids should be scored. No compose rate exists for them.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
