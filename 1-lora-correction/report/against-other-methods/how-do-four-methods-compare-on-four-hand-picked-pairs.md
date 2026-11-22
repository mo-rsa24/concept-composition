# 📒 Report: how four methods compare on the four pairs of the paper's main grid

**No verdict or rate belongs to this grid, because its four cells were picked by hand.** It is the
figure the paper's results section opens with. On all four rows the adapter ("Ours") draws both
named things from the two concept prompts alone. CO3 also draws both on all four, and it reads the
sentence naming both while it draws.

## Table of contents

- [The grid](#the-grid)
- [CO3 on other animal pairs](#co3-on-other-animal-pairs)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The grid

Navigation: 📋 [TOC](#table-of-contents) | [Next](#co3-on-other-animal-pairs) ➡️

> [!NOTE]
> The grid is `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/four-pairs-four-methods.png`,
> not shown: one small figure in its cat and dog row could not be cleared against the subjects the
> pair ledger keeps off sheets.

**The plain product fuses cat with dog and elephant with penguin, and the adapter separates them.**
Rows are cat and dog (seed 9), elephant and penguin (seed 10), tent and snowy mountain (seed 9), and
lighthouse and stormy sea (seed 9). Columns are the joint prompt, the plain product, CO3 and Ours
`verified` (seen on the picture; matches [the card](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/README.md)).

| Row | What each column draws, by eye |
|---|---|
| cat and dog | the plain product draws one animal with cat ears and a dog's muzzle; CO3 and Ours draw a separate cat and dog, and Ours adds small stray objects at the dog's feet |
| elephant and penguin | the plain product puts an elephant inside a penguin's outline; CO3 draws both as clip art; Ours draws both as a photograph |
| tent and snowy mountain | the plain product turns the mountain into a tent-like shape; CO3 and Ours draw both |
| lighthouse and stormy sea | all four columns draw a lighthouse in a stormy sea |

## CO3 on other animal pairs

Navigation: ⬅️ [Previous](#the-grid) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**CO3 draws both animals as separate figures on these three renders, which were kept by eye.**
[The script that pruned them](../../../scripts/local/prune_co3.sh) describes them as "the CO3 renders
chosen by eye" `stated`, so they say nothing about how often CO3 succeeds. No card records their
settings.

| Bison and horse, seed 42 | Tiger and parrot, seed 2 | Wolf and raven, seed 9 |
|---|---|---|
| <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/a_bison_and_a_horse/seed-42/a_bison_and_a_horse.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/a_bison_and_a_horse/seed-42/a_bison_and_a_horse.png" width="200" alt="CO3 render of a bison and a horse on a grass plain"></a> | <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/a_tiger_and_a_parrot/seed-2/a_tiger_and_a_parrot.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/a_tiger_and_a_parrot/seed-2/a_tiger_and_a_parrot.png" width="200" alt="CO3 illustration of a tiger beside a parrot on a branch"></a> | <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/a_wolf_and_a_raven/seed-9/a_wolf_and_a_raven.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/tiles/a_wolf_and_a_raven/seed-9/a_wolf_and_a_raven.png" width="200" alt="CO3 illustration of a wolf facing a raven on a branch"></a> |

> [!NOTE]
> The sheet of all these renders is
> `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/what-co3-does-on-our-pairs/co3-audition.png`,
> not shown: holds a subject the pair ledger keeps off sheets.

## Where the sources disagree

Navigation: ⬅️ [Previous](#co3-on-other-animal-pairs) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The grid's card and its picture differ on the lighthouse row.** No code decides this one, so both
readings stay here for the owner.

<a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/tiles/a_lighthouse__x__a_stormy_sea/seed09/poe.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/tiles/a_lighthouse__x__a_stormy_sea/seed09/poe.png" width="200" alt="Plain product render of a white lighthouse on a stormy sea"></a>

| What | The card | The picture |
|---|---|---|
| the plain product on lighthouse and stormy sea, seed 9 | "the plain product fuses or drops one" across the four cells `stated` | this tile, the grid's plain-product cell for the row, shows a lighthouse and a stormy sea `verified`. [The Langevin sheet's card](../../artifacts/results/against-other-methods/samplers-and-correctors/does-a-langevin-corrector-fix-the-product/README.md) reports both things in every column for this pair at seed 10 `stated` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The four "Ours" tiles come from three different adapters.** Cat and dog uses the cross+self
adapter, elephant and penguin uses v58-05 at step 15,000, and the two object-and-place rows use the
v58 self adapter `stated` (the card, from the `ROWS` list in `scripts/paper_main_grid.py`). The
one-checkpoint comparisons are on [the benchmark page](how-does-the-adapter-score-on-the-attend-and-excite-prompts.md)
and [the held-out page](how-does-every-sdxl-method-compare-on-the-held-out-pairs.md).

**CO3 starts from different noise in this grid.** The other three columns of a row share one cached
starting noise, and CO3 draws its own `stated` (the card). The later review runs start CO3 from the
shared noise ([line 123 of the baseline script](../../scripts/review_baseline_cell.py)).

**This page quotes no [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) or sharpness number**, so neither known measurement fault applies
to it.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

The grid was built by `scripts/paper_main_grid.py` from tiles rendered by
`scripts/figure_candidates.py`. The CO3 tiles in the grid were rendered on 2026-09-25 by
`scripts/showcase/run_co3_targets.sh` (Slurm job 59714) `stated` (the card).

Nineteen pictures are held out of the artifact drain: the main grid, the other tiles of this result and one fidelity sheet. They repeat a tile described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so their folder is not linked here.

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Whether the small figure in the grid's cat and dog row may be shown on a report page.
- [ ] Which reading of the lighthouse row the card should carry.
- [ ] A card for the other CO3 renders, recording the settings they were made with.
- [ ] Where the single CO3 render of a lynx and a crane came from, since no card or script names it.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
