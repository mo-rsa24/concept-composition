# 📒 Report: which checkpoint to take each figure cell from

**No verdict was recorded, and no file names the checkpoint chosen.** On 2026-09-24 a sweep drew saved
checkpoints of eight training runs on 13 pairs, so a person could pick by eye which training step to
use for each picture in a figure. The result is one contact sheet per run and seed group. This folder
has no card, and `BUILDS.md` is its only written record.

## Table of contents

- [How to read one sheet](#how-to-read-one-sheet)
- [What the final sheets hold](#what-the-final-sheets-hold)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## How to read one sheet

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-final-sheets-hold) ➡️

<a href="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/history/20260924-1455/which-checkpoint-v58-01-plain.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/history/20260924-1455/which-checkpoint-v58-01-plain.png" width="200" alt="An early sheet with three rows and six columns: joint prompt, plain product and step 10,000, for cat and dog, chess board and hourglass, and dog and dog. Most step 10,000 cells read not rendered."></a>

**Each row is a checkpoint and each column is one pair at one seed, with two reference rows on top.**
This is the first saved state of the `v58-01-plain` sheet, at 14:55, when one adapter picture had
landed. The first row is the joint prompt the correction aims at, the second is the plain product it
repairs, and an empty cell reads "not rendered" `verified`.

## What the final sheets hold

Navigation: ⬅️ [Previous](#how-to-read-one-sheet) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The sweep finished seeds 9 and 42 for six of its eight runs, and the other seed groups stopped part
way.** One seed group is 13 pairs at 2 seeds, so 26 pictures per checkpoint.

| Run | Checkpoints drawn (training step) | Seeds 9, 42 | Seeds 3, 5 | Seeds 13, 15 | Seeds 43, 45 | Full group |
|---|---|---|---|---|---|---|
| `refl-03-self-contrast-15k` | 250 | 16 | 16 | 0 | 9 | 26 |
| `v58-01-plain` | 10,000, 20,000, 30,000, 40,000, 47,500 | 130 | 39 | 39 | 0 | 130 |
| `v58-02-contrast` | 10,000, 20,000, 27,500 | 78 | 36 | 39 | 0 | 78 |
| `v58-03-self` | 10,000, 20,000, 30,000, 40,000, 50,000 | 130 | 25 | 0 | 0 | 130 |
| `v58-04-freezenull` | 10,000, 20,000, 30,000, 40,000, 45,000 | 130 | 37 | 0 | 0 | 130 |
| `v58-05-self-contrast` | 10,000, 20,000, 30,000, 37,500 | 104 | 36 | 33 | 0 | 104 |
| `v58f-01-plain` | 10,000, 20,000 | 26 | 34 | 10 | 0 | 52 |
| `v58f-05-self-contrast` | 10,000 | 26 | 8 | 0 | 0 | 26 |

Each count is the number of files named `ours_<run>_s<step>_w50.png` in the sweep's tile folder for
that run and seed group. They add up to 1,001, the last probe-tile count in `BUILDS.md` `verified`.

> [!NOTE]
> The 32 final sheets in `latest/` are not shown. All 32 were opened, but at 150 pixels a tile some
> shapes on the chess board and saxophone cells could not be cleared against the
> [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md). The tile and
> snapshot folders are not linked, because a reader clicking through would see tiles nobody checked.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-the-final-sheets-hold) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The commits explain the first row, and the second shows the sweep cannot be the whole record of how
the review's candidates were chosen.**

| What | One source | The other source | What the code or files show |
|---|---|---|---|
| how many runs each build held | `BUILDS.md` reads 7 runs until 18:14, then a broken cell holding "0" twice from 21:34 | `latest/` holds sheets for 8 runs | commit `c174299e` at 21:28 changed the sheet script's output to "checkpoints, seeds". [The watcher](../../../scripts/local/sweep_watch.sh) kept counting "checkpoints ->" until commit `47db12e9` at 23:34, after the last row at 23:19. With no match, `grep -c` prints 0 and the `\|\| echo 0` fallback prints a second 0 `verified` |
| whether the review's candidates come from this sweep | [the protocol](../paper-review/protocol.md) lists nine candidate checkpoints | these sheets hold four of them: `v58-03-self` at 30,000 and 50,000, `v58-01-plain` at 47,500 and `v58-02-contrast` at 27,500 | `v58-03-self` at 25,000 and 27,500, `v58-05-self-contrast` at 15,000, `v58f-05-self-contrast` at 25,000 and `v57-11b-x0-self-decay` have no row here `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The sheets show what each checkpoint drew and do not say which one was judged best.** The sheet
script writes `shortlist_template.json` to be cut down by hand, and no cut-down shortlist was found in
this repository. No score is attached to any tile, and no compose rate is quoted here. The run names
are not defined in this folder.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The joint prompt, the plain product and the adapter are taught in
[the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses). The learning tree
has no page on a training checkpoint in this sense.

**A checkpoint and a probe cell**

A cook saves the recipe after every 10 tastings, so tastings 10, 20 and 30 are three saved versions.
Each is a checkpoint: the model's weights as they stood at that training step. To compare them, cook
the same 2 dishes from the same 2 batches of ingredients. One dish from one batch is a probe cell.

| | Dish 1, batch 1 | Dish 1, batch 2 | Dish 2, batch 1 | Dish 2, batch 2 |
|---|---|---|---|---|
| version 10 | plate | plate | plate | plate |
| version 20 | plate | plate | plate | plate |
| version 30 | plate | plate | plate | plate |

That is 3 versions by 4 cells, or 12 plates. On this project's sheets, 13 pairs at 2 seeds make 26
cells, so `v58-03-self` with 5 checkpoints needs 5 × 26 = 130 pictures per seed group.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| how the sheets were built | [the sheet script](../../../scripts/sweep_sheets.py) and [the watcher](../../../scripts/local/sweep_watch.sh) | `stated` |
| the build history | `BUILDS.md` in this result's folder, 109 rows from 14:55 to 23:19 on 2026-09-24 | `verified` |
| the steps each run has | [the shortlist template](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-checkpoint-to-take-each-figure-cell-from/latest/shortlist_template.json) | `verified` |
| what was opened | all 32 final sheets, 5 tiles and 2 snapshot sheets, 2026-11-22 | `verified` |
| 2,845 pictures held out of the drain | the raw tiles, build history and repeat layouts in this result's folder. They repeat a picture or sheet described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Write this folder's card, defining each run and saying which checkpoint was taken for which figure cell.
- [ ] Decide whether any of the 32 final sheets may be shown, after clearing their tiles one by one.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
