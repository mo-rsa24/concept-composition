# 📒 Report: how were the adapter's training pairs chosen?

**A person kept a pair when the joint prompt drew both named things as separate objects, and the final list
holds 48 pairs at 182 pair-and-seed cells over 23 seeds.** The rule is `stated` in
[the paper's source](../../../writing/restoring-plurality-in-compositional-diffusion/iclr2027_conference_revision.tex),
lines 710 and 1121, and the counts are `verified` in [the training cell list](../../../artifacts/_shared/cross_pair_pool_configs/cells_v58.json).
No verdict was recorded for this question, and the rule has no numeric pass mark.

## Table of contents

- [What a kept and a dropped pair look like](#what-a-kept-and-a-dropped-pair-look-like)
- [The round that added nineteen pairs](#the-round-that-added-nineteen-pairs)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What a kept and a dropped pair look like

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-round-that-added-nineteen-pairs) ➡️

**The joint prompt's picture is the training target, and CO3's picture is at most a second view.**

| Kept: joint prompt | Second view: CO3, same pair and seed | Dropped: joint prompt |
|---|---|---|
| <a href="../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles/a_horse__x__a_pig__seed_1.png"><img src="../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles/a_horse__x__a_pig__seed_1.png" width="200" alt="A dark brown horse standing behind a pink pig in a field."></a> | <a href="../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles/co3__a_horse__x__a_pig__seed_1.png"><img src="../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles/co3__a_horse__x__a_pig__seed_1.png" width="200" alt="A chestnut horse with a pale mane standing behind a pink pig."></a> | <a href="../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles/a_crow__x__a_raven__seed_1.png"><img src="../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles/a_crow__x__a_raven__seed_1.png" width="200" alt="Two similar black birds perched on one branch."></a> |
| Horse and pig, seed 1: both animals, separate. This cell is in the training list `verified` | The same pair from CO3, another composition method. No CO3 picture is a training target `stated` | Crow and raven, seed 1: two black birds. The paper drops the pair because they are hard to tell apart `stated` |

The folder holds 117 such pictures, 91 from the joint prompt and 26 from CO3, over 23 pairs `verified`. Three
sheets lay them out: [kept beside left out](../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/how-training-targets-are-chosen.pdf),
[nine example targets](../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/training-target-examples.pdf)
and [the joint prompt beside CO3 on five pairs](../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/joint-prompt-beside-co3.pdf).

## The round that added nineteen pairs

Navigation: ⬅️ [Previous](#what-a-kept-and-a-dropped-pair-look-like) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**One screening round on 2026-09-23 drew candidate pairs with the joint prompt alone, and 19 of them are in
the final training list.** No adapter ran: all 185 saved records have an empty checkpoint field `verified`.

| Seed 2: in the training list | Seed 11: an evaluation seed | Seed 42: in the training list |
|---|---|---|
| `mono.png` is not shown because it failed the subject check | <a href="../../artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles/a_badger__x__a_magpie/seed11/mono.png"><img src="../../artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles/a_badger__x__a_magpie/seed11/mono.png" width="200" alt="Two badgers on grass with no magpie."></a> | <a href="../../artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles/a_badger__x__a_magpie/seed42/mono.png"><img src="../../artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles/a_badger__x__a_magpie/seed42/mono.png" width="200" alt="A painted badger in long grass with two magpies on a branch."></a> |
| Badger and magpie, both present `verified` | Two badgers and no magpie `verified` | A badger and two magpies `verified` |

The same prompt can draw both things on one seed and miss one on another, which is why a cell is kept per
seed. Seeds 9 to 16 are never trained on, so seed 11 could not have entered training whatever it drew `stated`.

| Stage of the round | Pairs | Pictures | How it is known |
|---|---|---|---|
| drawn onto the contact sheet | 73 | 207 | rows and filled cells counted on the sheet `verified` |
| no longer in the tile folders | 7 | 22 | sheet minus folders; each names a subject on the [subject guard's lists](../../../scripts/shared/disallowed_subjects.py) `verified` |
| left in the tile folders | 66 | 185 | counted `verified` |
| of those, in the final training list | 19 | 28 at training seeds | every one at seed 2 or 42, in [the training cell list](../../../artifacts/_shared/cross_pair_pool_configs/cells_v58.json) `verified` |
| of those, on the held-out list | 2 | | piano and candle, skateboard and helmet, in [the pair pool](../../../artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml) `verified` |
| of those, in neither list | 45 | | counted `verified` |

The contact sheet is not embedded here because it still shows the 7 pairs the guard holds back.

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-round-that-added-nineteen-pairs) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded statements conflict, and the record explains each one.**

| What | One source | The other source | What the code or history shows |
|---|---|---|---|
| whether the screened pairs were ever trained on | [the screening card](../../artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/README.md) calls them pairs the adapter never trained on | 19 of the 66 are in the final training list | the previous list (`cells_v57.json`) holds none of the 66, and commit `155f6105` adds the new pairs on seeds 2 and 42. The card's "adapter" being the one before this list is `inferred` |
| how many pairs the round added | commit `155f6105` says 20 pairs, 49 training pairs, 184 cells | the list now holds 19 new pairs, 48 and 182 | commit `1c9264c1` removed one pair, on a subject the guard holds back, the same day |
| what a green frame means | the hidden caption for the joint-beside-CO3 sheet: a picture used as a training target (paper source, line 1142) | the shown caption for the screening figure: the clearest picture for that seed (line 1133) | they are two figures with one layout. The screening figure puts green on CO3 pictures at seed 3, and CO3 pictures are never targets, so its green cannot mean target `verified` |
| whether bison and horse is held out | commit `155f6105` adds it to the held-out list, and [the frozen evaluation set](../paper-review/held-out-set.md) calls it held out at training | [the pair pool](../../../artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml) has not listed it since commit `08d27d9b` (2026-10-21) | it is absent from the training cell list, so it was never trained on under either reading `verified`. The commit gives no reason for the removal |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Why any one pair was kept or left out.** The per-pair calls are not stored beside these pictures, so the
45 screened pairs outside both lists have no recorded reason here.

**Whether every kept seed was checked.** The paper says not every kept seed was checked on its own, and that a
few targets show only one of the two things `stated`.

**Whether this selection helps or hurts transfer.** Keeping only pairs the joint prompt can draw shapes what the
adapter learns, and no experiment here varies that choice.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The compose rate is taught on [the sibling page on transfer](does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md#concepts-this-uses).

**A joint prompt**

Ask for "a cup and a spoon" in one sentence and the model gets one instruction naming both. That sentence is
the joint prompt: "joint" because both things share it. The alternative asks for "a cup" and "a spoon"
separately and combines the two predictions, which is [product-of-experts composition](../../../context/world/poe-composition.md#what-poe-composition-is).
The adapter trains to make the separate version land where the joint prompt lands.

**A pair-and-seed cell**

Two pairs at three seeds each make 2 x 3 = 6 cells, and a cell is one row of that grid: one pair, one seed, one
starting noise. Keeping cells lets a pair enter on its good seeds only. Here 48 pairs hold 182 cells, so a pair
averages 182 / 48 = 3.8 seeds.

**Seeds kept for evaluation**

If seed 5 trained the model on any pair, a test at seed 5 would reuse noise it has seen, because the cached
starting noise for a seed is the same across pairs `stated` (paper, line 1121). So seeds 9 to 16 are kept out of
training. The 23 training seeds are 1 to 8, 17 to 22, and nine others from 26 to 58 `verified`.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the training cells | [the training cell list](../../../artifacts/_shared/cross_pair_pool_configs/cells_v58.json), read 2026-11-22 | `verified` |
| how the screening pictures were made | [the candidate renderer](../../scripts/figure_candidates.py) and each picture's saved settings, read 2026-11-22 | `verified` |
| how the contact sheet was built | [the sheet builder](../../../scripts/local/build_sheet.py), read 2026-11-22 | `verified` |
| the selection rule | [the paper source](../../../writing/restoring-plurality-in-compositional-diffusion/iclr2027_conference_revision.tex), lines 710 and 1121 | `stated` |
| decisions on held-back subjects | [the pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md), read 2026-11-22 | `stated` |
| 296 tiles held out of the drain | 182 in [the new-pair tiles](../../artifacts/results/does-it-reach-unseen-pairs/the-transfer/which-new-pairs-compose-cleanly/tiles) and 114 in [the training-pair tiles](../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/how-the-training-pairs-were-chosen/tiles), all in the same joint-prompt layout as the tiles and sheets linked above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide whether the contact sheet stays, since it still shows 7 pairs the subject guard holds back.
- [ ] Write the missing cards for the training-pair folder and the pair list. The [card for the pairs](../../artifacts/results/does-it-reach-unseen-pairs/the-pairs/README.md) says "no card yet" on both rows.
- [ ] Correct the screening card's "never trained on" for the 19 pairs now in the training list.
- [ ] Record why bison and horse left the held-out list.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
