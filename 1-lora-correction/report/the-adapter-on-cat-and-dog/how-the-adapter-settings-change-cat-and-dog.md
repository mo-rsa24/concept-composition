# 📒 Report: how the adapter's settings change a cat and a dog

**No verdict has been recorded.** These are the tiles behind the paper's adapter settings figure and
its limitations figure. The first varies the adapter's rank and the steps it was trained on, for a
cat and a dog at three held-out seeds. The second shows what the adapter does to a joint prompt and
how a corrected render can go wrong.

## Table of contents

- [Rank and training steps at seed 12](#rank-and-training-steps-at-seed-12)
- [How a corrected render goes wrong](#how-a-corrected-render-goes-wrong)
- [What the adapter does to a joint prompt](#what-the-adapter-does-to-a-joint-prompt)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Rank and training steps at seed 12

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-a-corrected-render-goes-wrong) ➡️

**At seed 12 every adapter separates the two animals more than the plain product does, and rank 8 is
the haziest** `verified` (by eye). All tiles start from the same noise. Rank 16 is left out of the row
because it looks almost the same as rank 32 on each of the three seeds.

| No adapter | Rank 8 | Rank 32 |
|---|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c0.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c0.png" width="200" alt="One fused white and brown animal lying down."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c1.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c1.png" width="200" alt="Two hazy pale animals in dry grass."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c3.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c3.png" width="200" alt="A white fluffy cat beside a mottled animal with a red collar."></a> |
| one fused animal | two animals, hazy | two animals, the second mottled |

| Trained on first 10 steps | Trained on first 25 steps | Trained on all 50 steps |
|---|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c4.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c4.png" width="200" alt="A white fluffy animal beside a black, brown and white cat-like animal with a red collar."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c5.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c5.png" width="200" alt="A smooth white animal with a dog-like snout beside a black and white cat."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c6.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c6.png" width="200" alt="A large white animal with a long snout beside a black and white cat."></a> |
| two animals | two animals, the white one smooth | two animals, the white one smooth |

Each column is one trained adapter, listed in `ROWS` in [the render script](../../../scripts/across_adapters.py) `stated`:

| Column | Training run | Checkpoint step |
|---|---|---|
| rank 8 | `phase1_r8_450k` | 450,000 |
| rank 16 | `phase1_r16_100k` | 30,000 |
| rank 32 | `phase1_r32_100k` | 30,050 |
| first 10 steps | `v57-09-x0-seeds`, which also differs in loss and seeds | 40,000 |
| first 25 steps | `pool43-early25` | 40,000 |
| all 50 steps | `pool43-all50` | 40,000 |

## How a corrected render goes wrong

Navigation: ⬅️ [Previous](#rank-and-training-steps-at-seed-12) | 📋 [TOC](#table-of-contents) | [Next](#what-the-adapter-does-to-a-joint-prompt) ➡️

**Five of the figure's six cat and dog outcomes are shown, each named by the
[figure script](../../scripts/figure7_limitations.py).** Red boxes on a tile are zoomed insets. The
sixth, labelled "Invented people", is `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c1.png`,
not shown because the animal in it could not be identified against the pair ledger.

| One fused animal | Grey mush, no animal in it | Flat line art of one animal |
|---|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c0.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c0.png" width="200" alt="One orange and white face mixing a cat and a dog, with a zoomed inset."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c2.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r0-c2.png" width="200" alt="A grey-brown haze with only a faint animal outline, with a zoomed inset."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c0.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c0.png" width="200" alt="A black line drawing of one cat-like face, with a zoomed inset."></a> |

| A third animal in the corner | Two animals of the wrong species |
|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c1.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c1.png" width="200" alt="A tabby cat and a white dog with a small third cat at the lower right, with zoomed insets."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c2.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c2.png" width="200" alt="A brown horse and a small white animal on grass, with a zoomed inset."></a> |

## What the adapter does to a joint prompt

Navigation: ⬅️ [Previous](#how-a-corrected-render-goes-wrong) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Run on the joint prompt itself, the adapter adds an hourglass the base model left out and adds
extra animals to tiger and dog.** The top row is the base model and the bottom row has the adapter
attached, from the same noise. The figure's note says "Adapter trained for 1,000 epochs" and does not
name the adapter `stated` ([the layout file](../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/figure7-limitations-spec.json)).

| | Tiger and dog | Chess board and hourglass |
|---|---|---|
| Base model | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r0-c1.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r0-c1.png" width="200" alt="A large tiger with a small black and white dog in front of it."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r0-c2.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r0-c2.png" width="200" alt="A flat chess board crowded with pieces and no hourglass."></a> |
| With the adapter | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r1-c1.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r1-c1.png" width="200" alt="A tiger, a tiger-striped dog, a pale dog and a small dog at the right edge."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r1-c2.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/joint-r1-c2.png" width="200" alt="Chess pieces on a board with a glass hourglass among them."></a> |

The cat and dog tile of this row, `limitations-tiles/joint-r1-c0.png`, is not shown: a coiled shape
in a shell at its lower edge could not be identified against the pair ledger.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-the-adapter-does-to-a-joint-prompt) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The code and a pixel comparison explain the first two rows below, and the third is a stale note.**

| What | One source | The other source | What the code or files show |
|---|---|---|---|
| what "first 10, first 25, all 50" means | the paper's caption says the correction runs "during the first 10, first 25, or all 50 denoising steps" and "after the adapter stops, the PoE completes the remaining steps" | the figure's heading says "Steps it trains on" | [the render script](../../../scripts/across_adapters.py) sets 50 adapter steps at line 92, passes that window at line 113 and stops at line 116 unless the adapter was on for exactly 50. The first-25 tiles match the saved tiles numbered 09 in `existing/sheet-tiles-seed-NN/`, with a mean grey-level difference of 0.48, 0.90 and 0.52 out of 255 at seeds 9, 11 and 12 (both images resized to 256 by 256) `verified`. Number 09 is the script's row for `pool43-early25`, so those tiles are that script's output and the window is the training range `inferred`. That the first-10 and all-50 tiles were drawn the same way follows from the [grouping script](../../../_archive/scripts/across_adapters_by_family.py), lines 51 to 54 `inferred` |
| whether the first limitations tile has the adapter | the layout file `figure6b-limitations-spec.json` titles the group "product with the adapter" | the paper's caption calls the top-left tile "the uncorrected hybrid" | outside its insets the tile matches the plain product at seed 9, with a mean grey-level difference of 1.19 out of 255 (both resized to 300 by 300, region x 170 to 300, y 215 to 300). The caption's reading fits `inferred` |
| whether the limitations appendix is in the paper | [the evidence index](../paper-evidence-index.md) says it "is commented out" | [the paper source](../../../writing/restoring-plurality-in-compositional-diffusion_overleaf/iclr2027_conference.tex) has the subsection and its figure live at lines 1359 to 1369 | the paper source is the current text `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Both groups of columns are mixed comparisons.** The rank columns differ in checkpoint step as well
(450,000 for rank 8 against about 30,000 for the others). The first-10 column also differs in loss
and seeds, as the [grouping script](../../../_archive/scripts/across_adapters_by_family.py) says.
So no column-to-column gap here can be credited to rank or training steps alone. Three seeds of one
pair give no rate. For rank measured another way, see
[how much rank the correction needs](../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The plain product, the joint prompt, the same starting noise and the
adapter are taught in [the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses).

**Rank**

A table of 9 numbers can sometimes be rebuilt from 6. The column (1, 2, 3) times the row (1, 0, 2) gives
the rows (1, 0, 2), (2, 0, 4) and (3, 0, 6). That table has rank 1, because one column-and-row pair
builds it. Rank r means r such pairs added together, so a higher rank can hold more patterns and
costs more numbers. The adapter's [low-rank update](../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md)
uses this form with r = 8, 16 or 32.

**A training window and a drawing window**

A pianist can practise only the first 10 bars and still play all 50 at the concert. The training
window is what the adapter practised, set by `--train-step-range` in
[the trainer](../../../poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py). The drawing
window is when it is switched on during a render. In this figure the first varies and the second
stays at all 50 steps.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what this folder holds | [its card](../../artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/README.md) | `stated` |
| column headings and row seeds | `figure6-spec.json` in this result's folder | `verified` |
| which run each column is | [the render script](../../../scripts/across_adapters.py) and [the grouping script](../../../_archive/scripts/across_adapters_by_family.py) | `stated` |
| outcome names | [the limitations figure script](../../scripts/figure7_limitations.py) | `stated` |
| the two pixel comparisons | computed 2026-11-22 as the mean absolute difference of greyscale pixels, on the sizes and regions given in the table | `verified` |
| 17 pictures held out of the drain | the other tiles and layouts in this result's folder. They repeat a picture placed above, or could not be cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Choose whether the paper's settings caption changes. It describes a drawing window, and the render script used a training window.
- [ ] Name the adapter and the source of the limitations tiles, which no file records.
- [ ] Decide whether the coiled shape in the shell and the animal in the invented-people tile may be shown. Both are in the paper's figures.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
