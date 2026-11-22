# 📒 Report: what the adapter ablations show

**Both ablation questions are recorded as inconclusive, and their findings live in the project on
understanding composition.** An ablation compares adapters that differ in a setting. One finding asks
[whether adapting self-attention changes the result](../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/does-adapting-self-attention-change-the-result.md),
and the other asks [how much rank the correction needs](../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md).
This page holds what they leave out: the per-seed render grids, and a mismatch between the rank grids'
pictures and the numbers printed under them.

## Table of contents

- [The two charts](#the-two-charts)
- [The per-seed render grids](#the-per-seed-render-grids)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The two charts

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-per-seed-render-grids) ➡️

**Neither chart orders the settings, and every comparison on them is mixed.** The numbers are read and
explained in the two findings linked above.

| Which layers the adapter covers | How high its rank is |
|---|---|
| <a href="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/which-layers-carry-the-adapter.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/which-layers-carry-the-adapter.png" width="200" alt="Horizontal bars of DINOv2 drift for four adapters on six tracking cells, with the zero line marked."></a> | <a href="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.png"><img src="../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.png" width="200" alt="Two lines of DINOv2 drift against adapter rank 8, 16 and 32, one for cat and dog at seed 9 and one at seed 10."></a> |
| x-axis: DINOv2 drift, lower is nearer the joint-prompt render. y-axis: six cells. One bar is one adapter on one cell. Numbers from [the layers sidecar](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/which-layers-carry-the-adapter.json) `verified` | x-axis: adapter rank. y-axis: DINOv2 drift. One point is one run on one cell. Numbers from [the rank sidecar](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.json) `verified` |

## The per-seed render grids

Navigation: ⬅️ [Previous](#the-two-charts) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Ten grids show what these adapters draw, and none is placed here because their tiles were not
cleared one by one against the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md).**
Each grid is one seed. Rows are five pairs: cat and dog, tiger and dog, cat and fox, chess board and
hourglass, picnic basket and watermelon. Columns are the joint prompt, the plain product and the
adapters. A drift number is printed only under the cat and dog row at seed 9, because the runs logged
drift for that pair at seeds 9 and 10 only and seed 10 has no grid `stated` ([the grid script](../../../_archive/scripts/adapter_ablation_grids.py)).

The files sit in `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/`:

| Files | Seeds |
|---|---|
| `which-layers-carry-the-adapter-seedNN.png` | 02, 09, 11, 12, 42 |
| `how-much-rank-the-correction-needs-seedNN.png` | 02, 09, 11, 12, 42 |

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-per-seed-render-grids) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The code explains the first row. The second and third need files that are not on this laptop, and
the fourth is a stale reference.**

| What | One source | The other source | What the code or files show |
|---|---|---|---|
| which checkpoint a rank grid's drift describes | the column headings name the rendered checkpoints: step 105k, 50k and 30k | the drift under them, -0.252, -0.516 and -0.408, is the [rank finding](../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md)'s value at the end of training: steps 200,000, 100,018 and 100,050 | `drift()` at lines 68 to 72 of [the grid script](../../../_archive/scripts/adapter_ablation_grids.py) reads the value [the extraction script](../../../scripts/local/extract_measures.py) keeps, which is the last one logged. Each tile's sidecar names its own earlier checkpoint. So on the seed 9 rank grid the picture and the number come from different checkpoints of one run `verified` |
| how many cells a [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) was scored on | the self-attention finding says the two held-out cat and dog cells | the rank finding says four cells per run | the trainer scores only pairs that have a saved anchor folder ([line 815](../../../poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py)). That folder and each run's `history.json` are not on this laptop, so the counts cannot be checked. The two findings read different runs, so both may hold `inferred` |
| whether the three rank runs differ in loss | [the card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/README.md) and the rank sidecar's `why_not_clean` say they differ in loss space | the same sidecar's run entries and the [rank finding](../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md) give all three the epsilon-space loss | [the chart script](../../../3-understanding-compositionality/scripts/adapter_ablation_figures.py) writes both: "epsilon loss" for all three at lines 40 to 42, and "differ in loss space" at line 164. The runs' configs are on the cluster, so this stays unsettled |
| whether the paper has this appendix | [the card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/README.md) says the grids are behind the appendix labelled `app:adapter-ablations` | [the current paper source](../../../writing/restoring-plurality-in-compositional-diffusion_overleaf/iclr2027_conference.tex) has no such label and names neither chart | the paper source is the current text `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**No gap between two adapters here can be credited to layers or rank.** The layers comparison also
differs in learning-rate decay and in a resume from step 5,000. The rank comparison also differs in
training length.

**The compose rate of 1.0 in both findings came from the trainer, which scored the render alone.**
Lines 779 to 791 of the trainer at commit `e289646` pass only the file
`<quadrant>__<pair>__seed<NN>.png`, so no `__cmp` comparison strip was scored `verified`. That every
run used this code is `inferred`, because the runs' logs are on the cluster. This score is 1 when the
render's DINOv2 embedding is nearer the joint-prompt anchor than either single-prompt anchor
([the scorer](../../../poe_repair/experiments/compose_scorer_validation/scorer.py)). It does not count
animals.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The plain product, the joint prompt and the adapter are taught in
[the four-pairs page](does-one-adapter-compose-four-pairs.md#concepts-this-uses), and rank in
[the adapter settings page](how-the-adapter-settings-change-cat-and-dog.md#concepts-this-uses).

**DINOv2 drift**

Suppose a corrected render is 0.2 away from the joint-prompt render and 0.5 away from the plain
product. Its drift is 0.2 - 0.5 = -0.3. Negative means it moved toward the joint prompt, and 0 means
the correction bought nothing.

| | Distance from the corrected render |
|---|---|
| to the joint-prompt render | 0.2 |
| to the plain product | 0.5 |
| drift, first minus second | -0.3 |

"DINOv2" is the image model that turns each picture into a list of numbers. "Distance" is
[cosine distance](../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this),
one minus the cosine of the angle between two such lists, so it runs from 0 to 2 and drift runs from
-2 to +2. Rank 16 on cat and dog at seed 9 reads -0.516 by this subtraction. The sidecar keeps only
the result, so its two distances cannot be shown.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| what each chart shows | [its card](../../artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/README.md) and the two charts, opened 2026-11-22 | `verified` |
| the checkpoint behind each rank tile | the sidecars `ours_rank8best_w50.json`, `ours_rank16best_w50.json` and `ours_rank32best_w50.json` for cat and dog at seed 9, in the checkpoint sweep's tile folder | `verified` |
| how the grids were built | [the grid script](../../../_archive/scripts/adapter_ablation_grids.py) | `stated` |
| what was opened | both charts and six of the ten grids (layers at seeds 9, 11, 42; rank at seeds 2, 9, 12) | `verified` |
| 4 pictures held out of the drain | the raw tiles, build history and repeat layouts in this result's folder. They repeat a picture or sheet described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide whether the rank grids should print the drift of the checkpoint they show. That value would have to be read from each run's `history.json` on the cluster.
- [ ] Settle whether a compose rate was scored on two cells or four, from the same logs.
- [ ] Settle from each run's config whether the three rank runs share one loss.
- [ ] Clear the ten render grids tile by tile if any should be shown.
- [ ] Repoint the card's paper link. It names a path and an appendix label the current paper source does not have.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
