# 📒 Report: how the adapter scores against other methods on the 276 Attend-and-Excite prompts

**The one comparison with a rule written before scoring has the recorded verdict SUPPORTS: the
adapter scores above plain product of experts on all three prompt sets.** [Product of experts](../../../context/world/poe-composition.md#what-poe-composition-is)
(PoE) draws two concepts from their two separate prompts, and often fuses them or drops one.
[The adapter](../../../context/world/lora-corrector.md) ("Ours" in the tables) is a small set of
trained weights that corrects PoE. Every number here uses one checkpoint, `v58-03-self` at training
step 30,000, on all 50 sampling steps.

## Table of contents

- [The scores by method and prompt set](#the-scores-by-method-and-prompt-set)
- [The comparison with a rule written first](#the-comparison-with-a-rule-written-first)
- [What a high and a low cell look like](#what-a-high-and-a-low-cell-look-like)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The scores by method and prompt set

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-comparison-with-a-rule-written-first) ➡️

**The adapter sits above PoE and below ToMe on BLIP-VQA in every set.** It is also below the
joint prompt and CO3 on the two sets with animals, and above both on objects. The first three number
columns are mean [BLIP-VQA](#blip-vqa), 0 to 1, over four seeds (9 to 12) per prompt. The last is
the [share of the 264 animal images where a detector counted two or more animals](#the-instance-count) `verified`
([the Table 1 result file](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/main_table.csv)).

| Method | Animals, 66 prompts | Animals with objects, 144 | Objects, 66 | Two or more animals counted |
|---|---|---|---|---|
| Joint prompt, one sentence naming both | 0.7067 | 0.8789 | 0.4806 | 0.8902 |
| PoE | 0.2251 | 0.5453 | 0.4524 | 0.0189 |
| CO3 | 0.7437 | 0.8688 | 0.4185 | 0.9394 |
| ToMe | 0.8059 | 0.8778 | 0.7020 | 0.9621 |
| PoE + Langevin | 0.2830 | not scored | not scored | 0.0492 |
| Ours | 0.5031 | 0.6753 | 0.5451 | 0.9659 |

Joint prompt has 575 of its 576 animals-with-objects images scored on BLIP-VQA, and PoE has 572
`verified` ([the tables' sidecar](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tables_sidecar.json)).

## The comparison with a rule written first

Navigation: ⬅️ [Previous](#the-scores-by-method-and-prompt-set) | 📋 [TOC](#table-of-contents) | [Next](#what-a-high-and-a-low-cell-look-like) ➡️

**The [95% interval](#an-interval-from-resampling-twice) on Ours minus PoE sits above 0 in all
three sets, which is what [the protocol's rule](../paper-review/protocol.md#the-bar-for-r1-fixed-before-any-evaluation-image-is-scored)
calls SUPPORTS.** The difference is in BLIP-VQA, paired on the same prompt, seed and starting noise.
The right-hand columns repeat it on the prompts that share no concept with the [48 training pairs](../paper-review/held-out-set.md#the-48-training-pairs).
No verdict was recorded for that subset `verified`
([the subset file](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/no_overlap_tables.json)).

| Prompt set | Prompts | Ours minus PoE | 95% interval | Prompts, no shared concept | Ours minus PoE | 95% interval |
|---|---|---|---|---|---|---|
| Animals | 66 | 0.278 | 0.1997 to 0.3551 | 28 | 0.2689 | 0.1621 to 0.3723 |
| Animals with objects | 143 | 0.1299 | 0.0726 to 0.1871 | 88 | 0.1722 | 0.0982 to 0.2434 |
| Objects | 66 | 0.0927 | 0.0287 to 0.158 | 55 | 0.1023 | 0.0318 to 0.1726 |

## What a high and a low cell look like

Navigation: ⬅️ [Previous](#the-comparison-with-a-rule-written-first) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**These two rows are the adapter's highest and lowest BLIP-VQA cells in the objects set, chosen by
[protocol rule 5](../paper-review/protocol.md#what-each-rule-means-in-practice) from the scores.**
Each caption gives the image's BLIP-VQA from the saved per-image score file `verified`. What each
picture shows is an eye read.

| PoE | CO3 | Ours |
|---|---|---|
| <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_yellow_bow__x__a_pink_bowl__s10__poe.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_yellow_bow__x__a_pink_bowl__s10__poe.png" width="200" alt="PoE render of a pink bowl with a yellow bow on it"></a><br>Yellow bow and pink bowl, seed 10: both drawn, 0.9540 | <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_yellow_bow__x__a_pink_bowl__s10__co3.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_yellow_bow__x__a_pink_bowl__s10__co3.png" width="200" alt="CO3 render of a yellow bowl with a pink bow, the two colours swapped"></a><br>The colours are swapped, 0.1899 | Not shown: one shape in this tile could not be cleared against the subjects the pair ledger keeps off sheets. The file is `tiles-rule5/a_yellow_bow__x__a_pink_bowl__s10__ours.png`. Its score is 0.9828 |
| <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_black_backpack__x__a_green_bow__s10__poe.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_black_backpack__x__a_green_bow__s10__poe.png" width="200" alt="PoE render of a green bow with no backpack"></a><br>Black backpack and green bow, seed 10: no backpack, 0.0078 | <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_black_backpack__x__a_green_bow__s10__co3.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_black_backpack__x__a_green_bow__s10__co3.png" width="200" alt="CO3 render of a black backpack below a green bow"></a><br>Both drawn, 0.9098 | <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_black_backpack__x__a_green_bow__s10__ours.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-rule5/a_black_backpack__x__a_green_bow__s10__ours.png" width="200" alt="Adapter render of a green backpack with no bow"></a><br>A green backpack and no bow, 0.0006 |

<a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/one-cell-bear-with-glasses-seed9.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/one-cell-bear-with-glasses-seed9.png" width="200" alt="Three renders of a bear with glasses: the joint prompt wears glasses, PoE has none, the adapter holds a cub and a misplaced lens"></a>

**A picture can keep a high [CLIP score](../measuring-composition/what-each-scorer-says-about-two-objects.md#clip-score) without drawing the glasses properly.** On "a bear with a glasses" at seed 9,
the adapter draws a bear holding a cub and a misplaced lens, and its CLIP score is 28.0 against 28.9
for the joint prompt. BLIP-VQA separates them: 0.3826 for the adapter, 0.9370 for the joint prompt
and 0.0436 for PoE. The sheet prints BLIP-VQA as queued, and these three values come from the saved
score file `verified`.

> [!NOTE]
> Five sheets for this table are not embedded or linked. All are in
> `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/`:
> `methods_grid.png`, `best-cases-benchmark.png`, `worst-cases-benchmark.png`, `ours-best-cases.png`
> and `methods-grid-preview-seed9.png`. Each is not shown: holds a subject the pair ledger keeps off sheets.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-a-high-and-a-low-cell-look-like) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict, and the code or the record accounts for each, with one step marked
`inferred`.** The means and the verdict agree wherever both sources hold them.

| What | One source | The other source | What the code shows |
|---|---|---|---|
| CO3's sampler and guidance | the Table 1 file labels it DDIM at guidance 7.5 | the per-image score file records DDIM with the CO3 corrector, CFG++, guidance 0.8 | the table builder writes DDIM and 7.5 for every method ([line 108](../../scripts/review_build_tables.py)); CO3 ran at 0.8 ([lines 38 and 129](../../scripts/review_baseline_cell.py)). ToMe carries the same written label, and its run settings are not in the saved score file |
| the intervals on Ours minus PoE | this page: 0.1997 to 0.3551, 0.0726 to 0.1871, 0.0287 to 0.158 | [the review folder's sidecar](../paper-review/tables-1-and-2-sidecar.json): 0.2005 to 0.352, 0.0709 to 0.1869, 0.0273 to 0.1579 | one random stream serves every interval ([line 76](../../scripts/review_build_tables.py)), and the held-out rows draw from it first (line 137, then line 161), so adding the CO3 and Langevin rows moves later draws. That this accounts for the whole change is `inferred` |
| which methods Table 1 holds | the result file has six | [the review folder's copy](../paper-review/table-1-r1-benchmark.csv) has joint prompt, PoE and Ours | the copy is byte-identical to `main_table.before-co3.csv` `verified` |
| whether the subset file matches Table 1 | [the card](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/README.md) says its full-set values equal the table cell for cell | the subset file has five methods and no ToMe | its input, the saved score file, has no ToMe rows among its 6,400 images `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**CO3, ToMe and the joint prompt read the sentence naming both concepts while drawing, and the
adapter does not.** They are [listed as references](../../scripts/refresh_paper_slots.py) for that
reason (line 140), so the gap between them and the adapter mixes method with input. PoE is the
matched comparison.

**A count of two animals does not say which two.** The detector is asked for "animal"
([the scorer](../../scripts/review_score_comp.py)), so two of the same animal count as a success.
That is why Ours can read 0.9659 on the count and 0.5031 on BLIP-VQA for the same 264 images.

> [!NOTE]
> The comparison-strip fault does not reach the count column. That fault was in `compose_rate.py`
> (fixed in commit `0811f029`, 2026-11-13). This column was written by `review_score_comp.py`, and
> none of the 6,400 images in the saved score file is a `__cmp` strip `verified`. ToMe's 0.9621 is
> outside that saved copy, so for ToMe the same statement rests on the shared code path `inferred`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above. The other
pages in this group link here.

### BLIP-VQA

Show a picture of a coffee cup and a plate to a model and ask two yes-or-no questions, one for each
thing the prompt names. Suppose it says yes to "a red cup?" with probability 0.9 and to "a blue
plate?" with probability 0.5. The score is 0.9 × 0.5 = 0.45, so one missing thing drags the whole
score down. BLIP is the model, VQA stands for visual question answering, which is its job here, and
the formula is the product of the yes-probabilities over the prompt's noun phrases ([the scoring script](../../scripts/review_score_blipvqa.py)).

Here, CO3's yellow-bowl render scores 0.1899 on "a yellow bow and a pink bowl" because the colours
are swapped. That reason is `inferred`, because the per-phrase answers are not saved. Read more in
[the BLIP-VQA walk](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/03-blip-vqa-asking-a-question-per-phrase.md)
and [the BLIP-VQA note](../../../../../Obsidian/deep-learning/12-scoring-and-rival-methods/tables/blip-vqa.md).

### The instance count

Take four pictures and let a detector box every animal it finds: 2, 1, 3 and 2 boxes. Each picture
with two or more scores 1, giving 1, 0, 1, 1, and the mean is 3 / 4 = 0.75. An instance is one
separate thing and the count is how many the detector boxed. The paper's column is called Comp.

Here the same steps run over 264 animal images per method, and Ours averages 0.9659. Read more in
[the note on object detectors](../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Object%20Detector.md)
and in [what a compose rate is](../../../context/world/compose-rate.md#what-a-compose-rate-is). The
learning tree has no page on this count itself.

### An interval from resampling twice

Say two prompts have seed differences A: 0.2, 0.4 and B: 0.0, 0.2. Draw two prompts with replacement
(say B and B), then redraw each one's seeds with replacement (0.2, 0.2 and 0.0, 0.2). That draw's mean
is 0.15. Repeat many times and keep the middle 95% of the means. Resampling is the drawing again,
twice because prompts and then seeds are drawn, and the interval is the range left.

Here it is 10,000 draws over 66 animal prompts with four seeds each, and the range is 0.1997 to
0.3551. Read more in [the percentile bootstrap walk](../../../../../goal-setting/learning/mathematics/statistics/sampling/plans/13-the-percentile-bootstrap-derived.md)
and [the bootstrap note](../../../../../Obsidian/Mathematics/Statistics/Bootstrap.md).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| File | What it holds | Written by |
|---|---|---|
| [main_table.csv](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/main_table.csv) | mean CLIP score, BLIP-VQA, ImageReward and the count, per method and set | [the table builder](../../scripts/review_build_tables.py) |
| [tables_sidecar.json](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tables_sidecar.json) | images scored per cell, the intervals and the verdict | the same |
| [no_overlap_tables.json](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/no_overlap_tables.json) | the same tables on prompts sharing no concept with training | [the subset script](../../scripts/review_no_overlap_tables.py) |
| `artifacts/drips/present-revision-to-supervisor/scores/scores_long.csv` | one row per image, 6,400 images, no ToMe rows | a copy of the cluster's score file |
| 217 pictures held out of the drain | the tile folders and summary sheets of `how-the-adapter-scores-against-other-methods`. They repeat the pictures placed above, or hold a subject the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folders are not linked here | `stated` |

The held-out half of this comparison is on [how every SDXL method compares on the held-out pairs](how-does-every-sdxl-method-compare-on-the-held-out-pairs.md),
and the GenEval and T2I-CompBench scores are on [how the adapter scores on the standard benchmarks](how-does-the-adapter-score-on-the-standard-benchmarks.md).

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] A render grid for this table that a reader can open, built without the rows the pair ledger keeps off sheets.
- [ ] The sampler and guidance labels on the CO3 and ToMe rows of the Table 1 file.
- [ ] The image list behind ToMe's count, which the saved score file does not hold.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
