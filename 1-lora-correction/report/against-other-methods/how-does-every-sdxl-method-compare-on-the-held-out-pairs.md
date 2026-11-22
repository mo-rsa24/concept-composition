# 📒 Report: how every SDXL method compares on the 53 held-out pairs

**No verdict was recorded for this table, because the protocol's pass rule covers only
[the benchmark prompts](how-does-the-adapter-score-on-the-attend-and-excite-prompts.md#the-comparison-with-a-rule-written-first).**
Over all 53 pairs the adapter ("Ours") has a mean [BLIP-VQA](how-does-the-adapter-score-on-the-attend-and-excite-prompts.md#blip-vqa)
of 0.6306, between PoE at 0.5187 and the joint prompt at 0.7655. These pairs were kept out of the
adapter's training, and each was rendered at eight seeds (9 to 16).

## Table of contents

- [The scores by method and group](#the-scores-by-method-and-group)
- [What the best and worst cells look like](#what-the-best-and-worst-cells-look-like)
- [Pairs sharing no concept with training](#pairs-sharing-no-concept-with-training)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The scores by method and group

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-best-and-worst-cells-look-like) ➡️

**The adapter is above PoE on hard and easy pairs and slightly below it on not-competing pairs,
0.7319 against 0.7395.** The hard group's intervals are too wide to separate any method. Each cell is mean BLIP-VQA (0 to 1) ± the half-width of its
[95% interval](how-does-the-adapter-score-on-the-attend-and-excite-prompts.md#an-interval-from-resampling-twice),
resampling pairs and then seeds `verified`
([group scores](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/held_out_ablation.csv),
[pooled scores](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/held_out_pooled.csv)).

| Method | Hard, 3 pairs | Easy but unseen, 29 | Not competing, 21 | All 53 pairs |
|---|---|---|---|---|
| Joint prompt | 0.4585 ± 0.3984 | 0.7367 ± 0.0769 | 0.8492 ± 0.0602 | 0.7655 ± 0.0586 |
| PoE | 0.3301 ± 0.3854 | 0.3783 ± 0.0865 | 0.7395 ± 0.0977 | 0.5187 ± 0.0789 |
| CO3 | 0.5937 ± 0.2742 | 0.7716 ± 0.0640 | 0.8074 ± 0.0815 | 0.7757 ± 0.0512 |
| PoE + Langevin | 0.3774 ± 0.3634 | not scored | not scored | not in the file |
| Ours | 0.4543 ± 0.4163 | 0.5755 ± 0.0934 | 0.7319 ± 0.0831 | 0.6306 ± 0.0699 |

**What the groups mean.** A competing pair is hard when its joint prompt shows both named things on
fewer than 4 of 8 seeds, and easy but unseen otherwise. Object-and-place pairs are not competing
`stated` ([the frozen evaluation set](../paper-review/held-out-set.md#the-held-out-groups)).

## What the best and worst cells look like

Navigation: ⬅️ [Previous](#the-scores-by-method-and-group) | 📋 [TOC](#table-of-contents) | [Next](#pairs-sharing-no-concept-with-training) ➡️

**The adapter draws both things on its best cells and loses or swaps one on its worst, where CO3
usually keeps both.** In all three sheets the columns are joint prompt, PoE, CO3, PoE + Langevin and
Ours, and each row is one pair at one seed. The rows were picked from the scores by
[protocol rule 5](../paper-review/protocol.md#what-each-rule-means-in-practice), so they show the
range and say nothing about how often each case happens `stated`.

| Highest and lowest cell per group | Best two pairs per group | Worst two pairs per group |
|---|---|---|
| <a href="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-every-sdxl-method-compares-on-held-out-pairs/methods_side_by_side.png"><img src="../../artifacts/results/against-other-methods/the-head-to-head-tables/how-every-sdxl-method-compares-on-held-out-pairs/methods_side_by_side.png" width="200" alt="Six held-out pairs by five methods, the highest and lowest adapter cell for each of three groups"></a> | `best-cases-held-out.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/best-cases-held-out.png` | `worst-cases-held-out.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/worst-cases-held-out.png` |
| Ours is 0.95 on seal and walrus at seed 11 and 0.01 on cow and horse at seed 13, where it draws two cows `stated` (the card) | Giraffe and lion, drum set and snowman, and tent and snowy mountain each show both things in the Ours column `verified` | Ours has no bowl, no glacier and no park bench on three rows, and CO3 shows both things on those rows `verified` |

## Pairs sharing no concept with training

Navigation: ⬅️ [Previous](#what-the-best-and-worst-cells-look-like) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**On the 23 of 53 pairs that share no concept with the 48 training pairs, the adapter's mean is above
PoE's in all three groups, and each pair of intervals overlaps.** Same score and interval as above.
The hard group here is a single pair, so its interval resamples seeds only `verified`
([the subset file](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/no_overlap_tables.json)).

| Method | Hard, 1 pair | Easy but unseen, 10 | Not competing, 12 |
|---|---|---|---|
| Joint prompt | 0.8871 ± 0.0323 | 0.667 ± 0.1525 | 0.8408 ± 0.0808 |
| PoE | 0.791 ± 0.1009 | 0.4693 ± 0.1489 | 0.6796 ± 0.1381 |
| CO3 | 0.8978 ± 0.0224 | 0.6919 ± 0.139 | 0.7837 ± 0.1161 |
| PoE + Langevin | 0.7453 ± 0.0739 | not scored | not scored |
| Ours | 0.8591 ± 0.0697 | 0.5737 ± 0.1545 | 0.7119 ± 0.1035 |

## Where the sources disagree

Navigation: ⬅️ [Previous](#pairs-sharing-no-concept-with-training) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three recorded facts conflict, and none changes a mean on this page.**

| What | One source | The other source | What the code or record shows |
|---|---|---|---|
| the half-widths for Ours | this page: 0.4163, 0.0934, 0.0831 | [the review folder's copy](../paper-review/table-2-r3-held-out.csv): 0.4189, 0.0939, 0.0840 | the copy is byte-identical to `held_out_ablation.before-co3.csv` `verified`. One random stream serves every row ([line 76](../../scripts/review_build_tables.py)), so adding the CO3 and Langevin rows before Ours moves its draws. That this is the whole change is `inferred` |
| CO3's guidance | the group file labels it 7.5 | the per-image score file records CFG++ at 0.8 | the table builder writes 7.5 for every method ([line 132](../../scripts/review_build_tables.py)) |
| which seeds the render grid shows | [the layout sheet](../paper-review/layout-sheet.md) says seeds 9 and 10 | the sheet's rows use seeds 10, 11, 13 and 16, as [protocol rule 5](../paper-review/protocol.md#what-each-rule-means-in-practice) asks | the folder `tiles-r4/` holds cells at seeds 9 and 10, and the sheet is built from `tiles-rule5/` `verified` |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The hard group holds three pairs, so its means are close to anecdotes.** Its half-widths run from
0.2742 to 0.4163 on a 0-to-1 score.

**CO3 and the joint prompt read the sentence naming both concepts while drawing, and the adapter
does not.** PoE is the matched comparison, as on [the benchmark page](how-does-the-adapter-score-on-the-attend-and-excite-prompts.md#what-this-cannot-tell-you).

**No [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) is quoted here.** This table holds BLIP-VQA only, so the `__cmp` strip fault does
not apply to any number on this page. How often a pooled adapter composes on unseen pairs is on
[does one pooled adapter compose on pairs it never trained on](../does-the-fix-reach-unseen-pairs/does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md).

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| File | What it holds | Written by |
|---|---|---|
| [held_out_ablation.csv](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/held_out_ablation.csv) | mean BLIP-VQA and half-width per method and group | [the table builder](../../scripts/review_build_tables.py) |
| [held_out_pooled.csv](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/held_out_pooled.csv) | the same over all 53 pairs, without PoE + Langevin | no script in the repository names this file |
| [no_overlap_tables.json](../../artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/no_overlap_tables.json) | the 23-pair subset | [the subset script](../../scripts/review_no_overlap_tables.py) |
| 82 pictures held out of the drain | 54 tiles in `how-every-sdxl-method-compares-on-held-out-pairs`, repeated by the sheets above or used by no sheet, and 28 CO3 pictures in `what-co3-does-on-our-pairs`, one of which holds a subject the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] The script that wrote the pooled file, which the repository does not hold.
- [ ] The guidance label on the CO3 rows of the group file.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
