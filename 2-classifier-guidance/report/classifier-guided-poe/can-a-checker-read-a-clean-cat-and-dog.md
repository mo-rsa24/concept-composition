# 📒 Report: can a small checker tell a clean cat and dog from a fused animal?

**No verdict covers the whole question, and the best reading of the sampler's own early states is
0.730 against a pass mark of 0.75.** A checker is a small network that reads a half-drawn SDXL picture
and answers "is this heading for one clean cat beside one clean dog?". Its gradient later becomes
[the early push](does-an-early-push-separate-the-cat-and-dog.md). Every score here is balanced
accuracy, the average of the hit rate on clean pairs and the hit rate on everything else, so 0.5 is a
coin flip and 1 is perfect.

## Table of contents

- [Discriminators trained on called pairs](#discriminators-trained-on-called-pairs)
- [Readout checkers on cat and dog](#readout-checkers-on-cat-and-dog)
- [Finished pictures and half-drawn guesses](#finished-pictures-and-half-drawn-guesses)
- [Training on the sampler's own states](#training-on-the-samplers-own-states)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Discriminators trained on called pairs

Navigation: 📋 [TOC](#table-of-contents) | [Next](#readout-checkers-on-cat-and-dog) ➡️

<a href="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-all-checkers.png"><img src="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-all-checkers.png" width="200" alt="Line chart of held-out balanced accuracy against sampler step 0 to 19 for four discriminators. Only the first rises above the 0.75 line, from step 7."></a>

**Every discriminator with hybrids among its negatives failed the held-out check.** The chart plots
held-out balanced accuracy (y) against sampler step 0 to 19 of 50 (x), one line per discriminator.
Only the first clears the dashed 0.75 line, and its check held no hybrids `verified` (read on the
chart; matches [the rung 2 review](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md#asked-after-the-result)).

The pass mark for the first four rows is balanced accuracy of at least 0.75 on at least 10 of the
first 20 sampler steps.

| Folder | What changed | Recorded result | Verdict |
|---|---|---|---|
| `discriminator`, first head | 17 clean pairs against lone animals and product renders | 13 of 20 steps on held-out seeds 7 and 8 | met, on a check with no hybrids `stated` ([plan 21](../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md)) |
| `discriminator`, second head | pair-prompt failures added as negatives, 64 tiles | 0 of 20 steps, 0.53 to 0.63 | failed `stated` (plan 21) |
| `dataset-v3`, head only | 81 clean pictures over 38 pairs | 0 of 20 steps, 0.53 to 0.63 | failed `stated` (commit `db448f7e`) |
| `dataset-v3`, head and adapters | the same data, adapters on SDXL's trunk | 0 of 20 steps, 0.55 to 0.69 | failed `stated` (commit `db448f7e`) |
| `discriminator/probe.json` | a linear layer on frozen CLIP, or on DINOv2 two ways | 0.710, 0.613 and 0.638 on 43 held-out pictures | below the 0.75 mark `verified` (the file) |
| `api-labels/drawings.jsonl`, `dataset-v4`, `result-sheet-v4c`, `result-sheet-v4c-early` | [plan 23](../../plans/23-rung2-a-discriminator-trained-the-kim-et-al-way-at-scale.md)'s 1,920 drawings labelled, then a readout checker built on them | no held-out reading is written in any plan, walk or card | no verdict recorded |

## Readout checkers on cat and dog

Navigation: ⬅️ [Previous](#discriminators-trained-on-called-pairs) | 📋 [TOC](#table-of-contents) | [Next](#finished-pictures-and-half-drawn-guesses) ➡️

**Adding 350 SDXL joint-prompt drawings moved the reading most, from 0.639 to 0.737.** A
[readout checker](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/11-readout-guidance-by-hand.md#1--the-two-taps-one-grid-and-the-time)
reads SDXL's internal features. Each row is one change, read on 220 held-out cat and dog pictures
(109 clean, about ±0.05). The last column counts seeds 9 to 16 with two or more bodies under that
checker's push, from draft calls `stated` ([the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md)).

| Folder | What changed | Balanced accuracy | Seeds of 8 with two bodies |
|---|---|---|---|
| `result-sheet-catdog` | first version, 753 tiles | 0.639 | 3 |
| `result-sheet-catdog-coco` | COCO photos added | 0.600, dropped | 3 |
| `result-sheet-catdog-more2` | 350 new joint drawings added, COCO left out | 0.737, kept | 5 |
| `result-sheet-catdog-more3` | 500 more joint drawings | 0.741, kept | 4 |
| `result-sheet-catdog-more3-long` | the same data, 2,000 training steps in place of 750 | 0.755, kept | about 7 |
| `result-sheet-catdog-more3-hard` | half of each batch's rejects are wrong pairs | 0.756, kept as the loop's result | 3 |
| `result-sheet-catdog-identity` `inferred` | the 2,000-step checker plus "cat present" and "dog present" outputs | 0.721, kept as a push | 5 |

The loop's own mark was at least 0.75 and more two-body tiles than plain product and a random push of
the same size. It closed on the 0.756 row `stated`.

## Finished pictures and half-drawn guesses

Navigation: ⬅️ [Previous](#readout-checkers-on-cat-and-dog) | 📋 [TOC](#table-of-contents) | [Next](#training-on-the-samplers-own-states) ➡️

**A CLIP checker reads finished pictures at 0.90 and the first 10 steps' guesses at about chance.**
Networks fine-tuned on 1,603 finished pictures, read on the 220 held-out ones: DINOv2 ViT-B/14 0.846
and 0.847, CLIP ViT-L/14 0.865 to 0.900, CLIP ViT-bigG/14 0.873 and 0.909 `stated`. Two seeds of CLIP
ViT-L/14 differ by 0.035, as large as any change tried `stated`. The misses are in `finetune-misses` and
the guesses in `half-drawn-guesses`.

| Balanced accuracy at sampler step | 0 | 5 | 10 | 15 | 20 | 30 | finished |
|---|---|---|---|---|---|---|---|
| CLIP, trained on finished pictures | 0.50 | 0.50 | 0.51 | 0.58 | 0.62 | 0.81 | 0.90 |
| CLIP, then trained on 4,966 guesses | 0.51 | 0.54 | 0.72 | 0.82 | 0.84 | 0.87 | 0.90 |
| readout checker, noised finished pictures | 0.63 | 0.74 | 0.78 | 0.79 | | | |

## Training on the sampler's own states

Navigation: ⬅️ [Previous](#finished-pictures-and-half-drawn-guesses) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Training on 1,972 labelled runs' own states raised the early reading from 0.677 to 0.730, past the
required gain and short of the mark.** Scores are on 195 held-out joint-prompt runs (105 clean, 90
rejects), labelled by the 0.900 CLIP checker `stated` ([plan 25's review](../../plans/review/25-the-readout-checker-on-poes-own-states.md#answers)).
Its label sample is in `plan25-labels`.

| Sampler step | 0 | 2 | 4 | 6 | 8 | 10 | mean |
|---|---|---|---|---|---|---|---|
| before, trained on noised finished pictures | 0.510 | 0.548 | 0.644 | 0.747 | 0.794 | 0.817 | 0.677 |
| trained on the sampler's own states | 0.591 | 0.647 | 0.732 | 0.774 | 0.803 | 0.835 | 0.730 |

| Plan 25's question | Pass mark | Recorded | Verdict |
|---|---|---|---|
| Q1: do its own states teach the early steps? | mean at least 0.75, and at least 0.05 above the earlier checker | 0.730, +0.053 | in between |
| Q2: does its push make more clean pairs on seeds 9 to 16? | at least 2 more "two clear" seeds than plain product and the random push | this grouping holds no seeds 9 to 16 sheet for this checker | no verdict recorded |
| Q3: do 1,000 more joint runs help? | at least +0.02 | 0.724 against 0.730 | null |

## Where the sources disagree

Navigation: ⬅️ [Previous](#training-on-the-samplers-own-states) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three records differ, and the code explains two of them.**

| What | One source | The other source | What the code shows |
|---|---|---|---|
| the 2,000-step checker's reading at step 0 | 0.63 (checker loop) | 0.510 (plan 25's review) | two different tests. One adds noise to finished latents (`train_readout_pair.py` line 328), the other reads saved sampler states (lines 367 to 388). That this difference causes the gap is `inferred` |
| what plan 23 trains on | the plan: no labels, joint drawings against product drawings | the script: only joint drawings the API called a faithful pair count as real | `train_readout_pair.py` lines 16 and 17 |
| whether plan 22 ran | its task boxes are all unticked | commit `db448f7e` and the rung 2 review record both failed checks | no code bears on this |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**A higher reading does not promise a better push.** The 0.755 and 0.756 checkers gave two bodies on
about 7 and 3 of 8 seeds `stated`. The seed counts are draft calls, most at tile size.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named beside them.

**Balanced accuracy**

Ten photos, 4 of them a clean pair. The checker's calls:

| | Checker says clean | Checker says not clean |
|---|---|---|
| truly clean | 3 | 1 |
| not clean | 3 | 3 |

The top row is a hit rate of 3 of 4, or 0.75. The bottom row is 3 of 6, or 0.50. "Balanced" means each
row counts equally however many photos it holds, so the score is (0.75 + 0.50) / 2 = 0.625. For the
frozen CLIP probe in `discriminator/probe.json` the rows are 11 of 20 and 20 of 23, so (0.550 + 0.870)
/ 2 = 0.710. The [readout guidance chapter](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#balanced-accuracy-the-hit-rate-on-each-class-averaged)
works this measure through.

**A held-out check**

A student who memorised 10 flashcards scores 10 of 10 on them and 6 of 10 on new cards. Only the second
score says what was learnt. Here, a head trained on seeds 1 to 6 passed 13 of 20 steps on seeds 7 and
8, while the final head fit its own pictures on 20 of 20 `stated` (plan 21). The
[discriminator chapter](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/03-the-discriminator-and-the-density-ratio.md#31-two-piles-and-the-best-discriminator)
explains what such a network learns from two piles of examples.

**A noised finished picture against a real early state**

A reader names 8 of 10 finished drawings after they are smudged, and 5 of 10 real first sketches. A
smudged drawing still holds the finished layout, and a first sketch may not have one yet. The same
checker here reads 0.63 on noised finished pictures at step 0 and 0.510 on the sampler's own step 0.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the first discriminators | [plan 21](../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md), [plan 22](../../plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md) and [the rung 2 review](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md) | `stated` |
| the readout rows and fine-tunes | [the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md) | `stated` |
| the frozen probes | `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/probe.json`, read 2026-11-22 | `verified` |
| how each reading is computed | [the readout checker's training script](../../scripts/rung2/train_readout_pair.py) | `verified` |
| the `inferred` folder | `result-sheet-catdog-identity` is matched to its checker by name; no plan or walk names it | `inferred` |
| 13,105 pictures held out of the drain | the render sheets, tiles and image pools of `does-a-small-discriminator-steer-cat-x-dog`. Their counts are on this group's pages. They have not been cleared one by one against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, and some name such a subject, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Record plan 23's held-out reading, or state that it never ran. The check files are on the cluster's scratch folder.
- [ ] Decide whether plan 25's Q2 is still worth drawing. Its checker's 128-seed result is on [the 128-seed page](how-close-the-kept-push-comes-to-the-joint-prompt.md).
- [ ] Write card entries for this grouping's folders. Its card covers 5 of 72.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
