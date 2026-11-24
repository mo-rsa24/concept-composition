# 📒 Report: does an early checker push separate the cat and the dog?

**Yes on the 32 seeds kept for this test: two separate animals on 28 of 32, against 1 of 32 for plain
product and for a random push of the same size.** The [product of experts](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this)
of "a cat" and "a dog" normally draws one fused animal. The early push adds a
[readout checker](can-a-checker-read-a-clean-cat-and-dog.md#readout-checkers-on-cat-and-dog)'s
gradient on sampler steps 0 to 15 of 50, at strength 30, limited to 0.3 of the size of the product's
own prediction. Steps 16 to 49 are plain product.

## Table of contents

- [The result on 32 new seeds](#the-result-on-32-new-seeds)
- [Which kind of push moves the animals](#which-kind-of-push-moves-the-animals)
- [How hard and for how long](#how-hard-and-for-how-long)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The result on 32 new seeds

Navigation: 📋 [TOC](#table-of-contents) | [Next](#which-kind-of-push-moves-the-animals) ➡️

**Plan 29's verdict is support: both pass marks and both control limits held.** Seeds 3301 to 3332
were never used for training, evaluation or tuning when this ran. Calls are by eye from full-size
tiles `stated` ([plan 29's review](../../plans/review/29-does-the-early-push-hold-on-new-seeds.md#answers)).
The early push's counts were recounted from `scripts/rung2/calls_plan29.json` `verified`.

| Call, of 32 seeds | Plain product | Random push, same size | Early push | Pass mark |
|---|---|---|---|---|
| two separate animals | 1 | 1 | 28 | at least 19, controls at most 2 |
| clean or near-clean cat and dog | 0 | 0 | 14 (9 clean) | at least 10 |

The folder is `result-sheet-catdog-general`. Of the 18 seeds short of clean or near-clean, 11 show two
bodies with a borrowed face or hidden head, 3 show two of one kind and 4 are not separate.

`not shown: not cleared against the subjects the pair ledger keeps off sheets`:
`result-sheet-catdog-general/views/pushed-1.jpg` to `pushed-4.jpg`.

## Which kind of push moves the animals

Navigation: ⬅️ [Previous](#the-result-on-32-new-seeds) | 📋 [TOC](#table-of-contents) | [Next](#how-hard-and-for-how-long) ➡️

<a href="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/03-does-an-early-push-separate-them/attention.png"><img src="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/03-does-an-early-push-separate-them/attention.png" width="200" alt="Line chart of how much the cat and dog attention maps overlap against sampler step, one line per seed. Product lines sit above most joint prompt lines from the early steps."></a>

**A push through SDXL's own features moved the layout, and pushes through the decoded picture changed
faces and fur.** The chart plots the overlap of the top 10% of "cat" and "dog" attention (y,
intersection over union, 1 means the same places) against sampler step (x), one line per seed 1 to 8.
The median over the 8 seeds at each of steps 2 to 40 runs 0.506 to 0.814 for the product and 0.225 to
0.394 for the joint prompt `verified` (computed from `attention/overlap.json`). That gap is why a
push on attention was tried.

| Folder | The push | Recorded result, seeds 9 to 16 unless noted (draft calls) | Verdict |
|---|---|---|---|
| `result-sheet-layout` | pulls the two attention maps apart, steps 0 to 19 | 0 of 8 with two animals; strong pushes zoom into one face | none recorded |
| `clip-push-sheet-from0` | fine-tuned CLIP through the decoded guess, all steps | two bodies on 1 of 8 | none recorded |
| `clip-push-sheet-from10` | the same, steps 10 to 49 | 0 of 8 | none recorded |
| `clip-push-sheet-halfdrawn` | CLIP trained on half-drawn guesses, steps 10 to 49 | 0 of 8 | none recorded |
| `result-sheet-catdog-backward` | CLIP with three backward steps, steps 0 to 15, seeds 3301 to 3308 | layout stays plain product on 8 of 8; the early push separates the same 8 | dropped, mark of 7 clean pairs of 32 out of reach |
| `result-sheet-catdog-identity` `inferred` | readout checker, all steps, cap 0.1 | two or more bodies on 5 of 8 at strength 100 | kept as a push |
| `push-then-pick` | four pushed tries per seed, CLIP picks one | 3 of 8 picks called clean; by eye 2 near-clean and 1 two dogs. Best of four plain tries: 0 | none recorded |

## How hard and for how long

Navigation: ⬅️ [Previous](#which-kind-of-push-moves-the-animals) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**A cap of 0.3 on the first 16 steps kept the separation and most of the natural colour.** Rows are
seeds 9 to 16 at strengths 30 and 100, from draft calls at tile size, unless noted `stated`
([the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md#a-larger-cap)).
No written pass mark exists for these rows except where shown.

| Folder | What changed | Recorded result | Outcome |
|---|---|---|---|
| `result-sheet-catdog-identity-cap0.3` `inferred` | cap 0.1 raised to 0.3, all 50 steps | a separate cat and dog on most seeds, colour damaged | best of three caps |
| `result-sheet-catdog-identity-cap0.5` `inferred` | cap 0.5, all 50 steps | like 0.3 at strength 30 with more damage, broken at 100 | not used |
| `result-sheet-catdog-identity-cap0.3-early` | cap 0.3 on steps 0 to 15 only | same seeds separate, colour closer to a photograph; near-clean on 4 of 8 | kept |
| `result-sheet-catdog-identity-cap0.3-late` | cap 0.1 to step 9, then 0.3 to step 49 | separate on most seeds; near-clean on 3 of 8, on different seeds | not used |
| `result-sheet-catdog-identity-recur3` `inferred` | cap 0.1, each of steps 0 to 15 taken 3 times | two animals on about 4 to 5 of 8 | short of cap 0.3 |
| `result-sheet-catdog-cap0.3-early-corrector` | the early push with 3 re-noised repeats per step | clean or near-clean on about 3 of 8, more damage | not used |
| `result-sheet-catdog-general-cap0.2` | cap 0.2 on seeds 3301 to 3332, by eye at 640 px | 22 of 32 separate, 10 clean or near-clean (5 clean), against 28 and 14 at cap 0.3 | cap 0.3 stays |

## Where the sources disagree

Navigation: ⬅️ [Previous](#how-hard-and-for-how-long) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**One recorded range cannot be reproduced in full.**

| What | One source | The other source | What the code shows |
|---|---|---|---|
| median attention overlap for the joint prompt | [the walk](../../artifacts/drips/rung2-pair-aware-discriminator/EXECUTE_WALK.md): 0.22 to 0.44 over steps 2 to 40 | `attention/overlap.json`: 0.225 to 0.394, the median over 8 seeds at each step | `attention_overlap.py` saves the curves and computes no median. At the three steps it prints (2, 10, 40) the product's medians give the walk's 0.53 to 0.81 and the joint prompt's give 0.23 to 0.39, so the walk's 0.44 is unexplained |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**This is one pair, and pushed renders do not repeat pixel for pixel.** The same seed and settings gave
three different pictures in three runs, so compare pushed runs by their counts `stated`. Viewing size
moves the calls too: at half size plan 29 counted 17 clean or near-clean, and 14 held at full size
`stated`. The "4 of 8" for the early push appears to pool strengths 30 and 100 `inferred`, while plan 29
used strength 30 alone. Other pairs are on [the other pairs page](does-the-early-push-carry-to-other-pairs.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The push itself is taught on
[the code walk page](which-line-computes-which-part-of-guided-sampling.md#concepts-this-uses).

**The cap on a push**

A cart is rolling with a force of 10, and you may add at most 0.3 of that, so 3. A shove of 8 is cut to
3 in the same direction, and a shove of 2 passes unchanged. "Cap" is that ceiling: new push =
push × min(1, 0.3 × size of prediction / size of push). On the 128-seed sheet `result-sheet-catdog-big`
the cap cut the push on 54% of steps 0 to 15, averaged over seeds, from `capped_per_step` in its
`result.json`.

**A random push of the same size**

If any shove of force 3 knocks two stuck magnets apart, the direction of your shove deserves no credit.
So you also shove in a random direction with the same force at every moment. In plan 21 the random push's
size divided by the real push's size was 1.0000 at every step of all 24 random renders `stated`, and
in plan 29 it separated 1 of 32 seeds against 28 for the real push.

**A window of steps**

Painting a room takes 50 strokes, and the layout is settled by the first few. Pushing only on strokes 0
to 15 spends 16 of 50 on layout and leaves the rest alone. Here the window is steps 0 to 15, and
the colour damage from pushing on all 50 steps mostly went away `stated`.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the 32-seed test | [plan 29](../../plans/29-does-the-early-push-hold-on-new-seeds.md) and [its review](../../plans/review/29-does-the-early-push-hold-on-new-seeds.md) | `stated` |
| the push and cap rows | [the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md) | `stated` |
| the layout push's result | `artifacts/drips/rung2-pair-aware-discriminator/stock-take.md`, section 1 | `stated` |
| each sheet's settings | `result.json` in each folder, read 2026-11-22 | `verified` |
| the three `inferred` folders | matched to their experiment by name; no plan or walk names them | `inferred` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Confirm the draft calls on the seeds 9 to 16 sheets. Plan 29's calls are the ones made from full-size tiles.
- [ ] A figure showing plain product beside the early push on the same seeds, built only from tiles checked at stored size.

## Cross-references

- The mention of **the early push finding** in [the context page on the readout checker, one pushed sampling step, section "What is measured where"](../../../context/methods/02-the-readout-checker/01-overview.md#what-is-measured-where).
- The mention of **the early push finding** in [the context page on the readout checker, the push and its cap, section "The push, by what moves"](../../../context/methods/02-the-readout-checker/03-the-push-and-its-cap.md#the-push-by-what-moves).
- The mention of **the early-push report** in [the context page on the word-map checker, one sampler step, section "What is established and what is proposed"](../../../context/methods/06-the-word-map-checker/01-overview.md#what-is-established-and-what-is-proposed).
- The mention of **the early-push report** in [the context page on the word-map checker, the code that would run it, section "Our sampler: the attention push already run"](../../../context/methods/06-the-word-map-checker/05-the-code.md#our-sampler-the-attention-push-already-run).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
