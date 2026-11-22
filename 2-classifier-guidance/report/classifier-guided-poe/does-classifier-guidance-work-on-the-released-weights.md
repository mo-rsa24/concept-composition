# 📒 Report: does classifier guidance work on the authors' released weights?

**The route records all four targets as reproduced, and a strict look at each picture also meets the
condition the route wrote for "inconclusive" on three of them.** The test ran Dhariwal and Nichol's
[classifier guidance](https://arxiv.org/pdf/2105.05233#page=7) on their own 256 px ImageNet model and
classifier, with nothing retrained. The scale is the dial on the classifier's push: 0 turns it off,
and larger values steer harder toward the asked-for class.

## Table of contents

- [Scale 0 changes nothing](#scale-0-changes-nothing)
- [Scale 10 draws the asked-for class](#scale-10-draws-the-asked-for-class)
- [Which reading of the pass rule applies](#which-reading-of-the-pass-rule-applies)
- [Pictures kept off this page](#pictures-kept-off-this-page)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Scale 0 changes nothing

Navigation: 📋 [TOC](#table-of-contents) | [Next](#scale-10-draws-the-asked-for-class) ➡️

**At scale 0 the render matches plain sampling byte for byte on 4 of 4 seeds.** Seeds 0 to 3 have
equal sha256 checksums on the 8-bit image and on the 32-bit sample. Plain sampling run twice on seed 0
also matches itself, so the GPU repeats exactly here `verified` ([the checksums](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/identity_checksums.csv)).

## Scale 10 draws the asked-for class

Navigation: ⬅️ [Previous](#scale-0-changes-nothing) | 📋 [TOC](#table-of-contents) | [Next](#which-reading-of-the-pass-rule-applies) ➡️

**The released classifier calls 46 of 48 renders the asked-for class at scale 10, against 9 of 48 at
scale 1, and a strict tile-by-tile look agrees on 30 of the 48.** Each cell counts renders out of 16,
one per seed, with seed i starting from the same noise at both scales.

| Class asked for | Classifier yes, scale 1 | Classifier yes, scale 10 | Mean probability of the class, scale 10 (0 to 1) | Eye, sheet read, scale 10 | Eye, tile by tile, scale 1 | Eye, tile by tile, scale 10 |
|---|---|---|---|---|---|---|
| Pembroke Welsh corgi (263) | 2 | 16 | 0.854 | 13 | 0 | 9 |
| tabby cat (281) | 2 | 15 | 0.737 | 12 | 0 | 11 |
| golden retriever (207) | 5 | 15 | 0.907 | 14 | 3 | 10 |
| all three, of 48 | 9 | 46 | | 39 | 3 | 30 |

The classifier columns are recounted from the 96 rows of [the per-render calls](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/classifier_calls.csv),
columns `hit` and `p_asked` `verified`. The tile-by-tile columns are that file's `your_call`, one
reading by Claude at 256 px that the owner accepted `inferred`. The sheet read is recorded in the
route's targets T2 to T4 `stated`. The bottom row adds the three rows above it.

**All 22 disagreements over the 96 renders are the classifier saying yes and the eye saying no.** The
same classifier steers each render and then grades it, so its count is partly self-fulfilling.

> 🖼️ **Figure wanted**: classifier count beside tile-by-tile eye count at scales 1 and 10, one pair of
> bars per class, from `classifier_calls.csv`. Render into
> `artifacts/results/does-classifier-guidance-work-on-the-released-weights/`, then embed here. A chart
> carries this result without showing any sampled image.

## Which reading of the pass rule applies

Navigation: ⬅️ [Previous](#scale-10-draws-the-asked-for-class) | 📋 [TOC](#table-of-contents) | [Next](#pictures-kept-off-this-page) ➡️

> [!IMPORTANT]
> The route's table gives each class a "reproduced" bar and an "inconclusive" condition, and it does
> not say which wins when both hold. Both hold on all three classes under the tile-by-tile count. The
> choice is the owner's.

| How the rule is read | What is compared | Corgi | Tabby | Golden retriever | That reading gives |
|---|---|---|---|---|---|
| the reproduced bar alone | classifier yes at scale 10 (needs at least 14 of 16) and at scale 1 (needs fewer) | 16 and 2 | 15 and 2 | 15 and 5 | reproduced on all three |
| the inconclusive condition, with the sheet read as the eye count | classifier minus eye at scale 10 (inconclusive at 4 or more) | 3 | 3 | 1 | not triggered |
| the inconclusive condition, with the tile-by-tile count | the same subtraction | 7 | 4 | 5 | inconclusive on all three |

The condition names "the eye count in instruction 13", and instruction 13 is the tile-by-tile call
`stated` (`2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`, step 4 and
instruction 13). The route still ticks T2 to T4 as reproduced.

## Pictures kept off this page

Navigation: ⬅️ [Previous](#which-reading-of-the-pass-rule-applies) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**This model draws any ImageNet class, so no sample sheet appears here.** The
[pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) rules which subjects
may be shown. All paths start at `2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/`.

| File | Why it is not shown |
|---|---|
| `sheet-identity-scale-0-against-plain.png` | not shown: holds a subject the pair ledger keeps off sheets |
| `identity/identity_seed0_plain.png`, `identity/identity_seed0_scale0.png` | not shown: holds a subject the pair ledger keeps off sheets |
| `sheet-tabby-cat-scale-1-and-10.png`, `sheet-golden-retriever-scale-1-and-10.png` | left unopened, because a class call on one scale 1 tile of each could not be cleared |
| `eye-against-classifier-disagreements.png` | a tile could not be cleared against the pair ledger |
| `sheet-pembroke-welsh-corgi-scale-1-and-10.png` | a scale 1 tile could not be cleared |

The grouping's card and the route embed the first of these, so this page names both by path and links
neither.

## Where the sources disagree

Navigation: ⬅️ [Previous](#pictures-kept-off-this-page) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts conflict.**

| What | One source | The other source | What explains it |
|---|---|---|---|
| clear renders by eye at scale 10 | route targets T2 to T4: 13, 12 and 14 of 16 | route instruction 13 and `your_call`: 9, 11 and 10 of 16 | the card says the sheet read counted corgi-coloured and Labrador-like dogs, and the tile-by-tile read says yes only to a clear tile of the asked-for breed `stated` |
| whether every target passed | the card: all four targets passed | the route's inconclusive condition holds on three under the tile-by-tile count | no source says which column takes priority, as the table above shows |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Image quality on the paper's scale**

The paper's FID numbers need 50,000 samples each and were not attempted. Each cell here holds 16
samples and carries no confidence interval `stated` (the card).

**The sampler the product uses**

Only ancestral sampling ran. The product sampler ports the DDIM form, which this run did not exercise
`stated` (the card).

**Anything about composition**

Every render asks for one class. Nothing here tests two concepts at once.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The push and its strength are taught on
[the code walk page](which-line-computes-which-part-of-guided-sampling.md#concepts-this-uses), and the
term index has no note for classifier guidance itself.

**A classifier's top call**

A classifier gives one picture 0.5 for corgi, 0.3 for fox and 0.2 for beagle. "Top" is the largest
number and "call" is the class it belongs to, so the top call is corgi. If corgi was asked for, the
render counts as a hit. On this project's first row (corgi, scale 1, seed 0), the top call is Chihuahua
at 0.2618, corgi gets 0.0093, and `hit` is 0.

**A checker grading its own push**

A coach writes an exam, drills a student on it and marks it 10 of 10. An outside marker gives the same
paper 6 of 10. The gap of 4 is the part only the coach believed. Here the classifier gives the corgis
16 of 16 and the tile-by-tile read gives 9 of 16, a gap of 7.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| classifier counts, probabilities and eye calls | [the per-render calls](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/classifier_calls.csv), read 2026-11-22 | `verified` |
| the scale 0 match | [the checksums](../../artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/identity_checksums.csv), read 2026-11-22 | `verified` |
| how the renders were made | [the run script](../../scripts/guided_diffusion_reproduce.py) and [the sheet script](../../scripts/guided_diffusion_sheets.py), upstream commit `22e0df8` | `stated` |
| the bars and the sheet-read counts | `2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`, step 4 | `stated` |
| what was not attempted | `2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md` | `stated` |
| 96 pictures held out of the drain | the other tiles, sheets and probes in this result's folder. They repeat a picture described above, or have not been cleared against the subjects the [pair ledger](../../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off sheets, so the folder is not linked here | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Choose which reading of the pass rule applies to T2, T3 and T4. The owner decides.
- [ ] Decide what happens to the identity sheet, which the card and the route still embed. The owner decides.
- [ ] Draw the bar chart named above, so this result has a picture that is safe to show.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
