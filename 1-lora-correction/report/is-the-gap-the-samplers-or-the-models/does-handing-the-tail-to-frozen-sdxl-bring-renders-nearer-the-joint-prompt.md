# 📒 Report: does handing the last denoising steps back to frozen SDXL bring a render nearer its joint prompt?

**No verdict has been recorded for this sweep.** It asked whether switching the trained adapter off
for the last stretch of a 50-step render brings the picture nearer that seed's joint-prompt render
("a cat and a dog") and keeps both animals. The script printed `null` for two checkpoints and
`composition broke` for the third. The answer boxes in [the review file](../../plans/09-designing-the-correction-loss/review/06-fitting-the-whole-path-and-handing-back-the-tail.md#written-before-the-run-answered-after)
are still empty.

## Table of contents

- [What the sweep changed](#what-the-sweep-changed)
- [What came back](#what-came-back)
- [The rank 16 result under each reading of the rule](#the-rank-16-result-under-each-reading-of-the-rule)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What the sweep changed

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-came-back) ➡️

**One number moved: `k`, the step at which the adapter is switched off.** The adapter runs on steps
0 to k-1 and the frozen model's plain product-of-experts step draws the rest. `k` took the values 25,
29, 33, 36, 40 and 50, and `k = 50` is the adapter on the whole path, which is the control `stated`
([the sweep script](../../../scripts/shared/clean_tail_k_sweep.py), lines 115 to 116).

Each of three saved adapters rendered ten cells, where a cell is one animal pair at one seed: cat and
dog on held-out seeds 9 to 16, elephant and penguin on seeds 9 and 10. Every render used 50 DDIM steps,
guidance 7.5, correction strength 1.0 and 1024 by 1024 pixels `stated` (the script, lines 84 to 86,
117 and 396).

## What came back

Navigation: ⬅️ [Previous](#what-the-sweep-changed) | 📋 [TOC](#table-of-contents) | [Next](#the-rank-16-result-under-each-reading-of-the-rule) ➡️

**No seed on the two rank-32 checkpoints moved the required 0.05 nearer its joint-prompt render, and
one seed did on the rank-16 checkpoint.** Gain is the control's distance to the joint-prompt render
minus this render's distance, so a positive gain is nearer. Only 5 of the 10 cells have a joint-prompt
render showing both animals, so only those 5 count `verified` (read from the six sidecar files).

| Adapter | Readable cells clearing 0.05, at the best `k` | Largest gain on a readable cell | Cells with two or more animals, `k = 50` then `k = 25` | Script printed |
|---|---|---|---|---|
| rank 16, 30,000 training steps | 1 of 5 (cat and dog seed 9) | +0.093 (seed 9, `k = 25`) | 6 then 4, of 10 | `composition broke` |
| rank 32 pooled, 40,000 steps | 0 of 5 | +0.037 (seed 9, `k = 29`) | 9 then 9, of 10 | `null` |
| rank 32 pooled, 60,000 steps | 0 of 5 | +0.039 (seed 12, `k = 25`) | 9 then 9, of 10 | `null` |

In every sheet below, rows are seeds and columns are the joint prompt followed by the six values of
`k`. Each tile prints its animal count `n`, its gain and its sharpness `lap`, and a green frame means
a count of two or more `verified` (seen on the sheets; layout at lines 542 to 570 of the script).

| | Rank 16, 30,000 steps | Rank 32, 40,000 steps | Rank 32, 60,000 steps |
|---|---|---|---|
| Cat and dog, seeds 9 to 16 | <a href="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-v1-freeze-null-r16-s0-25-30k-a-cat-x-a-dog.png"><img src="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-v1-freeze-null-r16-s0-25-30k-a-cat-x-a-dog.png" width="200" alt="Eight seeds by seven columns for the rank 16 adapter on cat and dog. Seeds 15 and 16 lose their green frames at the earliest switch-off steps."></a><br>Seed 9 is the one cell that clears 0.05, at `k` 25 to 36. | <a href="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-40k-a-cat-x-a-dog.png"><img src="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-40k-a-cat-x-a-dog.png" width="200" alt="Eight seeds by seven columns for the rank 32 adapter at 40,000 steps on cat and dog. Each row looks almost the same across columns, and only seed 10 has red frames."></a><br>Each row barely changes across `k`. Seed 10 stays under two animals. | <a href="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-60k-a-cat-x-a-dog.png"><img src="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-60k-a-cat-x-a-dog.png" width="200" alt="Eight seeds by seven columns for the rank 32 adapter at 60,000 steps on cat and dog. Each row looks almost the same across columns, and only seed 14 has red frames."></a><br>Each row barely changes across `k`. Seed 14 stays under two animals. |
| Elephant and penguin, seeds 9 and 10 | <a href="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-v1-freeze-null-r16-s0-25-30k-an-elephant-x-a-penguin.png"><img src="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-v1-freeze-null-r16-s0-25-30k-an-elephant-x-a-penguin.png" width="200" alt="Two seeds by seven columns for the rank 16 adapter on elephant and penguin. Seed 9 has green frames and seed 10 has red frames."></a><br>Seed 10 draws one elephant at every `k`. | <a href="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-40k-an-elephant-x-a-penguin.png"><img src="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-40k-an-elephant-x-a-penguin.png" width="200" alt="Two seeds by seven columns for the rank 32 adapter at 40,000 steps on elephant and penguin, every frame green."></a><br>Every frame is green. Seed 9's joint-prompt render shows an elephant alone, so it has no usable distance. | <a href="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-60k-an-elephant-x-a-penguin.png"><img src="../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-60k-an-elephant-x-a-penguin.png" width="200" alt="Two seeds by seven columns for the rank 32 adapter at 60,000 steps on elephant and penguin, every frame green."></a><br>Every frame is green, and seed 10's gain stays between -0.014 and -0.003. |

## The rank 16 result under each reading of the rule

Navigation: ⬅️ [Previous](#what-came-back) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Four values of `k` tie for best on rank 16, and the printed result depends on which one is taken.**
Each clears the bar on the same single cell. The counts below come from the two rank 16 sidecar files.
A cell is lost when it holds two or more animals at `k = 50` and fewer at this `k`, and gained the
other way round `verified`.

| `k` | Readable cells clearing 0.05 | Cells with two or more animals, of 10 | Lost against `k = 50` | Gained |
|---|---|---|---|---|
| 25 | 1 | 4 | cat and dog seeds 15, 16 | none |
| 29 | 1 | 3 | cat and dog seeds 9, 15, 16 | none |
| 33 | 1 | 5 | cat and dog seeds 15, 16 | cat and dog seed 13 |
| 36 | 1 | 6 | cat and dog seed 16 | cat and dog seed 13 |
| 40 | 0 | 6 | none | none |
| 50 (control) | 0 | 6 | | |

> [!IMPORTANT]
> The choice between these readings is the owner's. The recorded outcome is the first row, and this
> page does not replace it.

| How the rule is read | Applied to rank 16 | That reading gives |
|---|---|---|
| the script as run: the tied `k` listed first wins, one clearing cell is enough, and a net fall of more than one cell breaks composition (lines 473 to 474 and 483 to 489) | `k = 25`, cells fall from 6 to 4 | `composition broke` (recorded) |
| the same script logic, with the tie going to `k = 33` or `k = 36` | net fall of 1 or 0 cells | `support`, on one cell of five |
| the review's support rule: five or more of the eight cat and dog seeds must clear 0.05 | at most 1 of 8 clears at any `k` | `null` |
| the review's composition rule: two or more composed seeds lost "at the `k` that wins on fidelity" | two or more lost at `k` 25, 29 and 33; one lost at `k` 36 | depends on the tied `k` chosen |

The review's five-seed rule cannot be met on readable seeds alone, because only four of the eight cat
and dog seeds (9, 11, 12 and 16) have a usable joint-prompt render. The two rank-32 checkpoints give
`null` under every reading, because none of their readable cells clears 0.05 at any `k`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-rank-16-result-under-each-reading-of-the-rule) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Seven recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| what `k` means | the card and the review: the step after which the adapter is detached | the script and the sheets: the adapter is on for steps 0 to k-1 and off from step `k`, so `k = 25` reads "adapter on 0-24" |
| whether the animal count changes on rank 32 | [the card](../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/README.md#handing-the-denoising-tail-back-to-frozen-sdxl-2026-09-19): "the instance count is unchanged at 9 of 10 cells at every k" | the sidecars: the number of cells with two or more animals stays 9 of 10, while the raw count differs from `k = 50` on 5 of the 50 cell-and-`k` comparisons at each checkpoint |
| how widely sharpness falls | the card: as `k` falls "on almost every cell" | the sidecars: sharpness at `k = 25` is below `k = 50` on 8 of 10 cells at 40,000 steps and 7 of 10 at 60,000 |
| where the three reference reads are stored | the card: each sheet's sidecar carries them | the sidecars hold only `reference_valid` per seed (the script, line 583). The three reads are written to `verdict.json` on the cluster (line 432) |
| how many cat and dog references are broken | the card: "all four broken" seeds | the script: three broken (10, 13, 14) and one ambiguous (15), lines 102 to 108 |
| whether the sweep ran | the review's opening line: "Nothing has run" | the review's own run table records it on 2026-09-19, 22:27 to 23:24 |
| what the `lap` number measures | its name, Laplacian variance | [the function](../../../scripts/shared/lambda_window_grid.py) (lines 327 to 330) sums the variances of two mixed row-and-column differences. It returns 0 on a straight vertical or horizontal edge, where a 4-neighbour Laplacian variance returns 2097.6 (a 64 by 64 test image, half at grey level 0 and half at 255) |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The bar may be the wrong question.** It scores closeness to the joint prompt's own picture, and
[the review](../../plans/09-designing-the-correction-loss/review/06-fitting-the-whole-path-and-handing-back-the-tail.md#still-open)
records that this is not the standard the project judges by: a held-out pair has no joint prompt at
inference, and the judgement is whether the picture plainly shows both concepts and looks good `stated`.

**These gains cannot be quoted beside the earlier hand-off result.** [The earlier finding](can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md)
used a different checkpoint, correction strength 1.2, and a mean over eight seeds. Its 0.038 and the
per-seed gains here measure different things `stated` (the review).

**Rank 16 and rank 32 are not compared.** Each checkpoint is read only against its own `k = 50`.

**The review's explanation of the null is one seed.** It measures the adapter lowering the contrast
of the picture the composition aims at by 56% over steps 0 to 9 and 1% over steps 40 to 49, and reads
this as the flattening being finished before step 25. That was measured on the 40,000-step checkpoint,
cat and dog seed 9 `stated`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the sidecar files.

**Handing back the tail**

Ten people paint a wall in turn, and a helper guides the first six. Here `k = 6`: the helper works on
turns 0 to 5 and the painters finish turns 6 to 9 alone. In this sweep, `k = 25` means the adapter
guides steps 0 to 24 and frozen SDXL draws steps 25 to 49. The formula is "adapter on for steps less
than `k`".

**Cosine distance and gain**

Two arrows (1, 0) and (1, 1) have a cosine of 1 / 1.414 = 0.71, so their distance is 1 - 0.71 = 0.29.
Cosine says how closely two arrows point the same way, and distance is what is left of 1. If the
control sits 0.29 from the target and a new picture sits 0.20, the gain is 0.29 - 0.20 = +0.09. For
cat and dog seed 9 on rank 16, the control is 0.684 from its joint-prompt render and `k = 25` is 0.590,
so the gain is +0.093. The arrows here are DINOv2 image embeddings. The vault's note on
[cosine similarity](../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this)
teaches the measure, and no learning page was found for DINOv2.

**A usable reference**

Marking a test against the wrong answer key gives a number that means nothing. A seed's joint-prompt
render is the answer key here, so it must show both animals. Three reads have to agree: the detector,
the saved quality file and a person's look. Five of the ten cells pass. Cat and dog seed 10 draws two
dogs, so its distance is left out of the count.

**The sharpness number**

Take the change from each pixel to the next along a row, then ask how that change differs from one
row to the next. A straight vertical edge has the same change on every row, so it scores 0. This is
what the code computes, and the name Laplacian variance usually means something else: the variance
of each pixel minus its four neighbours. On cat and dog seed 9 at 60,000 steps the number is 435 at
`k = 50` and 299 at `k = 25`. No learning page was found for this measure.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| per-seed gains, counts and printed results | the six `clean-tail-k-sweep-*.json` sidecars beside the sheets, read 2026-11-22 | `verified` |
| the lost and gained cells | compared `composed` at each `k` against `k = 50` for each of the ten cells in those sidecars | `verified` |
| the sheets | [the results card, tail section](../../artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/README.md#handing-the-denoising-tail-back-to-frozen-sdxl-2026-09-19), all six opened 2026-11-22 | `verified` |
| how the renders, scores and branches were made | [the sweep script](../../../scripts/shared/clean_tail_k_sweep.py) | `stated` |
| the question and its rule | [the design](../../plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md) and [its review file](../../plans/09-designing-the-correction-loss/review/06-fitting-the-whole-path-and-handing-back-the-tail.md) | `stated` |
| the run | `mscluster109` device 1, RTX A6000, 2026-09-19 22:27 to 23:24, no W&B run, per the review | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Choose how the pass rule reads for the rank 16 checkpoint, including what a tie between values of `k` means.
- [ ] Decide whether this page is the write-up the review asks for at `1-lora-correction/report/designing-the-correction-loss/08-fitting-the-whole-path-and-handing-back-the-tail.md`. That write-up also carries the verdict and the plan's two figures.
- [ ] Answer the review's open questions: which `k` wins, whether composition held there, whether the eye and the distance picked the same `k`, and whether the training stage was needed.
- [ ] Look at these renders by the project's own standard, whether the picture plainly shows both concepts and looks good. The review records that this has not been done.
- [ ] Decide whether the sharpness function keeps its name or its formula. Every `lap` value on this page comes from it.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
