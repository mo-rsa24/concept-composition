# 📒 Report: does a fused cat-dog fit both captions better than a real photograph

**Fused renders score 0.014 above real photographs on the weaker of "a photo of a cat" and "a photo of a dog", and renders of one sentence naming both animals score 0.017 above them, so no verdict is recorded and the written rule can be read two ways.** The experiment tests one prediction of the repo's explanation for the fused animal: multiplying "is this a cat?" by "is this a dog?" scores highest on one animal that is ambiguously both. [CLIP](../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Contrastive%20Image-Text%20Training.md#where-to-learn-this), a model that scores how well a picture fits a phrase, scored 137 images against three phrases. Nothing was trained.

## Table of contents

- [The same seed number, fused and as two animals](#the-same-seed-number-fused-and-as-two-animals)
- [Four sets of images on two scores](#four-sets-of-images-on-two-scores)
- [What would count](#what-would-count)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The same seed number, fused and as two animals

Navigation: 📋 [TOC](#table-of-contents) | [Next](#four-sets-of-images-on-two-scores) ➡️

| Product of "cat" and "dog", seed 2 | One sentence naming both, seed 2 | All 32 one-sentence renders |
|---|---|---|
| <a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed02/poe.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed02/poe.png" width="200" alt="One white animal with pointed ears, a mottled face and a red collar"></a> | <a href="../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/joint-tiles/seed02.png"><img src="../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/joint-tiles/seed02.png" width="200" alt="A brown dog and a white cat lying side by side on paving"></a> | `0-reference-where-the-joint-prompt-succeeds.png` is not shown because it failed the subject check |
| One fused animal `verified` (opened at 1024 px). Weaker single-animal score 0.239, pair-phrase score 0.237 `verified`. | A separate dog and cat `verified` (opened at 320 px). Weaker single-animal score 0.251, pair-phrase score 0.318 `verified`. | Each tile holds at least two animals at sheet scale, where a fused animal can pass for a pair `verified`. Seed 8 is two dogs and seed 13 is a dog, a cat and a kitten (both opened at 320 px). |

> 🖼️ **Figure wanted**: a two-panel dot chart drawn from `similarities.json`, with no image tiles. The y-axis is cosine similarity, the x-axis is the four sets, and one dot is one image. The left panel is the weaker single-animal score and the right panel is the pair-phrase score, with a bar at each set's mean. Render into `artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/`, update the card, then embed it here. The three figures already in that folder include real photographs whose licences and contents are unchecked, so they stay off this page.

## Four sets of images on two scores

Navigation: ⬅️ [Previous](#the-same-seed-number-fused-and-as-two-animals) | 📋 [TOC](#table-of-contents) | [Next](#what-would-count) ➡️

**The weaker single-animal score barely separates the sets, and the pair phrase puts every fused render below every one-sentence render.** Both scores are [cosine similarities](../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this) between an image and a phrase, on a scale of -1 to 1. The weaker score is the smaller of an image's similarities to "a photo of a cat" and "a photo of a dog". The pair phrase is "a photo of a cat and a dog". Means and sample standard deviations are from the [summary file](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/summary.json) `verified`.

| Set | Images | Weaker single-animal score, mean ± sd | Pair-phrase score, mean ± sd |
|---|---|---|---|
| real photographs of a cat and a dog (COCO, passed the by-eye audit) | 91 | 0.226 ± 0.021 | 0.285 ± 0.024 |
| product of "a cat" and "a dog" (the fused renders) | 7 | 0.240 ± 0.017 | 0.241 ± 0.018 |
| one sentence, "a cat and a dog" | 32 | 0.243 ± 0.016 | 0.310 ± 0.015 |
| product of "cats" and "dogs" | 7 | 0.226 ± 0.020 | 0.233 ± 0.015 |

| Difference between set means | Weaker single-animal score | Pair-phrase score |
|---|---|---|
| fused minus photographs | +0.014 | -0.043 |
| one sentence minus photographs | +0.017 | +0.026 |
| fused minus one sentence | -0.003 | -0.069 |

On the pair phrase, the highest fused render scores 0.264 and the lowest one-sentence render scores 0.270, so the two sets do not overlap `verified` (recomputed from the [per-image similarities](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/similarities.json)). The differences are the set means in the summary file subtracted.

## What would count

Navigation: ⬅️ [Previous](#four-sets-of-images-on-two-scores) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The plan's rule gives a different answer depending on whether the one-sentence renders are used as a control.** [The plan](../../plans/10-what-poe-loses-when-it-composes/does-the-hybrid-fit-both-captions.md#quick-context-where-you-are) says the claim holds if the plane "puts the product renders up and to the right of the photographs", and fails if "the photographs sit at least as high on the weaker caption". It sets no minimum size for the gap.

| How the rule is read | What was measured | That reading would give |
|---|---|---|
| fused renders against photographs, as written | weaker score 0.240 against 0.226; mean cat similarity 0.262 against 0.234; mean dog similarity 0.249 against 0.239 | the claim holds |
| the same gap, compared with the one-sentence renders, as [the card](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/README.md) reads it | renders with two separate animals also beat the photographs, by 0.017 against the fused renders' 0.014 | the test neither supports nor refutes the explanation |

> [!IMPORTANT]
> Which reading applies is the owner's choice, and no finding records a verdict.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Five recorded facts conflict, and a sixth looks like a conflict until the code explains it.**

| What | One source | The other source |
|---|---|---|
| whether seed numbers mean the same starting noise | the card pairs fused and one-sentence renders by seed number and reports a difference of -0.006 ± 0.021, with 3 of 7 seeds positive | [the set-building script](../../scripts/hybrid_caption_fit/build_sets.py), line 100, matches the numbers only. [The joint render script](../../scripts/render_joint_references.py), lines 6 to 9, says a tile from an old contact sheet differs from this repo's render of the same seed: at seed 19 the mean absolute pixel difference is 26.47 on a 0 to 255 scale. That this is the same sheet is `inferred` from its filename, and seeds 1 to 7 have not been compared |
| the range of the scores | the card and the label at line 100 of [the column plot script](../../scripts/hybrid_caption_fit/plot_columns.py) say 0.165 to 0.331, and the card calls 0.01 about a twentieth of it | the per-image file runs 0.165 to 0.336, the top value being a photograph's pair-phrase score. The label is typed by hand, and 0.01 is 5.9% of that range, about a seventeenth |
| how far fused and one-sentence renders overlap on the pair phrase | the card says they "barely overlap" | they do not overlap: 0.264 is the fused maximum and 0.270 the one-sentence minimum |
| what the one-sentence renders hold | the card says two separate animals | seed 8 is two dogs and has the set's lowest cat similarity, 0.184. Seed 13 holds three animals |
| whether the later steps ran, and on how many images | the plan leaves tasks 2.1 to 4.1 unticked and its considerations say 138 images | their output files exist and hold 137 records, the sum of the plan's own set table |
| how many "cats" and "dogs" product renders exist (explained by the code) | line 10 of the set-building script says 8 seeds | the set holds 7. Line 62 collects only `poe.png`, and the seed 8 folder holds only the render of the one sentence "cats and dogs" |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The photograph comparison is mixed.** Generated against photographed changes together with fused against two animals, so the weaker-score gap cannot be read as an effect of fusing `stated` (the card).

**The sets differ in resolution before CLIP resizes them to 224 px.** Fused renders are 1024 px, one-sentence renders are 320 px crops of a contact sheet, and photographs are about 640 px `stated` (the card).

**The fused set is 7 images, and two are unusual.** Seed 4 holds a wallpaper of flat silhouettes and no animal, and seed 1 has a fused animal in front of a second striped body `stated` (the card). Both stay in the means.

**A similarity is one number per phrase.** It cannot say where in the image the match came from, so a high score does not show two animals are present and is not a species call `stated` (the card).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the two result files named above.

**Cosine similarity between a picture and a phrase**

Give a picture the numbers (3, 4) and a phrase the numbers (4, 3). Multiply matching entries and add: 3 × 4 + 4 × 3 = 24. Each list has length 5, so the cosine is 24 / (5 × 5) = 0.96. Cosine is that sum divided by the two lengths, so it measures whether the lists point the same way and ignores their size. CLIP gives every image and phrase such a list. For the fused render on seed 2, the cosine with "a photo of a cat" is 0.265.

**The weaker of two scores**

One coffee scores 0.8 for strong and 0.3 for sweet, and another scores 0.5 for both. The weaker score is the smaller of the pair: 0.3 for the first and 0.5 for the second, so the balanced coffee wins even though the first has the larger total. Here the fused render on seed 2 scores 0.265 for cat and 0.239 for dog, so its weaker score is 0.239. The one-sentence render on seed 2 scores 0.251 and 0.253, so its weaker score is 0.251.

**A gap measured in standard deviations**

Two classes differ in mean height by 10 cm, and heights within a class typically vary by 5 cm. The gap is 10 / 5 = 2 [standard deviations](../../../../../Obsidian/Mathematics/Statistics/Variance.md#where-to-learn-this), so the classes are easy to tell apart. Divide the gap by the spread of the reference group. Against the photographs, the fused gap on the weaker score is 0.014 / 0.021 = 0.66 standard deviations, and on the pair phrase it is 0.043 / 0.024 = 1.80.

**A paired difference**

Two runners race on the same three days, and the first finishes +1, -2 and +1 seconds ahead. A [paired difference](../../../../../Obsidian/Mathematics/Statistics/Paired%20Difference.md#where-to-learn-this) subtracts within each day, so a slow day hits both runners, and then averages: (1 - 2 + 1) / 3 = 0. Here the seven differences, fused minus one sentence on the weaker score, are +0.014, -0.013, -0.013, -0.009, -0.043, +0.020 and +0.001, averaging -0.006. The pairing only cancels shared noise if each seed number means the same starting noise, which is [the first disagreement above](#where-the-sources-disagree).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| set means, spreads and gaps | [summary file](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/summary.json), read 2026-11-22 | `verified` |
| per-image scores, range and overlap | [per-image similarities](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/similarities.json), recomputed 2026-11-22 | `verified` |
| how the sets, scores and figures were made | [the scripts for this experiment](../../scripts/hybrid_caption_fit/summarise.py), with `build_sets.py`, `embed_clip.py` and `plot_columns.py` beside it, read 2026-11-22 | `verified` |
| settings of the fused renders | [the seed 2 settings file](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed02/poe.json): 50 steps, guidance 7.5, 1024 px | `verified` |
| the claim and its rule | [the plan](../../plans/10-what-poe-loses-when-it-composes/does-the-hybrid-fit-both-captions.md) | `stated` |
| what the images hold and the limits | [the folder's card](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/README.md) | `stated` |
| 38 pictures held out of the drain | 8 COCO photos in [the photo thumbnails](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/photo-thumbs) and the 3 figures in [this result's folder](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions) that paste them in, none opened, plus 27 [joint-prompt tiles](../../artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/joint-tiles) that the contact sheet above repeats | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Choose which reading of the plan's rule applies.
- [ ] Decide whether the three figures that include real photographs may appear on a page.
- [ ] Draw the dot chart from the per-image file, with no image tiles.
- [ ] Render the one-sentence prompt from the cached noise on seeds 1 to 7, so the paired read compares the same starting noise. The plan allows only the unpaired means if the seed numbers do not correspond.
- [ ] Record what the highest-scoring fused render and photograph show at full size, which is the plan's instruction 6.1.
- [ ] Correct the score range on the card and the plot label.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
