# 📒 Report: what four scorers say about a mug and a wine glass

**No verdict has been recorded for this probe.** It runs four published scorers on one pair of images
for the prompt "a mug and a wine glass" at seed 11, to check each scorer before the paper's review
tables use it. All four rank the image from the joint prompt above the image from the plain product
`verified` (the three score files).

## Table of contents

- [The two images and their four scores](#the-two-images-and-their-four-scores)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The two images and their four scores

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| Joint prompt | Plain product |
|---|---|
| <a href="../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/mug-and-wine-glass-seed11-joint.png"><img src="../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/mug-and-wine-glass-seed11-joint.png" width="200" alt="A tall glass with a mug handle standing on a goblet stem and foot, beside a smaller wine glass, on a mauve background."></a> | <a href="../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/mug-and-wine-glass-seed11-product.png"><img src="../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/mug-and-wine-glass-seed11-product.png" width="200" alt="One wine glass with a swirl of red wine on a grey background, and no mug."></a> |
| a tall glass with a mug's handle on a goblet's stem, beside a wine glass `verified` | one wine glass and no mug `verified` |

**All four scorers put the joint image first, and GenEval makes the difference a pass and a fail.** Each row
is one scorer, and the number in each cell is that scorer's output for the image. The scorers are
explained [below](#concepts-this-uses), and [the four-scorer figure](../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/four-scorers-on-mug-and-wine-glass.png)
draws the same comparison.

| Scorer | What the number is | Joint prompt | Plain product | Source |
|---|---|---|---|---|
| CLIP score | 100 times the image and prompt's cosine, 0 to 100 | 26.7271 | 23.5954 | [CLIP and ImageReward scores](../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/probe_clip_ir.csv) |
| ImageReward | predicted human preference, higher is preferred, can go below 0 | 0.5668 | -2.2534 | the same file |
| BLIP-VQA | product of the yes-probabilities for each noun phrase, 0 to 1 | 0.3641 | 0.0044 | [BLIP-VQA scores](../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/probe_blipvqa.csv) |
| GenEval two-object | 1 if a box of each named class is found, else 0 | 1 | 0, "expected cup>=1, found 0" | [GenEval scores](../../artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/probe_geneval.csv) |

## Where the sources disagree

Navigation: ⬅️ [Previous](#the-two-images-and-their-four-scores) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**One recorded description differs from the picture, and it is shown here unresolved.**

| What | One source | The other source | What the files show |
|---|---|---|---|
| what the joint image holds | [the step 9 session note](../../../artifacts/notes/session-merges/2026-11-11-route-step-9-retro.md) says it "shows both objects" | the picture's left object has a mug's handle on a goblet's stem and foot | GenEval's file puts a `cup` box (confidence 0.945) and a `wine glass` box (0.889) on almost the same pixels of that object, and a second `wine glass` box (0.982) on the right-hand glass |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**This is one pair of images.** It shows that each scorer runs and orders these two images the same way.
It gives no error rate for any scorer.

**A pass from GenEval does not mean a clean mug.** The joint image passes because a `cup` box exists, and
that same object also receives a `wine glass` box.

**Scales differ between scorers**, so a gap on one row cannot be compared with a gap on another.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the score files. Counting boxes with a
detector is taught on [the page about the 32 scored images](do-the-scored-successes-contain-both-animals.md#counting-animals-with-a-detector).

### CLIP score

Turn an image and a caption into two arrows and measure how closely they point the same way. That
measure is the [cosine](../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this).
A cosine of 0.25 gives a score of 100 x 0.25 = 25. The formula is 100 x max(cosine, 0). The joint image
scores 26.7271, so its cosine is 0.267. More in [the CLIP score table](../../../../../Obsidian/deep-learning/12-scoring-and-rival-methods/tables/clip-score.md).

### ImageReward

People compare images for a prompt, and a model learns to predict which one they would prefer. If tasters
keep picking coffee A over coffee B, a model trained on those choices scores A higher. The score has no
fixed range. Here the joint image gets 0.5668 and the product image -2.2534. More in
[the ImageReward table](../../../../../Obsidian/deep-learning/12-scoring-and-rival-methods/tables/imagereward.md).

### BLIP-VQA

VQA is visual question answering: the model is asked a question about an image. BLIP-VQA asks one question
per noun phrase and multiplies the yes-probabilities. With "a coin?" at 0.9 and "a die?" at 0.5, the score is
0.9 x 0.5 = 0.45, so one missing object pulls the whole score down. The per-phrase values for this probe are
not saved; only the products 0.3641 and 0.0044 are. More in [the BLIP-VQA table](../../../../../Obsidian/deep-learning/12-scoring-and-rival-methods/tables/blip-vqa.md).

### GenEval two-object

A detector looks for each object the prompt names, and the image passes when each has at least one box. For
"a coin and a die", finding a coin and no die is a fail. Each concept is named by its COCO class, so a mug is
looked for as `cup`. The product image has one `wine glass` box and no `cup`, so it fails. More in
[the GenEval table](../../../../../Obsidian/deep-learning/12-scoring-and-rival-methods/tables/geneval.md).

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the scores and boxes | the three score files linked in the table, read 2026-11-22 | `verified` |
| how each score is computed | [the scorer section of the review protocol](../paper-review/protocol.md#the-scorers) | `stated` |
| which image is which | the score files name `mono.png` and `poe.png`; the session note identifies the joint and product renders | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Write the card for this folder, which its index row lists as "no card yet".
- [ ] Decide whether a pass on a mug fused to a goblet should count as holding a mug.

## Cross-references

- The mention of **CLIP score** in [the report on the 276 Attend-and-Excite prompts](../against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md).
- The mention of **GenEval two-object** in [the report on the GenEval and T2I-CompBench benchmarks](../against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md).
- The mention of **GenEval two-object** in [the finding "Do the rival methods reproduce the scores their papers print?"](../against-other-methods/do-the-rivals-reproduce-their-printed-scores.md).
- The mention of **GenEval two-object** in [the context page on Latent CLIP, how its push differs from NoisyCLIP's](../../../context/methods/05-latent-clip/03-how-its-push-differs-from-noisyclips.md#beside-how-its-authors-used-the-gradient).
- The mention of **the scorer probe finding** in [the context page on the BLIP-VQA score, how one picture earns it, section "How it fails and how you know it worked"](../../../context/methods/08-the-blip-vqa-score/01-overview.md#how-it-fails-and-how-you-know-it-worked).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
