# 🧭 Progress: what we know about both papers, 2026-11-22

**Neither paper has a headline number that is safe to quote yet.** Both headline counts sit on the [void list](void-list.md) until you decide them. The tables below hold the 22 results with no open void question, and the 8 that have one are [held apart](#held-apart-until-you-decide).

Each status is the word the owning page records. † marks a compose rate from the instance-count scorer: the void list questions rates scored before 2026-11-13 on folders holding comparison strips, and nobody has checked which results those are.

## Table of contents

- [Why plain composition fails](#why-plain-composition-fails)
- [Paper 1: experts that never met](#paper-1-experts-that-never-met)
- [Paper 2: restoring plurality in one model](#paper-2-restoring-plurality-in-one-model)
- [What the text already carries](#what-the-text-already-carries)
- [Held apart until you decide](#held-apart-until-you-decide)

## Why plain composition fails

Navigation: 📋 [TOC](#table-of-contents) | [Next](#paper-1-experts-that-never-met) ➡️

**The product of two experts loses the pair, and the correction that restores it works best in the earliest steps tested.**

| Finding | Measured on | Status | Evidence |
|---|---|---|---|
| With exact scores on a toy, adding the two experts' noised scores draws 15 crisp pairs in 300 draws. The true score of the noised product draws 300 in 300. | two exact Gaussian-mixture experts | no verdict recorded | [the exact toy](../../../3-understanding-compositionality/report/the-exact-toy/how-much-co-occurrence-the-product-needs.md) |
| The plain product of "a cat" and "a dog" gives two clear animals on 0 of 8 seeds at PixArt's sampler settings, the same as at SDXL's own. | SDXL, cat × dog, by eye | support | [SDXL at PixArt's settings](../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md) |
| A 10-step injection of the true correction composes best at steps 0 to 10 on all 8 pairs: 21 of 32 runs, against 8 of 32 at steps 5 to 15 and 3 of 32 at steps 10 to 20. † | SDXL, 8 pairs, 4 seeds | no: the best window does not move with the pair | [the timing finding](../../../1-lora-correction/report/what-the-correction-is/when-does-the-correction-have-to-act.md) |
| Checks that avoid the compose scorer: 56% of 32 runs best match the two-animal caption at correction strength 0.75, up from 3% with none. The image checks are positive and the two text checks are null. | SDXL, cat × dog | two yes, one yes with a caution, two no | [the five checks](../../../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md) |

## Paper 1: experts that never met

Navigation: ⬅️ [Previous](#why-plain-composition-fails) | 📋 [TOC](#table-of-contents) | [Next](#paper-2-restoring-plurality-in-one-model) ➡️

**Separately trained experts miss their bars once the halves are no longer fixed.** No shared latent space has been built, and the one strong count, on fixed halves, is [held apart](#held-apart-until-you-decide).

| Finding | Measured on | Status | Evidence |
|---|---|---|---|
| Two MNIST experts trained apart, each free to draw on either side, make 4 clean pairs of 320 with both pushes and 0 with the plain product. The bar was 224. | five tuning pairs, judge | check failed | [plan 32](../../../2-classifier-guidance/report/classifier-guided-poe/32-mnist-two-experts.md) |
| With no fixed halves, the best two-model run on one shared canvas draws 121 clean pairs of 320 (38%). The bar was 288. | five tuning pairs, judge | inconclusive (proposed) | [plan 33, shared canvas](../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md) |
| Two small networks trained apart on circles and on squares draw 0 separate pairs in 300 samples at every shared fraction from 0 to 0.5. | small diffusion toy | null | [the small-model toy](../../../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md) |

## Paper 2: restoring plurality in one model

Navigation: ⬅️ [Previous](#paper-1-experts-that-never-met) | 📋 [TOC](#table-of-contents) | [Next](#what-the-text-already-carries) ➡️

**The adapter and the checker push both raise composition over the plain product, and on the three benchmarks below the adapter stays under CO3 and the joint prompt.** The joint prompt is available to Paper 2, so doing without it is no contribution. The checker's 4,400-picture pool came from Attend-and-Excite renders, so a checker trained on it cannot be scored fairly on those prompts.

| Finding | Measured on | Status | Evidence |
|---|---|---|---|
| An early checker push draws two separate animals on 28 of 32 new seeds, against 1 of 32 for the plain product and for a random push. Only 14 of the 32 are clean or near-clean. | SDXL, cat × dog, by eye | support | [plan 29](../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md) |
| After retraining the checker on its own push's pictures, one MNIST model draws two clear right digits on 285 of 320 canvases (89%) over five pairs never seen together. The page does not say whether that retraining used those five pairs. | fresh seeds, blind eye count | no verdict recorded | [one-model MNIST](../../../2-classifier-guidance/report/classifier-guided-poe/24-mnist-one-model.md) |
| The pooled adapter composes 7 of 8 held-out cat and dog seeds where the plain product composes 0 of 8. † | SDXL, instance count | no verdict recorded | [the pooled adapter](../../../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md) |
| Switched off after step 10, the adapter puts two or more animals in 75 of 160 images (46.88%), against 147 (91.87%) on for 25 steps and 153 (95.63%) on for all 50. † | SDXL, 53 held-out pairs, detector | no verdict recorded | [adapter timing](../../../1-lora-correction/report/what-the-correction-is/when-does-the-correction-have-to-act.md) |
| The adapter's fit to the true correction is 0.969 (cosine) on training pairs, 0.839 on unseen pairings of learned words and 0.800 on pairs of unseen words, so its gap on new pairs is a pair problem. | SDXL, rank 8, cached states | support: a pair problem; null on drift | [fit, drift or pair](../../../1-lora-correction/report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) |
| On the 66 Attend-and-Excite animal prompts, the adapter scores 0.278 above the plain product on BLIP-VQA (0 to 1), with a 95% interval of 0.1997 to 0.3551. It stays below CO3, the joint prompt and ToMe on the animal sets. | SDXL, 4 seeds per prompt | SUPPORTS | [Attend-and-Excite](../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md) |
| Over 53 held-out pairs, mean BLIP-VQA is 0.6306 for the adapter, 0.5187 for the plain product, 0.7655 for the joint prompt and 0.7757 for CO3. | SDXL, 8 seeds per pair | no verdict recorded | [the 53 held-out pairs](../../../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md) |
| On GenEval two objects, the fraction of 396 images judged correct is 0.4571 for the adapter, 0.2247 for the plain product, 0.7626 for the joint prompt and 0.7500 for CO3. These carry no intervals. | SDXL, seeds 9 to 12 | no verdict recorded | [GenEval](../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md) |
| Pushing the cat and dog attention maps apart produces two animals on 0 of 8 seeds. | SDXL, draft calls at tile size | no verdict recorded | [the attention push](../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md) |
| The best of eight text-space third experts puts two separate animals in frame on 6 of 7 judgeable seeds, and a whole cat beside a whole dog on 0 of 7. The pass mark was 4 of 7. | SDXL, dog × cat, by eye | no verdict recorded | [the third expert](../../../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md) |
| In a toy of 6,000 pictures, relabelling 182 of each caption's 2,000 pictures as both shapes (9.1%) lets the product draw 13, 5 and 72 pairs in 300 across three trainings. At 49 or fewer it draws at most 2. | toy diffusion model | yes, once about 9% | [toy relabelling](../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md) |
| On MNIST, inclusive labels at the highest setting that keeps each single digit working give 19, 16 and 11 of 256 correct pairs, a mean of 15.33 against a line of 39. | three trainings, digit judge | NULL in the counts file, plan line unticked | [MNIST labels](../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md) |
| After fine-tuning SDXL on 550 photos, 150 of them with both animals, the product still draws two animals on 0 of 6 seeds. | SDXL, cat × dog, by eye | no verdict recorded | [the SDXL pilot](../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md) |

## What the text already carries

Navigation: ⬅️ [Previous](#paper-2-restoring-plurality-in-one-model) | 📋 [TOC](#table-of-contents) | [Next](#held-apart-until-you-decide) ➡️

**The text encoder holds a real one-to-two direction, and the joint prompt sits near its middle.**

| Finding | Measured on | Status | Evidence |
|---|---|---|---|
| In SDXL's larger text encoder, "a cat" and "a dog" sit near the one-thing end of a one-to-two direction (-0.40 and -0.38), and "a cat and a dog" sits near the midpoint (+0.05). | 49 animal pairs, text only | no verdict recorded | [what the text says](../../../5-text-embedding-third-expert/report/what-the-text-already-says.md) |
| That one-to-two direction is real: on 41 held-out nouns its step is 43.24 times that of a random direction, where the plan required 3 times. | 200 random directions | no verdict recorded | [the direction check](../../../5-text-embedding-third-expert/report/what-the-text-already-says.md) |

## Held apart until you decide

Navigation: ⬅️ [Previous](#what-the-text-already-carries) | 📋 [TOC](#table-of-contents)

**These results are on the void list, so no finding above rests on them.** Each stays as recorded until you decide it.

| Result | Why it is held apart | Evidence |
|---|---|---|
| **Paper 1 headline.** 1,200 of 1,280 clean digit pairs on 20 unseen pairs, with fixed halves in pixel space | The plan counts by eye and the card leads with the judge. Final seeds are 64 to 127 in one source and 128 to 191 in another. Claude's 15 eye calls give 1,201 to 1,208 and wait for you | [two-model MNIST](../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md) |
| **Paper 2 headline.** The kept push draws 35 of 128 faithful pairs by the scorer and 54 by eye, on cat × dog | Each round was kept or dropped on these same 128 seeds, so the count may flatter it | [the kept push](../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md) |
| On those seeds the joint prompt gets 57 and 69, and CO3 gets 89 and 101 | Same selection question, and CO3's sampler settings are unconfirmed | [the same 128 seeds](../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md) |
| Adding the correction back raises composition from 3% to 94% over 32 runs per point | One index says 7% to 93%, and the curve file reads 3.1% and 93.8% | [the strength curve](../../../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md) |
| The pooled adapter draws two animals in 58 of 64 pictures (0.906) on unseen pairs and 80 of 88 (0.909) on training pairs | These rescored rates replace 0.961 and 0.960, which counted comparison strips. Your call is still recorded as open | [the transfer finding](../../../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md) |
| One MNIST model reaches 84 of 320 on five pairs never seen together | The review puts 84 at push strength 3 and 82 at strength 10, and the page reads as strength 10 | [one-model MNIST](../../../2-classifier-guidance/report/classifier-guided-poe/24-mnist-one-model.md) |
| On released ImageNet weights the classifier calls 46 of 48 right at scale 10, against 9 of 48 at scale 1 | The eye agrees on 30 of 48, so the route's inconclusive condition also holds | [the released weights](../../../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-work-on-the-released-weights.md) |
| The product holds two clear animals on 0 of 8 seeds in three setups, including one with no CLIP | A linked heading says the product fuses on every seed, while that page calls 2 of 8 a clean single cat | [what the fusion is not](../../../3-understanding-compositionality/report/what-sdxl-already-knows/what-the-fusion-is-not-caused-by.md) |

<details>
<summary>The other 13 entries on the void list</summary>

- Sharpness scores labelled "Laplacian variance" in the noise-search finding, the corrector and clean-tail finding and the hand-the-tail finding
- The 1.10 sharpness bar as applied in the render-time fix finding
- "Removes four of the five" differences, in the same finding's Still open list
- Rank 16 at step 10k and λ 0.5 in the SuperDiff adapter finding
- Rank 32 at step 20k, recorded 0 of 4 seeds, in the same finding
- "Every tile keeps its butterfly" in the corrector and clean-tail finding
- "96.9% compose on six unseen pairs", in what each figure argues (lines 503 and 514), the cached-runs review and the subspace picture `geometry_vs_transfer.png`. The paper's `.tex` files do not contain it
- Any other compose rate scored by `compose_rate.py` before commit `0811f029` (2026-11-13) on a folder holding `__cmp` strips
- "No control reaches two animals at any strength", in the `writing/iclr` manuscript, line 252
- Seal and walrus at seed 9, filed as a correct failure in the compose-score check
- "After the adapter stops, the PoE completes the remaining steps", the adapter settings caption in the revision manuscript, line 1314
- The three-concepts figure, R11, in the revision manuscript, line 1388
- The list of runs that draw an unrequested person, in the whole-path review

</details>

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
