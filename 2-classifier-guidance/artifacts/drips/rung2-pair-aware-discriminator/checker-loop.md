# Cat × dog checker loop

One change per iteration, kept only if it helps. A keep or drop call needs a difference larger than the evaluation can resolve, so the evaluation set grew to 220 pictures, 109 clean (±0.05), `eval_set_catdog_v2.json`, and every run is read on it. The number is held-out balanced accuracy on `eval_set_catdog.json` (70 cat × dog pictures never trained on: 35 clean pairs, 35 rejects), averaged over the first 20 sampler steps; about ±0.09 at this size. The push call is from full-size tiles on held-out seeds 9 to 16 and is a draft until the user calls it.

**Good** means at least 0.75 on that number, and a push sheet with more "two bodies or better" tiles than plain PoE and than the size-matched random push.

| Iteration | One change | Run | Held-out (35 + 35) | Push test | Kept |
| --- | --- | --- | --- | --- | --- |
| reference | CLIP features plus a linear layer on finished pictures, no noise | probe/catdog.json | 0.771 on the base training data; **0.877** when fitted on iteration 3's 1,603 pictures and read on the 220-picture evaluation (probe/per_animal.json) | none | reference |
| reference | readout on 25% of the cat × dog training set | readout-catdog-frac0.25 | 0.544; 0.562 on the 220-picture evaluation | none | reference |
| reference | readout on 50% of the cat × dog training set | readout-catdog-frac0.5 | 0.695; 0.658 on the 220-picture evaluation | none | reference |
| 0 | readout on the cat × dog training set (753 tiles, 167 clean) | readout-catdog | 0.655; **0.639 on the 220-picture evaluation** | two bodies or heads on 3 of 8 seeds (10, 14, 15), all hybrids, none clean; plain PoE 0 of 8, random push 0 of 8 (draft, full-size tiles) | baseline |
| 1 | add COCO photos to training (44 clean, 725 rejects) | readout-catdog-coco | 0.611; **0.600 on the 220-picture evaluation** | two or more bodies on 3 of 8 seeds (12, 13, 14); seed 12 at s = 30 near a clean pair (white dog and kitten); seed 14 two cats; plain PoE 0 of 8, random push 0 of 8 (draft, full-size tiles) | dropped: 0.039 below the base on the 220-picture evaluation |
| 2 | add 350 new joint drawings to the base (155 clean, 195 rejects; COCO left out) | readout-catdog-more2 | **0.737** on the 220-picture evaluation (0.745 on stature-rule clean pairs) | two or more bodies on 5 of 8 seeds (9, 11, 13, 14, 16); seed 11 at s = 100 a tabby cat before a white-and-grey dog, the first near-clean pair; seed 14 two cats; plain PoE 0 of 8, random push 0 of 8 (draft, full-size tiles) | kept: 0.098 above the base |
| 3 | add 500 more joint drawings (seeds 751 to 1,250; 254 clean, 246 rejects) to iteration 2's list: 1,603 pictures, 576 clean | readout-catdog-more3 | **0.741** on the 220-picture evaluation (0.749 stature-rule; 8 of 20 steps at 0.75 or above on the v3 tiles) | two or more bodies on 4 of 8 seeds (9, 12, 13, 14), none clean; seeds 13 and 14 show two of the same animal, seed 14 has a blue smear; plain PoE 0 of 8, random push 0 of 8 (draft, full-size tiles) | kept, but +0.004 over iteration 2: the first gain under 0.02 |
| lever 4 | per-animal reading on finished pictures: GroundingDINO boxes, CLIP names each crop | probe/per_animal.json | rule 0.588; per-animal logistic 0.769; whole-picture CLIP 0.877 on the same split | none | dropped: the detector finds exactly two boxes on 82 of 111 held-out rejects, so counting cannot separate fused or two-cat pictures |
| lever 5 probe | CLIP linear checker (0.877 on finished pictures) read on SDXL's product estimate of the finished picture | probe/estimate.json | 0.540 averaged over steps 0 to 19 (0.495 at steps 0 and 2, 0.657 at step 20, 0.873 at step 49) | none | dropped as a push checker: early estimates are too blurry to judge |
| 4 | iteration 3's data, trained 2,000 steps instead of 750 (its curve still rose: 0.699, 0.711, 0.741) | readout-catdog-more3-long | **0.755** on the 220-picture evaluation (0.763 stature-rule); **the original bar passed: 0.751 on the v3 tiles, 11 of 20 steps** | two or more bodies on about 7 of 8 seeds; seed 10 at s = 30 a small fluffy dog beside a large cream cat, the second near-clean cat and dog; more of the two-body tiles are a cat with a dog than in iteration 5; seed 16 two dogs; plain PoE 0 of 8, random push 0 of 8 (draft, full-size tiles) | kept: the best checker and push so far; the base for the identity push |
| 5 | iteration 3 exactly, with half of each batch's rejects drawn from the 233 wrong-pair pictures (two cats, two dogs) | readout-catdog-more3-hard | **0.756** on the 220-picture evaluation (0.763 stature-rule; 0.718 and 7 of 20 steps on the v3 tiles); the first final at or above 0.75 | two or more bodies on 3 of 8 seeds (11, 14, 16), none clean: seeds 11 and 14 two cats, seed 16 two dogs; plain PoE 0 of 8, random push 0 of 8 (draft, full-size tiles) | kept as the loop's result: meets both parts of good; +0.015 over iteration 3, inside ±0.05 |
| 6 | identity push: iteration 4 plus two outputs, a cat present and a dog present, trained on the API's per-picture answers; the push follows the sum of all three | readout-catdog-identity | **0.721** on the 220-picture evaluation (0.668 and 4 of 20 steps on the v3 tiles, so the original bar fails); cat present 0.695, dog present 0.728; held-out wrong pairs read as missing an animal 0.407 | two or more bodies on 5 of 8 seeds at s = 100 (9, 11, 12, 14, 16); near-clean cat and dog at seed 9 s = 30 (a cat with a dog behind it) and seed 16 s = 100 (a kitten against a large dog); seeds 11 and 12 a cat beside a kitten or a cat-dog; seed 13 one dog, seed 10 one cat with a dog's muzzle; the push also turns the drawn seeds (11, 14, 15) into photographs; plain PoE 0 of 8, random push 0 of 8 (draft, tile-size calls) | kept as a push, not as a reader: 0.034 below iteration 4 on the evaluation, but two near-clean pairs against iteration 4's one |

## Outcome

The loop stopped at iteration 5, 08:50 UTC on 2026-11-18, because both parts of good held: held-out balanced accuracy 0.756 on the 220-picture evaluation (`eval_set_catdog_v2.json`, the ±0.05 successor to the 70-picture set the loop first named), and a push sheet with two or more bodies on 3 of 8 held-out seeds against 0 for plain PoE and 0 for the size-matched random push.

What carried it was data of the right kind: 350 clean SDXL joint drawings took the checker from 0.639 to 0.737, the next 500 added 0.004, and half-wrong-pair rejects added 0.015. Photos did not help (COCO 0.600), reading each animal separately did not beat a whole-picture read (0.769 against 0.877 on finished pictures), and scoring SDXL's estimate of the finished picture is too blurry over the first 20 steps (0.540).

What is still wrong: no push has made a clean cat and dog beyond one near miss (iteration 2, seed 11 at strength 100). The push mostly makes two animals of the same kind: two cats or two dogs on most of its two-body tiles, even after the checker saw half its rejects as wrong pairs. The next idea is a push that knows which animal is which, for example a cat-and-dog term that rewards one body read as a cat and the other as a dog, rather than more of the same checker.

## Finished-picture fine-tunes

Every weight of a pretrained network trained on the 1,603 finished training pictures (`scripts/rung2/finetune_checker.py`), the epoch chosen on a 10% split of training, read on the 220 held-out pictures. This measures how well the clean-pair label can be learned at all, before any noise; it is not yet a push checker. Logits per picture are in `probe/finetune_<run>.json`; the misses are drawn in `probe/finetune-misses/`.

| Run | Held-out (220) | Wrong pairs | Hybrids | Damaged | Clean kept |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-L/14 linear layer only, frozen (reference) | 0.877 | | | | |
| DINOv2 ViT-B/14, random crops | 0.847 | 35/46 | 19/29 | 13/22 | 107/109 |
| DINOv2 ViT-B/14, no crops | 0.846 | 37/46 | 21/29 | 13/22 | 103/109 |
| CLIP ViT-L/14, random crops, seed 0 | **0.900** | 46/46 | 25/29 | 15/22 | 98/109 |
| CLIP ViT-L/14, random crops, seed 1 | 0.865 | 36/46 | 23/29 | 11/22 | 108/109 |
| CLIP ViT-L/14, no crops | 0.896 | 42/46 | 24/29 | 11/22 | 107/109 |
| CLIP ViT-bigG/14, random crops, seed 0 | 0.873 | 44/46 | 22/29 | 13/22 | 100/109 |
| CLIP ViT-bigG/14, random crops, seed 1 | **0.909** | 46/46 | 24/29 | 16/22 | 100/109 |

Two CLIP seeds differ by 0.035, as large as any change tried, because the 160-picture validation split sometimes picks epoch 1. Most "wrong pair" rejects are a cat and a dog with a third animal, usually a kitten. The damaged rejects are the hardest kind for every run.

## Reading half-drawn pictures

A push never sees the finished picture: at each step the checker reads SDXL's decoded guess at it. `scripts/rung2/make_estimates.py` drew 4,966 such guesses (training pictures at two steps between 10 and 30, the 220 held-out pictures at steps 0 to 40), and `scripts/rung2/finetune_estimates.py` scored the CLIP ViT-L/14 checker (seed 0, 0.900 on finished pictures) on them, then trained it on them for 8 epochs (epoch 4 kept on the validation split). Balanced accuracy on the 220 held-out pictures; 0.50 is a coin flip. Per-step numbers are in `probe/finetune_estimates_clip.json`; the strip of one held-out pair's guesses is `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/half-drawn-guesses/what-the-checker-sees-by-step.png`.

| Sampler step | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 40 | finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP, finished pictures only | 0.50 | 0.50 | 0.51 | 0.58 | 0.62 | 0.74 | 0.81 | 0.89 | 0.90 |
| CLIP, then trained on guesses | 0.51 | 0.54 | 0.72 | 0.82 | 0.84 | 0.89 | 0.87 | 0.91 | 0.90 |
| Readout checker, iteration 4 (noisy latents) | 0.63 | 0.74 | 0.78 | 0.79 | | | | | |

Before step 10 the decoded guess holds no second animal, and no checker trained on it rises above 0.54; the readout checker, which reads SDXL's features, is the better reader there.

## Pushing with the fine-tuned CLIP checker

`scripts/rung2/clip_push.py` turns a fine-tuned CLIP checker into a push: at each step it decodes the sampler's guess at the finished picture, scores it, and takes the gradient back to the noisy latent with the product's noise prediction held fixed (`render_results.py --clip`). Same cap (0.1 of the product prediction), same held-out seeds 9 to 16, strengths 10, 30 and 100. The overview is `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/push-sheets-compared.png`.

| Sheet | Checker | Push on | Two or more bodies, of 8 | What the pictures show (draft, tile-size calls) |
| --- | --- | --- | --- | --- |
| clip-push-sheet-from0 | CLIP ViT-L/14, finished pictures (0.900) | steps 0 to 49 | 1 (seed 16 at s = 30) | the same picture as plain PoE on 7 of 8 seeds, with eyes, mouths and fur redrawn |
| clip-push-sheet-from10 | the same | steps 10 to 49 | 0 | the same picture as plain PoE on every seed; seed 12 scored +0.46 by CLIP with no change a person would see |
| clip-push-sheet-halfdrawn | CLIP trained on half-drawn guesses | steps 10 to 49 | 0 | the same picture as plain PoE on every seed; seed 16 scored +0.10 the same way |

The CLIP push raised its own checker's score on every seed (by 1 to 13.5 logits over the size-matched random push) without moving where the animals are. A gradient through the decoded guess, at the same push size, changes texture; the readout checker's gradient, read off SDXL's own features, changes the layout.

## Push, then pick

`scripts/rung2/push_then_pick.py`: four starting noises per held-out seed (9 to 16), each drawn as plain PoE and with the identity push (iteration 6, strength 100, cap 0.1); the CLIP checker fine-tuned on finished pictures (0.900) picks the best of four in each set. Sheet and tiles in `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/push-then-pick/`.

| Set | Picks CLIP calls clean, of 8 | By eye (draft, full-size tiles) |
| --- | --- | --- |
| plain PoE, best of 4 | 0 (best logit −10.9 to −12.0) | one fused animal on every seed |
| identity push, best of 4 | 3 (seeds 10 +10.2, 16 +6.3, 11 +2.4) | 2 near-clean cat and dog: seed 10 a dog with an extra pair of pointed cat ears beside a tabby kitten; seed 16 a dog with one stray white ear beside a clean grey cat. Seed 11 is two dogs, a false clean call |

Picking among four pushed tries doubles the near-clean count of one pushed try per seed and finds none among plain tries. CLIP still calls two dogs clean, a mistake its evaluation set (where "wrong pair" was mostly a third animal) did not test.
