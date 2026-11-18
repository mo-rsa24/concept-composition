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

## A larger cap

The identity push (iteration 6) with the cap on each step's push raised from 0.1 to 0.3 of the product prediction's norm, strengths 30 and 100, seeds 9 to 16, plain PoE and best-of-2 reused from the cap-0.1 sheet. Overview: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/cap-0.1-vs-0.3.png`.

| Cap | What the pictures show (draft, tile-size calls; two checked at full size) |
| --- | --- |
| 0.1 | two bodies on 5 of 8 seeds at strength 100, mostly two cats, a cat and a kitten, or a hybrid pair |
| 0.3 | a separate cat and a separate dog on most seeds at both strengths (seeds 10, 11, 12, 13, 14, 15, 16); the pictures are damaged: flattened, oversaturated colour, blocky banding, a painted blue patch on seed 10 |
| 0.3, random push of the same size | plain PoE's fused animal on every seed |
| 0.5 | at strength 30 a separate cat and dog on about the same seeds as cap 0.3, with more colour damage; at strength 100 the pictures break down (seed 14 blocky colour patches, seed 13 a yellow blob, seed 12 black-and-white shapes). Overview: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/cap-0.3-vs-0.5.png` |
| 0.5, random push of the same size | plain PoE's fused animal on every seed |

At full size, seed 11 at strength 100 is a ginger cat beside a St Bernard: a clean cat and dog from one trajectory, overexposed and hazy. Seed 13 at strength 30 is a tabby cat beside a black-and-white dog whose head is hidden, with the cat's face garbled. The push was too small at 0.1: three times the room turns two cats into a cat and a dog, and costs picture quality; five times the room at strength 100 destroys the picture. Cap 0.3 is the best trade-off of the three.

## Two pushes: bodies early, species late (plan 26)

The species checker (CLIP ViT-L/14 fine-tuned with two outputs, a clear cat present and a clear dog present, on finished pictures and half-drawn guesses) reads, at step 15 on the 220 held-out pictures' guesses, cat 0.711 and dog 0.847: Q1 inconclusive (both had to reach 0.80; null below 0.70). The combined push (iteration 6's readout checker on steps 0 to 9, the species checker from step 10, cap 0.1) is in `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan26-both-vs-identity.png`.

The bodies push alone (the readout checker on steps 0 to 9, nothing after) draws almost the same pictures as the combined push at every strength (`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan26-bodies-vs-both.png`): one cat-dog face on most seeds at strengths 30 and 100, two animals on a few seeds at strength 10 (13 a cat beside a dog). The species push from step 10 changes little, as the species push alone shows: every seed stays plain PoE's fused animal with faces and eyes redrawn (`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan26-species-alone.png`), as every push through the decoded picture has. What merges the two bodies is the readout push stopping at step 9: the identity push over all 50 steps keeps two bodies on 5 of 8 seeds, the same push over steps 0 to 9 keeps them on 1 or 2. PoE pulls the animals back together once the push stops.

## Pushing hard early, then letting PoE finish

The identity push at cap 0.3 on steps 0 to 15 only, plain PoE on steps 16 to 49 (`result-sheet-catdog-identity-cap0.3-early`), beside the same push on all 50 steps: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/cap0.3-all-vs-early.png`.

A separate cat and dog holds on the same seeds as the push over all 50 steps (9, 10, 11, 12, 13, 14, 16 at one strength or both), and the colours are closer to a normal photograph on most of them; seed 15 at strength 100 is still posterised. At full size, seed 11 at strength 100 is a fluffy ginger cat beside a beagle in natural colour, a near-clean pair whose cat face has a little fox in it; seed 13 at strength 100 is a brown cat beside a brown-and-white puppy, normal colour, with the cat's face garbled and cat whiskers on the puppy. The random push of the same size over the same steps stays plain PoE.

The layout is settled by step 15 when the push is strong enough: cap 0.3 to step 15 keeps two animals where cap 0.1 to step 9 (plan 26's bodies push) let PoE pull them back together. Most of the colour damage comes from pushing on the late steps. What remains is face detail: borrowed features (a cat's whiskers on a dog, a garbled cat face).

## Self-recurrence, and NOT-hybrid guidance (plan 27)

Overview: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan27-and-recurrence.png`.

**Self-recurrence** (Universal Guidance's per-step repeat): the identity push at cap 0.1 with each of steps 0 to 15 taken 3 times. Two animals on about 4 to 5 of 8 seeds in natural colour (seed 9 at strength 100 a kitten beside a black-and-white dog, seed 11 a ginger kitten beside a black-and-white cat-dog), about what the identity push gives without repeats, and short of cap 0.3 on steps 0 to 15.

**NOT-hybrid guidance** (plan 27): PoE minus SDXL's own direction for "a hybrid animal, half cat and half dog". The λ 0 loop matches plain PoE pixel for pixel on all 8 seeds. The checker's weight sat at its clamp (k about 19) on every seed, so the checker-weighted runs equal the size-matched fixed control. No setting gives a separate cat and dog; the term moves the style instead, to ornate line drawings and round frames, and at λ 0.1 to a full-frame pattern with no animal. SDXL's "hybrid animal" direction is a style direction, not the direction of fusion.

## Strong push late instead of early

The identity push at cap 0.1 on steps 0 to 9 and cap 0.3 from step 10 to 49 (`result-sheet-catdog-identity-cap0.3-late`), beside cap 0.3 on steps 0 to 15 only: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/early-vs-late-strong-push.png`.

Two separate animals on most seeds again, in softer and more natural colour than the strong-early run; the random push of the same size stays plain PoE. Near-clean cat and dog (draft, tile size) on 3 seeds (9 a tabby cat beside a black-and-white dog, 12 the same, 14 a tabby cat beside a fluffy dog) against 4 for strong-early (9, 11, 15, 16), on different seeds. Seed 11 gains a third animal (a dog with two kittens); seeds 10 and 16 keep a cat-dog beside a kitten. A strong push need not be early: pushing hard over steps 10 to 49 after a gentle start separates the animals as often, and which seeds come out clean depends on when the push is strong.

## Each animal toward its own species (plan 28)

The early push (cap 0.3, steps 0 to 15), then from step 16 each detected animal pushed toward its own species by zero-shot CLIP on its box of the decoded guess (cap 0.1). Per-seed comparisons with the early push alone, at full size: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-catdog-boxes/compare/`.

The detector found two or more animals at step 16 on all 16 pictures. Clean or near-clean cat and dog (draft, full size) on 4 seeds (9, 11, 12, 16), against 4 for the early push alone (9, 11, 15, 16). The per-animal push changes species outright: seed 9 at strength 30 a puppy-faced animal becomes a black cat beside a spaniel; seed 11 at 30 a sketch-like cat becomes a grey tabby beside a red dog; seed 12 at 30 a garbled cat becomes a fluffy puppy beside a black-and-white cat. It also turns the wrong animal: seed 15 at 30 the tabby becomes a small dog (two dogs), seed 13 at 100 the puppy becomes a cat. Those follow a wrong naming at step 16, where the decoded guess is still blurry. New artifacts appear on seeds 10 (a flat gold cutout, a black hole for a muzzle) and 16 at strength 100 (a yellow cloth over the dog's face).

## Plan 25's runs, labelled

1,200 cat × dog seeds (2001 to 3200) drawn by plain PoE and by the joint prompt, each finished picture judged by the CLIP checker fine-tuned on finished pictures (clean above a logit of +2, reject below −2). Plain PoE ended clean on 0 of 1,200 seeds; the joint prompt on 613 of 1,200 (508 of 1,000 training, 105 of 200 held out), with 33 left out as unsure. Spot check of 32 training runs (`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan25-labels/spot-check.png`): 16 of 16 clean calls right, at most 1 doubtful reject.

**The confident version** (a box pushed only when zero-shot CLIP gives its species at least 0.8 at step 16; comparison of all three versions: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan28-confident-compare.png`). Both boxes were confident on about a third of the pictures; on most of the rest the dog box read as a cat (p(dog) 0.08 to 0.21), so the rule skipped exactly the two-cat pictures that needed a push toward dog. At full size: clean or near-clean on 4 of 8 again (9, 11, 12 a tabby cat beside a dog that looks like a porcelain figurine, 16); seed 10 at strength 30 a clean ginger cat beside a long-haired dog with a cat's face; seed 13 at 100 a clean pug puppy beside a cat with a garbled face; one wrong-species turn remains (seed 14 at 100 the shaggy cat becomes a small dog in a sweater, two dogs). Many tile-size "clean" calls did not survive full size.

## The early push on 32 new seeds (plan 29)

The recipe unchanged (iteration 6's readout push at strength 30, capped at 0.3, on steps 0 to 15, then plain PoE) on seeds 3301 to 3332, never used for training, evaluation or tuning. Calls by eye from full-size tiles (`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-catdog-general/views/full/`): a separate cat and dog on 28 of 32, clean or near-clean on 14 of 32 (9 clean), against 1 of 32 separate and 0 clean for plain PoE and for the size-matched random push. Of the other 18: two bodies with a borrowed face or a hidden head on 11, two of a kind on 3 (cats on 3313 and 3327, dogs on 3331), not separate on 4 (a flat cutout or shape as the second animal on 3309 and 3326, one animal on 3316 and 3325). The pass marks written before the run (separate on 19, clean or near-clean on 10, controls at most 2) all hold. Half-size calls had counted 17 clean or near-clean; 3 did not hold at full size.

## Corrector steps on the early recipe

The early recipe (cap 0.3, steps 0 to 15) with each of steps 0 to 15 taken 3 times, re-noising between repeats (`result-sheet-catdog-cap0.3-early-corrector`), beside the recipe without them: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/corrector-compare.png`. Draft calls at tile size: two separate animals on about the same seeds, clean or near-clean on about 3 of 8 (10 and 11 at strength 30, 16), against 4 without corrector steps. The corrector adds damage: flat cutouts and shapes as the second animal (seed 9 a stone statue, seed 10 at strength 100 flat colour planes), stronger oversaturation at strength 100, and one animal on seed 13 at 100. The random push with the same corrector steps stays one animal and turns 4 of 8 seeds into line drawings, the same drift to illustration the Langevin corrector grid found. More push per noise level does not add clean pairs.

## The early push at cap 0.2 on the 32 new seeds

The recipe with the cap at 0.2 instead of 0.3 (`result-sheet-catdog-general-cap0.2`), each seed beside its cap-0.3 picture: `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-catdog-general-cap0.2/views/`. Calls by eye at 640 px.

| | Cap 0.3 | Cap 0.2 |
| --- | --- | --- |
| Separate cat and dog | 28 of 32 | 22 of 32 |
| Clean or near-clean | 14 of 32 (9 clean) | 10 of 32 (5 clean: 3307, 3315, 3319, 3320, 3325) |
| Two of a kind | 3 | 4 (3312, 3317, 3327, 3331) |
| Not separate | 4 | 10, including an object as the second animal on 4 (a chrome cat on 3311, a black cat statue on 3313, a cloth cat house on 3326, a paper cone on 3330) |

The smaller cap does not remove the cutouts; it trades separations for single animals and objects. Cap 0.3 stays. The two caps give clean pairs on partly different seeds (cap 0.2 adds 3315, 3316, 3319, 3324, 3325), so the push's size changes which seeds succeed, not only how many.

## The cat × dog checker on two other pairs (plan 30, step 0)

The recipe unchanged, with the checker reading each picture with the pair's own joint prompt, on seeds 3301 to 3316. Calls by eye from full-size tiles.

| Pair | Plain PoE separate | Random push separate | Recipe: separate, right two kinds | Recipe: clean |
| --- | --- | --- | --- | --- |
| lion × meerkat | 1 of 16 | 1 of 16 | 6 of 16 (10 with two animals, 4 of them two meerkats) | 3 of 16 |
| typewriter × cactus | 1 of 16 | 1 of 16 | 14 of 16 | 6 of 16 |

A checker trained only on cats and dogs separates two things on new pairs. It cannot tell which body should be the lion, so the animal pair loses seeds to two meerkats; the object pair has no such confusion. Plan 30 step 1 trains a checker across 78 pairs to test whether that confusion goes away.

## An automatic scorer for push sheets

`scripts/rung2/score_sheet.py` labels each tile of a push sheet with the Claude API labeller (`api_label.py`, default rules, effort low; about 2,170 input and 99 output tokens per picture) and counts, per column, tiles with two or more bodies and "faithful pair" verdicts. Checked against Claude's eye calls from full-size tiles on plan 29's 96 pictures (`calls_plan29.json`; labels in `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/scorer-check/plan29.jsonl`):

| | Eye | Scorer | Agree |
| --- | --- | --- | --- |
| pushed, two bodies | 28 of 32 | 29 of 32 | 29 of 32 seeds |
| plain, two bodies | 1 | 1 | 30 of 32 |
| random push, two bodies | 1 | 2 | 31 of 32 |
| pushed, clean (eye: clean or near-clean; scorer: faithful pair) | 14 | 5 | 23 of 32 |

Two bodies agree on 91% of seeds; the three disagreements are a flat cutout counted as a body (3309, 3326) and a cat's head on a dog's body counted as one (3319). The scorer's "faithful pair" is stricter than the eye and never more lenient: all 5 are among the eye's 14, and each of the 9 it rejects names a borrowed feature (cat whiskers on the dog, a dog with a cat-like face, a kitten with a puppy's muzzle). It stands in for eye calls on large sheets as a count of two bodies and a conservative count of clean pairs; plain PoE and the random push give 0 faithful pairs of 32.

## The readout checker trained on PoE's own states (plan 25), final

`readout-catdog-traj`, iteration 4's settings plus half of each batch from the 1,972 labelled training runs' saved states: mean over sampler steps 0 to 10 on 195 held-out joint runs 0.730 at step 2,000, against 0.677 for iteration 4 (0.59 against 0.51 at step 0, 0.73 against 0.64 at step 4). In between Q1's marks: +0.053 over iteration 4, under 0.75. The older check on finished pictures with noise added falls to 1 of 20 steps. Its push sheets follow.

## The improvement loop (from 2026-11-18, 17:50 UTC)

One change per round, each judged on cat × dog evaluation seeds 6001 to 6128 (never used for training or tuning) with plain PoE and the size-matched random push beside it, scored by `score_sheet.py` (two bodies, faithful pair) and spot-checked by eye. A change is kept when faithful pair rises by at least 4 of 128 and two bodies does not fall. The baseline is the recipe with iteration 6's checker (`result-sheet-catdog-big`).

| Round | One change | Run | Two bodies of 128 | Faithful pair of 128 | Kept |
| --- | --- | --- | --- | --- | --- |
| baseline | the recipe, iteration 6's checker | result-sheet-catdog-big | pending | pending | baseline |
| 1 | plan 25's checker (`readout-catdog-traj`, trained on PoE's own states) in place of iteration 6's | result-sheet-catdog-big-traj | pending | pending | pending |

**Rescoring plan 30's step 0 with the scorer** (48 pictures each, about 2,180 input and 84 output tokens per picture):

| Pair | Plain PoE: two bodies / faithful | Random push | Recipe | Eye (recipe) |
| --- | --- | --- | --- | --- |
| lion × meerkat | 3 / 0 | 0 / 0 | 10 / 0 | 10 two animals, 6 lion and meerkat, 3 clean or near-clean |
| typewriter × cactus | 6 / 0 | 7 / 0 | 16 / 1 | 14 separate, 6 clean |

On objects the scorer counts loose cacti as extra bodies, so "two bodies" is high for plain PoE and the random push (6 and 7) and does not separate them from the recipe as cleanly as the eye calls; its faithful pair calls a typewriter with cacti growing on it a hybrid. For lion × meerkat it matches the eye on two bodies (10) and finds no faithful pair, stricter than the eye's 3.

## What counts as a clean training picture (from the benchmark pool on)

New checker data comes from the 276-prompt Attend-and-Excite benchmark renders (`collect_benchmark_pool.py`: the joint prompt, the adapter, CO3 and ToMe at seeds 9 to 12; 4,400 pictures across 275 pairs; plain PoE left out as almost all hybrids) and from plan 30's 78-pair joint runs. Every picture is labelled by the strict Claude API labeller (`label_pool.py`, 768 px, about 1,640 input tokens per picture):

| Verdict | Training label |
| --- | --- |
| faithful pair (two bodies, a clear A and a clear B, no borrowed feature or body shape, an intact photograph) | clean |
| hybrid, single, wrong pair | reject |
| damaged, or any verdict on a picture the labeller marks not an intact photograph | left out, so style cannot carry the label |

Left out of training by rule: the test pairs lion × meerkat and typewriter × cactus, and cat × dog at seeds 9 to 16. Each source is spot-checked by eye (16 clean, 16 rejects) before training uses it.

**The benchmark pool, labelled.** 4,400 pictures, about 7.24 million input tokens (1,646 per picture). The first rule left drawings in the rejects: the labeller calls a picture "damaged" only when no other verdict applies first, so a cartoon of two lions came out "wrong pair" while nearly every clean picture was a photograph (spot-check sheets `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/benchmark-pool/spot/`). Every picture not marked an intact photograph is now left out whatever its verdict. Kept: 611 clean and 645 rejects over 200 pairs (joint 178 / 165, adapter 30 / 161, CO3 157 / 185, ToMe 246 / 134); the clean calls in the spot-checks are right on all four methods.

**More runs of the same kind (plan 25 Q3): null.** `readout-catdog-traj-plus1000`, plan 25's training with 1,000 more cat × dog joint-prompt runs: 0.724 over steps 0 to 10 at step 2,000, against 0.730 without them (mark: +0.02). Faster early (0.728 against 0.671 at step 500), no higher at the end. The loop's data lever moves to new pairs and new methods (the benchmark pool, the 78-pair runs).

**Next architecture round: a LoRA inside the checker.** `train_readout_pair.py --trunk-lora 16` trains rank-16 adapters on a separate copy of SDXL's down and middle blocks (attention q, k, v, out) together with the head, so the features the checker reads can adapt instead of staying frozen; the push uses that adapted copy (`render_results.py` loads it from the checkpoint). Run `readout-multipair-lora16` differs from `readout-multipair` only by the adapters. Plan 22 tried trunk adapters with 81 positives over 38 pairs and failed its check at every step; this run has about 1,190 clean pictures over about 200 pairs. Kept if, on lion × meerkat seeds 3301 to 3316, it gives more seeds with a lion and a meerkat than `readout-multipair`, and on the 128 cat × dog seeds at least 4 more faithful pairs.

**Round: the fine-tuned CLIP with backward guidance.** Universal Guidance's backward step (`render_results.py --clip <checker> --backward-steps 3 --backward-lr 1.0`): at each pushed step, 3 Adam steps on a change to the estimate of the finished latent, scored as a finished picture by the CLIP checker trained on half-drawn guesses (`finetune/estimates-clip/best.pt`), carried into the noise prediction and capped at 0.3 of PoE's prediction as the recipe is; steps 0 to 15, then plain PoE. Judged on seeds 3301 to 3332 against plan 29's readout recipe there (scorer: two bodies 29, faithful pair 5). Kept as the better push if faithful pair reaches 7 of 32 with two bodies at 29 or more and the random push of the same size at 0 faithful pairs.
(The first launch, 5 Adam steps at learning rate 0.05, was stopped after one seed: the cap fired on 0% of steps and the checker's logit moved by 0.9 over the random push, because carrying a change to the finished estimate into the noise prediction multiplies it by √(ᾱ/(1−ᾱ)), about 0.07 at step 0. The relaunch uses learning rate 1.0 so the push reaches the recipe's cap, and 3 steps to keep it near 8 minutes a seed on a shared card.)
**Backward guidance: dropped after 8 of 32 seeds** (`result-sheet-catdog-backward`, overview `artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-catdog-backward/first-8.jpg`). The push reaches the cap on 18 to 20% of the 50 steps (most of the 16 pushed steps) and raises the CLIP checker's score on the finished picture by 0.3 to 3 logits over the random push, yet on all 8 seeds the layout is plain PoE's: one fused animal, redrawn in the face and fur, where the readout recipe gives two animals on 8 of 8 of the same seeds. Stopped there, since no separation on 8 seeds cannot reach the pass mark of 7 faithful pairs of 32 and the card is needed for the multi-pair checkers. A pixel checker pushed through the decoded estimate moves texture, not layout, with one forward step or several backward steps.

## What the MNIST session (step 7b) carries over

`report/classifier-guided-poe/a2b-mnist-digits-on-a-canvas.md` and its review file. On a digit model where the plain product fuses, a pair-aware readout push composes the seen pair (47 of 64 against 1 for plain, random and best-of-2) and, after three changes, the five pairs never seen together. The changes that carried most, each untried on SDXL in the form it took there:

| MNIST change | Effect on MNIST | Its SDXL form |
| --- | --- | --- |
| a partly random sampler (DDIM eta 1.0) under the push | 85 to 120 of 192 on the tuning pairs; eta alone 0 to 1 | `--eta 1.0` on the recipe (round below) |
| the checker retrained on its own push's renders, labelled by the judge, two passes | 77 then 120 of 192; the weakest held-out pair 4 to 37 of 64 | retrain the readout checker on the recipe's own renders labelled by the scorer |
| the separate-label push added to the pair-aware push, both at s = 10 | 215 to 285 of 320 on held-out pairs (89%) | the identity push already sums "clean pair", "cat present" and "dog present"; a separate checker per animal is untried |

Two MNIST readings match SDXL's: two checkers at the same accuracy composed 47 and 27 (accuracy does not decide the push), and the misses are a wrong second digit or one digit alone (two meerkats, two of a kind).

**Round: the recipe with a partly random sampler.** `render_results.py --readout --head <iteration 6> --eta 1.0`, strength 30, cap 0.3, steps 0 to 15, on seeds 3301 to 3332; the sampler's noise is fixed per seed (one trajectory) and shared by the random push row. Kept if the scorer gives faithful pair on at least 7 of 32 with two bodies at 29 or more, against the recipe at eta 0 there (29 and 5), and the random push at eta 1.0 at most 2 of 32 two bodies.

**Round: what the presence outputs add (ablation).** Iteration 6 sums "clean pair", "cat present" and "dog present" in one head; MNIST's gain came from a separate presence checker at its own strength. First, iteration 4 (one output, no presence) in the recipe on seeds 3301 to 3332 (`result-sheet-catdog-general-iter4`), against iteration 6 there (scorer 29 two bodies, 5 faithful pairs). The presence outputs earn their place if iteration 6 leads by 2 or more faithful pairs. Next, written before it runs: a presence-only readout checker pushed at its own strength beside the pair-aware push.

**Round: a separate presence checker beside the pair-aware one (MNIST's 215 to 285 of 320).** A second readout checker, `readout-catdog-presence-only`, is trained on iteration 6's data with only two outputs, "cat present" and "dog present" (the 1,823 labels in `presence_catdog.json`). Its push is added to iteration 6's at the same strength (weight 1), both reading one pass of SDXL, inside the recipe (strength 30, cap 0.3, steps 0 to 15), on seeds 3301 to 3332 (`result-sheet-catdog-presence-added`). Kept if faithful pairs reach 7 of 32 or more (iteration 6 alone: 5) and two bodies stays at 29 or more; the random push of the same size stays the control.

**Dropped: the trunk LoRA checker (`readout-multipair-lora16`).** Rank-16 adapters on a separate SDXL copy's down and mid attention, trained with the readout head on the multi-pair list. At step 500 it had not learnt: "cat present" and "dog present" at 0.500, clean against reject 0.525 on the check tiles, a mean of 0.522 on held-out joint runs over steps 0 to 10, and a loss of about 2.1 (chance for the three outputs). The same list without the adapters (`readout-multipair`) read 0.686 by step 500. Stopped at step 500 (about 7 s a step on mscluster112); its lion × meerkat and 128-seed renders never ran. A trunk LoRA would need fp32 adapter weights or a lower learning rate before it is tried again.

**Dropped: the recipe with DDIM eta 1.0 (`result-sheet-catdog-eta1`, seeds 3301 to 3332).** Scorer: two bodies 28 of 32 against 29 at eta 0, faithful pair 2 against 5; plain PoE 1 and 0, the random push 1 and 0. Below the mark (faithful 7 or more, two bodies 29 or more). By eye the partly random sampler keeps the animals apart but turns the photograph into a flat illustration or a posterised, smeared picture on most seeds (the scorer calls 7 of 32 damaged); seeds 3307 and 3329, clean at eta 0, become flat drawings. MNIST's gain from eta 1.0 does not carry to SDXL at this cap and window. Side by side for four seeds: `result-sheet-catdog-eta1/eta1-against-eta0-four-seeds.png`. API tokens: 2,168 in per tile, 96 tiles.

**Found: a pushed render is not reproducible run to run.** Seed 9 under the recipe, rendered three times with the same code path and settings (`result-sheet-catdog-identity-cap0.3-early`, `plan31-identity-recipe`, and `plan31-identity-w0` with plan 31's summed push at species weight 0), gives three different pictures (mean pixel difference 19 to 22 of 255; a fused puppy-cat, a cat with a toy roll, a separate puppy and cat). Plain PoE, best-of-2 and strength 0 are byte-identical across runs, so the difference enters with the push: its size at step 0 already differs in the fourth decimal (40.578 against 40.579), which points at the gradient through SDXL not being bit-reproducible on the GPU, and the 16 pushed steps amplify it. Every per-seed comparison between two pushed runs therefore carries run-to-run noise as well as the change being tested. Measured next: the recipe re-rendered unchanged on seeds 3301 to 3332 (`result-sheet-catdog-general-rerun`), scored and called against plan 29's 28 separate, 14 clean or near-clean, scorer 29 two bodies and 5 faithful pairs.

**Result: what the presence outputs add (`result-sheet-catdog-general-iter4`, seeds 3301 to 3332).** Iteration 4 (clean-pair output only) in the recipe: scorer two bodies 26 of 32, faithful pair 3; iteration 6 (plus "cat present" and "dog present") on the same seeds in plan 29: 29 and 5. Plain PoE and the random push 1 and 0 on both sheets. Iteration 6 leads by 2 faithful pairs, which meets the mark written before the run, so the presence outputs stay. The lead equals the mark exactly and sits inside the run-to-run noise found on seed 9; the recipe re-render (`result-sheet-catdog-general-rerun`) gives its size. API tokens: 2,168 in per tile, 96 tiles.

**Result: round 1, plan 25's checker on 128 new seeds (`result-sheet-catdog-big-traj`, seeds 6001 to 6128).** Scorer: two bodies 101 of 128, faithful pair 14; plain PoE 1 and 0, the random push 1 and 0. Kept or dropped once the iteration 6 baseline on the same seeds (`result-sheet-catdog-big`) is scored. API tokens: 2,168 in per tile, 384 tiles.

**Baseline on 128 new seeds (`result-sheet-catdog-big`, seeds 6001 to 6128).** The recipe with iteration 6: scorer two bodies 114 of 128, faithful pair 25; plain PoE 1 and 0, the random push 1 and 0. API tokens: 2,168 in per tile, 384 tiles.

**Dropped: round 1, plan 25's checker (`result-sheet-catdog-big-traj`).** Two bodies 101 and faithful pair 14 of 128, against the baseline's 114 and 25 on the same seeds: both fall, so training on PoE's own states does not carry into the push. Dropped rounds in a row: the trunk LoRA checker, round 1 and DDIM eta 1.0 make three.

**Round: a longer window (`result-sheet-catdog-big-window25`, seeds 6001 to 6128).** The recipe with the push on steps 0 to 24 instead of 0 to 15, everything else unchanged; plain and best-of-2 tiles reused from the baseline sheet. On plan 31's seeds 9 to 16 the push over all 50 steps kept two animals on 8 of 8 but streaked the background, so this tries the middle. Kept if faithful pair reaches 29 or more of 128 and two bodies stays at 114 or more.

**Labelled: the 78-pair joint runs (`pair-runs-pool`, 3,120 pictures, seeds 5001 to 5040).** Strict API labeller: faithful pair 393, hybrid 1,381, wrong pair 725, single 55, damaged 566. The clean rule now sits in `label_pool.py` itself: a reject must also be an intact photograph, so drawings and etchings are left out on both sides (clean 393, reject 778, left out 1,949). Spot-checked by eye: the clean sample is a squirrel and a chipmunk, a hawk and a falcon, an alpaca beside a wheelbarrow, a lighthouse and a sailboat, a telescope and a beach ball; the photographic rejects are a horse beside a goat-horse, a cow-horse beside a horse, a gazelle beside a vulture-gazelle. API tokens: 5,131,080 in (1,645 per picture), 324,022 out.

**Round: the multi-pair checker with the 78-pair runs added (`readout-multipair2`).** `training_set_multipair2.json`: 4,030 tiles over 315 pairs (1,603 cat × dog, 1,256 from the benchmark pool, 1,171 from the 78-pair runs), lion × meerkat and typewriter × cactus held out. Same training as `readout-multipair`. Pushed in the recipe on lion × meerkat seeds 3301 to 3316 and cat × dog seeds 6001 to 6128. Kept if cat × dog faithful pair reaches 29 or more of 128 with two bodies at 114 or more (baseline 25 and 114), or if lion × meerkat faithful pair beats both the baseline checker's 0 of 16 and `readout-multipair`'s count by 2 or more without cat × dog falling below 21 faithful pairs.

**Measured: run-to-run noise of the recipe (`result-sheet-catdog-general-rerun`, seeds 3301 to 3332).** The recipe re-rendered unchanged: scorer two bodies 30 of 32, faithful pair 7, against 29 and 5 in plan 29's run of the same seeds and settings; plain PoE 1 and 0, the random push 2 and 0. Rendering the same thing twice moves the faithful-pair count by 2 of 32, so a +2 of 32 mark cannot separate a change from a re-run. The presence-output ablation's lead (5 against 3) sits inside this; the presence outputs stay because nothing argues for removing them, not because the ablation showed they help. Marks on 32 seeds need +4 from here on, and 128-seed sheets carry the decisions. API tokens: 2,168 in per tile, 96 tiles.

**Dropped: the window of steps 0 to 24 (`result-sheet-catdog-big-window25`, seeds 6001 to 6128).** Scorer two bodies 122 of 128 (baseline 114), faithful pair 24 (baseline 25); the random push 1 and 0. Pushing 9 more steps separates 8 more seeds but turns none of them into a clean pair, which repeats plan 31's seeds 9 to 16: a longer push makes two animals more often and leaves which animal each one is untouched. Below the mark (faithful 29 or more). API tokens: 2,168 in per tile, 256 tiles. Dropped in a row since the presence result: round 1, then this.

**Dropped by the scorer: the presence checker's push added to the recipe (`result-sheet-catdog-presence-added`, seeds 3301 to 3332).** `readout-catdog-presence-only` (step 2,000; "cat present" 0.705 and "dog present" 0.746 at step 1,000) pushed at weight 1 beside iteration 6, steps 0 to 15. Scorer: two bodies 27 of 32, faithful pair 4, damaged 9 (the recipe on the same seeds: 29 to 30 two bodies, 5 to 7 faithful, 3 damaged); plain PoE 1 and 0, the random push 2 and 0. Below the mark. A tile-size look disagrees in part: most seeds show a cat beside a dog, but the colours are burnt and the light harsh, several dogs look like plastic toys (3319, 3325, 3327) and two seeds are flat drawings (3307, 3318), which is what the scorer's "damaged" and "hybrid" calls pick up. The presence push names the animals and costs the photograph. Sheet with verdicts: `result-sheet-catdog-presence-added/pushed-tiles-with-scores.png`. API tokens: 2,168 in per tile, 96 tiles.

**Kept: the multi-pair checker (`readout-multipair`, `result-sheet-catdog-big-multipair`, seeds 6001 to 6128).** Iteration 6's training on the multi-pair list (cat × dog plus the strictly labelled benchmark pool, 243 pairs). In the recipe: scorer two bodies 116 of 128, faithful pair 29; the random push 0 and 0 (plain PoE on these seeds: 1 and 0). Against the iteration 6 baseline's 114 and 25 on the same seeds, faithful pair rises by 4 and two bodies does not fall, which meets the keep rule exactly. Spot-checked by eye: of the first 24 tiles the scorer calls faithful pairs, about 20 are clean photographs of a cat beside a dog (6001, 6010, 6015, 6024, 6029, 6039, 6046, 6066, 6081, 6089 among them); 6022, 6055, 6070 and 6093 carry a borrowed ear or face. The +4 equals the mark and the run-to-run noise on 128 seeds is not measured yet, so the iteration 6 sheet is being re-rendered unchanged (`result-sheet-catdog-big-rerun`) to size it. Lion × meerkat with this checker (`result-sheet-lion-meerkat-multipair`): two bodies 12 of 16, faithful pair 0; by eye mostly two meerkats, a lion beside a meerkat on about 4 seeds. The new baseline for later rounds is this checker in the recipe. API tokens: 2,168 in per tile, 256 + 48 tiles.

**Dropped: the multi-pair checker with the 78-pair runs added (`readout-multipair2`, `result-sheet-catdog-big-multipair2`).** Cat × dog seeds 6001 to 6128: scorer two bodies 115 of 128, faithful pair 23, damaged 19; the random push 1 and 0. Lion × meerkat seeds 3301 to 3316: two bodies 11, faithful pair 0 (by eye a lion beside a meerkat on 1 seed). Below `readout-multipair` (116 and 29) on both pairs, so adding 1,171 joint-prompt tiles from 78 more pairs makes the checker worse; its own reading of half-drawn runs at step 500 was already lower (0.639 against 0.686). API tokens: 2,168 in per tile, 256 + 48 tiles.

**Round: the multi-pair checker trained twice as long (`readout-multipair-long`).** `readout-multipair`'s data and settings with 4,000 steps instead of 2,000; its reading of half-drawn runs was still rising at step 2,000 (0.686, 0.691, 0.702 at steps 500, 1,000, 1,500). Kept if cat × dog faithful pair reaches 33 or more of 128 with two bodies at 116 or more.
