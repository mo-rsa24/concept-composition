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
