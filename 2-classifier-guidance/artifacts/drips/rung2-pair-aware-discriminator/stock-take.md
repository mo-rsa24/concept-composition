# Stock take: fixing PoE's cat-dog hybrids with a classifier's push

Where the rung 2 work on SDXL cat × dog stands: the idea, what was tried, what each attempt showed, and what
runs next. The per-run numbers live in [the checker loop file](checker-loop.md); this page is the map.

## The idea

PoE (product of experts) draws "a cat" and "a dog" from two separate prompts and combines them. On cat × dog it
almost always draws one fused animal.

The hypothesis: a **checker**, a small classifier that rewards "a clean cat and a clean dog" and penalises
hybrids, can steer PoE. While PoE draws, the checker's gradient nudges each step toward a higher score. That
nudge is the **push**.

Every push is judged on the same 8 held-out seeds (9 to 16), against plain PoE (two animals on 0 of 8) and a
random push of the same size (also 0 of 8), so any change comes from the checker. Calls are by eye from the
full-size pictures, on the ladder single < hybrid < two bodies or heads < two clear.

## 1. The first discriminator

| Tried | Result |
| --- | --- |
| A discriminator trained on 17 clean pairs against PoE's own pictures | 0 of 4 seeds with two animals; it learnt the joint prompt's style and looked at one fixed patch |
| Discriminators on more pairs | failed their accuracy check at every step |
| A layout push (separate the cat's and dog's attention maps) | 0 of 8; strong pushes zoomed into one face |

A judge trained on "joint prompt against PoE" learns the prompt's style. The checkers after this are trained on
what the picture shows: positive is a clean cat and dog, negative is anything else.

## 2. The data

- Training set: 1,603 pictures, 576 clean pairs. Evaluation set: 220 pictures (109 clean), never trained on.
- Labels from the Claude API under strict rules, each source spot-checked by eye.
- SDXL's own "a cat and a dog" drawings helped most; COCO photos made the checker worse.

## 3. The readout checker, iterations 0 to 6

A frozen SDXL reads the noisy half-drawn picture; a small head (about 780,000 weights) reads SDXL's internal
features and answers "clean pair?". It learnt from finished pictures with noise added back.
[Its three networks](figures/three-networks-in-rung-2.png).

| Iteration | Change | Accuracy, steps 0 to 19 | Push: seeds with two animals, of 8 |
| --- | --- | --- | --- |
| 0 | first version | 0.639 | 3, all hybrids |
| 2 | + 350 SDXL drawings | 0.737 | 5, one near-clean pair |
| 4 | trained longer | 0.755 | about 7, one near-clean pair |
| 5 | extra two-cats and two-dogs rejects | 0.756 | 3, mostly two of the same animal |
| 6 | + "cat present?" and "dog present?" outputs | 0.721 | 5, two near-clean pairs |

The push changes where the animals are. A clean cat and dog stays rare; the common results are two cats, a cat
and a kitten, or a hybrid beside a cat. Accuracy does not predict the push: iterations 4 and 5 read the same
and their pushes gave about 7 and 3.

## 4. A fine-tuned pretrained network

| Network | Finished pictures |
| --- | --- |
| DINOv2 | 0.847 |
| CLIP ViT-L | 0.865 to 0.900 |
| CLIP ViT-bigG | 0.873 to 0.909 |

As a judge of finished pictures it is far better than the readout checker. On SDXL's half-drawn guesses it reads
0.50 up to step 10, and 0.72 at step 10 and 0.82 at step 15 after training on 4,966 guesses
([what it sees by step](../../results/does-a-small-discriminator-steer-cat-x-dog/02-can-a-checker-read-a-clean-pair/half-drawn-guesses.png)).
Before step 10 the guess holds no second animal, so no picture judge can read it.

As a push it left the layout as plain PoE on 7 or 8 of 8 seeds, in all three versions, while raising its own
score (all pushes compared).
Its gradient passes back through the picture decoder and reaches texture; the readout checker's gradient comes
from SDXL's own features and reaches the layout. The fine-tuned network is a good judge and a poor steering wheel.

## 5. Push, then pick

Four tries per seed from four starting noises, each pushed by iteration 6; the fine-tuned CLIP picks the best.
Two near-clean cat-and-dog pictures of 8 seeds (a dog with extra cat ears beside a kitten; a dog with one stray
ear beside a clean cat), against 0 when picking among plain PoE tries. CLIP called one two-dog picture clean
([the three picks](../../results/does-a-small-discriminator-steer-cat-x-dog/03-does-an-early-push-separate-them/push-then-pick.png)).

## What the evidence says

1. A classifier's push turns PoE's single fused animal into two animals on up to 7 of 8 seeds.
2. A clean cat and dog is rare: the push settles how many bodies, not which animal each becomes.
3. The outcome is set over steps 0 to 10, where every checker reads worst (0.63 at step 0 for the readout).
4. Which checker steers matters more than how accurate it is.
5. Picking among pushed tries finds near-clean pairs; picking among plain tries finds none.
6. Hypothesis status: partly supported. The push breaks the hybrid often; it does not yet make the right two
   animals reliably.

## Running

| Work | Question | Pass mark |
| --- | --- | --- |
| [Plan 25](../../../plans/25-the-readout-checker-on-poes-own-states.md) | Does the readout checker, trained on PoE's own early states, read steps 0 to 10 better? | 0.75 over steps 0 to 10 on held-out joint runs, 0.05 above iteration 4 |
| [Plan 26](../../../plans/26-two-pushes-bodies-early-species-late.md) | Does a species push from step 10 ("clear cat?", "clear dog?") turn two bodies into a cat and a dog? | two clear on 2 more seeds than the bodies push alone. The species checker read cat 0.71 and dog 0.85 at step 15 (Q1 inconclusive) |
| Larger cap | Is the push too small to beat PoE's pull toward the fused animal? | the identity push at cap 0.3 and 0.5 against 0.1, by eye |
