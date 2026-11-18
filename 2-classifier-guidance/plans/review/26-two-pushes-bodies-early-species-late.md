# Review: two pushes, bodies early and species late

The questions and pass marks for [plan 26](../26-two-pushes-bodies-early-species-late.md), written on
2026-11-18 before the species checker was trained or any combined push was drawn.

## Words this file uses

**Bodies push.** The [readout checker](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md) of iteration 6 (`readout-catdog-identity`, step 2,000), which reads
SDXL's own features; its push moves where the animals are.

**Species checker.** The CLIP ViT-L/14 checker fine-tuned on finished cat × dog pictures, with its last layer
replaced by two outputs, a clear cat present and a clear dog present, trained on finished pictures and SDXL's
half-drawn guesses (`finetune_species.py`). Its push follows log p(cat) + log p(dog) through the decoded guess.

**Two clear.** A clean cat and a clean dog in one picture, called from the full-size tile on the ladder single
< hybrid < two bodies or heads < two clear.

## Questions

**Q1. Can a checker read "which animals are there" from the half-drawn guess by step 15?**

- Support: both outputs reach 0.80 [balanced accuracy](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#balanced-accuracy-the-hit-rate-on-each-class-averaged) at step 15 on the 220 held-out pictures' guesses.
- Null: either output below 0.70 at step 15 (iteration 6's readout presence outputs read 0.695 and 0.728).
- In between: inconclusive; the push sheets still run.

**Q2. Does adding a species push from step 10 turn two bodies into a cat and a dog?**

Four rows on held-out seeds 9 to 16 at strengths 10, 30 and 100: bodies push alone (steps 0 to 9), species
push alone (steps 10 to 49), both (bodies on 0 to 9, species on 10 to 49), and each row's [size-matched random
push](a3q-cat-x-dog-quick-discriminator-test.md#words-this-file-uses).

- Support: "two clear" on at least 2 more seeds with both than with the bodies push alone, at some strength,
  while that strength's random push adds none.
- Null: no more "two clear" seeds with both than with the bodies push alone.
- A second reading beside it: how many "two bodies" tiles with two of the same animal turn into a cat and a dog.

## Runs

| Run | What | Status |
| --- | --- | --- |
| species-clip | `finetune_species.py` on mscluster112, epoch 6 kept | done: at step 15 cat 0.711, dog 0.847 |
| result-sheet-catdog-bodies | `--readout --head <iteration 6> --window 0 10` | drawing |
| result-sheet-catdog-species | `--clip <species> --window 10 50` | done: every seed stays plain PoE's fused animal, faces redrawn |
| result-sheet-catdog-both | `--readout --head <iteration 6> --species <species> --split 10` | done: one cat-dog face on most seeds at strengths 30 and 100 |

Every sheet draws a size-matched random push at each strength (render_results.py's r10, r30, r100 columns), so Q2's
random-push condition is read from the same sheet.

## Answers so far

**Q1: inconclusive.** At step 15 "dog present" reads 0.847 and "cat present" 0.711: dog clears 0.80, cat sits between
the 0.70 null line and the 0.80 support line.

**Q2: pending the bodies-alone row; leaning null.** The combined push pulls two bodies back into one cat-dog face on
seeds 10, 11, 12, 15 and 16 at strengths 30 and 100, fewer separate animals than the identity push over all steps: a
hybrid scores high on both presence outputs. The draft calls are from tile-size sheets
(`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/plan26-both-vs-identity.png`).

## Cross-references

- The mention of **a species checker** in [plan 26, two pushes, bodies early and species late, goal](../26-two-pushes-bodies-early-species-late.md#goal).
