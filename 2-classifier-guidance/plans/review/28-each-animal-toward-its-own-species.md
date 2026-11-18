# Review: each animal pushed toward its own species, after the early push

The questions and pass marks for [plan 28](../28-each-animal-toward-its-own-species.md), written on 2026-11-18 before
any picture was drawn.

## Words this file uses

**Early push.** The readout checker of iteration 6 (`readout-catdog-identity`, step 2,000) pushing PoE on steps 0 to
15 with the push capped at 0.3 of PoE's prediction. On its own it gives two separate animals on most held-out seeds,
with borrowed face features (a cat's whiskers on a dog, a garbled face).

**Late species push.** From step 16 to 49, capped at 0.1: at step 16 the decoded guess at the finished picture is
boxed by the animal detector, the two largest boxes are named cat and dog by zero-shot CLIP, and each box is pushed
toward its own species (log p(cat) on the cat box plus log p(dog) on the dog box, zero-shot CLIP on a square crop of
the decoded guess).

**Clean or near-clean.** At full size, a cat and a dog, each recognisably one species, with at most one small
borrowed feature.

## Questions

**Q1. Does the late species push clean the faces?**

On held-out seeds 9 to 16, strengths 30 and 100, called by eye from full-size tiles, one trajectory per picture:

- Support: clean or near-clean on at least 6 of 8 seeds at some strength, against 4 of 8 for the early push alone
  (seeds 9, 11, 15 and 16, `result-sheet-catdog-identity-cap0.3-early`), with the size-matched random push over the
  same steps adding none.
- Null: no more clean or near-clean seeds than the early push alone.

**Q2. How often does the detector find two animals at step 16?**

Read from the log's "late start" lines. Fewer than 6 of 16 pictures with two boxes makes Q1 inconclusive, because the
late push never acts on most of them.

**Q3. Does pushing only confidently named boxes keep the fixes and drop the wrong-species turns?**

The same run with a box pushed only when zero-shot CLIP gives its assigned species at least 0.8 at step 16, written
after Q1's answer and before this run:

- Support: no picture where a cat turns into a dog or a dog into a cat, and clean or near-clean on at least 5 of 8 seeds.
- Null: wrong-species turns remain, or fewer clean or near-clean seeds than the first run's 4.

## Runs

| Run | What | Status |
| --- | --- | --- |
| result-sheet-catdog-boxes | `render_results.py --readout --head <iteration 6> --box-species --split 16 --push-cap 0.3 --cap-late 0.1 --cap-late-from 16` | done |
| result-sheet-catdog-boxes-confident | the same plus `--box-confidence 0.8` | drawing |

## Answers

**Q1: null.** Clean or near-clean on 4 of 8 seeds (9, 11, 12, 16), the same count as the early push alone (9, 11, 15,
16). The per-animal push changes species: right on seeds 9, 11 and 12 at strength 30, wrong on seed 15 at 30 and seed 13
at 100, where the boxes were named from a still-blurry guess at step 16. It adds artifacts on seed 10 and on seed 16 at
strength 100. Draft calls at full size (`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-catdog-boxes/compare/`).

**Q2: support.** Two or more animals at step 16 on 16 of 16 pictures (three on 2).
