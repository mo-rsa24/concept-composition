# Review: does the early push hold on 32 new seeds

The questions and pass marks for [plan 29](../29-does-the-early-push-hold-on-new-seeds.md), written on 2026-11-18
before any picture was drawn.

## Words this file uses

**The recipe.** The readout checker of iteration 6 (`readout-catdog-identity`, step 2,000) pushing PoE at strength 30 on
sampler steps 0 to 15, the push capped at 0.3 of PoE's prediction, then plain PoE on steps 16 to 49. One trajectory per
seed from the seed's own starting noise. Chosen on held-out seeds 9 to 16, where it gave a separate cat and dog on about
7 of 8 and a clean or near-clean pair on 4 of 8.

**Separate.** Two animals in the frame, not fused into one body. **Clean or near-clean.** A cat and a dog, each
recognisably one species, with at most one small borrowed feature. Calls by eye from full-size tiles.

## Questions

**Q1. Does the recipe separate the animals on seeds it was never tuned on?**

On seeds 3301 to 3332 (never used for training, evaluation or tuning):

- Support: separate on at least 19 of 32 (60%) and clean or near-clean on at least 10 of 32 (30%), with plain PoE and the
  size-matched random push each at most 2 of 32 separate.
- Null: separate on at most 8 of 32 (25%).
- In between: the recipe works on some new seeds but less often than on the tuning seeds.

## Runs

| Run | What | Status |
| --- | --- | --- |
| result-sheet-catdog-general | `render_results.py --readout --head <iteration 6> --seeds 3301 to 3332 --strengths 30 --push-cap 0.3 --window 0 16` | done |

## Answers

**Q1: support.** Calls by eye from full-size tiles (`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-catdog-general/views/full/`).

| | Plain PoE | Random push, same size | The recipe |
| --- | --- | --- | --- |
| Separate | 1 of 32 | 1 of 32 | 28 of 32 |
| Clean or near-clean | 0 of 32 | 0 of 32 | 14 of 32 (9 clean) |

| Outcome | Seeds |
| --- | --- |
| Clean | 3306, 3307, 3311, 3314, 3320, 3322, 3323, 3329, 3332 |
| Near-clean (one small flaw) | 3304 (a blue patch on the dog), 3305 (a flat dog ear), 3308 (a fox-faced pup), 3321 (a dog-muzzled cat), 3330 (a stray red object) |
| Two bodies, a face borrowed or a head hidden | 3301, 3302, 3303, 3310, 3312, 3315, 3317, 3318, 3319, 3324, 3328 |
| Two of a kind | 3313 and 3327 two cats, 3331 two dogs |
| Not separate | 3309 and 3326 (the second animal a flat cutout or shape), 3316 (one painted cat), 3325 (one cat beside a rock) |

The one plain and random separation (3313) is a seed where plain PoE already drew two animals.
