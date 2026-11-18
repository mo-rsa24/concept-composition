# Review: does the recipe work on other pairs

The questions and pass marks for [plan 30](../30-does-the-recipe-work-on-other-pairs.md), written on 2026-11-18 before any
picture was drawn.

## Words this file uses

**The recipe.** The readout checker of iteration 6 (`readout-catdog-identity`, step 2,000, trained only on cat × dog)
pushing PoE at strength 30, capped at 0.3 of PoE's prediction, on sampler steps 0 to 15, then plain PoE. On cat × dog it
gave a separate cat and dog on 28 of 32 new seeds (plan 29). Here the checker reads each picture with the pair's own joint
prompt ("a lion and a meerkat", "a typewriter and a cactus").

**Separate.** Both things in the frame as two distinct objects, not fused. **Clean.** Each recognisably its own kind,
with at most one small borrowed feature. Calls by eye from full-size tiles.

## Questions

**Q1. Step 0: does the cat × dog checker, unchanged, separate other pairs?**

On seeds 3301 to 3316 for lion × meerkat and for typewriter × cactus, against plain PoE and the size-matched random push:

- Support: separate on at least 10 of 16 for a pair, with plain PoE and the random push each at most 2 of 16.
- Null: separate on at most 4 of 16 more than plain PoE.
- Read per pair; a support on one pair and a null on the other is a finding in itself (an animal pair against an
  object pair).

**Q2. Step 1: does a checker trained across many pairs separate the test pairs?**

Written before the step 1 runs were drawn. A readout checker trained on joint-prompt runs of 78 pairs (plan 23's pairs
less cat × dog and lion × meerkat, 40 runs each, labelled by the Claude API) plus the cat × dog data, pushing with the
same recipe, on lion × meerkat and typewriter × cactus, seeds 3301 to 3316:

- Support: on a pair, a separate pair of things on at least 10 of 16 and more seeds with the right two kinds than step 0's
  checker gave (lion × meerkat: 6 of 16).
- Null: no more seeds with the right two kinds than step 0's checker.

## Runs

| Run | What | Status |
| --- | --- | --- |
| result-sheet-lion-meerkat | `render_results.py --readout --head <iteration 6> --prompts "a lion" "a meerkat" --seeds 3301 to 3316 --strengths 30 --push-cap 0.3 --window 0 16` | done |
| result-sheet-typewriter-cactus | the same with `--prompts "a typewriter" "a cactus"` | done |
| pair-runs | `render_pair_runs.py`, 78 pairs × 40 joint runs, seeds 5001 to 5040 | drawing |

## Answers

**Q1, lion × meerkat: in between.** Calls by eye from full-size tiles
(`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-lion-meerkat/views/`).

| | Plain PoE | Random push, same size | The recipe |
| --- | --- | --- | --- |
| A separate lion and meerkat | 1 of 16 (3314, a lion behind two meerkats) | 1 of 16 (3314) | 6 of 16 |
| Clean or near-clean | 0 of 16 | 0 of 16 | 3 of 16 (3311, 3314, 3316) |

| Outcome under the recipe | Seeds |
| --- | --- |
| Clean or near-clean lion and meerkat | 3311, 3314 (the lion's face turned away), 3316 |
| A lion and a meerkat, the lion with a meerkat's face | 3301, 3305, 3307 |
| Two meerkats | 3303, 3304, 3306, 3309 |
| One animal | 3302, 3308, 3310, 3312, 3313, 3315 |

Two separate animals on 10 of 16, but only 6 are a lion and a meerkat: separate on 6 is below the 10 of 16 support mark
and 5 above plain PoE, past the null margin of 4. The cat × dog checker, unchanged, separates bodies on a second animal
pair; it does not know which body should be the lion, so four seeds give two meerkats.

**Q1, typewriter × cactus: support.** Calls by eye from full-size tiles
(`artifacts/results/does-a-small-discriminator-steer-cat-x-dog/result-sheet-typewriter-cactus/views/`).

| | Plain PoE | Random push, same size | The recipe |
| --- | --- | --- | --- |
| A separate typewriter and cactus | 1 of 16 (3302, a pot beside the machine) | 1 of 16 (3302) | 14 of 16 |
| Clean, as a photograph | 0 of 16 | 0 of 16 | 6 of 16 (3308, 3311, 3312, 3314, 3315, 3316) |

| Outcome under the recipe | Seeds |
| --- | --- |
| Clean typewriter and cactus | 3308, 3311, 3312, 3314, 3315, 3316 |
| Separate, drawn as a flat illustration or collage | 3301, 3303, 3304, 3305, 3309, 3310 |
| Separate, with cacti also growing on the typewriter | 3302, 3306 |
| Fused, a typewriter growing spines or a cactus | 3307, 3313 |

Plain PoE draws a cactus growing out of a typewriter on almost every seed, and in illustration style on about half;
the illustration share under the recipe is no larger. The cat × dog checker, unchanged, separates an object pair more
reliably than the second animal pair: the object pair has no "which body is which kind" confusion.
