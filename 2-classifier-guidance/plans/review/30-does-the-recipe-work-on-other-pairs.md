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

## Runs

| Run | What | Status |
| --- | --- | --- |
| result-sheet-lion-meerkat | `render_results.py --readout --head <iteration 6> --prompts "a lion" "a meerkat" --seeds 3301 to 3316 --strengths 30 --push-cap 0.3 --window 0 16` | queued |
| result-sheet-typewriter-cactus | the same with `--prompts "a typewriter" "a cactus"` | queued |
