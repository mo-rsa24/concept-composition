# Review: cat AND dog AND NOT hybrid

The questions and pass marks for [plan 27](../27-cat-and-dog-and-not-hybrid.md), written on 2026-11-18 before
any picture was drawn.

## Words this file uses

**NOT-hybrid term.** Each sampler step subtracts λ · p/(1 − p) · 7.5 · (SDXL's prediction for "a hybrid animal,
half cat and half dog" minus its prediction for the empty prompt) from plain PoE's prediction. p is the
readout checker's probability (iteration 4, read with no gradient) that the state is not heading for a clean
cat and dog. The direction comes from SDXL, the weight from the checker (Logical Guidance's NOT rule).

**Damaged.** At full size, flattened or oversaturated colour, blocky banding, painted patches, or a broken face.

## Questions

**Q1. Does the checker-weighted NOT-hybrid term give a separate cat and dog on one trajectory per seed?**

On held-out seeds 9 to 16, at λ 0.01, 0.03 and 0.1, called from full-size tiles:

- Support: a separate cat and dog on at least as many seeds as the identity push at cap 0.3 at its better strength,
  with fewer damaged pictures, and on at least 2 more seeds than plain PoE.
- Null: no more seeds with a separate cat and dog than plain PoE at every λ.

**Q2. Does the checker's weight matter?**

On PoE's own trajectory the checker's probability sits at its 0.95 clamp on most steps, so its weight k is near 19.
Two controls with no checker: the hybrid prompt at that same size (k fixed at 19, λ 0.03) and an ordinary negative
prompt (k = 1, λ 1.0).

- Support: the weighted term at λ 0.03 gives a separate cat and dog on at least 2 more seeds than the size-matched
  fixed term (k 19, λ 0.03).
- Null: the size-matched fixed term does as well; the hybrid direction alone carries it, and the checker adds nothing
  beyond setting its size.

The loop at λ 0 must match plain PoE to within fp16 noise (mean pixel difference under 1 on the 0 to 255 scale), or
nothing on the sheet is read.

## Runs

| Run | What | Status |
| --- | --- | --- |
| not-hybrid, first launch | λ 0.1, 0.3, 1.0; stopped after seed 9 with no picture read: the λ 0 loop differed from plain PoE (mean 2.27, max 210) because it batched four prompts, and k sat at 19 | stopped |
| not-hybrid | PoE's three prompts as one batch and the hybrid prompt apart; λ 0.01, 0.03, 0.1; controls k 1 and k 19 | drawing |
