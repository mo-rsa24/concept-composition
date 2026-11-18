# Review: two pushes, bodies early and species late

The questions and pass marks for [plan 26](../26-two-pushes-bodies-early-species-late.md), written on
2026-11-18 before the species checker was trained or any combined push was drawn.

## Words this file uses

**Bodies push.** The readout checker of iteration 6 (`readout-catdog-identity`, step 2,000), which reads
SDXL's own features; its push moves where the animals are.

**Species checker.** The CLIP ViT-L/14 checker fine-tuned on finished cat × dog pictures, with its last layer
replaced by two outputs, a clear cat present and a clear dog present, trained on finished pictures and SDXL's
half-drawn guesses (`finetune_species.py`). Its push follows log p(cat) + log p(dog) through the decoded guess.

**Two clear.** A clean cat and a clean dog in one picture, called from the full-size tile on the ladder single
< hybrid < two bodies or heads < two clear.

## Questions

**Q1. Can a checker read "which animals are there" from the half-drawn guess by step 15?**

- Support: both outputs reach 0.80 balanced accuracy at step 15 on the 220 held-out pictures' guesses.
- Null: either output below 0.70 at step 15 (iteration 6's readout presence outputs read 0.695 and 0.728).
- In between: inconclusive; the push sheets still run.

**Q2. Does adding a species push from step 10 turn two bodies into a cat and a dog?**

Four rows on held-out seeds 9 to 16 at strengths 10, 30 and 100: bodies push alone (steps 0 to 9), species
push alone (steps 10 to 49), both (bodies on 0 to 9, species on 10 to 49), and each row's size-matched random
push.

- Support: "two clear" on at least 2 more seeds with both than with the bodies push alone, at some strength,
  while that strength's random push adds none.
- Null: no more "two clear" seeds with both than with the bodies push alone.
- A second reading beside it: how many "two bodies" tiles with two of the same animal turn into a cat and a dog.

## Runs

| Run | What | Status |
| --- | --- | --- |
| species-clip | `finetune_species.py` on mscluster112 | training |
| result-sheet-catdog-bodies | `--readout --head <iteration 6> --window 0 10` | queued |
| result-sheet-catdog-species | `--clip <species> --window 10 50` | queued |
| result-sheet-catdog-both | `--readout --head <iteration 6> --species <species> --split 10` | queued |
