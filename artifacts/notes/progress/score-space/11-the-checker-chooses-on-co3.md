# Does the checker help CO3 when it chooses instead of pushing?

Every push aimed at which body is which has given nothing on CO3: adding the kept push lowered faithful pairs from 11 to 7 of 96 ([Loop 6's M1](../method-search/01-beat-the-baseline/review.md)). The published methods that make a noise-aware score help mostly use it to choose: Dynamic CFG picks the guidance scale per step and scores 47.2 Gecko against 45.3 when it adds the same evaluators' gradients ([its note](../abstract-checkers/01-reading/dynamic-cfg-online-feedback.md)); LPO picks the best of 4 candidate next steps ([its note](../noise-text-critic/01-reading/noise-aware-latent-reward.md)). This design asks whether the readout checker, used the same way, lifts CO3. The owner chose this test on 2026-11-24. Nothing has run.

## The run

**Group.** Runs that try a new idea (group 2 of `~/.claude/EXPERIMENT_CONVENTIONS.md`). A result here changes no claim until a held-out read.

**Fixed, as M1 ran it.** The 12 tuning pairs in `2-classifier-guidance/scripts/checker_choice/splits.json` at seeds 6001 to 6008, 96 pictures per row, each prompt the pair's joint prompt. CO3 at the released settings in the replications hub's `co3_common.py`: CFG++ at guidance 0.8, 50 steps, 6 corrected steps. The checker is the kept readout checkpoint `readout_pair/readout-multipair/head_step_002000.pt`, and only its clean-pair output is read, because its two presence outputs never trained on these subjects ([M1's design](../method-search/01-beat-the-baseline/design.md)).

**Where the choice happens.** CO3 is deterministic once its starting noise is fixed: its corrector re-noises with the unconditional prediction (`composers/utils/attn_utils.py` lines 198 and 256 in the replications hub's `co3`), and the only random draw is the starting noise (`composers/Co3.py` line 373). So the candidates come from CO3's own setting. Its weight sharpness $\beta$ sets how hard the sample is pushed away from whichever single concept it already resembles (Eq. 14 and 15, [CO3 p. 5](https://arxiv.org/pdf/2509.25940v2#page=5)). The released script uses 0.9, and the paper's appendix runs values from 0.3 to 1.1 ([Loop 5's CO3 design](../reproductions/CO3/design.md)). At each of the 6 corrected steps, the step runs once for each of the K = 4 values $\beta \in \{0.3, 0.6, 0.9, 1.1\}$. The checker reads each candidate's SDXL features under the joint prompt, and the candidate with the highest clean-pair logit is kept. No gradient enters the sample. This is Dynamic CFG's move, with CO3's concept weighting in place of the guidance scale.

| Row | What it does | What it isolates |
|---|---|---|
| 1. CO3 alone | M1's row 1, reused: 11 faithful pairs of 96 | the baseline |
| 2. The checker chooses each step | 4 values of $\beta$ at each corrected step, keep the highest clean-pair logit | the method |
| 3. A random candidate each step | the same 4 candidates, one kept at random | 2 against 3: does the checker's pick matter, or only the extra noise? |
| 4. Choose once per picture | 4 CO3 runs from the same seed, one per value of $\beta$, run to step 15; the checker keeps one, which finishes | 2 against 4: does choosing at every step beat one choice of $\beta$ per picture? |

Each comparison changes one thing: rows 2 and 3 differ only in who picks, and rows 2 and 4 only in when the choice is made.

**Scoring.** As M1 scored: by eye at full size, blind, shuffled under neutral names, one call per picture on [the blind count rubric](../method-search/01-beat-the-baseline/blind-count-rubric.md), with faithful pair and damaged marked. Giraffe and tortoise stays off every sheet, as M1 holds it.

**The bar, M1's, on faithful pairs of 96.** The standard error comes from 10,000 bootstrap resamples of the 12 pairs. A lead under 5 of 96 is a tie, because a repeated render moves by about 2 of 96.

| Result | What it takes |
|---|---|
| Support | row 2 leads row 1 by at least 5 of 96 and by more than 1.96 standard errors, and leads row 3 by the same margin |
| Choosing helps, the checker does not | rows 2 and 3 both lead row 1, and row 2 does not lead row 3 |
| Null | row 2 does not lead row 1 |
| Choose once is enough | row 4 is within 5 of 96 of row 2 |

**What would surprise.** Row 2 beating row 4. Choosing once per picture is the simpler move, and on the plain product plan 11's best-of-16 matched a stepwise method.

**Controls.** With only $\beta = 0.9$ offered, row 2 reproduces CO3 alone, to a largest latent difference under 1e-5 on one card and dtype. A pair whose prompt CO3 cannot split into two concepts is excluded from every row, as in M1.

## Cost and what it buys

**Cost:** CO3 alone takes a measured median of 38.4 seconds per picture ([M1's design](../method-search/01-beat-the-baseline/design.md#cost-and-what-it-buys)), and its corrector is about two thirds of that: 40 corrector iterations of 4 to 5 U-Net rows against the sampler's 100 rows ([Loop 5's CO3 design](../reproductions/CO3/design.md)). Running each corrected step 4 times plus the checker's reads makes rows 2 and 3 about 3 times CO3, about 115 seconds a picture or 3.1 GPU-hours each for 96 pictures. Row 4 runs about 3.3 times CO3, 3.4 GPU-hours. **About 9.6 GPU-hours in all**, on `biggpu` or a 48 GB `bigbatch` card, since SDXL with the readout checker needs the memory M1 used. A cheaper first read runs rows 2 and 3 on 6 of the 12 pairs, about 3.1 GPU-hours, and adds row 4 and the other 6 pairs only if row 2 leads. These are estimates; the smoke rung measures the real time.

**Buys:** a measured answer to whether this project's checker can help CO3 at all without a gradient, which is the open route for Paper 2; and, if it can, the winning and losing candidates are preference pairs an LPO-style LoRA could train on.

## Who runs it

The CO3 runner and the readout checker live in Loop 6's area (`2-classifier-guidance/scripts/method_search/`), and this run uses M1's pairs, seeds and rubric. The owner decides who runs it (board question filed with this page).
