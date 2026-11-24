# Design: does each animal's own prediction, inside its own region, stop the borrowing? (step Z7)

## What this asks

Z3 corrected each animal's hand-drawn region late in sampling and changed nothing: 5 of 48 pictures with two clear subjects, the same as CO3 alone ([Z3's review](../03-one-fix/review.md)). Its rule, inside each box, took twice the joint prompt's guess minus the other animal's guess and gave the animal's own prediction weight 0, so it never tested the rule the [method review](../method-review.md) and the reviewer point at: put each animal's own prediction where that animal is. Prompted alone, "cougar" draws a cougar with no spots. This step asks whether, inside its own box, it draws one inside CO3's picture too. Board question 98, decided by the manager loop for the owner, asks for this test and for a free text-only check first.

## Stage 1: the free text-only check

**What it measures.** For each of the 12 tuning pairs, how far each noun's vector inside the joint prompt has moved from that noun's encoding alone: one minus the cosine between the noun's last token in "a {A} and a {B}" and the same noun's last token in CO3's single-subject prompt, on the 2,048 numbers per token the U-Net receives. CLIP's text encoder reads left to right, so the second noun can carry the first and the first cannot carry the second ([A Cat Is A Cat's note](../01-reading/a-cat-is-a-cat.md), pp. 4 and 5).

**The test, fixed here.** Spearman $\rho$ between the second noun's shift and W2d's borrowed count over the 12 pairs, one-sided permutation p from 100,000 permutations, the same machinery as Z2 (`z2_bars.py`). One test, judged at p below 0.05. **If it passes, the text-encoder fix is worth testing**, and goes to the owner if over 3 GPU-hours. **If it does not, the text-encoder fix stays held.** The first noun's shift is reported beside it. It needs no picture; the encoders run on the card for a few seconds inside a short job.

## Stage 2: each animal's own prediction in its box

**What changes from Z3.** Only the weights inside the boxes, and the window, which starts after CO3's own correction because on the recorded seed the borrowing body already reads "cheetah" by steps 4 and 5 ([the free check](../03-one-fix/free-check/README.md)).

$$\tilde x(\text{cell}) = W_0(\text{cell})\,\hat x^{C} + W_A(\text{cell})\,\hat x^{A} + W_B(\text{cell})\,\hat x^{B}$$

- **Inside animal A's box only:** $W_0 = 0$, $W_A = 1$, $W_B = 0$. The cell's guess is A's own guess.
- **Inside animal B's box only:** $W_0 = 0$, $W_A = 0$, $W_B = 1$.
- **Where the boxes overlap, and outside both:** CO3's corrector weights, $W_0 = 2$ and equation 15's weights at that pass ($\beta = 0.9$).

The weights at every cell sum to 1. Everything else is Z3's corrector pass unchanged: one pass per step, the code's largest-entry rescale, the empty prompt's noise added back, the rescale to the input latent's length (`z3_regions.py`). **Window:** sampler steps 5 to 30 (timesteps 881 down to 381), after CO3's six corrected steps. **Boxes:** Z3's, drawn by eye before any fix render ([the boxes file](../03-one-fix/boxes.json)).

| Row | What it is | Pictures | Source |
|---|---|---|---|
| 1. CO3 alone | CO3 at the released settings | 48 | Z2's one-card full run (Slurm job 94591); no new render |
| 2. Whole-picture correction | one CO3 corrector pass per step at steps 5 to 30, CO3's weights everywhere | 48 | new, in the same window as row 3 |
| 3. Own prediction in its box | the rule above at steps 5 to 30 | 48 | new |

**Varied axis.** What goes inside the boxes. Row 3 against row 2 changes only that; row 2 against row 1 is extra correction alone. Pairs, seeds, prompts and boxes are Z3's: the six borrowing tuning pairs at seeds 6001 to 6008.

## How pictures are counted

As in Z3 and by board question 92: **two clear subjects with their own parts, drawing style ignored**, on [Loop 6's rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), with `broken` (broken anatomy, drawing style not counted) as the guard and `faithful` and `damaged` reported. Two blind reviewers call all 144 pictures under neutral ids with the key held apart; their agreement is reported; a third blind call settles each `two_clear` disagreement.

**Per-subject part scores, reported beside the bar.** Each reviewer also marks, per picture, `a_own` and `b_own`: 1 when that subject is present once and carries no part of the other subject. Per row, the count of pictures with each mark is reported, with no bar, so a fix that cleans one subject and not the other shows.

## The bar, on 48 pictures

The bar on 96 pictures (a lead of 5 and more than 1.96 standard errors) does not apply to 48 (Loop 4's request, `requests.md`). On 48 matched pictures, as in Z3 (`z3_bars.py`, unchanged):

| Result | What it takes |
|---|---|
| Support | row 3 has at least 4 more `two_clear` pictures than row 1, an exact one-sided McNemar test on the 48 matched seeds gives p < 0.05 (in practice a lead of at least 5: 5 against 0 discordant seeds gives p = 0.031), row 3's `broken` count is at most row 1's plus 3, and row 2 does not also meet the first two conditions against row 1 |
| Extra correction explains it | row 2 meets the first two conditions against row 1 as well; the boxes' content is credited only if row 3 also leads row 2 by at least 4 with p < 0.05 |
| Null | row 3 does not lead row 1 by 4, or p ≥ 0.05 |
| Broken | row 3 meets the lead but its `broken` count exceeds row 1's by more than 3 |
| Inconclusive | either control below fails |

**What would surprise.** Row 3's pictures losing the borrowed parts and gaining seams at the box edges would show in `broken` and the notes. Row 3 clean on the per-subject marks while `two_clear` does not move would mean the fix cleans each animal but something else (an extra animal, a missing one) keeps pictures out of `two_clear`.

## Controls

1. **The own-prediction rule reduces to the whole-picture rule.** With both boxes empty, row 3's code must give row 2's picture pixel for pixel, in one process on one card.
2. **Steps 0 to 4 are untouched.** In one process, the latent entering step 5 with the added passes switched on must equal the latent entering step 5 of a plain CO3 render, to a largest absolute difference under 1e-5.

## Cost and what it buys

**Cost:** stage 1, a few seconds of GPU inside a short job, about 0.05 GPU-hours with the model load. Stage 2, about 1.7 GPU-hours: 96 renders at about 60 seconds (CO3's measured 49 to 56 seconds plus 26 corrector passes of batch 4) and 0.2 for the smoke test and controls. Board question 98 gave about 0.8 for the own-prediction row alone; the whole-picture control in the same window doubles it, and the total stays under the 3-hour line. Two blind reviewers plus a third. No API tokens.

**Buys:** the test Z3 meant to be: whether each animal's own prediction, placed where that animal is, removes borrowed parts on CO3. A support makes "own prediction in its region" Paper 2's candidate and sends the next step to a reader of regions; a null says borrowing survives even the right content in the right place, which bears on Paper 1 (experts trained apart cannot borrow) by contrast.

## The ladder

Rungs save under `/datasets/mmolefe/poe_repair_min/loop-12/z7/<rung>/` and log to W&B project `prime_lab/poe-repair-animals-compose`.

1. **Text check** (stage 1).
2. **Smoke.** Cheetah × cougar seed 6001, rows 2 and 3, both controls.
3. **Full.** All 48 per row, then the blind calls and the bar.

## What this cannot say

Tuning pairs and seeds only, so no claim in either paper. Boxes from the finished picture are an upper bound on any reader. One pass per step at steps 5 to 30; other windows and pass counts are untested.
