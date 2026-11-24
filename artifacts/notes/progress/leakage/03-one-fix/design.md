# Design: does correcting each animal's region later in sampling stop the borrowing? (step Z3)

## What this asks

Z2 found no reading that predicts which pairs borrow parts, so no measured place points the fix. The owner's decision on board question 91 (made by the manager loop for the owner) replaces a fix aimed by a reading with a positive control: if each animal's region is known exactly, from boxes drawn by eye on CO3's finished pictures, does correcting each region against the other animal's prediction, later in sampling, give pictures with two clear subjects that keep their own parts? If it does not work with the right regions, no reader of regions can make it work. If it does, step 3 looks for a reader that finds the regions without a person.

The [free check](free-check/README.md) shows why the correction acts later and why it needs a layout: on the recorded cheetah and cougar seed, CO3's own per-place signal reads "cheetah" on both bodies by step 5, so it marks the borrowing body and cannot say which body should be the cougar.

This file fixes the correction, the comparison and the bar before any picture of the fix exists.

## The correction

**Where it acts.** At sampler steps 15 to 30 of 50 (timesteps 681 down to 381), where the latent is 9.5% to 45% picture on SDXL's schedule, after the two bodies' layout is set and while fine detail is still being drawn. CO3's own correction at steps 0 to 5 runs unchanged before it.

**What it does at each of those steps.** One corrector pass, with the authors' corrector formula (`attn_utils.py` lines 236 to 257, including the code's largest-entry rescale and the rescale to the input latent's length), with one change: the weights become a map over the 128 by 128 latent.

$$\tilde x(\text{cell}) = W_0\,\hat x^{C}(\text{cell}) + W_A(\text{cell})\,\hat x^{A}(\text{cell}) + W_B(\text{cell})\,\hat x^{B}(\text{cell})$$

- $\hat x^{C}, \hat x^{A}, \hat x^{B}$ are the clean-picture guesses of the joint prompt and of each animal alone, as in CO3's corrector.
- $W_0 = 2$ everywhere, as in CO3's corrector.
- **Inside animal A's box:** $W_A = 0$ and $W_B = -1$. The other animal's prediction is subtracted where A is drawn, and A's own is left in.
- **Inside animal B's box:** $W_A = -1$ and $W_B = 0$.
- **Where the boxes overlap, and outside both:** CO3's own whole-canvas weights from equation 15 at that pass ($\beta = 0.9$).

The weights at every cell sum to 1, as in CO3's corrector. This is the per-place rule written down in [the free check](free-check/README.md#the-per-place-rule-written-down).

**The boxes.** One box per animal, drawn by eye on Z2's one-card CO3 picture of the same pair and seed (`/datasets/mmolefe/poe_repair_min/loop-12/z2/full/<pair>/seed-<seed>.png`), as pixel coordinates on the 1024 by 1024 picture, divided by 8 onto the latent. They are saved to `03-one-fix/boxes.json` with a sheet of every picture and its boxes, before any picture of the fix is rendered. CO3 is deterministic up to step 15, so the layout the boxes were drawn on is the one the correction meets. Boxes from the finished picture make this run an upper bound: no reader can do better than the right regions.

## The comparison

**Group.** Runs that try ideas (group 2). The result decides whether step 3 looks for a reader; it makes no claim in either paper.

**Pairs and seeds.** The six tuning pairs that borrow on at least 4 of 8 seeds in W2d's calls: cheetah × cougar, duck × goose and puma × condor (7 each), boar × pheasant, bobcat × quail and seal × gull (4 each), at seeds 6001 to 6008: 48 pictures per row. They were chosen for their CO3 labels, so CO3 alone is called again blind in the same set to keep regression to the mean out of the comparison. No held-out pair or seed, no benchmark prompt, and giraffe and tortoise is not among them.

| Row | What it is | Pictures | Source |
|---|---|---|---|
| 1. CO3 alone | CO3 at the released settings | 48 | Z2's full run, one card (Slurm job 94591); no new render |
| 2. Whole-canvas correction | row 3's passes at steps 15 to 30, with CO3's whole-canvas weights everywhere | 48 | new |
| 3. Drawn-box correction | the correction above | 48 | new |

**Varied axis.** The weights at steps 15 to 30. Row 2 against row 1 is extra correction alone; row 3 against row 2 is the regions.

## How pictures are counted

**The measure is two clear subjects with their own parts, drawing style ignored,** decided for the owner on board question 92 on 2026-11-24, before any picture of the fix existed, so every Paper 2 result is counted one way (board question 87). It replaces faithful pairs as the measure of this step. Each picture gets the rung of [Loop 6's blind-count rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), with drawing style ignored: `two_clear` is exactly one clear A and exactly one clear B as separate bodies, with no borrowed features.

**Two marks beside it.** `broken` is 1 for melted or broken anatomy, garbled faces or stray limbs, with drawing style not counted; it is the guard. `damaged` and `faithful` are called as in the rubric and reported only.

**Who calls.** Two reviewers, each blind to the row, the other's calls and the boxes, see all 144 pictures under neutral ids in a shuffled order, one at full size at a time, with only the prompt. Their agreement on `two_clear` is reported. A third blind call settles each disagreement.

## The bar, on 48 pictures

The bar on 96 pictures (a lead of 5) does not apply to 48. On 48 matched pictures:

| Result | What it takes |
|---|---|
| Support | row 3 has at least 4 more `two_clear` pictures than row 1, an exact one-sided McNemar test on the 48 matched seeds gives p < 0.05, row 3's `broken` count is at most row 1's plus 3, and row 2 does not also meet the first two conditions against row 1 |
| Extra correction explains it | row 2 meets the first two conditions against row 1 as well; the regions are credited only if row 3 also leads row 2 by at least 4 with p < 0.05 |
| Null | row 3 does not lead row 1 by 4, or p ≥ 0.05 |
| Broken | row 3 meets the lead but its `broken` count exceeds row 1's by more than 3 |
| Inconclusive | either control below fails |

A lead of 4 of 48 is about twice the change a repeated pushed render showed on 96 pictures (about 2 of 96, Loop 6's M1 design), scaled to 48. The bar and the McNemar test live in `2-classifier-guidance/scripts/leakage/z3_bars.py`, committed before the first render.

**What would surprise.** Row 2 rising as far as row 3 would mean any extra correction late in sampling helps and the regions add nothing. Row 3 rising while the borrowed parts stay (pictures moving from `two_bodies` to `two_clear` by losing a subject's identity) would show in the notes and the `broken` count.

## Controls

1. **The region rule reduces to the whole-canvas rule.** With both boxes empty, row 3's code must reproduce row 2's latent to a largest absolute difference under 1e-5, in one process on one card.
2. **The correction leaves steps 0 to 14 alone.** The latent at step 15, before the first added pass, must equal Z2's CO3 latent for the same seed on the same card type, or the picture-match check of Z2's design is applied.

## Cost and what it buys

**Cost:** about 1.7 GPU-hours on one 48 GB card: 96 renders at about 58 seconds (CO3's 49.6-second median on the card Z2 used, plus 16 corrector passes of batch 4), and 0.2 hours for the smoke test and controls. Drawing 96 boxes by eye on 48 pictures, and two blind reviewers plus a third on 144 pictures. No API tokens. The owner's decision on board question 91 gives the cost as about 1.7 GPU-hours.

**Buys:** the first test of whether correcting each region against the other animal's prediction, with the regions known, removes borrowed parts on CO3, and the answer to whether step 3 should look for a reader.

## The ladder

Each rung saves its config, numbers and commit under `/datasets/mmolefe/poe_repair_min/loop-12/z3/<rung>/` and logs to W&B project `prime_lab/poe-repair-animals-compose`.

1. **Smoke.** Cheetah × cougar, seed 6001, rows 2 and 3, and control 1. Seconds per picture.
2. **Small.** Cheetah × cougar and duck × goose, all 8 seeds, rows 2 and 3; the pictures are looked at, with no count.
3. **Controls.** Control 2 on every picture.
4. **The comparison.** All 48 per row, then the blind calls and the bar.

## What this cannot say

The six pairs and eight seeds are tuning data, so the result cannot support a claim in either paper. Boxes from the finished picture are an upper bound; a reader that finds regions earlier and less exactly would do worse. The rule acts at steps 15 to 30 with one pass per step; other windows and pass counts are untested.
