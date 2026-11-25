# Does the kept checker's push point where the dropped term points? (step M13)

## What this asks

The product of experts adds the two subjects' noise predictions and drops the rest of what the joint prompt would have predicted. That
dropped part is the interaction term, r_t:

$$ r_t = \epsilon(\text{joint}) - \big(\epsilon(A) + \epsilon(B) - \epsilon(\text{empty})\big) $$

The kept readout checker pushes CO3 along the gradient of its score with respect to the latent. If that gradient points along r_t, the
push supplies some of what composition drops; if it points elsewhere, a better-trained checker of the same kind has no reason to help. The
owner set this check on 2026-11-25, through Loop 13, before the identity head the reviewer chose under board question 110: the head runs
only if the gradients point the right way.

## Words this uses

**r_t, the dropped term.** At one sampling step, the joint prompt's noise prediction minus the plain product's prediction (first subject
plus second subject minus the empty prompt), as above. It is a vector the size of the latent, 4 by 128 by 128.

**The push direction.** The gradient, with respect to the latent, of the sum of the kept checker's three outputs (clean pair, first subject
present, second subject present), read under the joint prompt. This is what CO3's push follows (M1's design). The gradient of the
clean-pair output alone is reported beside it.

**Cosine.** The cosine of the angle between two latent-sized vectors: 1 when they point the same way, 0 when unrelated, minus 1 when
opposite. Two random directions in a space of 65,536 numbers have a cosine near 0, spread about 0.004.

## The run

**Group.** A check before a run: it decides whether the identity head runs, and makes no claim in either paper.

**Data.** CO3's saved states on the 12 tuning pairs, at steps 0, 5, 10 and 15 of 50 (the steps where the push acts and that were saved): the
96 tuning trajectories at seeds 6001 to 6008 (`loop-6/m5/co3-read-states-shard0/` and `shard1/`) and the 96 fresh trajectories at seeds
6009 to 6016 (`loop-6/m12/size-corrected/co3/`), 768 states in all. The owner's message also names plain-PoE trajectories; Loop 6 holds none
saved, so this check reads CO3's only, and says so.

**What runs.** For each state: SDXL's four noise predictions (empty, joint, first subject, second subject) give r_t; the kept checker
(`readout-multipair/head_step_002000.pt`) gives the push direction by one backward pass; a Gaussian direction drawn per state from a fixed
seed gives the random baseline. Recorded per state: the cosine of the push direction with r_t, of the clean-pair gradient with r_t, and of
the random direction with r_t. The code is `m5_read_co3.py gradcos`, launched by `m5_read.sbatch gradcos`; outputs under
`/datasets/mmolefe/poe_repair_min/loop-6/m13/`.

## The bar, written on 2026-11-25 before the run

The suggestion in Loop 13's request, adopted as written. Per step, the median over the 192 trajectories of the cosine between the push
direction and r_t.

| Result | What it takes | What follows |
|---|---|---|
| Points the right way | a median of at least 0.10 at some step from 0 to 15, 25 times the random spread | the identity head may run, as the owner set |
| Points elsewhere | a median under 0.02 in size at every step | the identity head does not run, by the owner's rule |
| Inconclusive | anything between | reported, and the owner decides |

A median of minus 0.10 or below at some step means the push points against the dropped term. It is reported as that and counts as not
pointing the right way.

Reported beside the bar, deciding nothing: the spread (25th and 75th percentiles) per step; the same for the clean-pair gradient alone;
the random direction's median and spread, which should sit near 0 and 0.004; and the median split by the blind calls, two clear against not
(M1's calls on the tuning seeds, the settled calls on the fresh seeds).

**What would surprise.** A push direction that points along r_t on the two-clear pictures and against it on the rest. That would mean the
checker reads composition and the push fails for another reason.

## Cost and what it buys

**Cost:** under 1 GPU-hour: 768 states, each with one four-prompt SDXL pass and one backward pass through the checker's part of SDXL.
**Buys:** whether a checker of this kind pushes along the term composition drops, before about 3 GPU-hours go into training an identity head.

## What this cannot say

A cosine compares directions over the whole latent, so a push that matches r_t on the bodies and not on the background reads as a small
cosine. r_t is CO3's own joint prediction minus the plain product at the same state, so it is the term on CO3's path, not on the plain
product's path.
