# Is the joint prompt the wrong target where it fails? (step M14)

## What this asks

M13 found the kept checker's push pointing against r_t, the part of the joint prompt's prediction that the product of experts drops
([M13's review](../09-gradient-direction-check/review.md)). Loop 15 reads that two ways
([limits and what to test](../../directions/04-limits-and-what-to-test.md#what-changes-when-the-joint-itself-is-wrong)). Either the checker is
wrong, and its push opposes r_t whether or not the joint prompt draws both subjects; or the joint prompt is the wrong target, and the push
opposes r_t mostly where the joint prompt itself fails, moving away from the joint's error. The two readings predict different things on the
same states. The owner answered board question 112 on 2026-11-25: run Loop 15's three experiments in order, this one first, and skip the
second if this one says the checker is wrong.

## Words this uses

**The joint render.** SDXL drawing the joint prompt ("a puma and a condor") alone, with classifier-free guidance 7.5 over the empty prompt,
in 50 DDIM steps at 1024 by 1024, from the same starting noise as CO3's trajectory at that seed. It is what r_t's first term would draw if
nothing else steered.

**Joint succeeds, joint fails.** A blind call on the joint render under the [blind count rubric](../01-beat-the-baseline/blind-count-rubric.md):
two clear (one clear A and one clear B, separate, no borrowed parts, drawing style ignored), or not.

**The cosine.** M13's per-state cosine between the kept checker's push and r_t on CO3's state, already saved in
`/datasets/mmolefe/poe_repair_min/loop-6/m13/cosines.jsonl`. No cosine is computed again.

## The run

**Group.** A check that decides the target for both papers; it makes no claim in either paper by itself.

**Renders.** The 192 seeds M13 read: the 12 tuning pairs at seeds 6001 to 6016, each from CO3's own starting noise for that seed
(`co3_common.start_noise`), the SDXL weights CO3 loads, decoded in fp32. The held-out pairs and seeds 7001 to 7008 are not touched.

**Labels.** The 192 joint renders are shuffled under neutral names and called blind at full size by two reviewers under the blind count
rubric, with their agreement reported and a third blind call deciding each picture they disagree on. Neither reviewer sees a cosine.
Giraffe and tortoise is counted and stays off every sheet.

**The split.** M13's push cosine on CO3's state, at each step 0, 5, 10 and 15, split by the joint render's call. The difference is the median
over joint-fails trajectories minus the median over joint-succeeds ones; its 95% interval comes from 10,000 bootstrap resamples of
trajectories within each group.

## The bar, written on 2026-11-25 before the run

| Result | What it takes | What follows |
|---|---|---|
| The joint is the wrong target where it fails | at some step, the difference is at most −0.05 with its interval below 0, and where the joint succeeds the median is above −0.02 | both papers move to a target measured from the picture; M15 runs |
| The checker is wrong | at every step the interval of the difference includes 0 or the difference is above −0.02, and where the joint succeeds the median is at or below −0.02 at some step | the checker push is dropped and the joint stays the teacher; M15 is skipped, by the owner's answer |
| Inconclusive | anything else, or fewer than 20 trajectories in either group | reported, and the owner decides |

Reported beside the bar, deciding nothing: the joint prompt's own two-clear rate per pair (no report page holds it); the same split for the
clean-pair gradient alone; and the joint render's call against CO3's own call on the same seed, as a 2 by 2 count.

**What would surprise.** The push opposing r_t more where the joint succeeds. That would mean the checker fights the joint exactly where the
joint is right.

## Cost and what it buys

**Cost:** about 1 GPU-hour: 192 joint renders at 50 steps, at the rate of M13's 48 plain-product trajectories drawn in 21.5 minutes; no
cosine recomputed; two blind reviewers on 192 pictures. **Buys:** which of the two readings holds, which sets the target for both papers, and
the joint prompt's own two-clear rate on 12 pairs.

## What this cannot say

The cosine mixes bodies and background over the whole latent. A joint render that succeeds from CO3's starting noise does not mean the
joint prompt's prediction is right at every step of CO3's own path, which differs from the joint render's.
