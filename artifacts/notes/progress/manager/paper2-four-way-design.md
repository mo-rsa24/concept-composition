# Paper 2: what it claims, and the test that decides it

The owner approved this direction on 2026-11-25. The reviewer read it the same day and asked for the
fixes now built in ([the reviewer's section](../reviewer/2026-11-24-review.md), "Paper 2's four-way
test, 2026-11-25").

## The contribution

Adding two prompts' noise predictions in classifier-free guidance (product of experts, PoE) fails to
draw two subjects for two separate reasons:

1. **The sampler does not sample the product.** Summed scores at a noisy step are not the score of
   the product distribution at that step, so even a perfect pair of experts is sampled wrongly.
2. **The exact product is not co-occurrence.** Multiplying "a cat" by "a dog" gives pictures that
   are both at once (a hybrid) unless pretraining saw the two together, so that support exists.

The ICLR work showed that the right correction, added early in sampling, is enough to turn a hybrid
into two subjects. Paper 2 asks whether that correction can be supplied **without changing the
pretrained model**: a classifier trained afterwards on the same data, reading "two separate things in
frame" from the noisy latent, guiding one trajectory early.

The claim stands on its own only if the correction lifts plain PoE and the joint prompt ("a cat and
a dog"). CO3 is reported beside them: the kept push already won 13 and lost 15 of 96 matched
pictures on CO3 on this measure, so a CO3 result alone cannot carry the claim.

**What is already published.** No paper the project has read claims that one learned checker lifts
several samplers. Each ingredient is published: composing scores (Du et al., Bradley et al.),
guidance from a reader of the model's features (Readout Guidance), early attention fixes
(Attend-and-Excite), and FKC's 0.78 on GenEval's two-object prompts on SDXL. CO3 and ToMe have no
reading notes yet, so they are unchecked.

## The three questions, in the order they decide things

| # | Question | Answered by | If no |
|---|---|---|---|
| 1 | Can an early frozen-model correction fix known failures beyond what a random push does? | the first test below | park the correction; Paper 2 needs a different mechanism or a different problem |
| 2 | Does it lift plain PoE and the joint prompt at a rate worth a paper? | the four-way test below | report which cause the correction misses |
| 3 | Does the noisy latent carry "two things in frame" early enough to steer? | the checker's ranking on steps 0 to 15, already measured by Loops 6 and 8 | the classifier route is limited by what the latent holds, which is itself a finding |

## First: can the correction fix a known failure at all?

This runs before anything larger, on cases already drawn, and answers whether the idea is worth sizing.

**Cases.** Eleven pair-and-seed cases from Loop 6's 12 tuning pairs, leaving out giraffe and
tortoise, which the subject guard holds back. Three CO3 failures, such as cheetah × cougar at seed
6001, and two cases CO3 already gets right are taken from W2d's hand labels of CO3's tuning renders.
No plain PoE or joint-prompt render of these pairs exists, so a screening pass draws both at seeds
6001 to 6004 (88 pictures, about 35 GPU-minutes), and three failures per base are picked by eye from
it. Cat × dog cannot be a case, because the kept checker was trained on it, and none may come from
the checker's 243 training pairs or the held-out seeds 7001 to 7008.

**The joint-prompt base.** The product with the joint prompt as both experts, each at half the
guidance scale, which equals plain classifier-free guidance on the joint prompt. An identity check
against plain guidance at the same precision runs first.

**Arms.** Each case is drawn again on the base it failed on, with the correction and with a random
push of the same size, four draws each, because pushed renders at one seed have differed between
runs. The existing picture is the "without" arm only when it was drawn on the same card type,
settings and seed; otherwise that one picture is drawn again.

**Cost.** About 88 pictures at 50 steps after the screening pass: plain PoE and the joint prompt
about 30 seconds a picture, CO3 about 60, about 65 GPU-minutes on a cluster card. With the screening
pass the first test is about 1.7 GPU-hours. The screening pass runs on mscluster107 card 1 under the
owner's approval in board question 103, and the test itself in `bigbatch`.

**Already known for the CO3 row.** Loop 6 ran the kept push on CO3 over all 96 tuning pictures:
7 faithful pairs with the push, 12 with the random push and 11 for CO3 alone
(`method-search/01-beat-the-baseline/paper-2-blind-result.json`). The review shows that result beside
this test's CO3 row.

**What it decides.** A base passes when the correction gives at least 4 more two-clear pictures than
the random push out of its 12 (three cases, four draws). Go to the four-way test when plain PoE and
the joint prompt both pass, and the two good cases break no more often with the correction than
with the random push. Otherwise the correction is parked with the eleven grids as the evidence. This
shows possibility, not a rate: no number from it goes in a paper.

## The four-way test

Runs only if the first test passes.

**Varied axes.** Base sampler (plain PoE, joint prompt, CO3) × correction (off, on, random push of
the same size). Pairs and seeds are fixed across all nine.

**Correction.** The kept early readout push (multi-pair readout checker, cap 0.4, steps 0 to 15),
because it exists and already runs on PoE and CO3; only the joint-prompt base is new code. The ICLR
adapter is drawn once per pair as a reference row: the trained correction that this frozen-model
correction tries to match.

**Baselines beside it.** Best of two draws without the correction, so a win is not just a second
chance, and Attend-and-Excite, the published early fix.

**Pairs.** 48 pairs named in a file the running loop writes before the first render: 36 that CO3
fails on, chosen from renders at seeds other than the test seed, and 12 it already draws as two
clear subjects. None from the checker's 243 training pairs. One seed per pair; a second seed only if
the test passes.

**Measure.** Two clear subjects, drawing style ignored, by the judge, with two blind reviewers on any
picture the judge marks uncertain. Every comparison is the same pair, same seed, same card type, at
50 sampling steps. Twenty-step renders may choose which correction to try, never decide the bar.

**Bars, written before running.**

- Support for a base: the correction wins on at least 6 more of the 36 hard pairs than it loses
  against the same base without it, an exact one-sided sign test on the pairs that differ gives
  p < 0.05, it also beats the random push and best of two, and it breaks no more of the 12 easy
  pairs than the random push does, plus 2.
- A method: support on both plain PoE and the joint prompt.
- A patch: support on CO3 only.
- Inconclusive: fewer than 8 pairs differ on a base, so the test could not have seen a win there.

**Cost.** About 4.8 GPU-hours at 50 steps on a cluster card. That is over the 3 GPU-hour line, so
this test needs the owner's yes before it runs; on a RunPod RTX 4090 it is about 2 hours.
