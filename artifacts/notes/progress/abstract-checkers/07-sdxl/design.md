# An identity checker pushing on CO3, on SDXL (step V4)

## What this asks

Can a small checker that reads whether the asked subjects are the ones present push CO3 to more pictures with two clear, right subjects? On SDXL, CO3 already draws two bodies in all 96 of its tuning pictures, and its misses are identity misses: a wrong subject, a borrowed face or parts, or the same subject twice. On digits, [V6](../10-reading-identity/review.md) found a checker that reads identity on digits it never trained on, at 0.861 on digits 5 to 9, once it sees the asked labels' noise-prediction differences beside other labels' and reads at the later noise levels. V4 tests the same kind of checker inside SDXL, against CO3 alone.

The owner set this form on board question 86 (2026-11-24). It replaces [Loop 7's V4](../03-design.md#step-v4-does-it-work-on-sdxls-unseen-objects), which pushed the count checker on six object pairs against plain PoE, and it starts from Loop 6's training list without waiting for step D3.

## Words this uses

**CO3.** The released CO3 sampler on SDXL (`co3_common.RELEASED_SETTINGS`: 50 DDIM steps, CFG++ at guidance 0.8, its corrector on), run once per seed. Loop 6's `method_search/co3_push.py` adds a push to CO3's one batch-2 call per step and leaves its corrector calls alone.

**Signature.** At one sampler state $z_t$ and timestep $t$, SDXL's noise prediction for the empty prompt, the two asked subjects alone ("a puma", "a condor"), and eight other subjects fixed in advance. Four inputs come from these: the asked subjects' signed differences from the empty prompt, $d_A$ and $d_B$; the mean of the eight others' differences; and their largest absolute value at each latent position. Each is 4 latent channels by 128 by 128, pooled 2 by 2 to 64 by 64, and all are divided by the mean absolute difference over the ten subjects. The checker is never told which subjects were asked, so it can carry over to subjects it never trained on.

**The eight other subjects.** "a horse", "a sheep", "a rabbit", "a deer", "an owl", "a bear", "a lion", "a wolf". Each passes `scripts/shared/disallowed_subjects.py`, and none appears in the 12 tuning pairs or the 12 held-out pairs of `checker_choice/splits.json`. They are fixed here, before any signature is computed, and do not change.

**Two clear subjects.** The `two_clear` rung of [the blind-count rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md): exactly one clear A and exactly one clear B as separate bodies, with no borrowed features. Drawing style is ignored, as board question 87 set for Loop 6's checker search, so V4 and that search report the same count.

**Early and later window.** SDXL sampler steps 0 to 15 and 16 to 31 of 50.

## Run settings

**Group.** Runs that try a new idea (group 2).

**Varied axis.** The push on CO3: none, the identity checker's, or a random push of the same length.

**Pairs and seeds.** Paper 2's 12 tuning pairs at seeds 6001 to 6008 (`splits.json`), 96 pictures per row. The held-out pairs and seeds 7001 to 7008 are not touched. Giraffe and tortoise are counted but kept off every sheet.

**Training data.** Loop 6's balanced list from step M5, once its review names it. Tiles called "positive" are label 1, tiles called "wrong pair" are label 0, and every tile naming an object is left out, so the checker trains on animal pairs only. Each finished latent is forward-noised to a timestep of its window, as `train_readout_pair.py` does, and the signature is computed with no gradient. No tuning or held-out pair is in the list (`composition.json`: 0).

**Checker.** A small convolutional head on the four signature inputs and the timestep, one per window, trained for 4,000 updates of 64 with half label 1 and half label 0. A 16-example overfit check runs first.

## The steps, each with its bar

| Step | What runs | Bar, fixed here | If it misses |
|---|---|---|---|
| 1. Read check on held-out tiles | both windows' checkers on M5's held-out pairs, forward-noised | balanced accuracy at least 0.75 in a window | that window is dropped |
| 2. Read check on CO3's states | CO3 alone at the 96 tuning seeds, states recorded at every step; each read against M1's blind-count call on its finished picture: `two_clear` is 1, `two_bodies` is 0 | balanced accuracy over the window's 16 steps at least 0.75 | stop and report to the owner; no push runs |
| 3. Memory check | one pushed step on one seed: the gradient through the three calls that carry the asked subjects (empty, A, B), the eight others' differences held fixed | fits on a 24 GB `bigbatch` card | stop and ask the owner for one `biggpu` job through `progress_board.py ask` |
| 4. The rows | CO3 alone, CO3 with the identity push, CO3 with a random push of the same length, and the zero-push check | below | |

Step 2 is the owner's first check. Its labels come from the existing M1 calls on CO3's tuning pictures (`paper-2-blind-calls.csv` with its key): 32 are `two_clear` and 64 `two_bodies`, so it costs one pass of CO3 with reads and no new counting. The push runs in the window that passes step 2 with the higher balanced accuracy.

**The push.** $\hat\epsilon = \epsilon - \lVert\epsilon_{\text{guided}}\rVert\, b\, g/\lVert g\rVert$ on the window's steps, where $g$ is the gradient of the checker's logit with respect to $z_t$, subtracted from both entries of CO3's batch-2 call as `co3_push.py` does. $b = 0.4$, the length V3 chose on digits; it is not tuned here, because each row of 96 costs about 1.5 GPU-hours.

**The score.** Each picture is called blind by two reviewers by eye at full size with `method_search/paper2_blind.py`'s neutral ids, their agreement is reported, and a third settles disagreements, as in M5. The count is `two_clear` pictures of 96 with drawing style ignored; faithful pairs (`two_clear` and not damaged) and the damage count are reported beside it.

| Result | What it takes |
|---|---|
| Support | the identity push leads CO3 alone by at least 5 of 96 on two clear subjects and by more than 1.96 standard errors (10,000 bootstrap resamples of the 12 pairs' matched differences, `beat_baseline_bars.clears`) |
| Null | it does not |
| Inconclusive | the zero-push check differs from CO3 alone by 1e-5 or more, or the random push clears the same bar over CO3 alone |

**What would surprise.** The push raising two clear subjects while the random push of the same length lowers it by as much. That would mean CO3's pictures sit on an edge any push tips, and the checker only chooses the direction.

**Cost:** about 7 GPU-hours. The signatures for training and the two trainings take about 1, the CO3 read check about 1 (96 runs with ten extra calls per step and no gradient), the identity row about 4 (96 runs with a gradient through three SDXL calls on 16 steps), and the random row about 1.2 at M1's measured 45 seconds a picture. CO3 alone reuses M1's 96 renders, made by the same code at the same seeds, and the zero-push check reruns one pair's 8 seeds. No single job passes 8 GPU-hours. Loop 8 has used about 11 of its 30. **Buys:** the first test of an identity checker against Paper 2's rival, by Paper 2's bar.

## What this cannot say

The tuning pairs are the pairs the push is judged on, and the window is chosen on them in step 2, so a support here still needs the held-out pairs before it is a Paper 2 result. The eight other subjects are one fixed choice; another set could read differently. The gradient leaves the eight others fixed, so the push does not steer away from them directly. A picture with two clear subjects can still be damaged; the damage count beside it says how often.
