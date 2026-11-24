# Design: what predicts which pairs borrow parts under CO3? (step Z2)

## What this asks

Under CO3, borrowing is a property of the pair. Cheetah and cougar borrow on 7 of 8 seeds and crow and raven on 0, though both pairs look alike. This step measures three things per pair, each read at a different place inside the model and at three stretches of the sampling path, and asks which one rises and falls with the borrowed count, and from which step. The place and the stretch that predict borrowing are where and when Z3's fix should act.

This file fixes the measurements, the outcome and the bars before any number exists.

## Words this uses

**Borrowed count.** For one pair, the number of its 8 CO3 pictures that W2d's hand labels call `borrowed`: two separate subjects, one carrying a clear part of the other. From [the part-binding calls](../../noise-text-critic/02d-binding/part-binding-calls.csv), final column.

**Corrected steps.** CO3's first 6 of 50 steps (timesteps 981 to 881), where its corrector runs the U-Net on one latent under the empty prompt, the joint prompt and each subject alone. Steps 0 to 2 use resampling and steps 3 to 5 the corrector, with 10 iterations at steps 0 and 1 and 5 at steps 2 to 5 (`co3-hybrid` in the authors' `Co3.py`, lines 302 to 333).

**Read steps and windows.** The predictors are read at 14 of the 50 steps: 0 to 5, then 10, 15, 20, 25, 30, 35, 40 and 45. They are tested in three windows: early, steps 0 to 5, where CO3 acts; middle, steps 10 to 20; and late, steps 25 to 45. Co-occurrence is decided early in this project's records (the correction composes 21 of 32 runs injected at steps 0 to 10 against 3 of 32 at steps 10 to 20), while a frozen caption scorer only separates the right caption after latent 20 in NoisyCLIP ([Loop 9's NoisyCLIP note](../../noise-text-critic/01-reading/noisyclip.md), p. 10). Part identity may therefore be set after CO3's window, and the late window tests that.

**$d_t$ map.** What one subject's prompt adds to the empty prompt's noise prediction on the same latent, $d_t(A) = \epsilon_\theta(x_t, t, A) - \epsilon_\theta(x_t, t, \varnothing)$, a 4 by 128 by 128 tensor.

## How much of the picture the input holds at these steps

On SDXL's schedule the noisy latent holds this share of its variance as picture: 0.6% at step 0, 1.7% at step 5, 4.3% at step 10, 9.5% at step 15, 18% at step 20, 30% at step 25, 45% at step 30, 62% at step 35, 78% at step 40 and 92% at step 45 ($\bar\alpha_t$ at DDIM timesteps 981 down to 81). The three predictors read things computed from that latent: the text encoder reads no latent at all, the attention maps read the U-Net's features at those steps, and the $d_t$ maps are U-Net outputs. On SDXL, a readout on the U-Net's features read a clean pair at 0.63 at step 0 and 0.78 at step 10 ([the checker loop](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md)), so the features carry layout information this early even though the latent carries little picture.

## Group and varied axis

**Group.** Runs that try ideas (group 2). The result chooses Z3's fix and makes no claim in either paper.

**Varied axis.** The pair. Model, sampler, settings, seeds and prompts are M1's CO3 row unchanged. No method is varied.

**Data.** Loop 6's M1 tuning pairs, the 12 in `2-classifier-guidance/scripts/checker_choice/splits.json`, at seeds 6001 to 6008, prompt "a {A} and a {B}" as CO3 parses it. No held-out pair, held-out seed (7001 to 7008) or benchmark prompt is used. Giraffe and tortoise is counted and kept off every sheet, page and message.

## The three predictors, one primary number each

Each predictor has one primary number, fixed here, and secondary numbers reported beside it without a bar.

| Predictor | Read from | Primary number per pair | Secondary, reported only |
|---|---|---|---|
| 1. Text similarity | SDXL's two text encoders, as the U-Net receives them (penultimate layers concatenated, 2,048 numbers per token) | cosine between the two subject-noun tokens inside the joint prompt; for a noun of several tokens, its last token | the same cosine for each noun encoded alone ("a cheetah", "a cougar"); each encoder separately; the pooled embeddings |
| 2. Attention overlap | the joint prompt's cross-attention in the U-Net, one extra call at each read step on the latent the sampler steps from | DAAM's overlap: each noun's map from every cross-attention layer, averaged over heads, upscaled to 128 by 128, summed over layers, rescaled to run from 0 to 1, cut at 0.4, then the intersection over union of the two cut maps; averaged over the window's read steps and the 8 seeds | the same per step; the 32 by 32 layers alone; a soft overlap (sum of the minimum of the two normalised maps) |
| 3. Score-map similarity | the same extra call, under the empty prompt and each subject alone | cosine between the flattened $d_t(A)$ and $d_t(B)$, averaged over the window's read steps and the 8 seeds | the cosine of their per-cell length maps (where each prompt acts, ignoring direction); per step |

DAAM's recipe is the published one ([DAAM's note](../01-reading/daam.md), p. 7) so the overlap has a counterpart in the literature, with one added step. DAAM cuts the raw sum at 0.4 of its maximum. On SDXL the raw sum sits within about a quarter of its peak everywhere: in the smoke run (eagle and fox, seed 6001, W&B `lmm75rgv`) eagle's map at step 0 had a mean of 10.97 against a maximum of 14.32, so every cell passed the cut and every overlap read 1.0. Each summed map is therefore rescaled to run from 0 to 1 before the cut. This was fixed after the smoke run and before any picture of a borrowing pair was read. With the rescaling, the smoke picture's two maps sit on their own animals from step 3 onward and still spread over the other animal, giving overlaps of 0.65 to 0.79. DAAM summed it over all steps of the image it scored; here each window is read separately, so the step at which the borrowing pairs first separate from the others can be named. A reading in the late window is closer to describing the picture than predicting it, and the review says so for any late support.

**Why an extra call and not a hook on CO3's own calls.** The capture needs an attention processor that keeps the probabilities, and replacing the sampler's processors can change its numbers. Loop 6's M9 met the same problem and read attention in a separate U-Net call whose output is discarded (`co3_bind.py`, lines 1 to 14). This step does the same: at each read step, just before the sampler's own call (after CO3's corrector at steps 0 to 5), one call with the batch [empty, joint, A, B] on the latent the sampler steps from, under `torch.no_grad()`, with the capturing processors attached only for that call. The sampler's own predictions are untouched, so each picture should match M1's.

## The outcome

**Primary.** The borrowed count of 8 per pair, from W2d's final calls:

| Pair | Borrowed of 8 |
|---|---|
| cheetah × cougar, duck × goose, puma × condor | 7 each |
| boar × pheasant, bobcat × quail, seal × gull | 4 each |
| crow × raven, eagle × fox, lighthouse × sailboat, zebra × flamingo, giraffe × tortoise, hippopotamus × swallow | 0 each |

**Secondary.** The borrowed share of pictures with two separate subjects (`borrowed` over `borrowed` plus `own_parts`), with hippopotamus × swallow left out because none of its 8 pictures shows two separate subjects. Its 0 in the primary outcome is a different failure, so the primary result is also reported with that pair left out.

## The test and the bars

**Statistic.** Spearman's rank correlation $\rho$ between each predictor's primary number and the borrowed count, over the 12 pairs. Its p-value comes from 100,000 random permutations of the borrowed counts across pairs, one-sided in the expected direction (more similarity or overlap, more borrowing). A 95% interval for $\rho$ comes from 10,000 bootstrap resamples of the pairs. The thresholds live in `2-classifier-guidance/scripts/leakage/z2_bars.py`, committed before the first run.

**Seven tests are made at once:** text similarity once, since it has no step, and attention overlap and score-map similarity in each of the three windows. Each test is judged at a one-sided p below 0.05 / 7 = 0.0071. With 12 pairs and the tied counts above, that needs $\rho$ of 0.706 or more, computed by `z2_bars.py` from 100,000 permutations of these counts. The owner chose the three windows on 2026-11-24, before any run.

| Result, per test | What it takes |
|---|---|
| Support | $\rho > 0$ and permutation p < 0.0071 |
| Null | p ≥ 0.0071 |
| Wrong way | $\rho < 0$ with the opposite one-sided p < 0.0071 |
| Inconclusive, all seven | any control below fails |

**Which predictor points where, and when.** The earliest window with a supported test names when the borrowing becomes readable; a fix acts from that window on. If support appears only in the late window, CO3's corrector, which acts at steps 0 to 5, cannot reach the failure, and the review says so. If only one predictor has support, Z3 aims at its place: the text encoder (a token or embedding fix, such as A Cat Is A Cat's or Follow the Flow's), cross-attention (Bounded Attention, the default), or the sum of scores (CO3's own weights). If more than one has support, the one with the larger $\rho$ is named, and the review says whether the two are correlated with each other across the pairs. If none has support, Z3 keeps its default and the review says the place is unknown.

**The sharper test, reported without a bar.** Three pairs hold both labels: bobcat × quail, seal × gull and puma × condor, 7 `own_parts` and 15 `borrowed` pictures. Within each, the review shows whether the seeds that borrowed had higher overlap or score-map similarity than the seeds that did not, in each window. That asks from which step a reading predicts the seed's outcome, which a per-pair average cannot. Twenty-two pictures are too few for a bar.

**What would surprise.** Text similarity predicting borrowing would mean the leak is fixed before sampling begins, so no sampler-side fix could remove it. Crow and raven reading as the highest overlap of the 12 while borrowing on 0 seeds would mean overlap is not the mechanism.

## Controls

1. **The picture is unchanged.** The smoke rung renders its picture twice in one job, with the capture off and on, so the capture's effect is measured on one card apart from any difference between cards. The smoke run on an RTX 3090 differed from M1's picture by 3.49 of 255, and most of M1's pictures were rendered on Quadro RTX 8000 cards. Every picture rendered here must match M1's CO3 picture for the same pair and seed, at `/datasets/mmolefe/poe_repair_min/loop-6/m1/paper-2/full/renders/co3/<pair>/seed-<seed>.png`, with a mean absolute difference under 3 of 255, the limit M9 used. A picture that fails is reported and its pair's readings are not used.
2. **The capture reads what it should.** On the smoke picture, each noun's summed map is shown over the decoded picture, and the overlap of a noun's map with itself is 1.0.
3. **The two subjects are the two CO3 concepts.** Each prompt must parse into exactly the two subject nouns, as M1's concept-parse check required.

## Cost and what it buys

**Cost:** about 1.8 GPU-hours on one 48 GB card. That is 96 CO3 renders at M1's measured median of 38.4 seconds, plus 14 extra U-Net calls of batch 4 per render (about 7 seconds), which comes to about 1.3 hours, plus 0.5 hours for the smoke test and controls. The text predictor runs on the same card in under a minute. The saved maps take about 400 MB under `/datasets/mmolefe/poe_repair_min/loop-12/z2/`. No API tokens. Scoring needs no new labels: the outcome is W2d's.

**Buys:** step Z2, the first measurement of which place inside SDXL with CO3, and from which step, predicts which pairs borrow parts, so Z3's one fix is aimed by evidence and not by the literature's guess.

## The ladder

Each rung saves its config, numbers and commit under `/datasets/mmolefe/poe_repair_min/loop-12/z2/<rung>/`, and logs to W&B project `prime_lab/poe-repair-animals-compose`.

1. **Smoke test.** One pair (eagle × fox), one seed (6001): the render, the picture check against M1, the maps over the picture, the self-overlap check, seconds per picture.
2. **A small read.** Two pairs at the ends of the outcome, eagle × fox (0 of 8) and cheetah × cougar (7 of 8), all 8 seeds. The three numbers are printed, with no bar.
3. **Controls.** The picture check on every picture, and the concept parse on all 12 prompts.
4. **The comparison.** All 12 pairs, then the correlation and the within-pair read.

## Two optional columns, the owner's choice

The reading suggested two more measurements. Neither is in the bars above. The padding-token overlap is reported beside the three, from the same call; the word-order swap is not run (the default of board question 82).

| Column | What it measures | Cost |
|---|---|---|
| Padding-token overlap | DAAM's overlap between each noun's map and the end-of-text and padding tokens' map, from the same extra call ([Magnet's note](../01-reading/magnet.md), [Bounded Attention's note](../01-reading/bounded-attention.md) p. 14) | none beyond the main run |
| Word-order swap | CO3 on "a {B} and a {A}" at the same 8 seeds for all 12 pairs, labelled blind by eye on W2d's rule. If which subject borrows follows the order of the names, part of the leak comes from the text encoder ([A Cat Is A Cat's note](../01-reading/a-cat-is-a-cat.md), pp. 4 and 5) | 96 more renders, about 1.1 GPU-hours, plus labelling 96 pictures by eye |

## What this cannot say

Twelve pairs are a small sample, and a correlation over them shows which reading travels with borrowing, not that it causes it. Z3 is the causal test. The outcome is W2d's labels of one picture per seed, so a pair's count carries the labellers' error, which W2d measured at 91 of 96 calls agreed before the third look.
