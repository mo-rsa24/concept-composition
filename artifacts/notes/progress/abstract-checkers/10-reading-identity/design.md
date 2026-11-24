# Can a checker read which digits are present (step V6)

## What this asks

V2's identity checker could not tell a canvas holding both asked digits from one holding an asked digit and a wrong one: balanced accuracy 0.555 on the pairs holding digit 4, against a pass mark of 0.75 (board question 55). Three results point at why. Its score rose from 0.48 at the noisiest step to 0.65 at the least noisy, so the window may be too noisy. Loop 10's probe on all ten digit labels' signed differences read the same distinction at 0.655 where the two normalised maps read 0.511 (`score-space/05-signal/review.md`), so the input may carry too little. And its training pool held only 1,004 wrong-digit canvases. The owner chose to test these before rerunning V5 (option A, 2026-11-24).

## Words this uses

**Identity checker.** A map reader trained on `apart` canvases (label 1) against `wrong` canvases (label 0), so layout gives it nothing and only which digits are present can separate the labels.

**Two maps.** V2's input: for each asked digit, $\lvert\epsilon_a - \epsilon_\varnothing\rvert$, normalised to add to 1 and multiplied by the 2,048 pixels.

**Signature.** Four channels from all eleven of the frozen model's noise predictions (no digit and digits 0 to 9): the asked digits' signed differences $d_a = \epsilon_a - \epsilon_\varnothing$ and $d_b$, the mean of the other eight differences, and the largest absolute value among those eight at each pixel. All four are divided by the canvas's mean absolute difference over all ten labels, so their sizes relative to each other stay and the overall scale does not. The checker is still never told a digit's name, so it can carry over to digits it never trained on.

**Early window.** The 16 toy timesteps that match SDXL's sampler steps 0 to 15, 712 down to 480, the window V2 used.

**Later window.** The 16 toy timesteps that match SDXL's sampler steps 16 to 31: 466, 451, 437, 423, 409, 395, 381, 368, 354, 341, 327, 314, 301, 288, 275 and 263, where the signal share $\bar\alpha$ runs from 0.108 to 0.487.

## Run settings

**Group.** Runs that try a new idea (group 2).

**Varied axes.** Two, crossed in a 2 by 2 grid: the input (two maps or signature) and the window (early or later). The four cells differ from each other in one axis at a time along each side of the grid. Every cell shares the rest.

**Shared settings.** `PixelHead` from `pixel_checker.py` with its first convolution taking the input's channels; binary cross-entropy; Adam at 0.001; 4,000 updates of 128, half label 1 and half label 0. The batch is half V2's 256 so the signature's eleven model calls per example fit the budget, so no cell repeats V2's 0.555 exactly.

**Training data for the grid.** Built afresh by `identity_grid.py` from the MNIST training file with V1's canvas code: the 6 pairs among digits 0 to 3, 1,250 `apart` and 1,250 `wrong` canvases per pair, the wrong digit drawn from 0 to 3. That is 7,500 wrong-digit canvases against V2's 1,004. The script prints the data's hash.

**Test data for the grid.** V1's `seen_test` canvases of kinds `apart` and `wrong` for the 4 pairs holding digit 4 (500 canvases, the set the 0.555 was read on), and for the 6 pairs among 0 to 3 (the trained pairs). Each canvas is read at its window's 16 timesteps with 4 fixed noise draws.

**The bar.** Balanced accuracy on the pairs holding digit 4 of at least 0.75, board question 55's bar, in `identity_grid.py` as `MIN_BA`. The gap to the trained pairs is reported beside it.

**If a cell passes.** The best passing cell retrains on the 10 pairs among digits 0 to 4 with the same recipe (pool built the same way from 0 to 4), is saved, and reads V1's `unseen_test` digits 5 to 9 once (`apart` against `wrong`, 1,000 against 250 canvases). The owner approved this read with option A. If it passes there too, V5 reruns with it as the right-digits checker, pushing in its own window.

**If no cell passes.** The result goes to the owner, and V5 is not rerun.

| Result | What it takes |
|---|---|
| Support | A cell reaches at least 0.75 on the pairs holding digit 4 |
| Null | Every cell's overfit check passes and none reaches 0.75 |
| Inconclusive | A cell's overfit check fails, so that cell has not learned its training set |

**What would surprise.** The signature in the early window passing while the two maps in the later window fail. That would mean what the input holds matters more than how noisy the step is.

**The ladder.** A smoke test of every cell on 1% of the training pool with 20 updates; the overfit check of 16 fixed examples per cell; then the grid.

**Cost:** about 1.8 GPU-hours for the grid (the two-map cells at about 11 minutes of training each, the signature cells at about 39 minutes each from eleven model calls per example), and about 0.7 for the read of digits 5 to 9 if a cell passes. Both fit inside Loop 8's remaining 22.7 of 30. **Buys:** whether identity can be read at all on this toy, and if so, at which noise and from which input.

## Robustness checks

The owner chose on 2026-11-24 (option A) to run three checks on the passing cell, the ten-label signature in the later window, before step V7. The bars below are in `identity_grid.py` as `MAX_SHUFFLED_BA` and `MIN_PAIR_LOWER`, written before these runs and before any per-pair score was opened.

| Check | What it rules out | Bar |
|---|---|---|
| Second training seed (seed 1 against 0), same pool and recipe | One lucky training run | at least 0.75 on the pairs holding 4 |
| Labels shuffled across the training pool (seed 8602), true labels at scoring | The signature carrying the answer some way other than which digits are present | at most 0.55 on the pairs holding 4 |
| Score per pair: mean over pairs, standard error from 10,000 bootstrap draws over pairs | One easy pair carrying the pooled score | mean minus 1.96 standard errors at least 0.75, on digits 5 to 9 (10 pairs); reported for the pairs holding 4 (4 pairs) beside it |

**Robust** means all three bars hold. Any failure goes to the owner before V7. Each retraining costs about 45 minutes (about 1.5 GPU-hours for both); the per-pair scores come from scores the runs already save.

## Two checks before V7

The owner chose on 2026-11-24 (option A, after Loop 13's review) to run two more checks before step V7. The bars are in `v6_baselines.py` as `MIN_LEAD_SE` and `SAMPLER_MIN_BA`, written before either run.

**A rule that needs no training.** The signature holds the frozen model's noise predictions for all ten digits. The model alone can classify a canvas with them: the digit whose prediction is closest to the noise actually added fits best ([Diffusion Classifier](https://arxiv.org/abs/2303.16203)). The rule calls a canvas "the asked pair" when both asked digits fit better than the best of the other eight, where a label's fit is the mean squared difference between its noise prediction and the added noise. It reads the same 1,250 canvases of digits 5 to 9 as V6's final read, at the same 16 later-window timesteps with the same 4 noise draws. It is scored two ways: per read, one canvas at one timestep with one noise draw, as the checker reads; and with each label's fit summed over all 64 reads of a canvas, as Diffusion Classifier does, which the push never has.

| Result | What it takes |
|---|---|
| The checker adds to the model's own knowledge | the checker's balanced accuracy beats the per-read rule's by more than 1.96 standard errors of the paired difference, resampling the 10 pairs 10,000 times |
| It does not | the lead is at most 1.96 standard errors; reported to the owner before V7 |

The summed rule is reported beside it.

**A read on the sampler's own states.** V6 read forward-noised pasted canvases; V7 pushes on the sampler's states. This check renders V3's pushed row (the count checker's push, length 0.4, sampler steps 15 to 25) on the 10 pairs among digits 5 to 9 at seeds 1000 to 1095, records each state at sampler steps 27 to 36, and reads it with V6's saved checker. Each canvas is labelled by the judge on its finished picture: 1 when the call is "two clear, right, apart", 0 when there are two blobs, one per half, each read at 0.9 or more, and they are not the asked pair; any other canvas is left out.

| Result | What it takes |
|---|---|
| Reads sampler states | balanced accuracy over the 10 steps at least 0.75 |
| Does not | below 0.75; reported to the owner before V7 |
| Inconclusive | fewer than 40 canvases in either label |

**Cost:** about 0.4 GPU-hours. **Buys:** whether the trained checker knows more than the model already does, and whether it still reads on the states V7 will push.

## Retraining on the sampler's own states

V6's checker reads the sampler's own states at 0.651, under the 0.75 bar for that check, while V7 pushes on those states. Board question 94, decided for the owner by Loop 13 on 2026-11-24 (option A), retrains it on sampler states and reads digits 5 to 9 again with the same bar, on seeds no earlier step used. V7 runs only if this read passes. The constants are in `v6_resample.py`, written before the run.

**The states.** Each canvas is rendered by the plain product with V3's count push (length 0.4, sampler steps 15 to 25), so most finished pictures hold two separate digits. Its state is recorded at each of sampler steps 27 to 36 (timesteps 449 to 265), where V7's identity push acts. The judge labels the finished picture: 1 when its call is "two clear, right, apart", 0 when it holds two blobs, one per half, each read at 0.9 or more, that are not the asked pair (this includes the same asked digit twice); every other canvas is left out.

**Seeds.** Earlier steps used seeds 0 to 31 and 1000 to 1095 (and 100,000 upward for V3's best of 4). Training uses the 10 pairs among digits 0 to 4 at seeds 2000 to 2959, 9,600 renders. The read uses the 10 pairs among digits 5 to 9 at seeds 3000 to 3095, 960 renders, drawn fresh for this read and used once.

**The checker.** V6's form: the ten-label signature computed on each recorded state, a `PixelHead` with four input channels, 4,000 updates of 128 with half label 1 and half label 0, each example one canvas at one of its 10 recorded steps. The 16-example overfit check runs first.

| Result | What it takes |
|---|---|
| Reads sampler states | balanced accuracy over the 10 steps at least 0.75 on the fresh digits 5 to 9 |
| Does not | below 0.75; V7 does not run, and the result goes to the owner |
| Inconclusive | the overfit check fails, or fewer than 40 canvases in either label in training or in the read |

Its read of V1's pasted digits 5 to 9 (the 1,250 canvases V6 read at 0.861) is reported beside it, to show what retraining costs there.

**Cost:** about 1.5 GPU-hours: 9,600 training renders with states (the sampler read rendered 960 in under 4 minutes), the training with eleven model calls per example, and the read. **Buys:** a checker V7 can push with, or the answer that this form of checker cannot read the sampler's own states.

## Training on pasted canvases and sampler states together

The checker retrained on 2,036 sampler states learned those states and read fresh digits 5 to 9 at 0.552. Board question 95, decided for the owner by Loop 13 on 2026-11-24 after the reviewer's second pass (option A with three additions, no bar changed), trains on both sources and adds two checks first. The constants are in `v6_mix.py`, written before the runs.

**The states again, with their seeds.** The saved states do not record which seed made them, so the 10 pairs among digits 0 to 4 at seeds 2000 to 2959 are rendered again exactly as before, keeping each canvas's seed. Pushed renders can differ slightly between machines, so the counts can differ a little from the 2,036 of the first collection.

**1. Is identity visible at these steps at all.** For each kept canvas and each of sampler steps 27 to 36, the one-step clean estimate $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_{\text{product}})/\sqrt{\bar\alpha_t}$ is read by the digit judge, the same judge that labels the finished picture. A read counts as "the asked pair" when the judge's call on $\hat x_0$ is "two clear, right, apart". This is a diagnostic only and never becomes the checker. If its balanced accuracy over the 10 steps is under 0.75, identity is not plainly present at those steps; then V7 is closed as not run and no checker is trained.

**2. A positive control.** Seeds 2000 to 2959 ending in 0 (96 seeds, about 10%) are held out of training on every pair. Every 500 updates the checker reads them, and the checker kept is the one with the highest balanced accuracy there. If that best is under 0.75, the recipe does not read even the digits it trained on, and the result is inconclusive.

**3. The mix.** Each batch of 128 holds 64 pasted canvases from V6's pool (digits 0 to 3, 32 label 1 and 32 label 0, forward-noised to a later-window timestep with fresh noise) and 64 sampler states (32 label 1 and 32 label 0, at one of steps 27 to 36). At most 4,000 updates.

| Result | What it takes |
|---|---|
| Reads sampler states | balanced accuracy over the 10 steps at least 0.75 on digits 5 to 9 at new fresh seeds 3100 to 3195, read once |
| Does not | below 0.75; V7 is closed as not run |
| Inconclusive | the positive control's best is under 0.75, or fewer than 40 canvases in either label |

The area under the curve is reported beside every balanced accuracy, and the read of V1's pasted digits 5 to 9 beside the fresh read. About 50 wrong-digit canvases in a read give a standard error near 0.04.

**Cost:** about 1.6 GPU-hours, bringing Loop 8 to about 15 of 30. **Buys:** whether one checker can read identity on pasted canvases and the sampler's states at once, and, if identity is not visible at these steps, the reason two recipes missed.

## What this cannot say

A pass says a checker reads digit identity on pasted canvases. Whether its gradient helps the sampler draw the right digits is V5's rerun. The later window is later in sampling, where the layout may already be fixed, so an identity push there may fix identity without moving the digits apart.
