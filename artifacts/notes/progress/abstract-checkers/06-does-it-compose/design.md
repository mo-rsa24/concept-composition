# Does the count checker's push compose unseen digits (step V3)


The verdict is in [the finding on whether the count checker's push composes unseen digits](../../../../../2-classifier-guidance/report/abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md).

## What this asks

Step V2 left a count checker that tells two digits sitting apart from the other layouts, on digits it never trained on. Step V3 tests whether its gradient, added to the plain product while sampling, makes the digit model draw two separate, right digits on the pairs among digits 5 to 9. By the owner's answer to board question 69, the push is the count checker's alone, and the plain product's two digit labels supply which digits appear.

The model in the first test never saw two digits in one picture, so a failed push could mean the checker is weak or that the model cannot draw two digits at all. A second test, run only after a failure, separates those two (board question 58).

## Words this uses

**Count checker.** V2's map reader trained to answer "two digits sitting apart?", saved at `/datasets/mmolefe/poe_repair_min/loop-8/v2/count-final/count-head-map.pt` (W&B run `tyhrpyov`).

**Push.** The checker's gradient with respect to the noisy canvas, scaled to a fixed share of the product's noise prediction and added to it.

**Clean pair.** A finished canvas the judge calls "two clear, right, apart": two blobs, each a confident single digit, the two asked digits, one in each half.

**Test 1.** V3 on the model trained with no two-digit canvases (`share-0.0`). **Test 2.** The same on the model trained with two digits on 1% of its canvases (`share-0.01`), run only when Test 1 is null.

## Run settings for Test 1

**Group.** Runs that try a new idea (group 2).

**Varied axis.** The push: none, the count checker's, or a random direction of the same length. A best-of-4 row is the control for selection without a gradient.

**Model and sampler.** `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.0/unet.pt`; DDIM with 50 steps and eta 0, the product of the two asked digit labels at guidance 1 (`canvas_unet.sample`), with `cap=1.0` so the sampler's own cap never fires.

**Push window.** The sampler steps whose timestep lies between 480 and 712, the noise levels the checker trained on: sampler steps 15 to 25, timesteps 693 down to 489, 11 steps. The original design pushed on steps 0 to 15, where V2 showed a checker reads at chance, and board question 64 moved the checker to these noise levels.

**Push.** $\hat\epsilon_t = \epsilon_{\text{product}} - \lVert\epsilon_{\text{product}}\rVert\, b\, g/\lVert g\rVert$, with $g = \nabla_{x_t}\ell$, the gradient of the checker's logit through the checker, the two maps and the frozen model's three calls. Norms are per canvas.

**Push length b.** Chosen once on the 10 seen pairs among digits 0 to 4 at seeds 0 to 31, among 0.1, 0.2 and 0.4, as the one with the most clean pairs; a tie goes to the smaller b. The frozen b then runs once on the 10 unseen pairs at seeds 1000 to 1031, 320 canvases per row.

| Row | What it controls for |
|---|---|
| Plain product | the baseline, the same sampler with no push |
| Count checker's push | the method |
| Random push of the same length, per canvas and step | a push that fails while the checker's works cannot have failed for being too small |
| Best of 4 starting seeds, picked by the checker's logit on the finished canvas noised to timestep 480 | selection with no gradient |

**Score.** The digit judge `judge-v3/judge.pt` through `two_experts.call_two`, the same call M1 used. The headline is clean pairs of 320. A second count, two separate digits whatever their identity (the clean call plus "two, not both clear and right"), is reported beside it and carries no bar, because the count checker reads layout and not identity.

**Standard errors.** 10,000 bootstrap resamples of the 10 pairs' per-pair differences (`beat_baseline_bars.se_over_groups`).

**Zero push.** The checker's push at b = 0 on pair 5 and 6, seeds 1000 to 1007, must match the plain product to under 1e-5 in the same process.

| Result | What it takes |
|---|---|
| Support | The checker's push beats the plain product by more than 1.96 standard errors, and the random push does not |
| Null | The checker's push does not beat the plain product by that margin |
| Inconclusive | The zero push differs from plain by 1e-5 or more, or the random push also beats plain by that margin |

**The ladder.** A smoke test on pair 0 and 1 at 4 seeds with every row; the b choice on the seen pairs; then the one run on the unseen pairs. The script refuses to run the unseen stage twice.

**Cost:** about 0.5 GPU-hours of this loop's 30, budgeted before timing: 7 sampler passes of 320 canvases on the unseen pairs, of which 4 are the best-of-4 row, plus 4 passes on the seen pairs; pushed steps add a backward pass through the frozen model. **Buys:** step V3.

**What would surprise.** The push raising two separate digits of any identity while clean pairs stay flat. That would mean the checker moves layout and the plain product cannot supply the right digits once the layout changes.

## Test 2, only if Test 1 is null

**Varied axis.** The diffusion model: `share-0.0` in Test 1 and `share-0.01` in Test 2. The count checker is retrained on `share-0.01`'s maps with V2's count settings, because it reads the model's own noise predictions. Its push window, b, data, seeds, rows and score stay frozen at Test 1's values, with no new tuning.

**Model.** `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.01/unet.pt`.

**Cost:** about 1 GPU-hour, spent only if Test 1 is null. **Buys:** a failed Test 1 no longer ends the work before SDXL unless the push also fails on a model that has seen two digits together.

| Test 1 | Test 2 | What it means | Next |
|---|---|---|---|
| Support | not run | The push composes unseen digits on a model that never saw two digits | V5, then V4 |
| Null | Support | The push composes unseen digits once the model has seen two digits together | V5, then V4 |
| Null | Null | The push does not compose unseen digits on either model | Stop and file an `ask` |
| Inconclusive | not run | The zero push changed the sampler, or the random push also passed | Fix the cause, then rerun Test 1 |
| Null | Inconclusive | The zero push changed the sampler, or the random push also passed | Fix the cause, then rerun Test 2 |

Test 1 keeps its own verdict in `review.md` whatever Test 2 shows.

## What this cannot say

The `share-0.01` model held out five pairs (0 and 3, 1 and 9, 3 and 4, 3 and 6, 4 and 7), so it saw all 10 pairs among digits 5 to 9 together during training. A pass in Test 2 means the push works on pairs the checker never saw and the model has seen. SDXL's test has the same shape. Test 2 says nothing about pairs the model itself never saw together.

The judge reads pixel canvases. A pass here is about a 32 by 64 digit canvas, and step V4 tests whether the same kind of checker helps on SDXL's objects.
