# Does each paper's method beat its simple baseline? (step M1)

## What this asks

Each paper needs a method that does more than the cheapest thing a reviewer would try first. Paper 1's cheapest thing is to
sample each digit model alone and paste the two results side by side. Paper 2's is to run CO3 and add nothing. This stage
measures both comparisons on tuning data, with the bars fixed here before any run.

## Words this uses

**Expert.** One of the two MNIST diffusion models trained apart: `e04-either` draws digits 0 to 4 and `e59-either` draws
digits 5 to 9, each on either side of a 32 by 64 pixel canvas.

**Checker.** A small classifier whose gradient pushes the sampler toward two separate, correct objects.

**Push.** The checker's gradient added to the noise prediction during sampling.

**Clean digit pair.** A canvas the digit judge (`judge-v3`) calls "two clear, right, apart": exactly two blobs, one in each
half, read as the asked digits with at least 0.9 confidence each.

**Faithful pair.** An SDXL picture with two separate bodies, a clear subject A, a clear subject B, no borrowed features and an
intact photograph.

**Size-matched random push.** A random direction scaled, step by step, to the length of the real push. It shows whether the
checker's direction matters or any disturbance of that size would do.

## Run settings shared by both papers

**Group.** Runs that try ideas (group 2). The result decides whether each method goes forward and makes no claim in either
paper. The baselines are frozen at the settings below.

**Data.** Tuning data only. No benchmark prompt, held-out pair or held-out seed is used.

**Noise rule.** A lead counts only when it exceeds 1.96 standard errors of the matched difference, the rule in the
[comparison protocol](../../targets/comparison-protocol.md). The bars live in
`2-classifier-guidance/scripts/method_search/beat_baseline_bars.py`.

## Paper 1: the method against sample-and-paste

**Varied axis.** How the two digits reach the canvas. Experts, pairs, seeds, sampler and judge are the same in every row.

**Pairs and seeds.** The five tuning pairs from plan 33 (1 and 9, 3 and 6, 4 and 7, 2 and 8, 0 and 5) at seeds 0 to 63, so
each row has 320 canvases. The 20 unseen pairs are not touched.

**Sampler.** DDIM with 50 steps and eta 1.0, each seed with its own fixed noise.

| Row | What it does |
|---|---|
| 1. Sample and paste | Each expert samples its digit alone on the full canvas. Expert A uses seed `s` and expert B uses seed `s + 100000`. The half holding more ink is cut from each, and the two halves are placed on opposite sides, with the side set by `default_rng(1000 + s)` |
| 2. Method, no push | `--compose inferred-lock`: the experts choose the side over the first 10 of 50 steps, hold it, and each supplies the prediction on its own half |
| 3. Method with the push | Row 2 plus the pair and separate pushes at strength 10, capped at 0.1 of the prediction |
| 4. Random push | Row 2 plus a size-matched random push |

**The checker follows the owner's data rule (board question 37).** It trains on expert samples only.

- The pool is 3,000 training and 500 held-out samples per digit, 35,000 canvases, drawn from seed 1,000 up. Each 28 by 28 crop
  is labelled by the digit its expert was asked to draw, and it is written in MNIST's file format, so
  `2-classifier-guidance/scripts/mnist/pixel_checker.py` can read it.
- `2-classifier-guidance/scripts/method_search/train_checker.py` runs that trainer with no pair of a 0 to 4 digit and a 5 to 9
  digit anywhere in its data, including the wrong-pair negatives, so all 25 cross-range pairs stay unseen.
- The earlier checkers' retraining rounds on their own pushed renders are left out, because the digit judge that labelled those
  renders trained on real MNIST.
- The judge only scores. It trains nothing and filters nothing in the pool.

**Controls.**

1. A zero push reproduces row 2 with a largest absolute difference under 1e-5, in one process on one card and dtype.
2. Sixteen random canvases per row are checked by eye at full size against the judge's calls.

**The bar, on clean digit pairs of 320.** The standard error comes from 10,000 bootstrap resamples of the matched canvas
differences, resampled within each pair, because five pairs are too few to resample as units.

| Result | What it takes |
|---|---|
| Support | row 3 minus row 1 is more than 1.96 standard errors above zero |
| Null | the difference is within 1.96 standard errors |
| Loses | row 1 minus row 3 is more than 1.96 standard errors above zero |
| Inconclusive | the zero-push check fails, or row 4 beats row 2 by more than 1.96 standard errors |

**What would surprise.** The experts' solo counts on these five pairs multiply out to about 285 of 320 for row 1. A paste count
under 250 would mean the paste step itself is damaging digits, and that would be checked before any verdict.

## Paper 2: CO3 with the push against CO3 alone

**Varied axis.** Whether the push is added to CO3. Model, prompts, seeds and CO3's own settings are the same in every row.

**Pairs and seeds.** The 12 tuning pairs in `2-classifier-guidance/scripts/checker_choice/splits.json` at seeds 6001 to 6008, so
each row has 96 pictures. The prompt is the pair's joint prompt, such as "a puma and a condor". The 12 held-out pairs and seeds
7001 to 7008 are not touched.

**CO3.** The released settings in the replications hub's `co3_common.py`, unchanged: CFG++ at guidance 0.8, 50 steps, 6
corrected steps.

| Row | What it does |
|---|---|
| 1. CO3 alone | CO3 as released |
| 2. CO3 with the push | CO3 plus the kept recipe's push: strength 30 on steps 0 to 15, capped at 0.4 of CO3's guided prediction |
| 3. Random push | CO3 plus a size-matched random push, using row 2's lengths for the same pair and seed |

**Where the push enters.** CO3's step is $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{x}_0(\epsilon_g) + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_u$, where
$\epsilon_g$ is the guided prediction and $\epsilon_u$ the unconditional one. The push is subtracted from both, after CO3's corrector
has run on that step. The latent then moves as it does in the kept recipe's DDIM step, where one prediction fills both roles.
Subtracting it from $\epsilon_g$ alone would move the latent much further for the same push. The code is a subclass in
`2-classifier-guidance/scripts/method_search/`, and no file of the authors' code changes.

**The checker.** The kept checkpoint, `readout_pair/readout-multipair/head_step_002000.pt` under
`/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2/`, reading SDXL's features under the joint
prompt. None of the 12 tuning pairs is in its training list. That list holds 54 tiles over 27 pairs naming a subject on the guard's
decision list, so this checkpoint can answer M1 and cannot be frozen as a recipe. D3 retrains on the cleaned list.

**What the push follows.** The checker is one network with three outputs: clean pair, first subject present and second subject
present. The push is the gradient of their sum. The two presence outputs trained only on the 1,603 cat and dog pictures, because
`presence_catdog.json` has no label for the other 1,256 pictures and the loss masks them. On the 12 tuning pairs those two outputs
never trained on the subjects asked for. This row therefore tests the kept push as it stands, and M2 tests which outputs should
drive it.

**Scoring.** By eye at full size, because D3's approved API tokens are for D3. All 288 pictures are shuffled under neutral names, and
each gets one call on the ladder: one thing, a fused hybrid, two bodies or heads, two clear things, with faithful pair and damaged
marked. The key that names the methods is opened only after every call is saved.

**Held from display.** Giraffe and tortoise is one of the 12 tuning pairs, and the guard holds tortoises back from anything shown.
Its pictures are counted and stay off every sheet, page and message.

**Controls.**

1. A zero push reproduces CO3's own loop with a largest absolute latent difference under 1e-5, in one process on one card and dtype.
2. Each prompt must parse into exactly two concepts for CO3. A pair that does not is reported and excluded from all three rows.

**The bar, on faithful pairs of 96.** The standard error comes from 10,000 bootstrap resamples of the 12 pairs, using each pair's
matched difference over its 8 seeds. A pushed render is not bit-reproducible and has moved by about 2 of 96 when repeated, so a lead
under 5 of 96 is a tie.

| Result | What it takes |
|---|---|
| Support | row 2 leads row 1 by at least 5 of 96 and by more than 1.96 standard errors |
| Null | neither side has that lead |
| Loses | row 1 leads row 2 by at least 5 of 96 and by more than 1.96 standard errors |
| Inconclusive | the zero-push check fails, or row 3 leads row 1 by at least 5 of 96 and by more than 1.96 standard errors |

**What would surprise.** A random push matching the real one would mean any disturbance helps CO3, and the checker would have earned
nothing.

## Cost and what it buys

**Cost:** about 6.8 GPU-hours, of this loop's 60. Paper 1 is budgeted at 1.5 GPU-hours for the sample pool, two checker trainings and
four rows. Paper 2 is 288 pictures at a budgeted 60 seconds (CO3 alone has a measured median of 38.4 seconds), which is 4.8 GPU-hours,
plus 0.5 for the smoke test and zero-push check. No API tokens. The smoke rung replaces these budgets with measured times.

**Buys:** step M1, the first measured answer to whether either paper's method does more than its cheapest baseline.

## The ladder

Each rung saves its config, numbers, sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-6/m1/<paper>/<rung>/`.

1. **Smoke test.** Paper 1: one batch through the pool, the checker and each row. Paper 2: one CO3 render with and without the push.
   Records seconds per picture.
2. **Overfit.** Paper 1's two checkers each memorise 16 canvases, reaching at least 15 of 16 right. Paper 2 trains nothing.
3. **1% of the data.** Paper 1's checkers train on 1% of the pool and the loss falls. Paper 2 renders one pair at one seed in all three rows.
4. **Controls.** The zero-push checks and, for Paper 2, the concept parse on all 12 prompts.
5. **Comparison.** The full rows, then the blind count for Paper 2.

## If a method loses

The result is recorded, an `ask` names what it means for that paper with continuing as the default only if a concrete fix exists, and
work carries on with the other paper.

## What this cannot say

Tuning pairs and seeds cannot support a claim in either paper. Paper 1 tests digits at 32 by 64 pixels with two fixed halves. Paper 2's
checker is a pre-D3 checkpoint, so its result says whether the kept push helps CO3, and the checker D3 chooses may do better or worse.
