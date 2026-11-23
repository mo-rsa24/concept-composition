# Does the checker's push compose unseen digits (step V3)

## What this asks

Step V3 tests whether the checker's gradient makes the plain product draw two right digits on digit pairs the checker never trained
on. The model in the first test never saw two digits in one picture, so a failed push could mean the checker is weak or that the model
cannot draw two digits at all. A second test, run only after a failure, separates those two explanations.

The rows, seeds, score and bar for the first test are in [the V3 section of the abstract-checker design](../03-design.md#step-v3-does-its-gradient-compose-unseen-digits).

## Words this uses

**Test 1.** V3 as the design writes it, on the model trained with no two-digit canvases (`share-0.0`).

**Test 2.** The same test on the model trained with two digits on 1% of its canvases (`share-0.01`). It runs only when Test 1's result is null.

**Unseen pairs.** The 10 pairs among digits 5 to 9. The checker never trains on them.

## Run settings for Test 2

**Group.** Runs that try a new idea (group 2).

**Varied axis.** The diffusion model: `share-0.0` in Test 1 and `share-0.01` in Test 2. The checker is retrained because it reads the model's own noise predictions. Its configuration, push length, step window, data, seeds, rows and score stay frozen at Test 1's values, with no new tuning.

**Model.** `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.01/unet.pt`.

**Pairs and seeds.** The 10 unseen pairs at seeds 1000 to 1031, 320 canvases per row.

**Rows.** Plain product, checker push on steps 0 to 15, a random push of the same length, and best of 4 starting seeds.

**Cost:** about 2 GPU-hours of this loop's 30, budgeted from V2's and V3's 1 GPU-hour each. It is spent only if Test 1 is null.

**Buys:** a failed Test 1 no longer ends the work before SDXL unless the checker also fails on a model that has seen two digits together.

## What counts as a result

Each test uses the same bar: the checker's push beats the plain product by more than 1.96 standard errors, and the random push does not.

| Test 1 | Test 2 | What it means | Next |
|---|---|---|---|
| Support | not run | The push composes unseen digits on a model that never saw two digits | V4 |
| Null | Support | The push composes unseen digits once the model has seen two digits together | V4 |
| Null | Null | The push does not compose unseen digits on either model | Stop and file an `ask` |
| Inconclusive | not run | The zero push changed the sampler, or the random push also passed | Fix the cause, then rerun Test 1 |
| Null | Inconclusive | The zero push changed the sampler, or the random push also passed | Fix the cause, then rerun Test 2 |

Test 1 keeps its own verdict in `review.md` whatever Test 2 shows.

**What would surprise.** Test 1 null and Test 2 null with the checker above the V2 bar on both models. That would mean a checker can read two digits on noisy canvases and its gradient still cannot draw them.

## What this cannot say

The `share-0.01` model held out five pairs (0 and 3, 1 and 9, 3 and 4, 3 and 6, 4 and 7), so it saw all 10 pairs among digits 5 to 9 together during training. A pass in Test 2 means the push works on pairs the checker never saw and the model has seen. SDXL's test has the same shape. Test 2 says nothing about pairs the model itself never saw together.
