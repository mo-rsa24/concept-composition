# Does a checker learn plurality on unseen digits (step V2)

## What this asks

Can a small checker that reads only where each asked digit acts on a noisy canvas tell whether both asked digits are present and apart, on digits it never trained on? The checker trains on the 10 pairs among digits 0 to 4 and is tested on the 10 pairs among digits 5 to 9. The inputs, network and bars come from [the abstract-checker design](../03-design.md#step-v2-does-a-checker-learn-plurality-on-unseen-digits), and the data from [step V1](../04-environment/review.md).

## Words this uses

**Map reader.** The checker under test. It reads two maps, one per asked digit, showing where that digit's label changes the model's noise prediction. It receives no digit label.

**Raw reader.** The comparison checker. It reads the noisy canvas itself with the asked pair as an input.

**Balanced accuracy.** The mean of the share of label-1 canvases and the share of label-0 canvases called correctly, from 0 to 1, where 0.5 is chance.

**Inner test.** A rehearsal that never touches digits 5 to 9. It trains on the pairs among digits 0 to 3 and tests on the pairs holding digit 4.

## Run settings

**Group.** Runs that try a new idea (group 2).

**Varied axis.** What the checker reads: the two maps or the raw noisy canvas. Data, labels, loss, updates and noise levels are the same.

**Model.** `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/models/share-0.0/unet.pt`, frozen, which never saw two digits on one canvas.

**Noise levels.** The first 16 of the 50 sampler steps, timesteps 999 down to 693. Each training example draws one uniformly.

**Training.** `PixelHead` from `pixel_checker.py`, binary cross-entropy on logits with no per-step weighting and no image augmentation, Adam at 0.001, 4,000 updates of 256 with half label 1 and half label 0.

**Map scale.** Each map adds to 1 as the design defines it, then is multiplied by the 2,048 canvas pixels before the network so its average value is 1.

**Scoring.** Every test canvas is read at all 16 steps with 4 fixed noise draws. A logit above 0 counts as label 1. Balanced accuracy is pooled over steps and draws, and is also saved per step.

**Inner-test data.** Training canvases holding digit 4 as the wrong digit are left out, so digit 4 is unseen in the rehearsal.

**Cost:** about 1.2 GPU-hours of this loop's 30 if the base configuration passes the inner test, and about 4 if all three variations run and the 12,000-update one is chosen. These come from the measured 337 ms per map-reader update on `mscluster83`, which is 22.5 minutes per 4,000 updates and 67 minutes per 12,000, plus an assumed 5 minutes of scoring per configuration.

**Buys:** step V2, and whether reading maps is what lets a checker carry over to digits it never saw.

## The ladder

1. **Smoke test.** 20 updates of each reader on the 1% data, printing the time per update.
2. **Overfit.** Each reader memorises 16 fixed examples, one at each noise level, to at least 15 right.
3. **1% of the data.** 1,000 updates of each reader on 160 canvases, scored on the 1% seen test split.
4. **Inner test.** The base map reader on the rehearsal split. If it misses, the three named variations run and the one with the highest smaller value of overall and `apart` against `wrong` balanced accuracy on the inner test is frozen.
5. **The comparison.** The frozen configuration retrains on digits 0 to 4 for both readers, and reads `unseen_test` once. The script refuses to run this stage a second time.

**The three named variations.** Add each map's log total as two extra input channels, use signed differences in place of absolute values, or train for 12,000 updates.

## What counts as a result

The bars sit in `2-classifier-guidance/scripts/abstract_checkers/v2_learn.py` as `MIN_BA`, `MAX_GAP` and `MIN_APART_WRONG`.

| Result | What it takes |
|---|---|
| Support | Map reader on `unseen_test`: balanced accuracy over steps 0 to 15 at least 0.75 and within 0.05 of `seen_test`, and at least 0.75 on `apart` against `wrong` |
| Null | The overfit rung passes and the frozen configuration misses any of those three |
| Inconclusive | The overfit rung fails, or the map reader scores below 0.55 on `seen_test` at every step |

**What would surprise.** The raw reader matching the map reader on `unseen_test`, because then the maps are not what makes the checker carry over.

## The rehearsal at SDXL's noise level (board question 64)

The first run read the toy's sampler steps 0 to 15, where the signal-to-noise ratio is 14 to 144 times lower than at SDXL's steps 0 to 15. This rehearsal asks whether the map reader learns which digits are present when the noise matches SDXL's.

**Group.** Runs that try a new idea (group 2).

**Varied axis.** The noise window, compared with the base map reader's row in the first inner test. Everything else stays at the base configuration: data, labels, loss, 4,000 updates, the network and the scoring.

**Noise window.** The 16 toy timesteps whose signal share ($\bar\alpha$) equals SDXL's at its timesteps 981, 961, ..., 681: 712, 696, 680, 664, 648, 632, 616, 601, 585, 570, 555, 539, 524, 510, 495 and 480. Their $\bar\alpha$ runs from 0.0058 to 0.0949. The list sits in `v2_learn.py` as `SDXL_MATCHED`.

**Data.** The rehearsal split only: training on the pairs among digits 0 to 3, scoring on those pairs and on the pairs holding digit 4. Digits 5 to 9 are not read.

**Readers.** The base map reader and the raw reader, each with its own overfit check at this window. The raw reader keeps its recipe. In the first run it answered label 0 on every read with a final loss of 0.6931, which is what a reader with no usable signal learns, and no code fault has been found. Each reader now reports the share of reads it calls label 1, and one that calls label 1 on under 1% or over 99% of reads is marked as not trained.

**Cost:** about 0.6 GPU-hours of this loop's 30: 24 minutes of map-reader training at the measured 360 ms per update, about 5 minutes of scoring, and a few minutes for the raw reader and the overfit checks. **Buys:** whether the null in the first run comes from the toy's noise level.

| Result | What it takes | Next |
|---|---|---|
| Pass | The map reader on the pairs holding digit 4: balanced accuracy at least 0.75 and within 0.05 of the trained pairs, and at least 0.75 on `apart` against `wrong` | Ask the owner for a new read of digits 5 to 9 at this window, before V3 |
| Miss | The overfit check passes and any of those three bars is missed | File the result with the owner, and the chain stays held |
| Inconclusive | The map reader's overfit check fails | Find the cause before anything else runs |

**What would surprise.** `apart` against `wrong` still at chance at this window. The toy's own timestep 480 keeps about 31% of the signal amplitude, so a reader blind to digit identity there would point at the maps and not the noise.

## What this cannot say

These canvases are pasted MNIST digits, so a pass says the checker reads pasted digits under noise. Step V3 tests whether its gradient helps on what the sampler draws. The steps are very noisy (at timestep 999 about 0.6% of the signal amplitude remains), so the per-step scores matter as much as the pooled one.
