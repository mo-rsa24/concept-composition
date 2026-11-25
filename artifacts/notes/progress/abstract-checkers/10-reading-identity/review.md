# Can a checker read which digits are present (step V6): review

**Status: support. A checker trained on digits 0 to 4 tells two right digits from one right and one wrong digit at 0.861 balanced accuracy on digits 5 to 9.** The bar, written before the run, is 0.75; 0.5 is chance. The frozen model alone reads the same canvases at 0.539, so the checker's reading is its own. On the sampler's own states it reads 0.651, under the 0.75 bar for that check, and a checker retrained on those states reads fresh digits 5 to 9 at 0.552, so step V7 does not run. 0.861 is the fair number: the checker, the ten-label signature read in the later window, is the best of four cells chosen on the pairs holding digit 4, where it scored 0.888. Digits 5 to 9 have been read by earlier steps too (V2, V3 and V5), so they are held out from training but not unread. The design is [the V6 design](design.md).

<a href="identity-grid.png"><img src="identity-grid.png" width="900" alt="The 2 by 2 grid. V2's identity checker at 0.555 leads into the early window, where two maps score 0.563 and the ten-label signature 0.612, both red. The later window holds two maps at 0.702, red, and the ten-label signature at 0.888, green, which leads to the retrain on digits 0 to 4 and the read of digits 5 to 9."></a>

## Contents

- [The grid](#the-grid)
- [What the grid shows](#what-the-grid-shows)
- [Digits 5 to 9](#digits-5-to-9)
- [The robustness checks](#the-robustness-checks)
- [Two checks before V7](#two-checks-before-v7)
- [Retraining on the sampler's own states](#retraining-on-the-samplers-own-states)
- [Is identity visible at these steps at all](#is-identity-visible-at-these-steps-at-all)
- [When the sampler settles which digits it draws](#when-the-sampler-settles-which-digits-it-draws)
- [The runs](#the-runs)
- [What this cannot say](#what-this-cannot-say)

## The grid

Each cell trained on the 6 pairs among digits 0 to 3 (15,000 canvases, half `apart` and half `wrong`, pool hash `2f6cc975...`) and was scored on V1's `seen_test` canvases: 500 on the 4 pairs holding digit 4 (400 `apart`, 100 `wrong`) and 750 on the trained pairs, each read at its window's 16 timesteps with 4 noise draws. The numbers are in [the result files](grid-files/), copied from `/datasets/mmolefe/poe_repair_min/loop-8/v6/`.

| Input | Window | Learned 16 fixed examples | Pairs holding 4, bar 0.75 | Trained pairs | Gap | `apart` right, pairs holding 4 | `wrong` right, pairs holding 4 |
|---|---|---|---|---|---|---|---|
| Two maps | early, t 712 to 480 | 16 of 16 | 0.563 | 0.586 | 0.023 | | |
| Two maps | later, t 466 to 263 | 16 of 16 | 0.702 | 0.827 | 0.125 | 0.716 | 0.687 |
| Ten-label signature | early | 16 of 16 | 0.612 | 0.628 | 0.015 | | |
| Ten-label signature | later | 16 of 16 | **0.888** | 0.892 | 0.004 | 0.926 | 0.850 |

Every cell learned its 16 fixed examples, so the design's table reads **support**: a cell reaches 0.75 on the pairs holding digit 4.

## What the grid shows

**The window matters more than the input, and the two add up.** With two maps, the later window adds 0.139 over the early one. With the early window, the signature adds 0.049 over two maps. Together they add 0.325. The bigger wrong-digit pool (7,500 against V2's 1,004) adds 0.008 over V2's 0.555 in the early window, so the pool size was not what held V2 back.

**The passing cell reads unseen pairs as well as its own.** Its gap is 0.004. The two-map cell in the later window fits its own pairs at 0.827 and reaches 0.702 on the pairs holding 4, so it learned partly which digits it trained on.

**It improves through the window.** On the pairs holding 4 it reads 0.80 at t 466 (SDXL's step 16), 0.88 by t 409, and 0.92 at t 263 (step 31).

## Digits 5 to 9

Job 94383 (55 minutes on mscluster87, W&B run `cnvjjrh2`) retrained the passing cell on the 10 pairs among digits 0 to 4 with the same recipe (pool hash `cd44808e...`), saved it as `/datasets/mmolefe/poe_repair_min/loop-8/v6/identity-head.pt`, and read V1's `unseen_test` digits 5 to 9 once: 1,250 canvases, 1,000 `apart` and 250 `wrong`, at 16 timesteps with 4 noise draws. The numbers are in [the final result file](grid-files/final.json).

| Digits | Balanced accuracy, bar 0.75 | `apart` right | `wrong` right | First step (t 466) | Last step (t 263) |
|---|---|---|---|---|---|
| 0 to 4, trained | 0.919 | 0.915 | 0.923 | 0.81 | 0.97 |
| **5 to 9, never seen** | **0.861** | 0.887 | 0.834 | 0.75 | 0.91 |

**It reads which digits are present on digits it never saw.** The design's table reads **support**. The gap between the two is 0.058. V6's design reports the gap beside the bar and sets no limit on it; V2's count checker was held to a gap of at most 0.05, and this gap is 0.008 over that.

**No single pair carries the score.** Every one of the 10 unseen pairs clears 0.75. The weakest is 6 and 9 at 0.824, the strongest 7 and 9 at 0.922. The mean over pairs is 0.861 with a standard error of 0.008 (10,000 bootstrap draws over pairs), so the mean minus 1.96 standard errors is 0.844, above the robustness bar of 0.75 written in [the design's robustness checks](design.md#robustness-checks).

| Pair | 5-6 | 5-7 | 5-8 | 5-9 | 6-7 | 6-8 | 6-9 | 7-8 | 7-9 | 8-9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | 0.878 | 0.846 | 0.852 | 0.833 | 0.876 | 0.850 | 0.824 | 0.858 | 0.922 | 0.868 |

## The robustness checks

The bars were fixed in [the design's robustness checks](design.md#robustness-checks) before these runs and before the per-pair scores were opened, and after the 0.888 on digit 4 was known. All three hold.

| Check | Job | Result | Bar |
|---|---|---|---|
| Second training seed (1 against 0), pairs holding 4 | 94471, 44 min | 0.877 (trained pairs 0.907); every pair 0.853 to 0.905 | at least 0.75 |
| Labels shuffled in training, pairs holding 4 | 94472, 47 min | 0.500 on every pair | at most 0.55 |
| Per pair on digits 5 to 9 | from job 94383 | mean 0.861, minus 1.96 standard errors 0.844 | at least 0.75 |

**The shuffled checker ignores the canvas.** It calls "the asked pair" on 56.25% of reads, which is 9 of the 16 timesteps on every canvas, so its answer depends on the timestep alone. That rules out a pipeline that hands the label to the checker. It does not test whether the frozen model's own ten predictions already carry identity; [the two checks before V7](#two-checks-before-v7) test that.

## Two checks before V7

Both ran on mscluster107's card 1, on another user's idle allocation, by the owner's yes on board question 90, through `/datasets/mmolefe/poe_repair_min/loop-8/v6/shared/launch-2.sh` (code commit `13c68afd`; log `logs/l8-v6-shared-2.log`, 16:07 to 16:24 UTC). A watchdog checked the card every 30 seconds and never fired. The bars are in [the design](design.md#two-checks-before-v7). The numbers are in [the rule's result file](grid-files/untrained-untrained.json) and [the sampler read's result file](grid-files/sampler-sampler.json).

| Check | Result | Bar | Verdict |
|---|---|---|---|
| The checker against the rule that needs no training, digits 5 to 9 | checker 0.861; rule 0.539 per read and 0.603 with each digit's fit summed over a canvas's 64 reads; lead 0.322, standard error 0.009 over pairs | lead more than 1.96 standard errors | **the checker adds to what the model knows** |
| The checker on the sampler's own states, sampler steps 27 to 36 | 0.651 over the 10 steps, on 123 clean against 49 wrong-digit canvases | at least 0.75 | **does not read sampler states**; reported to the owner before V7 |

**The frozen model alone barely reads identity.** The rule reads 0.52 to 0.58 on every pair and 0.50 to 0.57 at every step. Summing a canvas's 64 reads lifts it to 0.603. The trained checker's 0.861 is mostly its own.

**On sampler states the checker reads less.** Per step it reads 0.67 at step 27 (t 449), peaks at 0.737 at step 29 (t 408), and falls to 0.60 to 0.63 by steps 33 to 36. The states come from V3's pushed renders at seeds 1000 to 1095 on the 10 pairs among 5 to 9, 960 canvases. The judge calls 123 clean, 328 "two, not both clear and right", 443 one blob, 57 damaged and 9 same half. Only 49 of the 328 hold two confident digits that are not the asked pair, so most of them are unclear rather than wrong, and they are left out. Per pair the labels are uneven: 5 and 7 has 3 wrong and 1 clean, 6 and 8 has 1 wrong and 26 clean.

**What this cannot say.** The labels come from the finished picture, ten to twenty steps after the read, so a state the checker reads correctly can still end in a different picture. The training set's `wrong` canvases always hold one asked digit and one other digit; on sampler states a wrong canvas may hold the same asked digit twice, which the training never showed (inferred; the judge's per-canvas digits were not saved).

## Retraining on the sampler's own states

The checker was retrained on the sampler's own states, as [the design's retraining section](design.md#retraining-on-the-samplers-own-states) sets out (board question 94, option A). It **does not read sampler states**: 0.552 on fresh digits 5 to 9, against the bar of 0.75. Step V7 does not run, and the result goes to the owner. The numbers are in [the training data file](grid-files/collect-collect.json) and [the read file](grid-files/train-read.json).

| What | Result |
|---|---|
| Training canvases, digits 0 to 4, seeds 2000 to 2959 (job 94654, 15 min) | 9,600 rendered; 1,486 clean (label 1) and 550 with two confident digits that are not the asked pair (label 0); 3,738 one blob, 3,358 two digits not both clear, 904 damaged and 114 in one half were left out |
| Training (job 94655) | the 16-example check learned 16 of 16; 4,000 updates, final loss 0.0014 |
| Read on fresh digits 5 to 9, seeds 3000 to 3095, used once | 0.552 over the 10 steps (0.49 to 0.61 per step), on 108 clean against 50 wrong-digit canvases |
| Read on V1's pasted digits 5 to 9, beside it | 0.541: 0.750 right on two-right-digit canvases, 0.332 on one-wrong-digit canvases |

**It learned the training digits, not the property.** A loss of 0.0014 on 2,036 canvases, with both reads of unseen digits near chance, is what memorising digits 0 to 4 looks like. V6's original checker trained on 15,000 pasted canvases and carried over to unseen digits at 0.861; on these states it read 0.651. Retraining on 2,036 states lost the carry-over without gaining the states.

**What this cannot say.** One training size and one recipe were tried. A checker trained on pasted canvases and sampler states together, or on many more sampler states, has not been tested. The smoke test (job 94653) read seeds 2960 to 2967, which are outside both ranges.

## Is identity visible at these steps at all

Board question 95, amended by Loop 13 for the owner after [the method review](method-review.md), ends the digit identity work with one diagnostic and no further training: the digit judge reads the sampler's one-step clean estimate, $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_{\text{product}})/\sqrt{\bar\alpha_t}$, on the 2,036 saved digits 0 to 4 states at steps 27 to 36. A read counts as "the asked pair" when the judge calls the clean estimate "two clear, right, apart". Job 94717 (mscluster80, 2 minutes) wrote [the result file](grid-files/clean-check.json). The bar, written before the run, is 0.75.

| Step | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | 0.655 | 0.681 | 0.692 | 0.713 | 0.727 | 0.738 | 0.738 | 0.758 | 0.756 | 0.780 |
| Clean canvases called the asked pair | 0.55 | 0.55 | 0.55 | 0.58 | 0.62 | 0.67 | 0.72 | 0.77 | 0.81 | 0.84 |
| Wrong-digit canvases called the asked pair | 0.24 | 0.19 | 0.17 | 0.16 | 0.17 | 0.20 | 0.25 | 0.25 | 0.30 | 0.28 |

**Identity is not plainly visible at these steps: 0.724 over the 10 steps, under 0.75.** The judge on the clean estimate does better than either checker on the states (0.651 and 0.552) and still misses. At step 27 it finds the asked pair on only 55% of canvases that end clean, so on nearly half of them the digits they will end with are not yet drawn. The two checkers' misses are partly a property of the states, not only of the checkers.

## When the sampler settles which digits it draws

Board question 97 (Loop 12's audit of this loop's designs, row 8) asked whether V7's identity push at steps 27 to 36 would come after the sampler had settled which digits it draws. The check re-rendered V3's plain and pushed rows on the 10 pairs among digits 5 to 9 at seeds 1000 to 1031, 320 canvases each, and read the sampler's one-step clean estimate with the digit judge at all 50 steps. A canvas is settled from the first step after which the judge's digits on the clean estimate always match its finished picture. It ran on mscluster107's card 1 by the owner's yes on board question 99 (`v6/shared/launch-3.sh`, code commit `070118f6`, 3 minutes, log `logs/l8-settle-shared.log`). The numbers are in [the result file](grid-files/settle.json).

| Row | Settled by step 15 | by step 26 | by step 36 | Median step settled | First step where 90% match |
|---|---|---|---|---|---|
| Plain product | 4% | 16% | 46% | 37 | 44 |
| V3's count push | 0% | 15% | 45% | 38 | 46 |

**The digits are still open through the identity window.** At step 27, 39% of plain canvases and 32% of pushed ones already show the digits they end with; half settle only at step 37 or 38. A push at steps 27 to 36 comes before most canvases have settled, so V7's window was not too late. This agrees with the judge on the clean estimate at those steps reading 0.724: the right digits are not yet drawn on many canvases there.

**V7 is closed as not run.** No checker reads identity on the sampler's states at steps 27 to 36 (0.651, then 0.552 after retraining), and the judge itself reads only 0.724 there, because most canvases have not yet settled. Since board question 86 there is no SDXL removal test for V7 to feed.

## The runs

| Job | What | Node | Time | Result |
|---|---|---|---|---|
| 94335 | smoke test, all four cells on 1% of the pool, 20 updates | mscluster87 | 2 min | every cell learned its 16 fixed examples |
| 94336 | two-map cells | mscluster87 | 27 min | 0.563 and 0.702 |
| 94337 | both signature cells | mscluster60 | 38 min | ran out of memory scoring 512 canvases at once (eleven U-Net calls each), after training the first cell; no score |
| 94352 | signature, early window | mscluster60 | 44 min | 0.612 |
| 94353 | signature, later window | mscluster87 | 48.5 min | 0.888 |
| 94383 | the passing cell retrained on digits 0 to 4, read on digits 5 to 9 | mscluster87 | 55 min | 0.861 |
| 94471 | second training seed | mscluster60 | 44 min | 0.877 |
| 94472 | labels shuffled | mscluster88 | 47 min | 0.500 |
| shared run | the rule that needs no training, then the sampler read | mscluster107 card 1 | 17 min | 0.539 and 0.651 |
| 94654 | training states on digits 0 to 4 | mscluster60 | 15 min | 2,036 labelled |
| 94655 | retraining on states, then the fresh read | mscluster60 | 46 min | 0.552 |
| 94717 | the judge on the clean estimate of the saved states | mscluster80 | 2 min | 0.724 |
| shared run | when the sampler settles its digits | mscluster107 card 1 | 3 min | median step 37 and 38 |

Commit `ba6d84da` scores the signature's U-Net calls in pieces of 128, the size training uses, and gives each signature cell its own job. The grid used 2.66 GPU-hours, the failed job included, and the read of digits 5 to 9 0.92. Loop 8 has used about 10.9 of its 30.

## What this cannot say

These are pasted canvases. Whether its push makes the sampler draw the right digits is step V7. The later window is SDXL's steps 16 to 31, where the layout may already be fixed, so a push there may fix identity without moving merged digits apart.

On SDXL the signature has no fixed list of labels: "the other eight digits" becomes a chosen set of other prompts, and that choice could move the score. Step V4 has to fix that set before it runs. Each cell trained once, with one seed.
