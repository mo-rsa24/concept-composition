# Can a checker read which digits are present (step V6): review

**Status: support, on digit 4 and on digits 5 to 9; two robustness checks are queued (jobs 94471 and 94472).** One of the four checkers, the ten-label signature read in the later window, tells two right digits from one right and one wrong digit at 0.888 balanced accuracy on the pairs holding digit 4, which it never trained on. The bar, written before the run, is 0.75; 0.5 is chance. The design is [the V6 design](design.md).

<a href="identity-grid.png"><img src="identity-grid.png" width="900" alt="The 2 by 2 grid. V2's identity checker at 0.555 leads into the early window, where two maps score 0.563 and the ten-label signature 0.612, both red. The later window holds two maps at 0.702, red, and the ten-label signature at 0.888, green, which leads to the retrain on digits 0 to 4 and the read of digits 5 to 9."></a>

## Contents

- [The grid](#the-grid)
- [What the grid shows](#what-the-grid-shows)
- [Digits 5 to 9](#digits-5-to-9)
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

The second training seed (job 94471) and the shuffled-label control (job 94472) are queued on `bigbatch`; their results complete the robustness checks.

## The runs

| Job | What | Node | Time | Result |
|---|---|---|---|---|
| 94335 | smoke test, all four cells on 1% of the pool, 20 updates | mscluster87 | 2 min | every cell learned its 16 fixed examples |
| 94336 | two-map cells | mscluster87 | 27 min | 0.563 and 0.702 |
| 94337 | both signature cells | mscluster60 | 38 min | ran out of memory scoring 512 canvases at once (eleven U-Net calls each), after training the first cell; no score |
| 94352 | signature, early window | mscluster60 | 44 min | 0.612 |
| 94353 | signature, later window | mscluster87 | 48.5 min | 0.888 |
| 94383 | the passing cell retrained on digits 0 to 4, read on digits 5 to 9 | mscluster87 | 55 min | 0.861 |

Commit `ba6d84da` scores the signature's U-Net calls in pieces of 128, the size training uses, and gives each signature cell its own job. The grid used 2.66 GPU-hours, the failed job included, and the read of digits 5 to 9 0.92. Loop 8 has used about 10.9 of its 30.

## What this cannot say

These are pasted canvases. Whether its push makes the sampler draw the right digits is step V7. The later window is SDXL's steps 16 to 31, where the layout may already be fixed, so a push there may fix identity without moving merged digits apart.

On SDXL the signature has no fixed list of labels: "the other eight digits" becomes a chosen set of other prompts, and that choice could move the score. Step V4 has to fix that set before it runs. Each cell trained once, with one seed.
