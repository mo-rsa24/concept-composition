# Do inclusive labels help the product compose on MNIST?

**What this asks**

A small diffusion model learns MNIST digits on 32 × 64 canvases. A setting called the dial (0, 0.1, 0.3, 0.6 or 1) says how often a two-digit canvas is also trained under each single label. The question is whether a higher dial makes the product of "7" and "8" draw two separate right digits.

**The answer on record**

The verdict is null, set in task 8 of [the MNIST stage plan](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/02-the-mnist-stage.md) on 2026-11-22 `stated`. The dials that count are 0, 0.1 and 0.3, where the corrected 7 × 8 product draws a mean of 0, 0.33 and 8.0 of the first 64 seeds by eye, under the null line of 9.75 `verified`. Dial 0.6 stays out because "7" alone drew its lone digit on 205 of 256 seeds in training 1, under that dial's floor of 207 `verified`.

**What is here**

<a href="the-leak-read.png"><img src="the-leak-read.png" width="200" alt="Two panels against the dial. Top: percent of 256 seeds with two separate right digits for the 7 x 8 and 4 x 7 products, all under the support line. Bottom: percent of draws where 7, 8 and 4 alone draw their digit alone, beside dashed training lines"></a>

Both panels have the dial on x and one dot per training. The top y-axis is the percent of 256 seeds with two separate right digits, and the bottom y-axis is the percent of draws where a single label draws its digit alone, beside a dashed line for its training data `verified`.

**How it was made**

[The scorer](../../../../poe_repair/experiments/inclusive_captions/mnist/score.py) calls every tile, writes [the per-tile calls](scores.csv) (30,720 rows) and [the counts file](counts.json), and draws the figure `verified`. The floors and lines are in [the bar file](../../../../poe_repair/experiments/inclusive_captions/mnist/bar.py).

**Where it is used**

[The report page on inclusive labels](../../../report/do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md) gives every count and the places where its sources disagree.
