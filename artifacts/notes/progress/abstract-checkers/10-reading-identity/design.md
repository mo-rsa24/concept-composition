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

## What this cannot say

A pass says a checker reads digit identity on pasted canvases. Whether its gradient helps the sampler draw the right digits is V5's rerun. The later window is later in sampling, where the layout may already be fixed, so an identity push there may fix identity without moving the digits apart.
