# Review: does sample-and-paste already solve small CLEVR? (step M7)

This file holds what the runs in [the M7 design](design.md) produced. M7 has no verdict yet.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-6/m7/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Smoke render, 4 pictures each of position 1, empty and all three positions | 93710, 93711, 93712 | `337d1be4-15d8bdd` | `smoke/renders/` | passed |
| Full renders of the empty scene, 5,000 pictures | 93716 | `337d1be4-15d8bdd` | `full/renders/empty/` | running |
| Position 1, first attempt | 93713 | `337d1be4-15d8bdd` | none | stopped by a shared-disk race on `mkdir` |
| Positions 1, 2 and 3, first pass | 93718, 93714, 93715 | `8c0c129a-15d8bdd` for position 1, `337d1be4-15d8bdd` for 2 and 3 | `full/renders/` | ended at 1,000, 996 and about 880 pictures, because the generator's visibility check leaves one file open per call and the node's limit is 1,024 |
| Positions 1, 2 and 3, resumed from those pictures with the handle closed | 93907, 93908, 93909 | `100772b7-15d8bdd` | `full/renders/` | running, and positions 1 and 2 have each rendered past 1,024 new pictures in one process |
| Scorer subsets s0 to s7, 250 pictures each, one job at a time | 93949 to 93956 | `100772b7-15d8bdd` | `full/renders/` | finished, 250 of 250 in each subset |
| EDM2 smoke test on the empty scenes, 16,384 images seen, then 16 samples | 94044 | `beaf1219-15d8bdd` | `smoke/models/empty/` | passed, 3 minutes 31 seconds |
| EDM2 overfit on 16 pictures of position 1, EDM2's fixed ramp | 94047 | `50d1d1a1-15d8bdd` | `overfit/models/p1-16/` | failed, the samples are noise |
| Scorer: overfit on 16 pictures, then 1,600 train and 400 held out, on CPU | 93978 | `a6c06e83-15d8bdd` | `full/scorer/` | passed: 16 of 16 on the overfit, and 100% at all three positions on the 400 held-out renders against a bar of 99% (W&B `guf6vtho`) |
| EDM2 overfit on 16 pictures of position 1, ramp scaled to the run | 94055 | `0306b4bf-15d8bdd` | `overfit-scaled-ramp/models/p1-16/` | running |

## The smoke render

<a href="smoke-objects-at-their-positions.png"><img src="smoke-objects-at-their-positions.png" width="600" alt="Twelve CLEVR renders at 64 pixels: four with one object at position 1, four empty scenes, and four with objects at all three positions, each asked position outlined by its 18-pixel paste window"></a>

All 16 placed objects landed within 0.002 of the image of where they were aimed. A picture took 4.9 to 5.3 seconds with objects and 4.5
without. Large objects at position 3 reach the edge of the 18-pixel paste window, so the window control in the design decides whether the
window needs fixing before any sample is scored.

## Training speed

The EDM2 smoke test trained `edm2-img64-xs` at 6.82 seconds per thousand images on one RTX 3090, with 128 images per GPU pass and 22 GB in use.
At that rate one model takes about 3.8 GPU-hours for 2 million images and the four models take about 15.2, against the design's budget of 8 to
10.

**EDM2's fixed ramp does not train a short run.** With the ramp at its fixed 10 million images, a model trained on 16 pictures for 256,000 images
reached a learning rate of 0.0003, which is 2.5% of the full 0.012. Its loss moved from 1.13 to 1.03, and its samples are noise with the scene's
rough brightness.

<a href="overfit-unscaled-ramp-real-and-sampled.png"><img src="overfit-unscaled-ramp-real-and-sampled.png" width="420" alt="Top two rows: the 16 real pictures of one object. Bottom two rows: 16 samples from the model, which are colour noise"></a>

Each model now keeps the preset's ramp as a fraction of training, 0.93%, through `edm2_scaled_ramp.py`, which changes no file of EDM2.

## The window control

**Result: it misses its mark of 95% on all three positions.** Over 256 real renders per position, the share of changed pixels inside the 18 by 18
window is 94.4% at position 1, 93.9% at position 2 and 91.7% at position 3. A pixel counts as changed when a colour channel differs from the mean
empty scene by more than 0.1 on a 0-to-1 scale. The numbers are in `window-control-written-rule.json`.

<a href="window-control-written-rule.png"><img src="window-control-written-rule.png" width="420" alt="How often each pixel changes, for the three positions, with the 18 by 18 window drawn: position 3's object runs past the top and bottom of its window"></a>

**Why it misses.** Position 3 sits nearer the camera, so its objects reach 24 pixels tall against a window of 18, and 2,301 of its 3,170 outside
pixels are on the object's own footprint. Positions 1 and 2 have no object pixels outside the window. Their 820 and 977 outside pixels are scattered
across the frame, about 3 per picture, against 0.39 per picture in an empty scene. The breakdown is in `window-diagnosis.json`, where a pixel is on
the footprint if it changes in at least 2% of renders and connects to the position.

**What follows.** No sample has been scored, and the design says the window is fixed before one is. The choice of fix is with the owner, because this
paste is the frozen rival.

