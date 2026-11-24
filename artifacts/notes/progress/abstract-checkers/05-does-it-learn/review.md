# Review: does a checker learn plurality on unseen digits (step V2)

**Status: done, by the count checker.** The count checker passes on digits 5 to 9, which it never trained on (W&B run `tyhrpyov`), and by the owner's answer to board question 69 step V3 runs with its push alone. The first design's single checker, which had to tell both layout and digit identity, is null on digits 5 to 9 (W&B run `a562r799`), and the identity checker trained apart misses its rehearsal; both verdicts stay as recorded below.

**Which bars the step met.** Board question 55 made V2 also need balanced accuracy of at least 0.75 on `apart` against `wrong`. Board question 67 split that work into two checkers: the count checker carries the first two bars, and the identity checker carries question 55's bar. The count checker meets its two bars. Question 55's bar is met by no checker here: the identity checker reads 0.555 on the pairs holding digit 4, and the right-digits checker trained the same way reads 0.556 on digits 5 to 9 (step V5). The board marks V2 done on the count checker by the owner's answer to question 69.

**Noise levels.** The board's proof for V2 names steps 0 to 15. The single checker was read at the toy's own sampler steps 0 to 15 (timesteps 999 to 693). After board question 64, every later checker here was read at the 16 timesteps 712 down to 480 that match SDXL's first 16 steps, which lie between the toy's sampler steps 14 and 25.

## The count checker on digits 5 to 9

Job 94108 on `mscluster79` took 27 minutes 32 seconds, from the code snapshot of commit `0974c41e`. It trained the count checker on all 10 pairs among digits 0 to 4 at the 16 timesteps matching SDXL's noise, read `seen_test` and `unseen_test` once, and saved both checkers to `/datasets/mmolefe/poe_repair_min/loop-8/v2/count-final/`.

| Score over the 16 steps, 0.5 is chance | Bar | Map reader | Raw reader |
|---|---|---|---|
| Digits 0 to 4, `seen_test` | none | 0.782 | 0.787 |
| Digits 5 to 9, `unseen_test` | at least 0.75 | **0.791** | 0.754 |
| Gap | at most 0.05 | **0.009** | 0.033 |
| Overfit check, of 16 | at least 15 | 16 | 16 |

The verdict is support. The map reader's score on digits 5 to 9 rises from 0.63 at the noisiest step to 0.92 at the least noisy, and every one of the 10 unseen pairs scores between 0.77 and 0.81. By kind it calls 96% of overlaid canvases correctly and 65% of single-digit canvases, again its weakest kind.

The raw reader, which is told the asked digits, loses 0.033 moving to digits it never saw, while the map reader loses nothing. On this one read, reading where each asked digit acts carries over better than reading the pixels with the digit names.

The same job wrote `count-explained.npz`: the saved map reader's real maps, score and push gradient on the zoom ladder's four example canvases at timesteps 712, 616, 539 and 480.

## Two checkers at SDXL's noise level

Job 94085 on `mscluster42` took 49 minutes 54 seconds, from the code snapshot of commit `6c3f8a16`, and wrote `/datasets/mmolefe/poe_repair_min/loop-8/v2/sdxl-split/split.json`. Both checkers trained on the pairs among digits 0 to 3 at the 16 toy timesteps matching SDXL's noise, and were scored on those pairs and on the pairs holding digit 4. Digits 5 to 9 were not read. All scores are balanced accuracy over the 16 steps, where 0.5 is chance.

| Checker | Reader | Trained pairs | Pairs holding 4, bar 0.75 | Gap | Overfit, of 16 | Verdict |
|---|---|---|---|---|---|---|
| Count: two digits apart? (11,504 training canvases, 800 test) | map | 0.781 | **0.774** | 0.007, bar 0.05 | 16 | **pass** |
| | raw | 0.793 | 0.775 | 0.018 | 16 | |
| Identity: the asked digits? (7,004 training canvases, 500 test) | map | 0.603 | **0.555** | 0.048 | 16 | **miss** |
| | raw | 0.605 | 0.556 | 0.049 | 16 | |

The count checker's score on the pairs holding digit 4 rises from 0.64 at the noisiest step to 0.91 at the least noisy. By kind it calls 95% of overlaid canvases correctly and 65% of single-digit canvases, its weakest kind.

The identity checker rises from 0.48 to 0.65 across the window, so it reads a little identity at the least noisy steps and none at the noisiest. It says no to 63% of reads. Its training pool holds 6,000 right-digit canvases and 1,004 wrong-digit ones, drawn half and half, so each wrong-digit canvas is drawn about 510 times. The raw reader, which is told the asked digits, scores the same, so the limit is the noise level and the training data and not the maps.

## The rehearsal at SDXL's noise level

<a href="two-noise-levels.png"><img src="two-noise-levels.png" width="600" alt="Balanced accuracy on pairs holding digit 4 across the 16-step window. At SDXL's noise, all canvas kinds rise from 0.59 to 0.81; at the toy's own early noise they rise from 0.50 to 0.60. Right digits against a wrong digit stays near 0.5 at both noise levels"></a>

Job 94080 on `mscluster42` took 32 minutes 48 seconds, from the code snapshot of commit `6ad01702`, and wrote `/datasets/mmolefe/poe_repair_min/loop-8/v2/sdxl-window/rehearse.json`. Both readers trained on the pairs among digits 0 to 3 at the 16 toy timesteps that match SDXL's noise (712 down to 480), as the design's section on board question 64 sets out. Digits 5 to 9 were not read. Solid lines are this run and dashed lines are the base map reader at the toy's own early noise.

| Score over the 16 steps, pairs holding digit 4, 0.5 is chance | Bar | Map reader | Raw reader |
|---|---|---|---|
| Overfit check, 16 fixed examples | at least 15 right | 16 | 16 |
| Balanced accuracy, all canvas kinds | at least 0.75 | **0.704** | 0.713 |
| Gap to the trained pairs | at most 0.05 | 0.029 | 0.037 |
| `apart` against `wrong` | at least 0.75 | **0.511** | 0.514 |
| Share of reads called label 1 | between 0.01 and 0.99 | 0.636 | 0.554 |
| Wrong-digit canvases called label 0 | none | 0.18 | 0.26 |

The verdict is a miss by the written rule. Both readers trained, and both learned where the digits sit: the map reader rises from 0.59 at the noisiest step to 0.81 at the least noisy. Neither learned which digits are present. They call 82% and 74% of wrong-digit canvases label 1, and the raw reader is told the asked digits and still scores 0.514 on `apart` against `wrong`, so the maps are not what blocks it. Wrong-digit canvases are 1 in 8 of the training mix, and the other three label-0 kinds can be told apart by layout alone.

**Read afterwards as a two-digits-apart counter**, with `apart` and `wrong` as label 1 and the other three kinds as label 0, the map reader's balanced accuracy is 0.766 on the pairs holding digit 4 and the raw reader's is 0.776. This reading was not planned before the run, so it supports no claim and only shows that the count half of the task is within reach at this noise level.

## The result

<a href="balanced-accuracy-by-step.png"><img src="balanced-accuracy-by-step.png" width="600" alt="Balanced accuracy against sampler step 0 to 15 on digits 5 to 9. All five canvas kinds rise from 0.50 to 0.62, and both right digits against a wrong digit stays at 0.50, with the pass mark at 0.75"></a>

Each point is the frozen map reader's balanced accuracy at one sampler step, over 2,000 canvases of digits 5 to 9 with 4 noise draws each. The numbers are in `/datasets/mmolefe/poe_repair_min/loop-8/v2/full/final.json`, from job 93905 on `mscluster63`, which took 39 minutes 50 seconds.

| Score over steps 0 to 15, 0.5 is chance | Bar | Map reader | Met | Raw reader |
|---|---|---|---|---|
| Balanced accuracy on digits 5 to 9 | at least 0.75 | **0.540** | no | 0.500 |
| Gap to digits 0 to 4, which scored 0.540 | at most 0.05 | **0.0002** | yes | 0.000 |
| `apart` against `wrong` on digits 5 to 9 | at least 0.75 | **0.501** | no | 0.500 |
| `apart` against `wrong` on digits 0 to 4 | none | 0.504 | | 0.500 |

The verdict is null by the written rule: the overfit rung passed and the frozen configuration missed a bar. The small gap between seen and unseen digits does not show transfer, because the reader is at chance on `apart` against `wrong` for both.

**The raw reader did not train.** Its final loss is 0.6931, the loss of a constant guess, and it answers label 0 on every read: 0.00 right on `apart` and 1.00 on the other four kinds. The map-against-raw comparison therefore has no result. `final.json` holds `raw_matches_map_on_unseen: true` only because 0.500 is within 0.05 of 0.540, so that flag does not show the surprise the design describes.

**The toy is noisier than SDXL at the same step numbers.** The toy's steps 0 to 15 run from timestep 999 to 693, where the signal share of the noisy image ($\bar\alpha$) goes from 0.00004 to 0.0076. SDXL's scheduler, a scaled-linear schedule with leading spacing, runs from timestep 981 to 681, where it goes from 0.0058 to 0.0946 (the toy's nearest timestep to 0.0946, 480, has 0.0949, the end value the design quotes). The signal-to-noise ratio is 144 times higher for SDXL at step 0 and 13.7 times higher at step 15. SDXL's window has the same $\bar\alpha$ as the toy's timesteps 712 to 480, which are toy sampler steps 14 to 25, so only steps 14 and 15 of this test overlap it. This null therefore says nothing direct about SDXL's early steps. Equal $\bar\alpha$ in pixels and in SDXL's latents is also not equal difficulty (inferred).

## The ladder

Job 93691 on `mscluster83` ran from the code snapshot of commit `34750ade` and took 18 minutes 6 seconds. Its outputs are in `/datasets/mmolefe/poe_repair_min/loop-8/v2/ladder/`.

| Rung | What it checks | Map reader | Raw reader | Result |
|---|---|---|---|---|
| Smoke test | 20 updates run and the loss is finite | 455 ms per update | 15 ms per update | passed |
| Overfit | 16 fixed examples, at least 15 right | 16 of 16 | 16 of 16 | passed |
| 1% of the data | balanced accuracy on the 80-canvas seen test split, steps 0 to 15, 0.5 is chance | 0.530 | 0.513 | ran |
| 1% of the data | the same score on `apart` against `wrong` canvases | 0.505 | 0.502 | ran |

The 1% run trains on 160 canvases for 1,000 updates, so it checks that the pipeline runs and carries no verdict. Its scores are close to chance for both readers. The map reader's balanced accuracy by step runs from 0.50 at step 0 to 0.58 at step 15, with a high of 0.61 at step 14.

## The inner test

Job 93702 on `mscluster83` took 2 hours 41 minutes. It trained on the pairs among digits 0 to 3 and scored both those pairs and the pairs holding digit 4, which it never saw. All scores are balanced accuracy over steps 0 to 15, where 0.5 is chance. The full results are in `/datasets/mmolefe/poe_repair_min/loop-8/v2/full/inner.json`.

| Configuration | Trained pairs | Pairs holding 4, bar 0.75 | Gap, bar 0.05 | `apart` against `wrong`, trained pairs | `apart` against `wrong`, pairs holding 4, bar 0.75 | All three bars met |
|---|---|---|---|---|---|---|
| Base map reader | 0.542 | 0.535 | 0.006 | 0.499 | 0.497 | no |
| Log totals added | 0.542 | 0.535 | 0.007 | 0.499 | 0.500 | no |
| Signed differences | 0.544 | 0.531 | 0.013 | 0.502 | 0.495 | no |
| 12,000 updates | 0.545 | 0.537 | 0.008 | 0.503 | 0.498 | no |

No configuration separates `apart` from `wrong`, on the pairs it trained on or on the pairs it never saw. The miss is in learning which digits are present, before any question of carrying over to new digits. Balanced accuracy over all canvas kinds rises from 0.50 at step 0 to between 0.60 and 0.61 at step 15 on the pairs holding digit 4.

The written rule freezes the configuration with the highest smaller value of the two scores on the pairs holding 4, which is **log totals added** at 0.4999. The four values span 0.4946 to 0.4999, so this choice separates configurations that are all at chance.

GPU-hours spent so far: 5.62 of 30 (0.14 on step V1, 0.30 on this ladder, 2.68 on the inner test, 0.66 on the final read, 0.55 on the rehearsal at SDXL's noise level, 0.83 on the two checkers and 0.46 on the count checker's read of digits 5 to 9).
