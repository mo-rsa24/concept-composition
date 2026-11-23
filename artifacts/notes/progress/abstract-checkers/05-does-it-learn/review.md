# Review: does a checker learn plurality on unseen digits (step V2)

**Status: open.** The smoke test, the overfit check and the 1% run are done. All four configurations missed the inner test. The one read of `unseen_test` has not run, so step V2's proof is not met.

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

| Configuration | Trained pairs | Pairs holding 4, bar 0.75 | Gap, bar 0.05 | `apart` against `wrong`, trained pairs | `apart` against `wrong`, pairs holding 4, bar 0.75 | All bars |
|---|---|---|---|---|---|---|
| Base map reader | 0.542 | 0.535 | 0.006 | 0.499 | 0.497 | missed |
| Log totals added | 0.542 | 0.535 | 0.007 | 0.499 | 0.500 | missed |
| Signed differences | 0.544 | 0.531 | 0.013 | 0.502 | 0.495 | missed |
| 12,000 updates | 0.545 | 0.537 | 0.008 | 0.503 | 0.498 | missed |

No configuration separates `apart` from `wrong`, on the pairs it trained on or on the pairs it never saw. The miss is in learning which digits are present, before any question of carrying over to new digits. Balanced accuracy over all canvas kinds rises from 0.50 at step 0 to between 0.60 and 0.61 at step 15 on the pairs holding digit 4.

The written rule freezes the configuration with the highest smaller value of the two scores on the pairs holding 4, which is **log totals added** at 0.4999. The four values span 0.4946 to 0.4999, so this choice separates configurations that are all at chance.

GPU-hours spent so far: 3.12 of 30 (0.14 on step V1, 0.30 on this ladder and 2.68 on the inner test).
