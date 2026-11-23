# Review: does a checker learn plurality on unseen digits (step V2)

**Status: open.** The smoke test, the overfit check and the 1% run are done. The inner test and the one read of `unseen_test` have not run, so step V2's proof is not met.

## The ladder

Job 93691 on `mscluster83` ran from the code snapshot of commit `34750ade` and took 18 minutes 6 seconds. Its outputs are in `/datasets/mmolefe/poe_repair_min/loop-8/v2/ladder/`.

| Rung | What it checks | Map reader | Raw reader | Result |
|---|---|---|---|---|
| Smoke test | 20 updates run and the loss is finite | 455 ms per update | 15 ms per update | passed |
| Overfit | 16 fixed examples, at least 15 right | 16 of 16 | 16 of 16 | passed |
| 1% of the data | balanced accuracy on the 80-canvas seen test split, steps 0 to 15, 0.5 is chance | 0.530 | 0.513 | ran |
| 1% of the data | the same score on `apart` against `wrong` canvases | 0.505 | 0.502 | ran |

The 1% run trains on 160 canvases for 1,000 updates, so it checks that the pipeline runs and carries no verdict. Its scores are close to chance for both readers. The map reader's balanced accuracy by step runs from 0.50 at step 0 to 0.58 at step 15, with a high of 0.61 at step 14.

GPU-hours spent so far: 0.44 of 30 (0.14 on step V1 and 0.30 on this ladder).
