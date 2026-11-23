# Review: does a checker learn plurality on unseen digits (step V2)

**Status: open.** The smoke test, the overfit check and the 1% run are done. The inner test is in job 93702, and the one read of `unseen_test` has not run, so step V2's proof is not met.

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

Job 93702 on `mscluster83` trains on the pairs among digits 0 to 3 and scores the pairs holding digit 4, read from its log at 20:01 on 2026-11-23. Its full results land in `/datasets/mmolefe/poe_repair_min/loop-8/v2/full/inner.json` when the job ends.

| Configuration | Balanced accuracy, bar 0.75 | `apart` against `wrong`, bar 0.75 | All bars |
|---|---|---|---|
| Base map reader | 0.535 | 0.497 | missed |
| Log totals added | 0.535 | 0.500 | missed |
| Signed differences | 0.531 | 0.495 | missed |
| 12,000 updates | training at 20:01 | not scored | not scored |

## Cluster access

Commands on `mscluster84` stopped returning between 20:01 and 20:22 on 2026-11-23. At 21:04 a new connection for this loop authenticated in 41 seconds, and a command sent through it then produced no output in 420 seconds. Job 93702's state after 20:01 is unknown.

GPU-hours spent so far: 0.44 of 30 (0.14 on step V1 and 0.30 on this ladder), plus job 93702, which had run for 1.75 hours at 20:01.
