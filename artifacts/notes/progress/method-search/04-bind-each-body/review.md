# Review: does giving each subject its own region fix identity? (step M9)

This file holds what the runs in [the M9 design](design.md) produced. Rung 1 has no verdict yet, and rung 2 has not started.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-6/m9/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

| Rung | Slurm job | Commits | Output | State |
|---|---|---|---|---|
| Rung 1 smoke, seed 9 | 93719 | `275c54f4-15d8bdd` | none | stopped by the GPU guard, because another process held the card on mscluster107 |
| Rung 1 smoke, seed 9 | 93724 | `275c54f4-15d8bdd` | `smoke/` | passed |
| Rung 1, all 8 seeds | 93794 | `275c54f4-15d8bdd` | `full/` | running |

## Rung 1 controls on the smoke seed

**Does the joint render match the picture its boxes came from?** ✅ Yes. The mean absolute difference on seed 9 is 0.0 of 255, against a limit of 3.

**Does the regional composer reduce to the plain product?** ✅ Yes. Masks of 0.5 at guidance 7.5 and the plain product at guidance 3.75 differ
by 1.53 of 255 in mean absolute pixel value, against a limit of 3.
