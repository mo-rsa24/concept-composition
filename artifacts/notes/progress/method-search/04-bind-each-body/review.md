# Review: does giving each subject its own region fix identity? (step M9)

This file holds what the runs in [the M9 design](design.md) produced. Rung 1 has a count of 0 clear pairs of 8 on right boxes, and its verdict is held until the sampler checks finish. Rung 2 has not started.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-6/m9/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

| Rung | Slurm job | Commits | Output | State |
|---|---|---|---|---|
| Rung 1 smoke, seed 9 | 93719 | `275c54f4-15d8bdd` | none | stopped by the GPU guard, because another process held the card on mscluster107 |
| Rung 1 smoke, seed 9 | 93724 | `275c54f4-15d8bdd` | `smoke/` | passed |
| Rung 1, all 8 seeds | 93794 | `275c54f4-15d8bdd` | `full/` | finished, 23 minutes 50 seconds on one RTX 3090 |

## Rung 1 controls on the smoke seed

**Does the joint render match the picture its boxes came from?** ✅ Yes. The mean absolute difference on seed 9 is 0.0 of 255, against a limit of 3.

**Does the regional composer reduce to the plain product?** ✅ Yes. Masks of 0.5 at guidance 7.5 and the plain product at guidance 3.75 differ
by 1.53 of 255 in mean absolute pixel value, against a limit of 3.

## Rung 1 count, before the sampler checks

All 48 pictures were called blind at full size by two reviewers, who agreed on 47. A third blind call settled the other one, a joint render on
seed 19, as two bodies with borrowed parts. The calls and key are in `rung-1-calls-first.csv`, `rung-1-calls-second.csv`, `rung-1-calls-third.csv`
and `rung-1-key.json`, and the counts are in `rung-1-result.json`.

<a href="rung-1-six-rows-eight-seeds.png"><img src="rung-1-six-rows-eight-seeds.png" width="420" alt="Eight seeds by six conditions: the joint prompt draws a cat beside a dog, and every region row draws one animal or a fused cat-dog"></a>

| Row | Clear pairs of 8 | Pictures with two separate animals, of 8 |
|---|---|---|
| Joint prompt | 7 | 8 |
| Plain product | 0 | 0 |
| Right boxes | 0 | 0 |
| Random boxes | 0 | 0 |
| Swapped prompts | 0 | 0 |
| Same prompt twice | 0 | 0 |

**What this count can say.** Fusing the two predictions by region from step 0 did not separate the animals on any of the 8 seeds, even with
the layout taken from a render that succeeded. Right boxes drew 6 fused cat-dogs and 2 single animals. The bar calls 0 of 8 a null, and the
oracle-layout plan holds a null until bootstrapping and then MultiDiffusion's own code have been run on the same layouts.
