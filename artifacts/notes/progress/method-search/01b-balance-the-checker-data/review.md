# Review: does balancing the checker's training data help? (step M5)

This file holds what the runs in [the M5 design](design.md) produced. M5 has no verdict yet. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m5/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`. A commit pair is the
repository commit, then the replications hub commit.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke, current list: 2 updates | 94579 | `ab7f6c55-15d8bdd` | `smoke/current/` | passed: one batch trained, a checkpoint saved, 11 minutes 42 seconds on mscluster89 |
| Smoke, balanced list: 2 updates | 94580 | `ab7f6c55-15d8bdd` | `smoke/balanced/` | passed, 10 minutes 34 seconds |
| CO3 read smoke: puma and condor, seed 6001 | 94593 | `58cee28a-15d8bdd` | `co3-read-smoke/` | passed: the rerun matches M1's picture within 2.38 of 255, all 7 steps read, 38.7 seconds |
| CO3 read: 96 tuning trajectories | 94594 | `58cee28a-15d8bdd` | `co3-read/` | inconclusive under the written bar: 75 of 96 reproduce, against 80 needed |
| Overfit, both lists | 94603, 94605 | `ee84f4aa-15d8bdd` | none | failed before training: the tile picker read a `latent` field that benchmark-pool tiles do not carry |
| Overfit, then 1% of the data, both lists | 94622 to 94625 | `9df9dc5b-15d8bdd` | `overfit/`, `tiny/` | queued; the picker now reads a pool tile's first augmented copy for its check |

## The read of CO3's trajectories

**Inconclusive under the written bar, because too few trajectories reproduce.** 75 of the 96 reruns match M1's picture within 3 of 255 in
mean absolute pixel value, against the 80 the bar needs. On those 75, of which 22 are two-clear pictures, the best balanced accuracy of the
clean-pair score's sign is 0.616, at step 25, under the 0.65 floor. The numbers are in `co3-read/read-score.json`.

| Sampler step | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced accuracy of the score's sign, 0.5 is chance | 0.523 | 0.559 | 0.542 | 0.603 | 0.603 | 0.616 | 0.588 |
| Area under the curve, 0.5 is chance | 0.416 | 0.661 | 0.687 | 0.681 | 0.697 | 0.716 | 0.696 |

The area under the curve, reported beside the bar and deciding nothing, sits near 0.7 from step 5 on. The score ranks two-clear pictures above
the others more often than not, and its zero point does not separate them.

**Why 21 did not reproduce.** All 21 come from the six pairs M1 drew on a Quadro RTX 8000 (its shard 0): eagle and fox 7, boar and pheasant 5,
puma and condor 4, giraffe and tortoise 4, seal and gull 1. The read ran on an RTX 3090, and all 48 trajectories M1 drew on an RTX 3090 match,
the largest difference 2.50 of 255. The 21 differ by 3.0 to 27.4. So the difference is the card type, and a rerun of the 21 on an RTX 8000
can meet the reproduction rule with no bar changed. `m5_read.sbatch reread` writes it to `co3-read-rtx8000/` and scores the merged set into
`co3-read/read-score-merged.json`. The only usable RTX 8000 is mscluster107's card 1, the shared card of board question 90, and Loop 13 is asked
for the slot.
