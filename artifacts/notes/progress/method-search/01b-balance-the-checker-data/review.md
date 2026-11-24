# Review: does balancing the checker's training data help? (step M5)

This file holds what the runs in [the M5 design](design.md) produced. M5 has no verdict yet. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m5/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`. A commit pair is the
repository commit, then the replications hub commit.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke, current list: 2 updates | 94579 | `7ac66879-15d8bdd` | `smoke/current/` | passed: one batch trained, a checkpoint saved, 11 minutes 42 seconds on mscluster89 |
| Smoke, balanced list: 2 updates | 94580 | `7ac66879-15d8bdd` | `smoke/balanced/` | passed, 10 minutes 34 seconds |
| CO3 read smoke: puma and condor, seed 6001 | 94593 | `58cee28a-15d8bdd` | `co3-read-smoke/` | passed: the rerun matches M1's picture within 2.38 of 255, all 7 steps read, 38.7 seconds |
| CO3 read: 96 tuning trajectories, on an RTX 3090 | 94594 | `58cee28a-15d8bdd` | `co3-read/` | 75 of 96 reproduce, against 80 needed |
| Overfit, both lists | 94603, 94605 | `ee84f4aa-15d8bdd` | `overfit/`, no checkpoint | failed before training: the tile picker read a `latent` field that benchmark-pool tiles do not carry |
| Overfit, then 1% of the data, current list | 94622, 94623 | `9df9dc5b-15d8bdd` | `overfit/current/`, `tiny/current/` | overfit passed: 16 of 16 right (balanced accuracy 1.0, bar 0.9375) from sampler step 2; 1% rung passed, the logged loss falling from 0.694 to 0.000 over 400 updates. The picker now reads a pool tile's first augmented copy for its check |
| CO3 reread of the 21, then overfit and 1% of the data, balanced list, one after another | shared card: mscluster107 card 1, wrapper process 87714 | `9df9dc5b-15d8bdd` | `co3-read-rtx8000/`, `overfit/balanced/`, `tiny/balanced/` | started 16:25:47 UTC on a Quadro RTX 8000 under board question 90, the Slurm copies 94624 and 94625 cancelled; the reread finished at 16:42:53 UTC with all 21 matching M1 exactly (difference 0.0 of 255); balanced overfit passed, 16 of 16 from sampler step 0 |
| Full training, current list: 2,000 updates | 94638 | `09f8256b-15d8bdd` | `full/current/` | running under Slurm on mscluster89 |
| Saved states, the six pairs M1 drew on an RTX 3090, for scoring new checkers (question 93) | 94637 | `09f8256b-15d8bdd` | `co3-read-states-shard1/` | done: 48 of 48 match M1 exactly, 43 MB of states, 33 minutes on mscluster60 |

## The read of CO3's trajectories

**The checker does not see CO3's failure, under the bar as written.** All 96 tuning trajectories reproduce M1's pictures, 32 of them two-clear
and 64 two bodies that are not both clear. At no step does the balanced accuracy of the clean-pair score's sign reach the 0.65 floor: the best
is 0.602, at step 5. By the design, the owner now decides whether D3 and M2 pause for a checker that reads identity. The merged numbers are in
`co3-read/read-score-merged.json`.

| Sampler step | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced accuracy of the score's sign, 0.5 is chance, bar 0.75, floor 0.65 | 0.547 | 0.602 | 0.563 | 0.594 | 0.594 | 0.586 | 0.578 |
| Area under the curve, 0.5 is chance, reported beside the bar | 0.535 | 0.707 | 0.712 | 0.698 | 0.689 | 0.702 | 0.682 |

**What the two rows say together.** From step 5 on, a two-clear picture outscores a picture with borrowed parts about 7 times in 10, so the
score carries some of the difference. Its zero point does not separate them, so the push, which follows the score toward "clean pair" on every
picture, has no reason to move a borrowed-part picture more than a good one. That fits M1, where the push made 7 faithful pairs of 96 against
11 for CO3 alone.

**What it cannot say.** Whether a checker trained on M5's lists would do better. The read is of the kept checker only, the one both of M5's
trainings start from in architecture and settings.

### The first read, on 75 trajectories

**Inconclusive by its count.** 75 of the 96 reruns match M1's picture within 3 of 255 in
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
`co3-read/read-score-merged.json`. The only usable RTX 8000 is mscluster107's card 1, the shared card of board question 90.

**The order of events.** The 75-trajectory result above (best 0.616 at step 25, inconclusive by its count) was read first. The rerun of the 21
on an RTX 8000 was decided after it, by Loop 13 for the owner, who may overrule: about 15 minutes, reversible, and meeting the read's own
written reproduction rule on the card type that drew the pictures, with no bar changed. The full 96 are reported against the bar exactly as
written, in the section above.
