# Review: can a score that names no subject rank CO3's pictures within a pair? (step M15)

This file holds what the runs in [the M15 design](design.md) produced. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m15/`.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Positive control: the cat x dog figure's 8 joint-prompt trajectories, reproduced within 0.05 at steps 5, 10 and 15 | 95425 | `3189b477-15d8bdd` | `m15/control/` | passed in 4 minutes 20 seconds on mscluster89: the largest gap to the figure's saved overlap is 0.024, under 0.05, on all 8 seeds |
| Smoke: one CO3 trajectory read | 95426 | `3189b477-15d8bdd` | `m15/read-smoke/` | passed in 48 seconds |
| Read: the 192 saved CO3 trajectories at steps 5, 10 and 15 | 95427 | `3189b477-15d8bdd` | `m15/read/` | done in 3 minutes 26 seconds, 192 trajectories, 576 states |

## The positive control

**The measure reproduces the cat-and-dog figure.** Rerunning the figure's 8 joint-prompt trajectories through this code gives the figure's
saved top-10% overlap to within 0.024 at steps 5, 10 and 15 (exactly on six seeds; 0.016 off on seed 1 at step 15, 0.024 on seed 4 at step
10, 0.011 on seed 5). As description, the seeds that came out apart (2, 3 and 7) have a mean overlap of 0.28, 0.25 and 0.26 at steps 5, 10
and 15, against 0.36, 0.33 and 0.34 for the other five, as in the figure. The numbers are in `m15/control/control.json`.

## The verdict

**The subject-free score does not rank CO3's pictures within a pair, under the bar as written.** On the 192 labelled CO3 pictures, over
415 within-pair couples, the score's within-pair area is at or below the shuffled control's 95th percentile at every step.

| Step | Within-pair area of the score (0.5 is chance, bar 0.70) | Shuffled control, 95th / 99th percentile | Overlap alone | Instance counts alone |
|---|---:|---|---:|---:|
| 5 | 0.540 | 0.599 / 0.634 | 0.511 | 0.547 |
| 10 | 0.516 | 0.599 / 0.637 | 0.507 | 0.524 |
| 15 | 0.570 | 0.594 / 0.629 | 0.588 | 0.543 |

For comparison, the judge on the decoded clean estimate reached 0.567, 0.610 and 0.744 at the same steps (the reviewer's pass), so the
attention score is at or under the judge at every step and falls further behind by step 15. The numbers are in `score.json` in the laptop's
`outputs/loop-6/m15/`.

**The overlap is lower on two-clear pictures overall, but not within a pair.** The median overlap is 0.010, 0.020 and 0.041 on two-clear
pictures against 0.046, 0.057 and 0.103 on the rest at steps 5, 10 and 15. Within a pair that difference vanishes (0.51 at steps 5 and 10),
so it comes from which pairs succeed, the same pair effect that carried the per-place readers. CO3's overlaps are also far below the cat-and-dog
joint prompt's (0.25 to 0.36 group means in the figure), so on CO3 the two nouns already look at different places on most pictures,
including those that borrow parts.

**By failure kind, as description only.** On the 96 tuning seeds with Loop 9's labels, own parts against borrowed parts gives 0.48, 0.71 and
0.52 at steps 5, 10 and 15 (31 couples), and own parts against "not two" gives 0.61, 0.46 and 0.56 (49 couples). With 31 and 49 couples these
swing by chance more than the bar allows, and no kind is read consistently across steps.

**What follows.** By the design, the where-map result stays a digits-only result: SDXL's attention at steps 5 to 15, scored without naming a
subject, does not tell CO3's two-clear pictures from the rest of the same pair.
