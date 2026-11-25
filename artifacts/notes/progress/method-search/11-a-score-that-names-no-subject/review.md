# Review: can a score that names no subject rank CO3's pictures within a pair? (step M15)

This file holds what the runs in [the M15 design](design.md) produced. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m15/`.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Positive control: the cat x dog figure's 8 joint-prompt trajectories, reproduced within 0.05 at steps 5, 10 and 15 | 95425 | `3189b477-15d8bdd` | `m15/control/` | queued after M11's shared-canvas diagnosis |
| Smoke: one CO3 trajectory read | 95426 | `3189b477-15d8bdd` | `m15/read-smoke/` | runs only if the control reproduces |
| Read: the 192 saved CO3 trajectories at steps 5, 10 and 15 | 95427 | `3189b477-15d8bdd` | `m15/read/` | runs after the smoke |
