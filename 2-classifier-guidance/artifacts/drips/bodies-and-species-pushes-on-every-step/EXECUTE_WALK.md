# Execute walk: bodies and species pushes on every step

Plan: [plan 31](../../../plans/31-bodies-and-species-pushes-on-every-step.md), opened with `--request` on 2026-11-18.

## Stop point

- Bar: Q1 in [the review file](../../../plans/review/31-bodies-and-species-pushes-on-every-step.md) printed as support, no change or worse, from eye calls on full-size tiles of seeds 3301 to 3332; judged by the user.
- Time box: 2 sittings, about 3 hours of render time.
- Steps at the open: 5 (depth full).
- Toy: seed 9.
- Sittings: 1 started 2026-11-18.

## Position

Round 5: the user's check of the 32 eye calls (`result-sheet-catdog-every-step-general/calls.csv`). Rounds 1 to 4 done; Q1 read as no change (15 of 32 clean or near-clean, 29 separate).

## Decisions

| Decision | Options | Who chose | Why |
| --- | --- | --- | --- |
| Third setting on seeds 9 to 16 | keep the presence-only readout checker on all 50 steps; drop it | user (defaults) | it is MNIST's arrangement exactly |
| Bodies push alone on all 50 steps as a fourth row on seeds 9 to 16 | add; leave out | Claude | separates the species push's share from the window change |
| Time box | 2 sittings, about 3 hours; own | user (defaults) | |
| Round 1 before the presence run lands | yes; wait | user (defaults) | writes files only |
| Commits | none until the user says done | user's rule | |
| The seeds 9 to 16 pick | presence checker; bodies alone; CLIP species 10 or 30 | Claude, user answered `next` | 6 of 8 clean or near-clean against 3; the CLIP rows fuse the pair |
| Run seeds 3301 to 3332 after the loop's presence round scored burnt colour | render; stop | user (`next`) | |

## Out of the session

| Id | What | Serves | State |
| --- | --- | --- | --- |
| plan31-general | `plan31-general.sh`: render_results.py --readout --head readout-catdog-identity --presence-head readout-catdog-presence-only --presence-weight 1 --seeds 3301..3332 --strengths 30 --push-cap 0.3 --window 0 50 | step 3.1 | done, 1,585 s; folded into the review file |
