# Walk map: route step 9, score every image and build the tables

Plan: [route step 9](../../../plans/route-answer-the-iclr-reviews.md#9--score-every-image-and-build-the-tables), walked with `/drip-execute-plan --request --depth objective`.
Board: [the paper review board](../../../report/paper-review/board.md). Protocol: [the comparison protocol](../../../report/paper-review/protocol.md).

## Stop point

Bar: `python3 scripts/refresh_paper_slots.py --check` prints PASS, the paper compiles, and every R1 cell on disk has a row for all five scorers or a recorded skip reason.
Time box: 60 minutes of your time at three checkpoints (scorer checks, CO3-style table, paired differences); scoring GPU time runs in the background and does not count.
Steps at the open: 5 (pieces 1 to 5, depth objective).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-11 | |

Toy: `a cat and a dog` and one animal-with-object R1 prompt, seed 9, joint, PoE and ours where they exist. Scorer probe images: mug and wine glass, seed 11, joint render (both objects, step 4 call) and plain product render, from the step 4 census.

## The cut

1. Scorers installed and proved on the two small-object images; versions and settings into the protocol. Checkpoint 1.
2. The bar written into the protocol and committed before any score.
3. Score every complete folder into one long results file with rule 9's axis columns; missing cells listed, never imputed.
4. The CO3-style table and the rate table with intervals (pairs resampled, then seeds). Checkpoint 2.
5. The paired differences, the slots filled, `--check` PASS, the paper compiles. Checkpoint 3.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| step 7 gate | piece 1 picks the two small-object images and writes the one-page scorer table | run step 7 first | default accepted | step 7's combine map was never written; the table it gates is made here |
| scope today | R1 and early_steps as they land; baselines, SD 1.5 and R3 listed as owed | wait for every job | default accepted | wave 1 at about 10%, steps 8 and 10 never run |
| GenEval and T2I-CompBench | official repos in separate envs, 30-minute box each, skipped and recorded on failure | three scorers only | default accepted | no code for either in the repo |
| stop point | the bar above, 60 minutes | | default accepted | |

## Position

Piece 2 of 5 closed (bar committed `3c65c634`); piece 3 waits on `r1_ours`, which has not started.

## Ticks

| Piece | What | Evidence |
|---|---|---|
| 1 | four scorers installed and run on the mug and wine glass probe, seed 11: all four rank the joint render (both objects) above the product render (mug missing) | `artifacts/results/what-each-scorer-says-about-two-objects/` (three CSVs and the figure); jobs 84587, 84588 |
| 2 | scorer section and R1 bar written into the protocol before any evaluation score | commit `3c65c634` |

## Out of the session

| Id | Kind | Serves | State |
|---|---|---|---|
| r1-ours on 111 (forward) and 112 (backward) | render | R1 `ours` column | not running: blocked by the render script's share guard; relaunch needs `FIGURE_CANDIDATES_MAXUSED_GB=30`, left to the user |

## Owed

- Baseline rows (step 8), SD 1.5 rows (step 10), R3 adapter renders (`r3_ours`, not started at the open).
