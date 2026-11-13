# Walk map: route step 9, score every image and build the tables

Plan: [route step 9](../../../../../plans/route-answer-the-iclr-reviews.md#9--score-every-image-and-build-the-tables), walked with `/drip-execute-plan --request --depth objective`.
Board: [the paper review board](../../../../report/paper-review/board.md). Protocol: [the comparison protocol](../../../../report/paper-review/protocol.md).

## Stop point

Bar: `python3 1-lora-correction/scripts/refresh_paper_slots.py --check` prints PASS, the paper compiles, and every R1 cell on disk has a row for all five scorers or a recorded skip reason.
Time box: 60 minutes of your time at three checkpoints (scorer checks, CO3-style table, paired differences); scoring GPU time runs in the background and does not count.
Steps at the open: 5 (pieces 1 to 5, depth objective).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-11 05:20 | 2026-11-11 13:30 |

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

Closed 2026-11-11 13:30: Tables 1 and 2 filled for joint, PoE and ours, the paper builds (22 pages), `--check` PASS; the R1 bar reads SUPPORTS. The CO3 row is owed to step 8's renders. Retrospective: [the retrospective](../../../../../artifacts/notes/session-merges/2026-11-11-route-step-9-retro.md).

## Ticks

| Piece | What | Evidence |
|---|---|---|
| 1 | four scorers installed and run on the mug and wine glass probe, seed 11: all four rank the joint render (both objects) above the product render (mug missing) | `1-lora-correction/artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/` (three CSVs and the figure); jobs 84587, 84588 |
| 2 | scorer section and R1 bar written into the protocol before any evaluation score | commit `3c65c634` |
| 3 | 4,584 images scored on CLIP score, ImageReward and BLIP-VQA, animal prompts counted; 5 images unreadable from mscluster108 listed in `scores/rows_hidden.csv` | `scores/scores_long.csv` on `/datasets`; jobs 84640, 84737, 84788, 84779, 84790; BLIP on 108 |
| 4 | Tables 1 and 2 built; bar SUPPORTS on all three sets | `1-lora-correction/report/paper-review/tables-1-and-2-sidecar.json` |
| 5 | slots R1 and R3 filled, captions rewritten to match, paper builds | commit `56b968b6`; `--check` PASS; 22 pages, no slot too wide |

## Out of the session

| Id | Kind | Serves | State |
|---|---|---|---|
| r1-ours on 111 (forward) and 112 (backward) | render | R1 `ours` column | not running: blocked by the render script's share guard; relaunch needs `FIGURE_CANDIDATES_MAXUSED_GB=30`, left to the user |

## Parked

- Figure 15 (R12), the image grid behind Table 1: parked on the author's word, Tables 1 and 2 come first. Protocol rule 8 asks for a grid beside every table, so it is owed rather than dropped. The seed 9 preview with 16 of 18 tiles is `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/methods-grid-preview-seed9.png`; it needs the two object-row Ours tiles and a CO3 column.

## Owed

- Baseline rows (step 8), SD 1.5 rows (step 10), R3 adapter renders (`r3_ours`, not started at the open).
