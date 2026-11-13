# Walk map: PRZE-2, prove the Langevin corrector and render the six-column figure

Plan: [prze-2-langevin-corrector.md](../../../plans/prove-the-langevin-corrector-for-the-six-column-figure.md), started by `--request --depth objective`.
On: closed 2026-11-11 on your `stop`; open lines are listed in the retrospective, `artifacts/notes/session-merges/2026-11-11-prze-2-langevin-corrector-retro.md`.

## Stop point

- Bar: the check prints `PASS: all four step sizes`, both drivers match the plain sampler at k = 0, and R6 is built from the frozen settings; judged by you from the table and the figure.
- Time box: 120 minutes of your time, one sitting.
- Steps at the open: 5 pieces (depth objective).
- Sitting 1: 2026-11-11 07:20 to about 10:55 SAST.

## Toy

N(-1, 0.5) × N(1, 0.8) = N(-0.2308, 0.3077), variance as the second number.

## Decisions

| Decision | Options | Chosen | Who | Why |
|---|---|---|---|---|
| Pass bar for the check | 3 Monte Carlo standard errors of the chain's own exact values; fixed 2% | 3 standard errors, at δ = 0.005, 0.02, 0.05, 0.1 | default you accepted | the chain's exact variance is V / (1 − δ/(2V)), not 0.3077 |
| Tuning grid and rule | the recommended grid and instance-count rule; your own | c ∈ {1, 3, 10} × k ∈ {10, 20}; HMC the same with 5 leapfrog steps; most of 32 cells with two or more animals and not destroyed, tie to lower k | default you accepted | rule 3 caps the grid at six settings |
| CO3's starting noise | its own from seed 10, marked mixed; patch it to take our latent | its own, marked mixed | default you accepted | rule 2 says official code runs unmodified |
| Stop point | the plan's bar, 120 minutes | as stated | default you accepted | |
| Bias column | keep; drop | keep | default you accepted | |
| Predictions before each test | ask you; I answer them | I answer them | you ("can't you do it for me") | |
| What HMC's k counts | UNet calls per level; iterations | iterations, L = 5, so 5k + 2 calls per level | you ("fix b") | |
| Which correctors the review shows | Langevin and unadjusted HMC; Langevin only | Langevin only | you ("fix langevin only") | half the GPU time; HMC stays built |
| Tuning grid for Langevin | three c values on 32 tuning cells; none | none, c = 3, k = 20 | you ("fix no grid and let's stick to what reviewer wants") | PrzE asked for the baseline, not a tuned grid |
| Where the corrector shows in the paper | appendix R6 alone; R6 with ours added; fold into Figure 5 | fold into Figure 5 (R2), R6 removed | you ("fix a") | R6 without ours duplicated Figure 5's rows and showed no contrast with our method |
| SuperDiff | keep; drop everywhere | drop everywhere | you ("we don't need superdiff because it does not claim to fix the issue") | the baselines session's protocol history row already records it |
| How to launch past the stale cluster checkout | a clean worktree on /datasets; /reconcile-machines first | the worktree | you ("run") | the cluster checkout is 71 commits behind with 19 untracked files those commits track |

## Ticks

- Piece 1 build: `1-lora-correction/scripts/check_langevin_two_gaussians.py`, answer key run 2026-11-11 07:28, exit 0, last line `PASS: all four step sizes`.

- Piece 1 test: you read the table in chat and said "done", 2026-11-11.
- Piece 2 build: `1-lora-correction/scripts/check_langevin_order.py`, 13 s, exit 0.
- Piece 2 test: you read the table and kept our order, 2026-11-11.
- Piece 3 build: drivers, composers and both identity checks, commits `945f48e5`, `5d358f1a`, `d0b582ed`; R6 layout draft `58063749`.

- Piece 5: R6 at `1-lora-correction/artifacts/results/against-other-methods/samplers-and-correctors/does-a-langevin-corrector-fix-the-product/five_columns.png` with `five_columns.json`; route rows added; board redrawn at step 8.

## Changes

- removed: the Langevin tuning grid and its contact sheet (you approved), piece 4.
- removed: the unadjusted HMC column (you approved), piece 5.
- added: one bigbatch job for the k = 0 identity checks (repair, piece 3); the review nodes 112, 108 and 111 were all busy.

## Out in a run, a thread or a hand

| Id | What | Serves | State |
|---|---|---|---|
| Slurm 84583 | identity job, worktree `/datasets/mmolefe/poe_repair_min/worktrees/prze2` at `d0b582ed`, log `paper_review/logs/prze2-identity-84583.out` | piece 3 | done: PASS, log read 2026-11-11 08:19; worktree kept for pieces 4 and 5 |
| mscluster107 card 0, nohup | R6 render, worktree at `379695a6`, log `paper_review/logs/prze2-r6-107.out` (Slurm 84596 cancelled: bigbatch cap of 6 held by the baselines session) | piece 5 | done 09:58: tiles 20, sheet exit 0 |


## How to approach this next time

- Piece 1: check a sampler against the stationary law of its own discretised chain, then report the gap to the target separately; the two answer different questions.
- Piece 2: compare orders on a toy whose every expert score is exact, so the only difference left is where the last correction falls.
