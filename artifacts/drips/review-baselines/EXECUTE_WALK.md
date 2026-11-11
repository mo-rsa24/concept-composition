# Walk map: CO3, ToMe and SuperDiff baselines

**Plan:** [the baselines plan](../../../plans/prze-review-baselines.md) · mode `--request --depth objective`
**On:** piece 2, grids and the winning rule

## Stop point

- **Bar:** every method has a frozen setting chosen by the written rule, and every evaluation job at both settings is running or finished with its count matching (1,104 for R1, 424 for R3), checked by a count on each launch node.
- **Time box:** 90 minutes of your time.
- **Steps at the open:** 5 pieces (depth objective).
- **Sittings:** 2026-11-11 07:24 to (open).

## Toy

Lion and meerkat at seed 9, SDXL, 1024 by 1024.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| CO3 noise | wrapper calls `sample_loop` with our latent | wrapper · own noise, mixed | default you accepted | keeps the column unmixed without editing the vendored code |
| ToMe noise | wrapper passes `latents=` (unused: ToMe dropped) | wrapper · generator only, mixed | default you accepted | same |
| ToMe | dropped | one more attempt with `huggingface_hub<0.26` · drop | you, 08:24 | `drop tome`, after it missed the 30-minute limit on an import error |
| Where piece 1 runs before 09:30 | `bigbatch` RTX 3090, sbatch shown first | bigbatch · wait for 111 | default you accepted | CO3 already ran there; no biggpu card free at 07:25 |

## Ticks

- Task 1, 08:25: CO3 default (two clear animals, by my call) and SuperDiff default (one hybrid) rendered lion and meerkat seed 9; ToMe dropped on an import error. Seeds 9 to 16 start from a seeded fp16 CPU draw, since the cache has no entry for them; identical bytes under torch 2.5.1 and 2.11.

## Out in runs, threads and handoffs

| Id | What | Serves | State |
|---|---|---|---|
| Slurm 84569 | `sbatch --partition=bigbatch ... launchers/review_baselines_onecell.sh`, log `outputs/paper_review/logs/baselines_onecell_84569.log` | task 1 | submitted 07:40 on your yes |
| login-node nohup | `tome` conda environment build, log `outputs/paper_review/logs/tome_env_build.log` | task 1, ToMe | started 07:31; 30-minute limit 08:01 |

## Teaching lines for the close
