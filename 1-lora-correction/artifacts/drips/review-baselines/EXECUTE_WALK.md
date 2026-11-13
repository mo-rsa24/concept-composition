# Walk map: CO3, ToMe and SuperDiff baselines

**Plan:** [the baselines plan](../../../plans/the-baselines-the-reviewers-asked-for.md) · mode `--request --depth objective`
**On:** closed 2026-11-11 14:20; instructions 1 and 2 are yours

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
| Tuning cells | 16 (seeds 9 to 12) | 32 · 16 | you, 08:40 | halves the grid's cost; separates gaps of about 35 points instead of 25 |
| SuperDiff knob | step count × κ clamp | mixing rule (not in our code) | you, 08:40, with the 16-cell change | our SuperDiff has only AND |
| Tuning grid | none; published defaults only | 32-cell grid · 16-cell grid · none | you, 08:50 | spend the compute on the evaluation, not on tuning |
| Shard order | leave it: R1 first, R3 last in each job | leave it · reorder and resubmit | you, 09:15 | finish times unchanged; restarting costs more than it buys |
| SuperDiff | out of every table and figure | mixed column · out | you, in the step 9 walk, 10:45 (`ef5494c9`) | it does not claim to fix hybrids |
| CO3 sharding | 2 old jobs plus 4 `NSHARDS=6` jobs | as launched · six shards | you, in the step 9 walk, 10:43 (`41f70c32`) | use the slots SuperDiff freed |
| Where piece 1 runs before 09:30 | `bigbatch` RTX 3090, sbatch shown first | bigbatch · wait for 111 | default you accepted | CO3 already ran there; no biggpu card free at 07:25 |

## Ticks

- Task 1, 08:25: CO3 default (two clear animals, by my call) and SuperDiff default (one hybrid) rendered lion and meerkat seed 9; ToMe dropped on an import error. Seeds 9 to 16 start from a seeded fp16 CPU draw, since the cache has no entry for them; identical bytes under torch 2.5.1 and 2.11.

- Task 2, 08:45: grids, rule and ToMe removal committed as `028d9d69`, before any tuning image.

- Protocol rule 3 rewritten to defaults only, 08:55; the grid launcher `1-lora-correction/scripts/review_baselines_grid.sh` stays committed and unused.

- Task 3, 14:10: CO3 complete, 1,104 R1 and 424 R3 tiles, counted from mscluster107; mscluster84's NFS view showed 35 missing that three other nodes see.

- Task 4, 14:20: jobs table rows 7 and 8, route step 8b and five findings rows.
- Sittings: 07:24 to 14:20, about 2 h 40 min of your time against a 90-minute box.

## Out in runs, threads and handoffs

| Id | What | Serves | State |
|---|---|---|---|
| Slurm 84569 | `sbatch --partition=bigbatch ... launchers/review_baselines_onecell.sh`, log `outputs/paper_review/logs/baselines_onecell_84569.log` | task 1 | submitted 07:40 on your yes |
| login-node nohup | `tome` conda environment build, log `outputs/paper_review/logs/tome_env_build.log` | task 1, ToMe | started 07:31; 30-minute limit 08:01 |

## Teaching lines for the close
| Slurm 84590, 84591 | `METHOD=co3 SHARD=0,1 NSHARDS=2 sbatch ... launchers/review_baselines_eval.sh`, logs `outputs/paper_review/logs/eval_eval-co3-*_8459?.log` | task 3, CO3 on R1 and R3 | submitted on your yes; 764 cells each, 15 to 21 h |
| Slurm 84592, 84593, 84595 | SuperDiff shards 0, 1, 3 | task 3, SuperDiff | cancelled 10:43 by the step 9 walk; SuperDiff is out |
| Slurm 84597 | SuperDiff shard 2 | task 3 | cancelled 10:43 by the step 9 walk |
| Slurm 84757, 84759, 84760, 84761 | `METHOD=co3 SHARD=2..5 NSHARDS=6`, launched by the step 9 walk | task 3, CO3 | running since 10:43 |
| mscluster85 nohup, pid 20830 | `GPU=0 nohup bash launchers/review_baselines_fig5.sh`, log `outputs/paper_review/logs/fig5_baselines.log` | task 3, R2 Figure 5 tiles for CO3 and SuperDiff | done 09:02, 8 of 8 (CO3 38 to 43 s, SuperDiff 84 to 87 s per image) |
| mscluster108 nohup, pids 127193 and 127483 | `GPU=0,1 CODE_COMMIT=cdd58a18 CELLS=logs/cells_co3_tail2way_{0,1}.txt nohup bash launchers/review_baselines_co3_tail.sh` | task 3, CO3's last 154 cells (54 R1, 100 R3), whole pairs from the end of the list | started 13:18 on your yes; 79 and 75 cells |
| mscluster108 nohup, pids 132805 and 133042 | `GPU=0,1 CODE_COMMIT=5345d3d0 CELLS=logs/cells_langevin_extra_{0,1}.txt nohup bash launchers/review_langevin_cells.sh` | PoE + Langevin tiles for the best- and worst-cases grids, 16 cells outside the subset, into `langevin_extra/` | started 14:54 on your yes |
| Slurm 85115, 85116, 85117 | CLIP and ImageReward, BLIP-VQA, and the animal count on `rows_co3.csv`, the step 9 walk's own scripts | rating CO3 for R1 and R3 | submitted 16:05 on your yes |
| mscluster112, 107 (cards 0 and 1), 108 card 1, nohup | `review_langevin_cells_v3.sh` with `LANGEVIN_TILES=langevin_subset`, 12, 3, 3 and 3 of the subset's last 25 cells | PoE + Langevin subset, alongside mscluster111 | started 16:14 on your yes; bigbatch helpers 85118 to 85120 cancelled while pending |
| Slurm 85157 | the animal count on CO3's 264 R1 animal cells under run name `r1_co3` (`rows_co3_r1animals.csv`) | CO3's Comp. column | done 16:44, `comp_co3.csv` 424 rows |
| table builder on mscluster112 | `bin/review_build_tables.py` with CO3 as `rows_batchco3.csv`, `clip_ir_batchco3.csv`, `blip_batchco3.csv` | R1 and R3 with CO3's row | done 16:47; old tables kept as `*.before-co3.csv` |
| Slurm 85188, 85189, 85190 | CLIP and ImageReward, BLIP-VQA and the animal count on `rows_batchlv.csv` (288 PoE + Langevin cells) | PoE + Langevin's rows in R1 and R3 | done 19:14; tables rebuilt and copied to the R1 and R3 slot files |
