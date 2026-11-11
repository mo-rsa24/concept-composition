# PRZE-10: every review job priced, and wave 1 rendered

Retrospective of the walk over [the PRZE-10 plan](../../../plans/prze-10-wave-1-renders.md), 2026-11-11, step 5 of [the review route](../../../plans/route-answer-the-iclr-reviews.md). The walk map is [EXECUTE_WALK.md](../../drips/prze-10-wave-1/EXECUTE_WALK.md), and the jobs table is in [the compute plan](../../../report/paper-review/prze-10-compute.md#the-jobs).

## 1. The compute table

**The request.** One table pricing every GPU job of the review response: node, memory, images, hours with their basis, priority, and what each waits on.

**The problem.** The rate the request carried, 11 to 16 s per image, came from training-time evaluation passes, and two of the planned cards turned out to be busy with the user's own trainings.

**What was done.** The per-image rate was read from step 4's census logs (the same columns on the same Blackwells), and every card was re-read with `nvidia-smi` and `pgrep` before placing a job.

**Where the results live.** `report/paper-review/prze-10-compute.md`, sections "Where each card stands", "How long one image takes" and "The jobs" (13 rows); commit `56debc5e`.

**Before and after.** Joint 5.0 to 5.2 s, PoE 5.7 to 5.9 s and ours 12.1 to 14.3 s per image measured, against the 11 to 16 s assumed; wave 1 priced at 9.7 to 11.3 h on one Blackwell.

**Depends on.** Nothing earlier.

## 2. The launcher gaps

**The request.** Close what stopped the existing launchers from rendering the benchmark as the protocol says.

**The problem.** Both render scripts built the joint prompt as "{a} and {b}", which misdraws the 36 benchmark prompts that read "with"; the 45 prompts naming a frog or a mouse were refused by the subject check.

**What was done.** `--joint-template` and `--allow-ask-first` were added to `scripts/concept_pair_sheet.py`, `--allow-ask-first` to `scripts/figure_candidates.py`, and three launchers written beside the step 4 one: `scripts/review_wave1_renders.sh`, `scripts/review_wave1_r1.sh` and `scripts/review_wave1_r1_rev.sh`, with the prompt lists `scripts/review_benchmark_pairs.txt` and `scripts/review_held_out_pairs.txt`.

**Where the results live.** Commits `56debc5e`, `62554919` and `0c2babc1`; copies run from `/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers/`, because `scripts/cluster.sh run` refused to sync over three uncommitted trainer files.

**Before and after.** The first R1 joint sidecar reads "a horse with a glasses"; 240 prompts join with "and" and 36 with "with".

**Depends on.** Chunk 1, for which jobs needed a launcher.

## 3. The launches

**The request.** Each wave 1 job launched by the shared-device path after a yes to its exact command, with an 80% prediction first.

**The problem.** Mid-morning the plan changed twice: the CO3 selection run on mscluster108 had already finished, and the user paused three species-fidelity trainings to free mscluster107 and 111 for the review.

**What was done.** Jobs 1 to 3 on 112, R1 joint and PoE on 108's two cards, each R1 job helped by a second card rendering its prompt list backwards (`review_wave1_r1_rev.sh`), and R1 ours queued behind R3 on 112 and started backwards on 111. The trainings were stopped by PID after writing their command lines to `/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/RESUME-2026-11-11.md`.

**Where the results live.** The logs under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/` (`112-jobs-1-to-3.log`, `108-r1-joint.log`, `108-r1-poe.log`, `111-r1-poe-rev.log`, `107-r1-joint-rev.log`, `112-r1-ours.log`, `111-r1-ours-rev.log`).

**Before and after.** Predicted 12.1 to 14.3 s for the adapter; measured 6.7 to 7.0 s with the adapter on part of the run and 11.3 s on all 50 steps, because the showcase rate had counted a model load per 16 images. RTX 8000 rates 17 s joint and 22 s PoE, inside their predictions. Wave 1 ended at 11:14 instead of the first estimate's 17:00 to 18:30.

**Depends on.** Chunk 2's launchers.

## 4. The counts

**The request.** Count every job's outputs against the plan and report anything stuck.

**The problem.** Where a forward and a backward run met, both created one pair folder at once over NFS, and the losing node read the folder as both "File exists" and "No such file"; mscluster108 kept that stale view and undercounted R1 PoE by 4.

**What was done.** A background watch counted tiles every 5 minutes; counts were moved to mscluster112, whose view was current, and the four "missing" PoE tiles were found present from 112 and 84.

**Where the results live.** The walk map's ticks; `paper_review/` folders `figure5`, `early_steps`, `r3_ours`, `r1_joint`, `r1_poe`, `r1_ours`.

**Before and after.** 3 of 3, 848 of 848, 424 of 424, and 1,104 of 1,104 three times, at 11:14: the bar's PASS.

**Depends on.** Chunk 3.

## 5. After the bar: sheets, previews and the corrector subset

**The request.** Show Figure 5 and the tables as results came in, and start PoE + Langevin.

**The problem.** No scorer had run on the corrector or on CO3, and the corrector needed a labelled subset that the protocol had not named.

**What was done.** Figure 5 assembled with all five current columns at seed 10; Tables 1 and 2 previewed with the scoring session's builder on a copy of its files; the corrector subset written into protocol rule 6 (your choice, 288 pictures) and launched in two shards on 111 and 112 from the step 8 worktree; shard 0 restarted after a folder collision with a retrying `make_dir` (commit `6edc1897`).

**Where the results live.** `artifacts/results/four-pairs-one-adapter/figure5-seed10-five-columns.png`; `/datasets/mmolefe/poe_repair_min/outputs/paper_review/preview_tables/2026-11-11-1210/out/`; protocol rule 6 and its history row (commit `14990299`); `paper_review/langevin_subset/`.

**Before and after.** Table 1 preview: Comp. on the animals set 0.02 for PoE, 0.89 for the joint prompt, 0.97 for ours; the R1 bar (ours minus PoE in BLIP-VQA) reads SUPPORTS on all three sets. The corrector runs at 99 to 102 s a picture.

**Depends on.** Chunk 4's renders.

## The stop point

The bar was every wave 1 job running or finished with its count matching the plan; the box was 45 minutes of your time. The bar was met at 11:14; the walk ran from 06:22 to about 13:20, far past the box, by your choice to keep it open for the corrector, the previews and the counts. Way on taken: ship, with the scoring and the corrector handed on.

## Decisions

| Decision | Options | Chosen | Who | Why |
|---|---|---|---|---|
| Joint text for "with" prompts | a template option; the full prompt | `--joint-template "{a} with {b}"` | default you accepted | protocol rule 4: the prompt as written |
| Where wave 1 runs | 112 and 111; 112 alone; 112 and 108 | 112 and 108 | you | 111 held a training at the time |
| Stop the trainings | none; 107 only; all three | all three, resumable | you | "the priority is the review" |
| R14's two-concept methods | two-object only; split with a mark; one prompt | two-object only | you | only two-object prompts split into two concepts |
| The corrector's subset | animals plus hard; animals only; all of R3 | animals plus hard, 288 | you | Comp. is defined there, and the hard group is where a corrector would matter |
| Step 13's form | edit the paper; a markdown of amendments | a markdown of amendments | you | discuss each change before it enters the paper |

## Dead ends and not done

- Instructions 1 to 3 of the plan (read one row, look at the smoke tiles, judge PASS) stay open: they are yours, and no "done" was given.
- CO3 on R1 is at 1,045 of 1,104 (13:08) and unscored; CO3 on R3 has not started. Both belong to the baselines session.
- PoE + Langevin on the subset runs until about 16:40 and is unscored.
- The three paused trainings are not resumed; their commands are in `RESUME-2026-11-11.md`.
- `artifacts/results/what-co3-does-on-our-pairs/` has no card.
- Your prediction that CO3 will beat ours was not written down before CO3's scores; it was offered and not taken.

## The family table

| Who | Id | What | State |
|---|---|---|---|
| this walk | session `a4d17e2c-8fab-4c9c-b8c3-c17cc665b327` | PRZE-10 | closed |
| step 8 | session `afd3106a-798a-4f57-a01b-e7ee95e53231` | the Langevin corrector, Figure 5's corrector and CO3 tiles | closed |
| step 9 | session `47281f64-fc21-4b71-b0fd-3b968913d9b3` | scoring and the tables; note `routes/to-scoring-session.md` | open |
| baselines | session `ece09d30-6979-43e8-9fcc-84e4db760176` | CO3 on R1 and R3 | open |
| corrector shard 0 | mscluster111 pid 50558 | 144 cells | running |
| corrector shard 1 | mscluster112 pid 55397 | 144 cells | running |
| CO3 on R1 | bigbatch jobs eval-co3-0 to 5of6 | 1,104 cells | finishing |
