# Walk map: PRZE-10, price every review job and launch wave 1

Plan: [prze-10-wave-1-renders.md](../../../plans/price-the-review-jobs-and-launch-the-first-renders.md), started by `--request`.
On: piece 4, polling and counting; every wave 1 job is launched or queued.

## Stop point

- Bar: every wave 1 job running or finished, output count matching the table (6, 848, 424, 1,104, 1,104, 1,104), judged by you from the table and the logs.
- Time box: 45 minutes of your time.
- Steps at the open: 4 (depth full).
- Sitting 1: started 2026-11-11 06:22 SAST.

## Toy

"a horse with a glasses" at seed 9: its joint tile's sidecar must record that text exactly.

## Decisions

| Decision | Options | Chosen | Who | Why |
|---|---|---|---|---|
| Where wave 1 runs | 112 and 111; 112 alone; 112 and 108 | 112 and 108 | you | 107 and 111 hold your trainings; 108 card 0 holds the CO3 selection run, so 108 takes R1 joint on card 1 |
| Joint text for "with" prompts | `--joint-template`; full prompt text | `--joint-template "{a} with {b}"` | default you accepted | rule 4: the prompt as written; 36 of 276 prompts read "with" |
| Order on 112 | Figure 5 first, then early-steps, R3, R1; R1 first | Figure 5 first | default you accepted | a 2 minute launch check, then the jobs Thursday's draft reads |
| Stop a training for cards | both on 107; one; none | none | you ("keep all") | the CO3 selection run had already finished, freeing both of 108's cards |
| How PoE and ours run R14's counting and spatial prompts | two-object only; split with a mark; one prompt as both | two-object only | you | only two-object prompts split into two concepts |
| Pause the three species-fidelity trainings for the review | stop all three; stop v58-08 only | all three, v58-06 after its next save | you (yes) | the review is the priority and every run resumes from its checkpoint |
| The corrector's labelled subset | R1 animals plus R3 hard; R1 animals only; all of R3 | R1 animals plus R3 hard, 288 cells | you | written into protocol rule 6 and its history before any corrector picture |
| Where R1 ours runs | 111 after its training; 112 after jobs 1 to 3 | 112 | you (yes) | 112 measured at 6.7 s per image and frees about 09:05, before 111 |

## Ticks

- Piece 1: the jobs table, `1-lora-correction/report/paper-review/prze-10-compute.md#the-jobs`, 13 rows, 2026-11-11.
- Piece 2: `--joint-template` and `--allow-ask-first`, `1-lora-correction/scripts/review_wave1_renders.sh`, `1-lora-correction/scripts/review_wave1_r1.sh`, commits `56debc5e` and `62554919`; toy sidecar reads "a horse with a glasses".
- Piece 3: jobs 1 to 3 on 112 (pid 36785), R1 joint on 108 card 0 (pid 59533), R1 PoE on 108 card 1 (pid 59822), R1 ours queued on 112 (waiting loop pid 37731), each after your yes. Job 1 finished, 3 of 3 tiles.

## Ticks, later in the sitting

- Wave 1 PASS at 11:14: Figure 5 3 of 3, early-steps 848 of 848, R3 ours 424 of 424, R1 joint, PoE and ours 1,104 of 1,104 each, counted on mscluster112.
- Preview Tables 1 and 2 (joint, PoE, ours) built at 12:10 from a copy of the score files, in `/datasets/mmolefe/poe_repair_min/outputs/paper_review/preview_tables/2026-11-11-1210/out/`; the R1 bar reads SUPPORTS on all three sets.

## Changes

- added: re-probe of 111 and 108 (repair), piece 1. 111 card 0 and 108 card 0 were found busy.

## Out in a run, a thread or a hand

| Id | What | Serves | State |
|---|---|---|---|
| 112 pid 36785 | jobs 1 to 3, log `paper_review/logs/112-jobs-1-to-3.log` | pieces 3, 4 | running |
| 112 pid 37731 | waits on 36785, then R1 ours, log `paper_review/logs/112-r1-ours.log` | pieces 3, 4 | queued |
| 108 pid 59533 | R1 joint, log `paper_review/logs/108-r1-joint.log` | pieces 3, 4 | running |
| 108 pid 59822 | R1 PoE, log `paper_review/logs/108-r1-poe.log` | pieces 3, 4 | running |
| 111 pid 43532 | R1 PoE backwards, log `paper_review/logs/111-r1-poe-rev.log` | pieces 3, 4 | running |
| 107 pid 204772 | R1 joint backwards on card 1, log `paper_review/logs/107-r1-joint-rev.log` | pieces 3, 4 | running |
| 107 waiting loop | stops `v58-06-bound-w40` (pid 97741) after its step 12,500 save; resume commands in `/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/RESUME-2026-11-11.md` | freeing 107 card 0 | waiting |
| 111 pid 50558 | PoE + Langevin subset, shard 0 of 2 (restarted after a folder collision), log `paper_review/logs/111-langevin-subset-0b.log` | rule 6's 288 cells | running, about 16:40 |
| 112 pid 55397 | PoE + Langevin subset, shard 1 of 2, log `paper_review/logs/112-langevin-subset-1.log` | rule 6's 288 cells | running, about 16:00 |
| watch b63zvk9q4 | tile counts every 5 min until a job finishes, fails or stalls | piece 4 | running |
