# Walk map: PRZE-10, price every review job and launch wave 1

Plan: [prze-10-wave-1-renders.md](../../../plans/prze-10-wave-1-renders.md), started by `--request`.
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
| Where R1 ours runs | 111 after its training; 112 after jobs 1 to 3 | 112 | you (yes) | 112 measured at 6.7 s per image and frees about 09:05, before 111 |

## Ticks

- Piece 1: the jobs table, `report/paper-review/prze-10-compute.md#the-jobs`, 13 rows, 2026-11-11.
- Piece 2: `--joint-template` and `--allow-ask-first`, `scripts/review_wave1_renders.sh`, `scripts/review_wave1_r1.sh`, commits `56debc5e` and `62554919`; toy sidecar reads "a horse with a glasses".
- Piece 3: jobs 1 to 3 on 112 (pid 36785), R1 joint on 108 card 0 (pid 59533), R1 PoE on 108 card 1 (pid 59822), R1 ours queued on 112 (waiting loop pid 37731), each after your yes. Job 1 finished, 3 of 3 tiles.

## Changes

- added: re-probe of 111 and 108 (repair), piece 1. 111 card 0 and 108 card 0 were found busy.

## Out in a run, a thread or a hand

| Id | What | Serves | State |
|---|---|---|---|
| 112 pid 36785 | jobs 1 to 3, log `paper_review/logs/112-jobs-1-to-3.log` | pieces 3, 4 | running |
| 112 pid 37731 | waits on 36785, then R1 ours, log `paper_review/logs/112-r1-ours.log` | pieces 3, 4 | queued |
| 108 pid 59533 | R1 joint, log `paper_review/logs/108-r1-joint.log` | pieces 3, 4 | running |
| 108 pid 59822 | R1 PoE, log `paper_review/logs/108-r1-poe.log` | pieces 3, 4 | running |
