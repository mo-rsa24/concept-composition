# Walk map: step 7b, MNIST digits on a canvas

Mode: `--request`. Design: [route step 7b](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#7b--mnist-digits-on-a-canvas-how-good-must-the-checker-be).
Review file: [the bar](../../../plans/review/a2b-mnist-digits-on-a-canvas.md). Plan file: [plan 24](../../../plans/24-mnist-digits-on-a-canvas.md).

## Stop point

- bar: one of section 7b's four readings printed against the review file, and the compose count against checker accuracy plotted with 0.655 and 0.771 marked
- time box: sittings M1 to M3, about 6 hours of the user's time (720 min at 2×); if M3 ends without the five-version sheet, the walk stops and looks at why
- steps at the open: 10 (objectives 0 to 9, depth full)

## Sittings

| Sitting | Start (UTC) | End (UTC) |
|---|---|---|
| M1 | 2026-11-18 05:35 | |

## Position

On objective 2: write `scripts/mnist/canvas_unet.py` and time 200 training steps on the laptop and one probed 3090; stop for the user's yes.

## Ticks

| Objective | Evidence |
|---|---|
| 1, the short plan | `plans/24-mnist-digits-on-a-canvas.md` written 2026-11-18; 24 was free (highest in plans/, plans/partial/, artifacts/plans/ was 23) |
| 0, the bar in a review file | `plans/review/a2b-mnist-digits-on-a-canvas.md` written 2026-11-18; accepted by the user ("defaults") |

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| a: where checkpoints live | `data/mnist-canvas/` (gitignored) · a named path | user, default | environment/ names no laptop output folder; the brief's fallback |
| b: which share the bar reads if several fuse | 0.01 · 0.1 · at the plain-product stop | user, default | lowest share with two-digit canvases; share 0 is the control |
| outputs while /datasets is down | `/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/` · laptop only | user, default | /datasets returned Input/output error at 05:40 UTC; outputs are small |
| commits for the cluster | commit and push on `exec/classifier-guidance-ladder` · laptop only | user, default ("I trust you") | `cluster.sh run` needs a committed, pushed tree; replaces the brief's "commit nothing until done" |
| who calls the sheets | Claude's draft calls with veto after · park at each sheet | user, default | judge gate reads "validated against Claude's calls" until the user checks the 64 tiles |
| orchestration | Slurm `--dependency=afterok` chain plus a `/loop` in this session · `/loop` alone | Claude (how the work runs) | the user asked whether the work survives the laptop losing power |
| c: the case the four readings leave unnamed | at the five-version stop, before counts are read · a fifth reading now | user, default | not invented by the walk |

## Out of the session (runs, threads, hands)

| Id | Kind | Serves | State |
|---|---|---|---|

## Rules carried from the brief

Seeds 0 to 63 never trained on. One knob per sweep run. Stop before each launch with the exact command. The judge never trains on canvases and never pushes. Filing on. Commit nothing until the user says done.
