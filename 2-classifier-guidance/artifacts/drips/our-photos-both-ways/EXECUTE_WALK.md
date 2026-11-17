# Walk: our photos, both ways

Walking [plan 20](../../../plans/20-our-photos-both-ways.md) with `/drip-execute-plan --request`,
driven by a self-paced `/loop`. Branch `exec/classifier-guidance-ladder` (kept, since other sessions
share this working tree).

## Where the walk is

Pieces 1 and 2 are done; both 64 px trainings run at about 0.7 s per step and land around 17:30
UTC. Piece 3 runs as one chain on `mscluster108` since 07:42 UTC (separate-label classifier, 512
product draws on both cards, pair-aware classifier), landing around 11:05 UTC. A waiter on `mscluster108` then draws
the released model's sheet on GPU 0 (about 40 minutes). The 64 px separate-label classifier trains on
`mscluster107` GPU 1 since 08:11 UTC. Next tick: the 64 px product sampler and piece 5's chain.

## The stop point

**Bar.** In at least one arm, on seeds 1 to 16, the pair-aware push gives two separate animals on at
least 4 more of 16 seeds than the separate-label push, with pictures no more damaged than the plain
product's; judged by the reader from full-resolution tiles.

**Time box.** 4 sittings, about 4.5 hours of the reader's time, plus the training wall clock.

steps at the open: 6 (depth full)

| Sitting | Started | Ended |
|---|---|---|
| 1 | 2026-11-17 09:24 SAST | |

## The toy

Seed 1, and one image per label, picked in piece 1.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| which images | the v2 pool, 200, 200, 150 | v2 COCO only · the old pool | you, by defaults | the request named 550 |
| pair-aware negatives | product draws plus single-animal images | product draws only | you, by defaults | SD3.5 renders on both sides |
| both-animal label in the all-images model | cat or dog at random per draw | a third label · null only | you, by defaults | the toy's both picture carried both labels |
| running their trainers | a launcher with one-process stand-ins | a new environment | you, by defaults | `co3` untouched |
| classifier start | fine-tuned from the released classifier | from scratch | you, by defaults | 550 images |
| where GPU work runs | free cards on 110 to 112 (`co3_bw`), then 106, 107, 108 (`co3`), then `bigbatch` | one job on 108 | you | "try blackwell, 106, 107, 108 on unused devices first before trying bigbatch" |
| pacing | a self-paced `/loop` | walk round by round | you | "next in a /loop" |
| launch path | scripts copied to `/datasets`, shared-device SSH launch | `cluster.sh run` | Claude | `cluster.sh run` refuses while other sessions hold uncommitted tracked edits |

## Ticks

- 1.1 the image set: 200, 200, 150 (COCO 100, 100, 38; SD3.5 100, 100, 112). By eye at sheet
  scale, `both_015` and `both_027` show only the dog, and `both_000` and `both_017` are collages;
  the reader's call whether to strike them. Test 2.1 open for the reader.
- 1.2 the trainings: timing 300 steps in 278 s on a Blackwell; 50,000 steps, about 12.9 h each.
- 1.3 smoke on `mscluster108`: 20 iterations and 2 draws at 25 steps ran end to end; the full
  separate-label run trains at 1.36 s per iteration at batch 32 (37 GB).
- 1.4 smoke of `sheet_256.py` on `mscluster108` GPU 1: 6 tiles at 25 steps in 26 s; at respaced
  step 12 the pair-aware push is 0.76 against the product's own step of 93.8. The first smoke run
  exposed an empty held-out pile in `train-pair`'s scoring, now skipped.

## Out of the session

| Id | Kind | What it serves | State |
|---|---|---|---|
| W&B `wubhxhse` | run, `mscluster111` GPU 0, PID 283386, log `classifier_guidance_photos/logs/train_64_no_both.log` | 1.2, the 64 px model without both-animal images | running since 07:35 UTC |
| chain PID 219860 | `piece3_256.sh` on `mscluster108` GPUs 0 and 1, log `classifier_guidance_photos/logs/piece3_256.log` | 1.3, the 256 px classifiers and the product's draws | running since 07:42 UTC |
| waiter PID 222024 | `piece4_256.sh` on `mscluster108`, log `classifier_guidance_photos/logs/piece4_256.log` | 1.4, draws the 256 px sheet when piece 3 prints `PIECE3 DONE` | waiting since 08:10 UTC |
| PID 710963 | `classifiers.py train-separate --size 64` on `mscluster107` GPU 1, log `classifier_guidance_photos/logs/separate_64.log` | 1.5, the 64 px separate-label classifier | running since 08:11 UTC |
| W&B `pnwlwh0f` | run, `mscluster112` GPU 0, PID 522361, log `classifier_guidance_photos/logs/train_64_all.log` | 1.2, the 64 px model on all images | running since 07:35 UTC |

## Closing pass owed

All of it.
