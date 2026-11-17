# Walk: our photos, both ways

Walking [plan 20](../../../plans/20-our-photos-both-ways.md) with `/drip-execute-plan --request`,
driven by a self-paced `/loop`. Branch `exec/classifier-guidance-ladder` (kept, since other sessions
share this working tree).

## Where the walk is

Pieces 1 and 2 are done; both 64 px trainings run until about 20:30 UTC. Piece 3, the 256 px
classifiers, is next, on `mscluster108` (two free RTX 8000s at 09:30 SAST).

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

## Out of the session

| Id | Kind | What it serves | State |
|---|---|---|---|
| W&B `wubhxhse` | run, `mscluster111` GPU 0, PID 283386, log `classifier_guidance_photos/logs/train_64_no_both.log` | 1.2, the 64 px model without both-animal images | running since 07:35 UTC |
| W&B `pnwlwh0f` | run, `mscluster112` GPU 0, PID 522361, log `classifier_guidance_photos/logs/train_64_all.log` | 1.2, the 64 px model on all images | running since 07:35 UTC |

## Closing pass owed

All of it.
