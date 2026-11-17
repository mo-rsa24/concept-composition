# Walk: our photos, both ways

Walking [plan 20](../../../plans/20-our-photos-both-ways.md) with `/drip-execute-plan --request`,
driven by a self-paced `/loop`. Branch `exec/classifier-guidance-ladder` (kept, since other sessions
share this working tree).

## Where the walk is

The released-model arm's product now uses the released classifier's cat and dog classes (the
reader's `a`). Pieces 1 and 2 are done; both 64 px trainings run at about 0.7 s per step and land around 17:30
UTC. Piece 3 runs as one chain on `mscluster108` since 07:42 UTC (separate-label classifier, 512
product draws on both cards, pair-aware classifier), landing around 11:05 UTC. A waiter on `mscluster108` then draws
the released model's sheet on GPU 0 (about 40 minutes). The 64 px separate-label classifier trains on
`mscluster107` GPU 1 since 08:11 UTC. Piece 5 waits on `mscluster111` and `112`: when each
training prints DONE (about 16:20 UTC), its own Blackwell draws 512 plain-product draws, trains that
model's pair-aware classifier and draws its sheet. Open question for the reader: the 64 px sheets
carry the separate-label push at scales 1 and 10, and the bar is read against the higher of the
two, as on the toy, unless the reader names one scale.

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
| the released-model arm's product | the released classifier's cat and dog classes at scale 10 | retrain our classifier · keep it and report the arm untestable | you (`a`) | our classifier's push draws food and objects; the released classifier's draws 4 of 4 animals, 2 of them hybrids |
| how the 64 px product is formed | cat + dog − null at weight 1, variance from the null call | a guidance weight of 3 · variance from the cat call | you (`options a`) | the plain product the bar names |
| who runs test 2.1 | Claude | the reader | you ("you do it") | |
| launch path | scripts copied to `/datasets`, shared-device SSH launch | `cluster.sh run` | Claude | `cluster.sh run` refuses while other sessions hold uncommitted tracked edits |

## Ticks

- 1.1 the image set: 200, 200, 150 (COCO 100, 100, 38; SD3.5 100, 100, 112). By eye at sheet
  scale, `both_015` and `both_027` show only the dog, and `both_000` and `both_017` are collages;
  the reader's call whether to strike them. Test 2.1 passed, run by Claude at the reader's word:
  every 64 px file within 0.20 grey levels of its 256 px crop shrunk, against 53.96 for an 8 px shift.
- 1.2 the trainings: timing 300 steps in 278 s on a Blackwell; 50,000 steps, about 12.9 h each.
- 1.3 smoke on `mscluster108`: 20 iterations and 2 draws at 25 steps ran end to end; the full
  separate-label run trains at 1.36 s per iteration at batch 32 (37 GB).
- 1.4 smoke of `sheet_256.py` on `mscluster108` GPU 1: 6 tiles at 25 steps in 26 s; at respaced
  step 12 the pair-aware push is 0.76 against the product's own step of 93.8. The first smoke run
  exposed an empty held-out pile in `train-pair`'s scoring, now skipped.
- 1.5 smoke of `sheet_64.py` on the all-images model's step-5,000 checkpoint, `mscluster108` GPU 1:
  8 draws in 6 s, 20-iteration 64 px classifiers, 10 tiles at 25 steps in 14 s; rows share their
  draw across columns, and the separate-label push at scale 10 already turns seed 1 into a cat.
- The two separate-label classifiers, held-out accuracy (cat present, dog present) at steps 0, 250,
  500, 900: 256 px (0.93, 0.95), (0.89, 0.93), (0.85, 0.78), (0.60, 0.51); 64 px (0.73, 0.84),
  (0.75, 0.80), (0.71, 0.76), (0.47, 0.58) · `classifiers/separate_{256,64}.pt.json`.
- The 256 px product draws run at about 27 s per draw per card, so the chain lands near 11:55 UTC
  and the 256 px sheet near 12:35 UTC.
- The bar on the 64 px sheets is read against the stronger separate-label scale, and each scale's
  comparison is reported beside it (the reader: "cant do both?").

- The released-model product is not about cats and dogs. The first 16 of its 512 draws (scale 10,
  `preview-product-256-draws.png`): by eye at grid scale about 5 hold a furry animal-like shape and
  none a clear cat or dog. Seeds 1 to 4 at scales 10, 30, 100 (`push-scale-check-256.png`,
  `mscluster107` GPU 1): scale 10 gives a dog and a bird among 4; 30 and 100 break into texture and
  give none. Our separate-label classifier's gradient does not carry cat and dog; asked the reader.
- The reader chose the released classifier's classes. Seeds 1 to 4 at scale 10
  (`imagenet-product-check-256.png`): 4 of 4 animals, draws 2 and 4 cat-dog hybrids and 1 and 3
  one dog-like animal, by eye at grid scale. The old chain (PID 219860), its draws and the piece 4
  waiter were stopped by PID at 10:17 UTC; the new chain draws, trains the pair-aware classifier and
  draws the sheet in one go.

## Out of the session

| Id | Kind | What it serves | State |
|---|---|---|---|
| W&B `wubhxhse` | run, `mscluster111` GPU 0, PID 283386, log `classifier_guidance_photos/logs/train_64_no_both.log` | 1.2, the 64 px model without both-animal images | running since 07:35 UTC |
| chain PID 228025 | `piece3_256.sh` on `mscluster108` GPUs 0 and 1, log `classifier_guidance_photos/logs/piece3_256.log` | 1.3 and 1.4, released-classifier product draws, pair-aware classifier, the 256 px sheet | running since 10:19 UTC, sheet about 14:05 UTC |
| waiter PID 285185 | `piece5_64.sh` ARM `no_both` on `mscluster111`, log `classifier_guidance_photos/logs/piece5_64_no_both.log` | 1.5, draws, pair-aware classifier and sheet for the model without both-animal images | waiting since 08:41 UTC |
| waiter PID 525583 | `piece5_64.sh` ARM `all` on `mscluster112`, log `classifier_guidance_photos/logs/piece5_64_all.log` | 1.5, the same for the model on all images | waiting since 08:41 UTC |
| W&B `pnwlwh0f` | run, `mscluster112` GPU 0, PID 522361, log `classifier_guidance_photos/logs/train_64_all.log` | 1.2, the 64 px model on all images | running since 07:35 UTC |

## Closing pass owed

All of it.
