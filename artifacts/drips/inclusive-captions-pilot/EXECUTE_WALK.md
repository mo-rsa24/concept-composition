# Walk: the inclusive-captions pilot

Walking [the pilot](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md)
with `/drip-execute-plan --request`. Branch `exec/inclusive-captions-pilot`.

## Where the walk is

Piece 2 of 6, the data build. Task 2.1 done, task 2.2 and 2.3 waiting on the cluster run below.

## The stop point

**Bar.** Run B's compose rate and run A's, each on 128 seeds, read against the review file's
three lines, judged by the user off the saved figure.

**Time box.** Three sittings, about 6 hours of the user's time.

| Sitting | Started | Ended |
|---|---|---|
| 1 | 2026-11-03 | open |

## The toy

Three photos, quoted by value through every piece: `cat_only/0000`, `dog_only/0000`, and
both-animal photo 0 off `both_sheet_00.png`. Sheet seeds 0 to 7, the same eight everywhere.

## Ticked this session

| Step | Evidence | Commit |
|---|---|---|
| 1.1 the criterion | six parts, each carrying a number; the band as a formula rather than a number | `4b394f1a`, amended by `5f3a1c06` |
| 7 approve the criterion | the user delegated the judgment ("you tell me"); two lines changed, veto offered and not taken | `5f3a1c06` |
| 2.1 captions into the manifest | `read_captions()` takes the shortest of each photo's five COCO captions; every row carries one | `f778597b` |

## Decisions

| Decision | Chosen | The options there were | Who chose | Why |
|---|---|---|---|---|
| where the plan lives | new scope `plans/11-do-inclusive-captions-help-poe-compose/` | that, or a plan inside scope 10 | the user, by `defaults` | it is its own question with its own verdict |
| W&B project | new `prime_lab/poe-repair-inclusive-captions` | that, or a group inside `poe-repair-animals-compose` | the user, by `defaults` | the runs share no axis with the animals-compose sweeps |
| which runs | A and B only, 128 seeds | that, or adding base SDXL and no-both as anchors | the user, by `defaults` | it is the whole experiment; C was already dropped |
| the trainer's base | this repo's own SDXL loader | that, or fetching the diffusers SDXL LoRA example | Claude, accepted | the loader already supplies both text encoders, the VAE and the UNet, so the training path matches the sampling path and needs no network on a compute node |
| the band | a formula fixed now, its number computed from the measured rates | that, or a flat 12.5 points fixed now | Claude, delegated by "you tell me", veto offered | 12.5 assumes rates near 0.5; this pair has composed near 0.1, where the band is 7.5, and a band twice as wide as the data warrants pushes a real effect into the inconclusive region |
| the word for B at or below A | "no support" | that, or "null" | Claude, same delegation | with a standard error of 6 points on the difference, a true gain of 6 can measure below zero, so "null" claims more than the data licenses |
| the averaging trigger | 5 of 8 sheet seeds, judged by eye | that, or a sharpness measure | Claude, flagged as the weakest line and left | Laplacian variance picked drawn texture over sharpness here once, and the detector called a dog in 64 of 64 single-cat renders; an eyeball read with an arbitrary threshold beats a precise measure of the wrong thing |
| the filler photos' caption | the shortest of each photo's five COCO captions | that, a single invented caption, or always blank | Claude, during 2.1 | an invented caption is a third condition A and B do not have, and blank would put the no-both run's rate at 26% against 10% |

## Repairs made to the plan

**The band and the no-support line**, swept through the plan file as well as the review file, so
no copy of the flat 12.5 bar survives. Verified with a grep.

## Out of the session

| What | Id | Serves | State |
|---|---|---|---|
| the pool rebuild with captions | background `bsxomuld7`, log `/home-mscluster/mmolefe/logs/inclusive-pool-captions-2026-11-03.log` | tasks 2.2 and 2.3 | in flight |

Launched as:

```bash
nohup /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
  scripts/census/inclusive_captions_pool.py \
  > /home-mscluster/mmolefe/logs/inclusive-pool-captions-2026-11-03.log 2>&1 &
```

## How to approach this next time

**Piece 1.** A criterion is three disjoint regions of one number, not a threshold. Fix the
formula, not the number, wherever the number depends on where the data lands.

**Piece 2.** An empty string is never a missing value when the field is a caption, because the
blank caption is a real training condition.

## Owed by the closing pass

- `verify-plan` and `xref-pillar` over the new scope, which the plan's group 0 leaves open until
  the end.
- The finding to `report/`, the figure filed with its card entry.
- `sync-plan-tree` from the scope root, and scope 11 entered into the root running order.
- The scope has no `diagram-prompts.md`; one is owed if any piece earns a picture.
