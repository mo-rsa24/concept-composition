# Walk: the inclusive-captions pilot

Walking [the pilot](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md)
with `/drip-execute-plan --request`. Branch `exec/inclusive-captions-pilot`.

## Where the walk is

Piece 5 of 6 is running: run A trains on `mscluster106` device 1 and run B on `mscluster107`
device 1, 2,000 steps each, sheets rendering into their own W&B runs as each checkpoint lands.
Instruction 11, watching the four columns move, is next. Piece 6 starts when both reach step
2,000.

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
| 2.1 captions into the manifest | `read_captions()` takes the shortest of each photo's five COCO captions; `missing_caption: 0` of 582 | `f778597b`, `dae7c3c9` |
| 2.2 every path opens | `missing_file: 0` of 582 chosen photos | `dae7c3c9` |
| 3.1 the trainer | `trainer.py`, 220 lines on this repo's SDXL loader; 46,448,640 trainable parameters at rank 32 on the four attention projections | `075bbdf1` |
| 3.2 the smoke run | 20 finite losses, `step_00000` on disk before any training, exit 0 in 225s on `mscluster111` | `075bbdf1` |
| 4.1 the watching sheet | `sheets.py`, four columns from one shared init latent, nine UNet forwards a step batched into one | `4a54bd6c`, `a2fdae38` |
| 4.2 the sheet proved | rendered off run B's `step_00000`; columns 3 and 4 differ on every one of the eight seeds | `a2fdae38` |
| 5.1 run A launched | `mscluster106` device 1, Quadro RTX 8000, 38 GB in use, W&B `inclusive-captions-run-A` | `96a28ad8` |
| 5.2 run B launched | `mscluster107` device 1, same card, W&B `inclusive-captions-run-B` | `96a28ad8` |
| 2.3 the shares on the real photos | PASSED on the real manifest: dog-present share of run B's cat draws 0.169, every caption share within 0.002 of design, blank rates spread 0.000 | `dae7c3c9` |

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

## The design picture

[The pilot's high-level design](../../scenes/how-the-inclusive-captions-pilot-works/the-inclusive-captions-pilot-high-level-design-explainer.svg),
also published at https://claude.ai/artifact/Uhp9tJfqpgHYHnQCfVaDNc. Twelve boxes, the two runs
identical except for the two arrows out of the caption rule, three boxes dashed as not yet built.
Its Figure Catalog row is in the plan. It is filed under `artifacts/scenes/` rather than
`artifacts/results/` because `.gitignore:40` ignores `results/`, so a card filed there would exist
on one machine only.

## Repairs made to the plan

**The band and the no-support line**, swept through the plan file as well as the review file, so
no copy of the flat 12.5 bar survives. Verified with a grep.

**The pool script's wanted counts** were still the design's assumed 200 both-animal and 200
filler photos, so `counts.json` recorded a shortfall of 69 against a number nobody wants any
more. They now read the counts that exist after the audit, and `chosen_total` is 582: the 491
every run trains on plus the no-both run's 91 filler photos.

## Out of the session

| What | Id | Serves | State |
|---|---|---|---|
| the pool rebuild with captions | background `bsxomuld7` | tasks 2.2 and 2.3 | landed, folded in, exit 0 |

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

**Piece 3.** A smoke run on the first N rows of a manifest exercises whichever set sorts first.
Take the toy one per set in turn, and print the composition at startup so the log says which
branches the run could have hit.

## Owed by the closing pass

- `verify-plan` and `xref-pillar` over the new scope, which the plan's group 0 leaves open until
  the end.
- The finding to `report/`, the figure filed with its card entry.
- `sync-plan-tree` from the scope root, and scope 11 entered into the root running order.
- The scope has no `diagram-prompts.md`; one is owed if any piece earns a picture.

## Lessons

**A sheet rendered beside the trainer fits on one card.** Training at 1,024 square with rank 32
and gradient checkpointing holds about 30 GB, and a 512-square sampling process beside it brings
the total to 39 GB of the Quadro RTX 8000's 48. Both arms run as one process tree each, so the
sheets arrive while the loss curve is still moving rather than at the end.

**Pin the free card rather than waiting for a free node.** `mscluster106` device 0 was at 100%
under another user and device 1 was untouched. `biggpu`'s one-job-per-user limit is a Slurm
limit, and a `nohup` launch pinned with `CUDA_VISIBLE_DEVICES` is not subject to it.

**Size is not species.** The pool's only screen on the 400 single-animal photos is a 0.25 floor on
the animal's box as a fraction of the image's longer side. That floor was chosen to match the
compose scorer, and it measures how much of the frame the animal takes. It cannot tell a live cat
from a cat on a television, a statue or a plush toy, which is exactly what the both-animal audit
rejected 33 photos for. The single-animal sets now get the same 448-pixel audit pages, 16 a page,
through `scripts/census/audit_both_animal_photos.py --set cat_only`.

**A bad photo cannot make the two arms differ.** Both train on the identical 491 photographs in
the identical order from the identical seed. Pool noise lowers both conditions together and
shrinks the gap the caption rule produces; it cannot invent one.

**SD 3.5 medium fails the two-animal prompt the same way SDXL does.** Asked for "a photo of a cat
and a dog" at 1,024 square, 28 steps, guidance 4.5, the first three seeds gave one clean cat and
dog, one single animal with a cat's ears and whiskers on a dog's muzzle, and one frame holding
three animals where the middle one is a hybrid. The fidelity is far above COCO's and the
composition is no better, which is the project's own premise showing up in a second model family.
Every generated photo is screened before it enters the pool.

**The RTX 8000 is Turing, so bf16 is emulated.** One 1,024-square SD 3.5 image took over seven
minutes in bf16 and 20.6 seconds in fp16 on the same card. Turing has fp16 tensor cores and no
bf16 ones. Any generation script on `biggpu`'s 106, 107 and 108 uses fp16.

**SD 3.5 needs its own environment here.** `co3` and `co3_bw` both carry diffusers 0.29.2, which
predates the dual attention layers `stable-diffusion-3.5-medium` uses, so the transformer loads
with a shape mismatch. `~/envs-sd35` is a venv over `co3` with diffusers 0.32.2, the newest that
works with torch 2.5.1, and `huggingface-hub` held below 1.0.
