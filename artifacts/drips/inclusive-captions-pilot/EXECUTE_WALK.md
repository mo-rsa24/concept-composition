# Walk: the inclusive-captions pilot

Walking [the pilot](../../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md)
with `/drip-execute-plan --request`. Branch `exec/inclusive-captions-pilot`.

## Where the walk is

Piece 5 is running again on the rebuilt pool: 550 photographs, 200 cat, 200 dog and 150
cat-and-dog, every one of them read by a judge that was itself validated against hand labels
first. Run A trains now on `mscluster106` device 1 and run B follows it on the same card, because
that is the only usable `biggpu` node left. Piece 6 starts when both reach step 2,000.

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


**A question that names the failure gets the failure back.** Asked "is each one an ordinary cat or
dog, with no features of the other species mixed into it", LLaVA answered no on 39 of 48 images,
including a clean cat beside a clean dog, where it said "the image shows a cat with a dog's head".
Its other three questions were right: the count said two on 44 of 48 and caught the three-animal
and single-animal frames. Wordings are measured against hand labels rather than chosen by how
careful they sound, which is what `inclusive_captions_judge_calibrate.py` does.

**Pushing to GitHub broke mid-session.** `origin` became an SSH URL this session has no key for,
and the HTTPS URL returns 403. Work continues by copying files straight to the cluster clone with
rsync. Everything is committed locally on `exec/can-a-classifier-supply-the-missing-term`.

**`mscluster107` went down at 09:48 on 2026-11-04 and took two jobs with it.** It stopped
answering ssh and `sinfo` moved it from the live list to `down*`. The calibration pass was 40 of
48 images in and wrote nothing, because it only wrote at the end; the generator had written 131
of its 175 images, and those survive because each is saved as it is drawn. Of `biggpu`'s seven
nodes, `107`, `108`, `110` and `112` are down, `109` is draining and `111` is the Blackwell `co3`
cannot drive, so `mscluster106` is the only usable node and everything queues on its one free
card. Any pass over hundreds of images writes a row at a time from here on.


**The pool as it was rebuilt.** 930 candidates judged, 763 kept.

| Set | Candidates | Kept | Into the pool |
|---|---|---|---|
| generated cat-and-dog | 306 | 243 | 112 |
| generated cat | 130 | 130 | 100 |
| generated dog | 130 | 128 | 100 |
| COCO cat-and-dog, floor 0.40 | 64 | 38 | 38 |
| COCO cat, floor 0.50 | 150 | 120 | 100 |
| COCO dog, floor 0.50 | 150 | 104 | 100 |

Nearly a third of the COCO dog photographs and a fifth of the cat ones fail a judge that only
asks whether the animal is a normal animal, clearly visible, and the only one there. That set had
never been audited, and a floor on box size was the only thing screening it.

**The quantity the pilot turns on went from 0.169 to 0.251.** Run B's cat-captioned draws now hold
a dog a quarter of the time against run A's zero, because the both-animal set grew from 91 to 150.
The effect the design predicts is larger on this pool than on the first one.

**A resumed W&B run id silently drops everything you log.** The relaunch reused
`inclusive-captions-run-A`, whose history was already at step 516 from the killed attempt, and
every new step logged was refused with "steps must be monotonically increasing". Each attempt now
takes a tag that names its checkpoints, its sheets and its W&B run together.

**The rebuilt pool made the renders worse, not better, and that rules the pool out as the whole
cause.** Run A at step 250 on the 550-photo pool draws washed-out, smeared, cartoon-like animals
with coloured fringing, where the same step on the cluttered first pool still drew photographs.
Two things could do that and they are separable: the learning rate of 1e-4 at rank 32 with
batch size 1 may simply be too strong for 250 steps, or the half of the pool that SD 3.5 drew is
teaching SDXL to look generated. The first is tested by two 250-step runs at 1e-5 and 3e-5 on the
same pool, which cost twenty minutes each against the three hours a full pair costs.

**The learning rate was the cause, not the generated half of the pool.** Same 550 photos, same 250
steps, 1e-5 instead of 1e-4: the animals come back with fur texture and photographic lighting on
seeds 0 to 3, where 1e-4 gave smeared pastel shapes. So training on SD 3.5 output is not what
flattened the renders, and the pool stands.

**Two separate kinds of strangeness in the watching sheet, and only one is the model.** Seeds 4
to 7 render as flat illustration, tiled pattern, a laptop and a picture frame at every checkpoint
including step 0, before any training. That is SDXL being asked for 512 square when it was
trained at 1024. The washed-out look on seeds 0 to 3 is the adapter. Judging anything off seeds 4
to 7 reads the sheet's resolution as a result.

**3e-5 is the rate, and it fixed more than the texture.** Three rates on the same 550 photos at
the same 250 steps. At 1e-4 the renders are smeared pastel shapes. At 1e-5 the texture returns
but four of the eight sheet seeds still draw tiled pattern, a laptop or a picture frame, which is
what the untrained model does at 512 square. At 3e-5 all eight seeds draw animals, the 512
artefacts are gone, and the joint prompt draws a cat beside a dog on six of the eight seeds. The
model has learnt what these photographs look like rather than only how they are lit.

**A shell chain written as one `bash -c` string lost its newline and silently dropped the second
run.** The two launcher lines arrived as one command, so run A was started with run B's whole
command line as trailing arguments it ignores, and run B would never have started. The launcher
takes four positional arguments and does not complain about a fifth. A chain of launcher calls
goes in a script file, never in an inline string.

**At 1,024 square the untrained baseline is clean, and the premise is visible without any
confound.** Run A's step-0 sheet, drawn before a single training step: all eight seeds are
photographic, the joint prompt puts a cat and a dog in frame on eight seeds of eight, and the
product of the cat and dog conditions puts one animal in frame on eight of eight, several of them
a single face carrying both species' markings. Every strange tile the 512-square sheets showed,
the tiled wallpaper, the laptops, the picture frames, is gone. The sheet is now measuring the
model rather than the resolution it was asked for.

**Run A at step 500 moves the product not at all, and costs the joint prompt something.** The
product of the cat and dog conditions puts one animal in frame on eight seeds of eight, the same
as step 0, most of them a single face carrying both species' markings. Meanwhile the joint prompt,
which drew one cat beside one dog on all eight seeds before training, now draws two cats on seeds
1 and 6 and a third animal on seeds 2 and 3. The control is learning that these photographs hold
two animals without holding which two, which is a result about the pool worth watching at step
2,000 and worth comparing against run B.

**Rank 16, and the memory it saves is 0.46 GB measured.** The pair restarted at rank 16 with alpha
matched to it, so the adapter's scale stays 1. The trainer reports 23,224,320 trainable parameters
against 46,448,640, and the process holds 29.38 GB against 29.84. Almost all of a 1,024-square run
is activations, which the rank does not touch, so halving it is a choice about the size of the
correction being learnt and never a way onto a smaller card. `bigbatch`'s 24 GB RTX 3090s remain
out of reach at either rank, and it had no idle node in any case.

**Both arms now launch from `scripts/inclusive_captions_both.sh`**, a file rather than an inline
string, after the inline form lost a newline and dropped run B.
