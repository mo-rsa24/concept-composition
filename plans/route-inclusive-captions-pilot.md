# 🎯 Does captioning both-animal photos under each animal's name help PoE compose?
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** a falsification criterion written before any run, then a figure of PoE
> compose rate (one cat and one dog, no hybrid, scored per class) for base SDXL and four LoRAs
> trained on the same COCO photos with different captions · **Time:** ~11 h to finish, ~14 h
> budget, one full day today plus a morning · **Stop point:** the figure exists and the
> hypothesis file's support, null or inconclusive line is ticked against it

Built to: route-file-template v2

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [The two lanes](#the-two-lanes)
- [Schedule](#schedule)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Later](#later)
- [Why this chain](#why-this-chain)
- [References](#references)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-know-so-far) ➡️

**Goal:** find out whether PoE's cat-and-dog failure on SDXL is partly made by what the training
captions teach each single-animal condition, so the paper can say whether co-occurrence in the
captions is one of its causes.

**Objective (the whole chain):** PoE compose rate on 128 shared seeds for LoRA A (one caption per
image) against LoRA B (both-animal photos also trained under each single name), with the
difference read against a criterion fixed beforehand.

**Outcome (the whole chain):** `HYPOTHESIS.md`, then a figure with compose rate per model per
generation strategy and renders beside it, and the ladder's rung 11b rewritten to carry this as
its second axis.

**Found on disk:**

- The [simple-distributions ladder](../artifacts/drips/simple-distributions-rebuild/PROMPTS.md)
  plans rung 11b, "curated cats and dogs, with the caption co-occurrence set by hand", and has no
  step file. Its design is in
  [what the evidence means for rung 11b](../artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b).
  That design trains from scratch and moves the *share* of both-animal photos. This experiment
  holds the photos and moves *which caption* they get.
- The ladder walk is at layer 2: layer 1 closed 2026-10-29 with its bar met
  ([execute walk](../artifacts/drips/simple-distributions-rebuild/EXECUTE_WALK.md)).
- Layer 5's toy fine-tune run: support for A∩B inherited from pretraining falls from 0.159 of
  the composed target to 0.001 within 30 fine-tune steps on the grid, and the samplers follow by
  step 100 ([figure](../artifacts/drips/simple-distributions-rebuild/figures/walk05-step7-finetune-decay.png)).
- COCO 2017 is on disk at `/datasets/mmolefe/coco2017/`, annotations under
  `annotations_trainval2017/annotations/`. `pycocotools` is not installed in `co3`, so the counts
  were read with plain `json`.
- No image-denoising LoRA trainer exists in the repo. `scripts/launch_sdlora.sh` and its siblings
  train residual-correction LoRAs on cached PoE trajectories, a different objective.
- HLDs in the register: "SDXL notebook in ten parts" (`~/PhD/SDXL`, laptop) and the
  "Simple distributions" scene in this repo. Neither draws this pipeline.

**Plan weight:** short, because the steps are known (subset, captions, train, render, score) and
the `--request` walk's open writes the plan file.

**Who you are for this route:** explain and defend it to a supervisor (1.5×). The AI lane writes
the trainer; you check the design, write the criterion and read the result.

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| COCO train2017: 3,906 cat-only, 4,177 dog-only, 208 both, 169 of those with exactly one cat and one dog | found | `json` count over `instances_train2017.json`, 2026-11-03 | steps 1, 3 |
| COCO val2017: 172 cat-only, 165 dog-only, 12 both, 9 with one of each | found | same count | no held-out real pairs; evaluation is by generation only |
| The project's PoE is `ε_uncond + gs·(ε_A + ε_B − 2ε_uncond)`; at gs 1 that equals the source's "corrected" `s_C + s_D − s_0` | found | `poe_repair/training_cache.py:11` | step 1: only the raw arm is new |
| The project's interaction residual is `gs·(ε_J − ε_A − ε_B + ε_uncond)` | found | `poe_repair/training_cache.py:13` | later: the per-LoRA residual read |
| The compose scorer counts GroundingDINO instances ≥ 2 and cannot tell a cat and a dog from two dogs | found | memory "FK steering baseline"; [compose-score validation](../artifacts/results/can-we-trust-the-compose-score/README.md) | step 3, piece 6: score per class |
| The source's sampler gives LoRA B (not A) a 10% non-specific caption on every image, so A and B would differ in two things | found | [source design](../artifacts/ideas/inclusive-captions-axis/source-design.md), "Data" and its sampler | step 1: every arm gets the same empty-caption rate |
| LoRA C is not defined well enough to train | found | source design, the C row | step 1 settles or drops it |
| The cat-and-dog joint prompt in the existing eval references is wrong on 4 of 8 held-out seeds | found | memory "Eval references broken on half the seeds" | base SDXL's joint column is measured here, not borrowed |
| Rank-8 residual LoRA step time: 0.706 s on an RTX A6000, 0.236 s on Blackwell | found | `environment/hpc/throughput.md` lines 63 and 72 | training time is an `estimate` from these until piece 3's smoke prints one |
| One job per user on biggpu; long runs go outside Slurm with `nohup` | found | project `CLAUDE.md`, harvest rule | step 3, piece 4 |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have written | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 the design checked | own | the claim marks in `IDEA_MAP.md` | one both-animal photo and the four captions it can be trained under, one arrow each into p(x\|cat), p(x\|dog), p(x\|cat, dog), p(x) | A against B: how many times "a photo of a cat" meets a dog in 2,000 steps | none | none |
| 2 the criterion | own | `HYPOTHESIS.md` | a bar chart of compose rate for A and B with the inconclusive band shaded | 128 seeds, rate near 0.5: two standard errors ≈ 12.5 points | none | none |
| 3 the pilot | gloss | nothing by hand; you steer the walk and read every printed count | the pipeline from COCO to the figure, one box per piece | the caption counts each arm actually trained on | none | none |
| **Outcome** | | **the compose-rate figure with renders beside it** | | | | **`/pressure-test` on the verdict, `/drip-retro` passed** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

▶ **Recommended:** `drip-idea → drip-idea --hypothesis → drip-execute-plan --request ‖ background
hld-cadence, training, renders ‖ you in the waits: hld-diagram, drip-read --combine --lean →
pair-figure → drip-retro → pressure-test → scene register`

This is proposed (no curated flow fits). It is a light splice of the figure-that-answers-a-question
rungs in workflow 20: the data looked at by hand, then the criterion, then the measurement, then
the figure. It skips workflow 20's scope axes, design atlas and master plan, because this is one
pilot cell, not a scope.

<details><summary>Alternatives</summary>

`drip-idea → drip-idea --hypothesis → stop`: better if you want no GPU time until rung 11b's
from-scratch design is also settled.

`drip-idea → init-master-plan → populate-plans`: better if this grows into the full sweep (both
axes, several pairs, checkpoints over steps) and other plans start depending on it.

</details>

## The two lanes

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

| Stage | The AI lane builds | Your lane learns | Quiz | Confidence bar | You rebuild | Compare on |
|---|---|---|---|---|---|---|
| 1 and 2 | nothing; the walks are yours | the design's claims, the criterion | none | every claim marked | none | none |
| 3 | subset builder, caption sampler, trainer, renderer, per-class scorer, five trainings, the renders | during training: the HLD, then the caption-co-occurrence reading | none at this depth | you can say what each printed count means | none at this depth | the caption counts per arm against the design table |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Today is one full day, about six hours, cut into three 2-hour blocks. The renders run overnight;
the figure and the close happen the next morning.

| Block | Run | Checkpoint at the end of the block |
|---|---|---|
| block 1, 2 h | step 1, then step 2 | every claim in `IDEA_MAP.md` marked; `HYPOTHESIS.md` written |
| block 2, 2 h | step 3, pieces 1 to 3 | subset manifest with counts printed; 12 both-animal photos looked at by you; caption counts per arm printed; a 50-step smoke training that saved a LoRA |
| block 3, 2 h | step 3, piece 4 launched (AI) ‖ you: step 4, then step 5 in the training wait; then piece 5 launched | five trainings finished; renders started overnight |

| Block | Run (AI) | You, while it runs |
|---|---|---|
| block 3, first hour | five LoRA trainings, ~25 min each on an A6000 at 2,000 steps (`estimate` from 0.706 s/step), two cards in parallel | step 4, the HLD (15 min) · step 5, the reading (45 min) |
| overnight | renders: 6 models × 5 strategies × 128 seeds (`estimate` ~2 to 4 h) | nothing |

| Morning block, 2 h | Run | Checkpoint |
|---|---|---|
| | step 3, piece 6; step 6; step 7; step 8; step 9 | figure filed, criterion ticked, retro done, ladder rewritten, page registered |

**Totals.** Completion ~11 h of your time (estimate: step 1 ~1.5 h, step 2 ~0.75 h, step 3 six
pieces × 30 min × 1.5 = 4.5 h, steps 4 and 5 ~1 h, steps 6 to 9 ~2 h). Budget ~14 h at 1.5× on
the teaching walks. AI ~5 to 7 h (training ~2 h, renders ~2 to 4 h, scoring ~0.5 h), of which
about 1 h overlaps your work and the rest runs overnight. The day's six hours reach the launched
renders; the morning closes it. The cut, if the day runs short: drop LoRA C (one fewer training
and render set) and render 64 seeds, which widens the inconclusive band to about 18 points.

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. 🧪 Check the design claim by claim

You ~90 min · AI ~15 min · Opus, high · moves: Achieve row 1

**Objective:** every claim the design rests on marked holds, needs a check, wrong as stated (with
its repair) or dead.
**Outcome:** `artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md` with the marks and a design
table the request walk can train from.
**Serves:** the design the criterion and the pilot are built on.

```text
/drip-idea artifacts/ideas/inclusive-captions-axis/source-design.md
Chain: a compose-rate figure for base SDXL and LoRAs A, B, C and no-both on the same COCO photos,
read against a criterion fixed first · this step delivers the checked design
Starts from: the source design file above; rung 11b's design in the section "What this means for
rung 11b" of artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md
Settled facts:
- COCO train2017: 3,906 cat-only, 4,177 dog-only, 208 both, 169 with exactly one of each;
  val2017 has 12 both (json count, plans/route-inclusive-captions-pilot.md, What we know)
- the project's PoE at gs 1 is already s_C + s_D - s_0 (poe_repair/training_cache.py:11); only
  the raw s_C + s_D arm is new, and it counts the prior twice
- the source gives LoRA B, not A, a 10% non-specific caption: two variables move between A and B
- layer 5's toy: fine-tuning erased inherited A-and-B support within 30 to 100 steps
  (artifacts/drips/simple-distributions-rebuild/figures/walk05-step7-finetune-decay.png)
- the compose scorer cannot tell a cat and a dog from two dogs (memory: FK steering baseline)
- rung 11b's design: hold the single-animal sets fixed, freeze the text encoder, label one
  animal / two animals / hybrid separately, report PoE and joint-prompt arms apart
Claims to walk, at least: (1) A against B differs in one variable only once every arm shares one
empty-caption rate; (2) LoRA C, defined or dropped; (3) 300/300/200 is reachable and the
single-animal sets stay identical across arms; (4) the no-both LoRA is rung 11b's share-zero
point, where the two axes meet; (5) the raw arm is worth its render cost; (6) 2,000 steps at rank
8 and 512 px is enough for LoRA A to show the layer-5 erosion; (7) the scorer reads per class.
When something is missing: a claim that needs a number not on disk gets "needs a check" and the
command that would print it, not a guess.
Outputs: IDEA_MAP.md in that folder with every claim marked, and a design table: arm, captions
per image group, empty-caption rate, images, steps, seeds.
Target: every claim marked; the design table has no cell reading "tbd" · judged by me
Proof: grep the map for unmarked claims prints nothing
Hand-off: the claim that the A-against-B difference tests goes to step 2 by its number
Controls: claim 1 example (the caption counts per arm over 2,000 steps); claim 6 prove
Side routes: claim 4 /drip-read --combine --lean "what training captions teach a single-concept
condition" (deferred to step 5 unless the claim cannot be marked without it)
```

- `added:` the COCO counts, the PoE identity, the two-variable catch, the layer-5 prediction, the
  scorer limit, rung 11b's design rules (from the files in What we know)
- `dropped:` the source's interactive dropout slider (why: the rates are a design-table cell);
  the 100 to 200 neither-animal background images (why: no claim needs them for A against B; in
  Later)
- [ ] **Gate:** LoRA C is defined or dropped, and the design table is complete
- **Leaves:** `artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md`

### 2. 📏 Write the falsification criterion

You ~45 min · AI ~10 min · Opus, high · moves: Achieve row 2

**Objective:** support, null and inconclusive written with numbers before any training.
**Outcome:** `artifacts/ideas/inclusive-captions-axis/HYPOTHESIS.md`.
**Serves:** the line the figure is read against.

```text
/drip-idea --hypothesis artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md claim <the claim
step 1 handed over>
Chain: the compose-rate figure read against a fixed criterion · this step delivers the criterion
Starts from: IDEA_MAP.md and its design table (step 1)
Settled facts: everything in step 1's list, plus the marks step 1 wrote
Comparison: PoE compose rate (one cat and one dog, no hybrid, per-class read) for LoRA B against
LoRA A, same 128 seeds, same sampler and guidance. Secondary: the joint-prompt rate for the same
pair, reported apart; base SDXL as the reference row
Counts: at a rate near 0.5 on 128 seeds per arm, two standard errors of the difference are about
12.5 points; on 64 seeds, about 18. Say which the bar uses and why
What would surprise: B's PoE rate unchanged while its joint-prompt rate moves (rung 11b's
surprise line, on this axis)
Outputs: HYPOTHESIS.md with comparison, support, null, inconclusive band, counts, surprise
Target: all six parts have a number · judged by me
Proof: the compile prints the file path and its six headed parts
Hand-off: the bar and the seed count go into step 3's prompt as its target
```

- `added:` the standard-error arithmetic (computed for this route), the surprise line (from the
  rung 11b design)
- [ ] **Gate:** the bar is fixed; nothing trains until it is
- **Leaves:** `HYPOTHESIS.md`

### 3. 🛠️ Build and run the pilot

You ~4.5 h over two blocks and a morning · AI ~5 to 7 h · Opus, high · moves: Achieve row 3 and
the Outcome

**Objective:** five LoRAs trained on one COCO subset with the caption counts printed per arm, and
six models rendered on 128 shared seeds under five strategies, scored per class.
**Outcome:** `artifacts/results/do-inclusive-captions-help-poe-compose/` with the scores CSV and
the renders; the plan file the open writes.
**Serves:** the figure.

```text
/drip-execute-plan --request Build and run the inclusive-captions pilot on SDXL from the design
table in artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md, judged against
artifacts/ideas/inclusive-captions-axis/HYPOTHESIS.md.
Chain: the compose-rate figure · this step delivers the trained LoRAs, the renders and the scores
Starts from: IDEA_MAP.md (design table), HYPOTHESIS.md (bar and seed count); COCO at
/datasets/mmolefe/coco2017/, annotations in annotations_trainval2017/annotations/
Settled facts:
- the COCO counts and the val shortage (plans/route-inclusive-captions-pilot.md, What we know)
- pycocotools is not in co3: read the annotations with json, or install it in a throwaway env
- PoE is eps_uncond + gs*(eps_A + eps_B - 2*eps_uncond) (poe_repair/training_cache.py:11);
  reuse the project's combine, add the raw arm beside it
- no image-denoising LoRA trainer exists here; launch_sdlora.sh trains a residual LoRA on
  cached trajectories, a different objective. Start from diffusers'
  examples/text_to_image/train_text_to_image_lora_sdxl.py with the caption picked per draw
- large outputs to /datasets only (environment/storage.md); disk guard on that mount
- biggpu: one job per user, so long runs go outside Slurm with nohup; bigbatch caps 6 jobs;
  exclude the faulted bigbatch nodes (memory: bigbatch faulted nodes)
- every render uses one sampler and step count for all models (memory: dose vs per-epoch mismatch)
- score per class and label hybrid apart; the instance-count scorer alone cannot tell a cat and
  a dog from two dogs
Pieces, in order:
1. subset manifest: image ids per group, identical single-animal sets in every arm, counts
   printed; a contact sheet of 12 both-animal photos for me to look at by hand
2. caption sampler per arm, with the counts each caption is drawn per image group over the step
   budget printed per arm before training
3. trainer, 50-step smoke run that saves a LoRA and prints its step time
4. five trainings (A, B, C or none, no-both, plus the step-time line), launched outside Slurm,
   harvested with pgrep and the log tails
5. renderer: base SDXL and each LoRA under cat, dog, joint, PoE raw, PoE corrected, 128 seeds
6. per-class scorer, the scores CSV with arm, caption rates, steps, seed beside every row
When something is missing: a node or card that is not free, ask; a training that diverges, stop
it and show the loss tail before relaunching
Outputs: the results folder above with scores.csv, renders/, manifest.json, captions-per-arm.txt
Target: <the bar from HYPOTHESIS.md> · every arm's caption counts match the design table exactly
· judged by the printed counts and by me
Progress: each piece prints its counts or its smoke line before the next starts
Proof: the scorer prints compose rate per model per strategy with its count out of 128
Route update: add the measured step time and render time to What we know, and edit the schedule
Hand-off: scores.csv and renders/ for step 6
Close: rewrite rung 11b's row in artifacts/drips/simple-distributions-rebuild/PROMPTS.md and the
rung 11b section of the evidence note so 11b carries two axes, share and caption assignment,
with this run as the caption axis on SDXL; show me the diff before writing
Controls: piece 1 example (the 12 photos); piece 2 example (counts per arm); piece 3 plumb (the
diffusers trainer, run alone); piece 3 draw (once COCO, sampler, trainer and LoRA are named);
piece 4 run; piece 6 prove (the per-class read against two-dogs renders)
Side routes: piece 3 /plumb train_text_to_image_lora_sdxl.py, hand back: which lines pick the
caption; piece 6 /pair-figure (step 6)
Background: after piece 3's draw, dispatch hld-cadence over this walk's dispatch.md, system
"inclusive-captions pilot", see_also the "SDXL notebook in ten parts" scene for the U-Net box
```

- `added:` the PoE identity and code path, the storage, biggpu and bigbatch rules, the sampler
  identity, the per-class scorer, the ladder write-back (from the files in What we know and the
  project `CLAUDE.md`)
- `dropped:` checkpoints at 250, 500 and 1,000 steps (why: one checkpoint keeps the pilot to a
  day; in Later); 1024 px (why: the source's own pilot choice is 512); bounding-box overlap
  analysis (why: 208 photos is too few to split by overlap; in Later)
- [ ] **Gate:** piece 2's counts match the design table before piece 4 launches
- **Leaves:** `artifacts/results/do-inclusive-captions-help-poe-compose/scores.csv`, `renders/`
- **Learning path for LoRA** (gloss: read rung 2 only), from `journey_lookup.py --path`:

  | # | | Open | Textbook | Watch |
  |---|---|---|---|---|
  | 1 | ⬜ | [Standing up the toy substrate](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/00-preflight.md) | | |
  | 2 | ⬜ | [The low-rank update, B times A](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md) | | |
  | 8 | ⬜ | [Attaching the LoRA to attn2 only](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/09-lora-attn2-only.md) | | |

  Rungs 3 to 7 (U-Net blocks, self-attention, two encoders) are link only at this depth. Empty
  Watch cells are listed for a `video-cadence` sweep.

- **Learning path for classifier-free guidance** (own, because the empty-caption rate is the
  unconditional branch every PoE arm subtracts):

  | # | | Open | Textbook | Watch |
  |---|---|---|---|---|
  | 1 | ⬜ | [Classifier-free guidance as base plus omega times a difference](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md) | | |

### 4. 🗺️ Draw the pipeline (in the training wait)

You ~15 min · AI ~15 min · Sonnet, medium · moves: Achieve row 3

**Objective:** one page showing COCO subset → caption sampler per arm → LoRA trainer → five LoRAs
→ renderer (five strategies) → per-class scorer → figure.
**Outcome:** a published HLD and its SVG under `artifacts/scenes/`.
**Serves:** what you show when defending the design.

```text
/hld-diagram plans/<the plan file step 3's open wrote> draw the inclusive-captions pilot, one box
per piece; mark the caption sampler as the only box that differs between arms; link the U-Net box
to the "SDXL notebook in ten parts" scene with see_also rather than redrawing it
```

- [ ] **Gate:** none; step 3 runs on
- **Leaves:** the HLD URL, added to What we know

### 5. 📖 Read what captions teach a single-concept condition (in the training wait)

You ~45 min · AI ~20 min · Sonnet, medium · moves: Achieve row 1 (defend the design)

**Objective:** one paragraph per source on what it found about co-occurrence in training data
and composition, and whether it predicts A against B.
**Outcome:** `READ_MAP.md` from the walk.
**Serves:** the paragraph you give a supervisor on why this is new.

```text
/drip-read --combine --lean "what the training captions teach a single-concept condition"
Sources: the papers cited in
artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md (Okawa, Liang,
Jeong, Zarei) and the ladder note in ../ideas/ideas-that-tie-the-ladder-together.md where it
names them
Question each source answers: does it change which caption a multi-object image is trained
under, or only how many such images there are? The rung 11b note says nobody has measured the
share dial; say whether anyone has measured the caption-assignment dial
Box: 45 min · stop: every source has its one-paragraph answer
```

- `added:` the question that separates this axis from the share axis (from the rung 11b note)
- **Leaves:** `READ_MAP.md` in the walk's folder

### 6. 📊 The figure

You ~20 min · AI ~20 min · Sonnet, medium · moves: Outcome

**Objective:** compose rate per model per strategy, with renders of the same seeds beside it.
**Outcome:** the figure and its sidecar in the results folder.
**Serves:** the outcome itself.

```text
/pair-figure artifacts/results/do-inclusive-captions-help-poe-compose/scores.csv
y: compose rate (one cat and one dog, no hybrid, count out of 128); x: model (base, A, B, C,
no-both), one bar group per strategy (cat, dog, joint, PoE raw, PoE corrected); the hypothesis's
inconclusive band drawn on the PoE-corrected group. Beside it: the same 6 seeds under PoE
corrected for A and B, with hybrid, one-animal and two-of-one renders labelled on the image
```

- **Leaves:** the figure, its sidecar JSON, the card entry in the results folder README

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 7. 👀 Look at the both-animal photos (inside step 3, piece 1)

- [ ] **Do:** open the contact sheet piece 1 writes (its path prints in the walk). For each of
  the 12 photos, say whether both animals are clearly visible, whether one is mostly hidden, and
  whether anything else dominates (a person, a sofa, a car).
- **You should see:** most photos busier than the single-animal ones. The rung 11b note warns
  that busier both-animal photos move scene complexity together with the caption.
- **Done when:** you have said a count, such as "8 of 12 show both clearly", and the walk has
  recorded it.

### 8. ✅ Tick the criterion against the figure

- [ ] **Do:** open `HYPOTHESIS.md` and the figure from step 6 side by side. Read the PoE-corrected
  bars for A and B and their difference.
- **You should see:** the difference in points and whether it falls above the support line,
  inside the inconclusive band, or below the null line.
- **Done when:** one of the three lines is ticked, with the two rates out of 128 written beside
  it.

### 9. 🧠 Explain it back

- [ ] **Do:** run the retrospective the walk's close prints, then pressure-test the verdict:

```text
/drip-retro <the retro file step 3's close prints>
```

```text
/pressure-test artifacts/ideas/inclusive-captions-axis/HYPOTHESIS.md and the ticked verdict:
could the difference come from scene complexity of the 208 photos, from the empty-caption rate,
or from the sampler, rather than from caption assignment?
```

- **Done when:** you can say the verdict, the two rates and the strongest alternative explanation
  without notes.

### 10. 🗂️ Register the pages built

- [ ] **Do:** for the HLD from step 4 and the figure from step 6, write each page's section in the
  scene library, then run `scene-cards.py --apply`, `link-concepts.py --refresh-marks ... --apply`
  and `build-artifact-index.py --apply`, per `~/.claude/ARTIFACT_REUSE_FORMAT.md`.
- **Done when:** the register's `Linked from` column reads something other than **none** for
  both pages.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

None at this depth. Explaining it back is step 9. If you raise the depth later, the quiz goes
after step 2, prepared by `/quiz-prep-pulse` and built by `quiz-cadence`.

## Later

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

- **Checkpoints at 250, 500 and 1,000 steps:** shows whether LoRA A's erosion happens early, as
  layer 5's toy predicts.
- **The interaction residual per LoRA:** `gs·(ε_J − ε_A − ε_B + ε_uncond)` measured under each
  LoRA with the project's residual code. A smaller residual under B would be the paper's link.
- **The source's three explanations** (wider overlap, corrected meaning, friendlier score
  geometry): the follow-up the source names, once the pilot shows an effect.
- **Neither-animal background images** and **bounding-box overlap splits**: need more both-animal
  photos than COCO's 208.
- **LoRA internals beyond the B-times-A update:** rungs 3 to 7 of the LoRA path above.

## Why this chain

Navigation: ⬅️ [Previous](#later) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

<details><summary>Why these skills, in this order</summary>

The design arrived from another conversation with five problems already visible, so a claim walk
comes before anything is built. The criterion is written before training, per the project's
experiment conventions. A `--request` walk fits because the steps are known and short. The HLD
and the reading sit in the training wait, where they cost no extra wall time. `drip-read` runs in
`--combine --lean` because the question is one idea across four papers, and the evidence note
already summarised them. `drip-execute-plan` is the walk that builds the pilot. The ladder
continues on its own from layer 2, and this pilot joins it at 11b without blocking it.

Checks run when routing: the catalog was stale for `drip`, `hld-diagram` and `video-pulse`, and the
mode graph was stale for 10 skills, so their modes were read live with `mode_index.py --skill`.

</details>

## References

Navigation: ⬅️ [Previous](#why-this-chain) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [The source design](../artifacts/ideas/inclusive-captions-axis/source-design.md)
- [The rung 11b design](../artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b)
- [The ladder guide](../artifacts/drips/simple-distributions-rebuild/PROMPTS.md)
- [Layer 5, the overlap dial](../artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md)
- [What PoE composition is](../context/world/poe-composition.md)
- [Measured throughput](../environment/hpc/throughput.md)

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

Paste step 1's prompt.
