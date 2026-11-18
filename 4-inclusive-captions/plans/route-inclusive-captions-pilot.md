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

- The [simple-distributions ladder](../../3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/PROMPTS.md)
  plans rung 11b, "curated cats and dogs, with the caption co-occurrence set by hand", and has no
  step file. Its design is in
  [what the evidence means for rung 11b](../../3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b).
  That design trains from scratch and moves the *share* of both-animal photos. This experiment
  holds the photos and moves *which caption* they get.
- The ladder walk is at layer 2: layer 1 closed 2026-10-29 with its bar met
  ([execute walk](../../3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/REBUILD_WALK.md)).
- Layer 5's toy fine-tune run: support for A∩B inherited from pretraining falls from 0.159 of
  the composed target to 0.001 within 30 fine-tune steps on the grid, and the samplers follow by
  step 100 ([figure](../../3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/figures/walk05-step7-finetune-decay.png)).
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
| In the 169 one-cat-one-dog train2017 photos, the smaller animal's segmentation area is a median 4.7% of the frame; it is under 5% in 90 photos (53%), under 2% in 42 (25%) and under 1% in 21 (12%) | found | `area` summed per category over `instances_train2017.json`, divided by width×height, 2026-11-03 | steps 1 and 3: half of what the inclusive arm captions under a single name is a photo in which that animal is a few percent of the frame |
| The project's PoE is `ε_uncond + gs·(ε_A + ε_B − 2ε_uncond)`; at gs 1 that equals the source's "corrected" `s_C + s_D − s_0` | found | `poe_repair/training_cache.py:11` | step 1: only the raw arm is new |
| The project's interaction residual is `gs·(ε_J − ε_A − ε_B + ε_uncond)` | found | `poe_repair/training_cache.py:13` | later: the per-LoRA residual read |
| The compose scorer counts GroundingDINO instances ≥ 2 and cannot tell a cat and a dog from two dogs | found | memory "FK steering baseline"; [compose-score validation](../artifacts/results/can-we-trust-the-compose-score/README.md) | step 3, piece 6: score per class |
| The source's sampler gives LoRA B (not A) a 10% non-specific caption on every image, so A and B would differ in two things | found | [source design](../artifacts/ideas/inclusive-captions-axis/source-design.md), "Data" and its sampler | step 1: every arm gets the same empty-caption rate |
| LoRA C is not defined well enough to train | found | source design, the C row | step 1 settles or drops it |
| The cat-and-dog joint prompt in the existing eval references is wrong on 4 of 8 held-out seeds | found | memory "Eval references broken on half the seeds" | base SDXL's joint column is measured here, not borrowed |
| Rank-8 residual LoRA step time: 0.706 s on an RTX A6000, 0.236 s on Blackwell | found | `environment/hpc/throughput.md` lines 63 and 72 | training time is an `estimate` from these until piece 3's smoke prints one |
| One job per user on biggpu; long runs go outside Slurm with `nohup` | found | project `CLAUDE.md`, harvest rule | step 3, piece 4 |
| The toy's trained caption model: plain product drew 0, 110, 103, 51 and 205 separate pairs of 300 at 0%, 10%, 30%, 60% and 100% relabelled; with 20 Langevin corrector steps 0, 300, 300, 300 and 300; with no two-shape picture under the single captions the corrector drew 0 of 300 in every model | found | `toy-caption-dial-corrector.json`, `toy-caption-corrector-no-single-captions.json`, in [the toy's co-occurrence card](../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md) | step 4: the prediction |
| The toy's exact-count sweep: with 50 or fewer relabelled pictures no training composed; at 200 one training of three did (153, 1 and 0 of 300); at 600 to 2,000 the counts ran from 17 to 252 across three trainings each | found | `toy-caption-threshold.json`, `toy-caption-threshold-600-1200-2000.json`, same card | step 4: three training seeds per setting |
| SDXL six-seed sheet: one animal in every product tile for plain SDXL, run A and run B; fused in run A, mostly a clean single animal in run B; run A at step 2,000 drew one animal in the product on 8 of 8 sheet seeds | found | [the pilot's results card](../artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/README.md), [the pilot walk](../artifacts/drips/inclusive-captions-pilot/EXECUTE_WALK.md) | step 3's scoring; step 10's pressure-test |
| Run B's "a photo of a cat" training draws held a dog 0.169 of the time on the real manifest | found | the pilot walk, task 2.3 | step 4: MNIST's "7" holds an 8 on 0.088 of draws at dial 0.3 |
| `/datasets` gave an I/O error on 2026-11-18; step 3's renders sit there | found | the brief for the MNIST route walk | step 3's scoring waits; step 4 does not |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have written | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 the design checked | own | the claim marks in `IDEA_MAP.md` | one both-animal photo and the four captions it can be trained under, one arrow each into p(x\|cat), p(x\|dog), p(x\|cat, dog), p(x) | A against B: how many times "a photo of a cat" meets a dog in 2,000 steps | none | none |
| 2 the criterion | own | `HYPOTHESIS.md` | a bar chart of compose rate for A and B with the inconclusive band shaded | 128 seeds, rate near 0.5: two standard errors ≈ 12.5 points | none | none |
| 3 the pilot | gloss | nothing by hand; you steer the walk and read every printed count | the pipeline from COCO to the figure, one box per piece | the caption counts each arm actually trained on | none | none |
| 4 the MNIST stage | gloss | nothing by hand; you steer the walk and read every printed count | one canvas holding a 7 and an 8, and the four labels it can be trained under | at dial 1, 1 in 6 "7" training draws holds a second digit | none | none |
| **Outcome** | | **the compose-rate figure with renders beside it** | | | | **`/pressure-test` on the verdict, `/drip-retro` passed** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

▶ **Recommended:** `drip-idea → drip-idea --hypothesis → drip-execute-plan --request ‖ background
hld-cadence, training, renders ‖ you in the waits: hld-diagram, drip-read --combine --lean →
pair-figure → drip-retro → pressure-test → scene register`

Beside step 3, waiting for nothing in it: step 4, the same question on MNIST digits (`drip-execute-plan --request`). Its verdict joins the chain at the pressure-test.

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
| 4 | canvas maker, pixel U-Net, 15 trainings, sampler with corrector, scorer | in the training wait: CFG trained on MNIST | none at this depth | you can say what each printed count means | none at this depth | label draws per kind against the design |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Today is one full day, about six hours, cut into three 2-hour blocks. The renders run overnight;
the figure and the close happen the next morning.

| Block | Run | Checkpoint at the end of the block |
|---|---|---|
| block 1, 2 h | step 1, then step 2 | every claim in `IDEA_MAP.md` marked; `HYPOTHESIS.md` written |
| block 2, 2 h | step 3, pieces 1 to 3 | subset manifest with counts printed; 12 both-animal photos looked at by you; caption counts per arm printed; a 50-step smoke training that saved a LoRA |
| block 3, 2 h | step 3, piece 4 launched (AI) ‖ you: step 5, then step 6 in the training wait; then piece 5 launched | five trainings finished; renders started overnight |

| Block | Run (AI) | You, while it runs |
|---|---|---|
| block 3, first hour | five LoRA trainings, ~25 min each on an A6000 at 2,000 steps (`estimate` from 0.706 s/step), two cards in parallel | step 5, the HLD (15 min) · step 6, the reading (45 min) |
| overnight | renders: 6 models × 5 strategies × 128 seeds (`estimate` ~2 to 4 h) | nothing |

| Morning block, 2 h | Run | Checkpoint |
|---|---|---|
| | step 3, piece 6; step 7; step 8; step 9; step 10 | figure filed, criterion ticked, retro done, ladder rewritten, page registered |

**Step 4, MNIST**, on the laptop's RTX 4070 (8 GB), any day, independent of `/datasets`:

| Sitting | You do | The AI runs meanwhile | Checkpoint at the end |
|---|---|---|---|
| MNIST 1, 1.5 h | pieces 1 and 2: the canvas maker, the network, a 200-step smoke run | nothing | the smoke run prints a measured step time, written into What we know |
| MNIST 2, 1.5 h | piece 3 launches the 15 trainings; then study [CFG trained on MNIST](/home/molef/goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/plans/08-cfg-trained-on-mnist.md) | 15 trainings, about 3 h (`estimate`: 30,000 steps at a guessed 25 ms) | all 15 trainings saved |
| MNIST 3, 1.5 h | pieces 4 to 6: the drawing, the judge's two checks with eye calls, the figure and the verdict | the drawing, about 1.5 h (`estimate`: about 108 million network passes at a guessed 20,000 a second; the corrected rows are 90% of it) | the two-panel figure filed and one verdict line ticked |

Step 4's totals: your time about 4.5 h (6 pieces × 30 min × 1.5), AI about 5 h (`estimate`
until the smoke run), most of it beside sitting 2 and between sittings.

**Totals.** Completion ~11 h of your time (estimate: step 1 ~1.5 h, step 2 ~0.75 h, step 3 six
pieces × 30 min × 1.5 = 4.5 h, steps 5 and 6 ~1 h, steps 7 to 10 ~2 h). Budget ~14 h at 1.5× on
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
**Outcome:** `4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md` with the marks and a design
table the request walk can train from.
**Serves:** the design the criterion and the pilot are built on.

```text
/drip-idea 4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/source-design.md
Chain: a compose-rate figure for base SDXL and LoRAs A, B, C and no-both on the same COCO photos,
read against a criterion fixed first · this step delivers the checked design
Starts from: the source design file above; rung 11b's design in the section "What this means for
rung 11b" of 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md
Settled facts:
- COCO train2017: 3,906 cat-only, 4,177 dog-only, 208 both, 169 with exactly one of each;
  val2017 has 12 both (json count, 4-inclusive-captions/plans/route-inclusive-captions-pilot.md, What we know)
- the project's PoE at gs 1 is already s_C + s_D - s_0 (poe_repair/training_cache.py:11); only
  the raw s_C + s_D arm is new, and it counts the prior twice
- the source gives LoRA B, not A, a 10% non-specific caption: two variables move between A and B
- layer 5's toy: fine-tuning erased inherited A-and-B support within 30 to 100 steps
  (3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/figures/walk05-step7-finetune-decay.png)
- the compose scorer cannot tell a cat and a dog from two dogs (memory: FK steering baseline)
- in 53% of the 169 both-animal photos the smaller animal is under 5% of the frame (area count,
  What we know): captioning those under that animal's name is what teaches its condition to put
  density on images where it is barely visible, which is the effect this axis is testing and the
  one a both-visible scorer discards
- rung 11b's design: hold the single-animal sets fixed, freeze the text encoder, label one
  animal / two animals / hybrid separately, report PoE and joint-prompt arms apart
Claims to walk, at least: (1) A against B differs in one variable only once every arm shares one
empty-caption rate; (2) LoRA C, defined or dropped; (3) 300/300/200 is reachable and the
single-animal sets stay identical across arms; (4) the no-both LoRA is rung 11b's share-zero
point, where the two axes meet; (5) the raw arm is worth its render cost; (6) 2,000 steps at rank
8 and 512 px is enough for LoRA A to show the layer-5 erosion; (7) the scorer reads per class; (8) a success
category that requires both animals plainly visible throws away the outcome inclusive captioning
most changes, a photo where one animal is a few percent of the frame, so the per-class read needs
a third outcome between both-visible and one-absent.
When something is missing: a claim that needs a number not on disk gets "needs a check" and the
command that would print it, not a guess.
Outputs: IDEA_MAP.md in that folder with every claim marked, and a design table: arm, captions
per image group, empty-caption rate, images, steps, seeds.
Target: every claim marked; the design table has no cell reading "tbd" · judged by me
Proof: grep the map for unmarked claims prints nothing
Hand-off: the claim that the A-against-B difference tests goes to step 2 by its number
Controls: claim 1 example (the caption counts per arm over 2,000 steps); claim 6 prove
Side routes: claim 4 /drip-read --combine --lean "what training captions teach a single-concept
condition" (deferred to step 6 unless the claim cannot be marked without it)
```

- `added:` the COCO counts, the PoE identity, the two-variable catch, the layer-5 prediction, the
  scorer limit, rung 11b's design rules (from the files in What we know)
- `dropped:` the source's interactive dropout slider (why: the rates are a design-table cell);
  the 100 to 200 neither-animal background images (why: no claim needs them for A against B; in
  Later)
- [ ] **Gate:** LoRA C is defined or dropped, and the design table is complete
- **Leaves:** `4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md`

### 2. 📏 Write the falsification criterion

You ~45 min · AI ~10 min · Opus, high · moves: Achieve row 2

**Objective:** support, null and inconclusive written with numbers before any training.
**Outcome:** `4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/HYPOTHESIS.md`.
**Serves:** the line the figure is read against.

```text
/drip-idea --hypothesis 4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md claim <the claim
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
table in 4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md, judged against
4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/HYPOTHESIS.md.
Chain: the compose-rate figure · this step delivers the trained LoRAs, the renders and the scores
Starts from: IDEA_MAP.md (design table), HYPOTHESIS.md (bar and seed count); COCO at
/datasets/mmolefe/coco2017/, annotations in annotations_trainval2017/annotations/
Settled facts:
- the COCO counts and the val shortage (4-inclusive-captions/plans/route-inclusive-captions-pilot.md, What we know)
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
   printed; each both-animal photo's per-animal area share recorded in the manifest, and the
   contact sheet of 12 both-animal photos chosen to span that range rather than at random, so
   what the inclusive caption is actually teaching is visible by hand
2. caption sampler per arm, with the counts each caption is drawn per image group over the step
   budget printed per arm before training
3. trainer, 50-step smoke run that saves a LoRA and prints its step time
4. five trainings (A, B, C or none, no-both, plus the step-time line), launched outside Slurm,
   harvested with pgrep and the log tails
5. renderer: base SDXL and each LoRA under cat, dog, joint, PoE raw, PoE corrected, 128 seeds
6. per-class scorer with three outcomes, not two: both animals plainly present; one plainly
   present and the other present but small or occluded; one absent. The middle outcome is scored
   on full-resolution tiles, never on a downscaled sheet, and the scores CSV carries arm, caption
   rates, steps, seed and all three counts beside every row. Reading only the top outcome would
   show the inclusive arm as worse precisely when it is working
When something is missing: a node or card that is not free, ask; a training that diverges, stop
it and show the loss tail before relaunching
Outputs: the results folder above with scores.csv, renders/, manifest.json, captions-per-arm.txt
Target: <the bar from HYPOTHESIS.md> · every arm's caption counts match the design table exactly
· judged by the printed counts and by me
Progress: each piece prints its counts or its smoke line before the next starts
Proof: the scorer prints compose rate per model per strategy with its count out of 128
Route update: add the measured step time and render time to What we know, and edit the schedule
Hand-off: scores.csv and renders/ for step 7
Close: rewrite rung 11b's row in 3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/PROMPTS.md and the
rung 11b section of the evidence note so 11b carries two axes, share and caption assignment,
with this run as the caption axis on SDXL; show me the diff before writing
Controls: piece 1 example (the 12 photos); piece 2 example (counts per arm); piece 3 plumb (the
diffusers trainer, run alone); piece 3 draw (once COCO, sampler, trainer and LoRA are named);
piece 4 run; piece 6 prove (the per-class read against two-dogs renders)
Side routes: piece 3 /plumb train_text_to_image_lora_sdxl.py, hand back: which lines pick the
caption; piece 6 /pair-figure (step 7)
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

### 4. 🔢 Ask the same question on MNIST digits

You ~4.5 h over three sittings · AI ~5 h (`estimate`) · Opus, high · moves: Achieve row 4

**Objective:** find out whether training two-digit canvases under each single label makes the
product of "7" and "8" draw two separate digits, on a small diffusion model trained from scratch,
so that the labels control everything each digit's condition has learnt.
**Outcome:** the two-panel figure, `scores.csv` and a card in
`4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/`,
and one verdict line ticked.
**Serves:** the half of the question SDXL cannot answer, because SDXL learnt "a cat" from its own
training data before any caption rule touched it.
**Sees:** [the label-conditional U-Net](../artifacts/drips/route-inclusive-captions-pilot/figures/the-label-conditional-unet.png),
[one sampling step of the product](../artifacts/drips/route-inclusive-captions-pilot/figures/one-sampling-step-of-the-product.png)
· **Ticket:** none · **Source page:** none

This step runs beside step 3 and waits for nothing in it. It needs no file on `/datasets`. It runs on the laptop's RTX 4070; its code goes in
`poe_repair/experiments/inclusive_captions/mnist/` and its data in `data/mnist/`, kept out of git. Its
verdict and step 3's meet in step 10's pressure-test.

- **The canvases and the two-digit share:** a 32 × 64 canvas, real MNIST digits pasted in
  28-pixel boxes, the same canvas as the classifier-guidance route's MNIST sheet. 90% of training
  canvases hold one digit at a random place. 10% hold two digits, one per half, with 0 to 4 pixels
  of jitter and the order swapped half the time. Half of the two-digit canvases are 7 and 8; the
  other half are spread evenly over the other 39 seen pairs. Five pairs never share a canvas:
  0 & 3, 1 & 9, 3 & 4, 3 & 6 and 4 & 7 (`random.Random(0)` over the 45 pairs). Each held-out pair
  is read against its own dial-0 rate, never against 7 × 8, because the two differ in how often the
  pair was seen and in whether it was seen at all. Sheets:
  [the canvas and the pairs](../artifacts/drips/route-inclusive-captions-pilot/figures/the-canvas-and-the-pairs.png),
  [what the "7" label sees](../artifacts/drips/route-inclusive-captions-pilot/figures/what-the-7-label-sees.png).
- **The dial:** the toy caption model's rule with the blank rate held fixed. Every canvas is
  trained under the blank label on 10% of its draws, which is the null label the product
  subtracts. A one-digit canvas takes its digit's label on the other 90%. A two-digit canvas
  holding digits a and b splits the other 90% by weight: the joint label "a and b" weight 1, and
  "a" and "b" each weight equal to the dial. The dial is 0, 0.1, 0.3, 0.6 or 1, and it applies to
  every two-digit canvas, 7 and 8 or not. At dial 1 a 7-and-8 canvas is drawn 30% "7 and 8", 30%
  "7", 30% "8" and 10% blank. The joint label is trained at every setting, so "7 and 8" alone is
  measured everywhere. The five held-out joint labels are never trained. "7" training draws hold
  a second digit on 0%, 5.1%, 10.3%, 14.5% and 17.6% of draws across the five settings, and an 8
  on 0%, 4.3%, 8.8%, 12.4% and 15.0% (400,000 simulated draws,
  [the sheet script](../scripts/mnist_training_sheets.py) with half of two-digit canvases 7 and
  8). The toy's plain-product counts at the same settings (0, 110, 103, 51 and 205 of 300) and its
  corrector counts (0, 300, 300, 300, 300) are the prediction.
- **The model and its training seeds:** one label-conditional U-Net per training, in pixel space
  on the 1 × 32 × 64 canvas, with no VAE, so a fused digit can only come from the denoiser or the
  sampler. The label is looked up in a table of 56 entries (ten digits, 45 joint labels, the blank
  label) and added to the noise-level embedding. The blank label is the null branch the product
  subtracts, from the same network. Three training seeds per dial setting, 15 trainings in all;
  each seed sets the starting weights and the canvas draws. Every training runs the same step
  count, 30,000 steps at batch 128 (`estimate`); a 200-step smoke run prints the step time, and
  an overrun cuts steps equally for all 15, never per setting.
- **The measurements:** every trained model is sampled the toy's way: reverse diffusion in 200
  steps on the cosine schedule, guidance 3, the product formed as blank + 3 × ("7" − blank) +
  3 × ("8" − blank). The corrected rows add 20 Langevin steps at every noise level, each of size
  0.05 of that level's noise variance (`toy_caption_dial_corrector.py`). Eight rows per model,
  each on seeds 0 to 255, the same starting noise in every model and every row; training draws
  from its own generator. The rows: "7" alone, "8" alone, "4" alone; "7 and 8" as one label; the
  product of 7 and 8, plain and corrected; the product of 4 and 7 (a held-out pair), plain and
  corrected. That is 2,048 pictures per model and 30,720 over 15 models; a corrected picture
  costs about 21 times a plain one. A single-label tile is called damaged, wrong digit, right
  digit with a second, or right digit alone. A two-digit tile is called damaged, one digit,
  fused, two digits with a wrong identity, or two separate right digits. The other four held-out
  pairs (0 & 3, 1 & 9, 3 & 4, 3 & 6) are drawn only if 4 and 7 moves above its dial-0 rate.
  Sheets: [what each model draws and how tiles are called](../artifacts/drips/route-inclusive-captions-pilot/figures/what-each-model-draws-and-how-tiles-are-called.png),
  [the label-conditional U-Net](../artifacts/drips/route-inclusive-captions-pilot/figures/the-label-conditional-unet.png),
  [one sampling step of the product](../artifacts/drips/route-inclusive-captions-pilot/figures/one-sampling-step-of-the-product.png).
- **The leak read:** each single label is judged against its own training data, never against a
  flat share. The share of each label's training draws holding a single digit (400,000 simulated
  draws of this design) is, at dial 0, 0.1, 0.3, 0.6 and 1: "7" 100%, 94.9%, 89.7%, 85.5%,
  82.4%; "8" 100%, 94.7%, 88.4%, 84.2%, 81.4%; "4" 100%, 99.1%, 98.1%, 97.2%, 96.7%. A model that
  learnt its data exactly draws a lone 7 at 82.4% at dial 1, so a flat 90% would fail it by
  construction. The result figure has two panels with the dial across both. The top panel shows
  how often the product draws two separate right digits out of 256, for 7 × 8 and 4 × 7, plain
  and corrected, one dot per training and a line through their mean. The bottom panel shows how
  often "7", "8" and "4" alone draw their digit alone, one dot per training, beside each label's
  training line drawn dashed. A label leaks at a setting when its dots sit below its training
  line by more than two standard errors at 256 seeds (3.75 points near 90%); round 8 fixes the
  number. Pictures: [why not a flat 90%](../artifacts/drips/route-inclusive-captions-pilot/figures/simple-why-not-a-flat-90.png),
  [what a leak looks like](../artifacts/drips/route-inclusive-captions-pilot/figures/simple-what-a-leak-looks-like.png),
  [the result figure's layout](../artifacts/drips/route-inclusive-captions-pilot/figures/the-leak-read-layout.png).
- **The judge:** one judge shared with the classifier-guidance route's MNIST stage, so a "two
  separate right digits" means the same on both routes. It is that route's
  `2-classifier-guidance/scripts/mnist/judge_sheet.py`, imported at a pinned commit and never
  edited from here: a small digit classifier trained only on single MNIST training digits, which
  finds each ink patch on a tile (flood fill, ink above 40 of 255, patches under 25 pixels
  ignored), reads it, and calls the tile. Its prototype reads 98.2% of MNIST's 10,000 test digits
  right but calls a fused 7-and-8 "one digit", so it needs a fused test. Before any count is used,
  step 4 runs two checks on it: on 500 pasted canvases per call with a known answer it must be
  right on at least 95% of each call; on one 64-tile sheet per dial setting (plain product,
  training 1) it must agree with eye calls, made on enlarged tiles and open to the user's
  overrule, on at least 58 of 64. If the judge is not committed when step 4 opens, or fails a
  check, step 4 stops and asks. Picture:
  [how the judge calls a tile](../artifacts/drips/route-inclusive-captions-pilot/figures/simple-how-the-judge-calls-a-tile.png).
- **The bar**, written before anything runs and put in the scorer's source as named constants.
  The judge's two checks pass first. A dial setting counts only if "7" alone and "8" alone stay
  within 5 points of their own training line in all three trainings (5 points covers two standard
  errors at 256 seeds, at most 4.8 points): "7" must draw a lone 7 on at least 244, 231, 217, 207
  and 199 of 256 at dial 0, 0.1, 0.3, 0.6 and 1, and "8" a lone 8 on at least 244, 230, 214, 203
  and 196. The verdict reads the corrected 7 × 8 product's "two separate right digits".
  Supported: at some counting setting, at least 128 of 256 in all three trainings, while dial 0
  stays at 12 or fewer of 256 in all three. Null: no counting setting reaches 39 of 256 on the
  average of its three trainings. Inconclusive: anything else. Stop and show the counts if the
  plain product at dial 0 already draws more than 12 of 256; the fallback is a lower two-digit
  share or a pair whose strokes overlap more, chosen by the user. Two side readings are reported
  without deciding the verdict: the corrector is not needed if the plain product also reaches 128
  at the supported setting; composition reaches a pair never seen together if, at that setting,
  the corrected 4 × 7 product reaches at least 39 of 256 in all three trainings and at least three
  times its own dial-0 count. Pictures: [the bar as a ruler](../artifacts/drips/route-inclusive-captions-pilot/figures/the-bar-as-a-ruler.png),
  [the order the checks run](../artifacts/drips/route-inclusive-captions-pilot/figures/the-order-the-checks-run.png).
- **The prompt:**

```text
/drip-execute-plan --request Build and run the MNIST stage of the inclusive-captions route, which
is step 4 of 4-inclusive-captions/plans/route-inclusive-captions-pilot.md.

Background
The route asks whether a product of experts composes two concepts more often when pictures
holding both are also trained under each single label. The SDXL pilot in step 3 cannot answer it
cleanly, because SDXL learnt "a cat" from its own data before any caption rule touched it. Its
128-seed scoring also waits on /datasets, which gave an I/O error on 2026-11-18. On MNIST a small
diffusion model trained from scratch sees only the canvases this step builds. The share of
two-digit canvases also trained under each single label is then a dial this step sets exactly.
Every choice below was settled in the route walk, with its reason, in
4-inclusive-captions/artifacts/drips/route-inclusive-captions-pilot/ROUTE_WALK.md.

Objectives
1. Build the canvas maker, which draws 32 x 64 canvases of real MNIST digits with the label each
   draw is trained under.
2. Build one label-conditional pixel U-Net and run a 200-step smoke run that prints a step time.
3. Train 15 models: dial 0, 0.1, 0.3, 0.6 and 1, three training seeds each, 30,000 steps each.
4. Draw 8 rows of 256 seeds from every model with the toy's sampler.
5. Put the shared judge through its two checks, then call every tile.
6. Draw the two-panel result figure and tick one verdict line against the bar in Rules.

What to read
- 4-inclusive-captions/plans/route-inclusive-captions-pilot.md: step 4 and the table "What we
  know so far".
- 4-inclusive-captions/artifacts/drips/route-inclusive-captions-pilot/ROUTE_WALK.md: the ten
  rounds that settled this step.
- 4-inclusive-captions/scripts/mnist_training_sheets.py: the canvas rule and the label rule,
  already written for the sheets; start the canvas maker from it.
- 3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md:
  the toy's caption model, its counts and its corrector, read whole.
- 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_caption_dial_ldm.py
  and toy_caption_corrector.py: the dial, the summed guidance and the corrector to copy.
- 2-classifier-guidance/scripts/mnist/judge_sheet.py: the shared judge; read it, never edit it.
- 4-inclusive-captions/artifacts/drips/route-inclusive-captions-pilot/figures/: the sheets of
  what each model draws, the U-Net, one sampling step, the leak read's layout and the bar.

What is already known
- MNIST is unzipped at /home/molef/PhD/replications/original-superdiff/data/MNIST/raw.
- The laptop's RTX 4070 has 8,188 MiB; /home/molef/micromamba/envs/sdxl-local/bin/python ran the
  toy's caption model on it in about 25 s per dial setting, and that environment has no
  torchvision.
- Nothing in this step reads or writes /datasets.
- A canvas is 32 x 64 with digits in 28-pixel boxes. 90% of canvases hold one digit at a random
  place. 10% hold two, one per half, with 0 to 4 pixels of jitter and the order swapped half the
  time.
- Half of the two-digit canvases hold a 7 and an 8. The other half are spread evenly over the
  other 39 seen pairs. The pairs 0 & 3, 1 & 9, 3 & 4, 3 & 6 and 4 & 7 never share a canvas
  (random.Random(0) over the 45 pairs, the classifier-guidance route's rule).
- Every canvas is trained under the blank label on 10% of its draws. A one-digit canvas takes its
  digit otherwise. A two-digit canvas holding a and b splits the other 90% by weight: "a and b"
  weight 1, and "a" and "b" each weight equal to the dial.
- The share of each label's training draws that hold one digit, at dial 0, 0.1, 0.3, 0.6 and 1:
  "7" 100, 94.9, 89.7, 85.5 and 82.4%; "8" 100, 94.7, 88.4, 84.2 and 81.4%; "4" 100, 99.1, 98.1,
  97.2 and 96.7% (400,000 simulated draws, mnist_leak_read_figures.py).
- Each training is one pixel U-Net. A 56-row label table (ten digits, 45 joint labels, blank) is
  added to the time embedding. The smoke run sets the channel widths. Training runs 30,000 steps
  at batch 128 for every model.
- The sampler is the toy's: reverse diffusion in 200 steps on the cosine schedule, guidance 3.
  The product is blank + 3 x ("7" - blank) + 3 x ("8" - blank). A corrected row adds 20 Langevin
  steps at every noise level, each of size 0.05 of that level's noise variance.
- Every model draws 8 rows on seeds 0 to 255, the same starting noise in every row and model:
  "7", "8" and "4" alone; "7 and 8"; 7 x 8 plain and corrected; 4 x 7 plain and corrected.
  Training draws from its own generator. A corrected picture costs about 21 times a plain one.
- A single-label tile is called damaged, wrong digit, right digit with a second, or right digit
  alone. A two-digit tile is called damaged, one digit, fused, two digits with a wrong identity,
  or two separate right digits.
- The toy's caption model at the same dial settings drew 0, 110, 103, 51 and 205 separate pairs
  of 300 plain, and 0, 300, 300, 300 and 300 with the corrector
  (toy-caption-dial-corrector.json). That is this step's prediction.
- Three trainings of one setting in the toy ranged from 17 to 252 of 300, which is why every
  setting here has three.
- The judge's prototype reads 98.2% of MNIST's 10,000 test digits right, but it calls a fused
  7-and-8 "one digit".
- artifacts/results/ is ignored by git (.gitignore line 40); earlier cards there were added with
  git add -f.

When something is missing
- The judge is not committed in the classifier-guidance route when piece 5 opens. Ask the user,
  and carry on with pieces 1 to 4 meanwhile.
- The judge fails one of its two checks. Stop and show the failing calls; the user chooses
  between a fix by that route's session and a judge of this step's own.
- The smoke run's step time puts 15 trainings past the schedule. Cut steps equally for all 15,
  never per setting, and say so.
- The laptop runs out of memory. Stop and ask; the fallback is a smaller batch, the same for all
  15 trainings.

Rules
- Never edit anything under 2-classifier-guidance/.
- Write the bar below into the scorer's source as named constants before any model draws.
- The judge's checks come first: at least 95% of each call right on 500 pasted canvases per
  call, and at least 58 of 64 agreeing with eye calls on one sheet per dial setting (plain
  product, training 1). Eye calls are made on enlarged tiles, and the user may overrule any.
- A dial setting counts only if "7" alone draws a lone 7 on at least 244, 231, 217, 207 and 199
  of 256, and "8" alone a lone 8 on at least 244, 230, 214, 203 and 196, at dial 0, 0.1, 0.3, 0.6
  and 1, in all three trainings.
- Supported: at some counting setting, the corrected 7 x 8 product draws two separate right
  digits on at least 128 of 256 in all three trainings, while dial 0 draws 12 or fewer in all
  three. Null: no counting setting reaches 39 of 256 on the average of its three trainings.
  Inconclusive: anything else.
- Reported beside the verdict: the corrector is not needed if the plain product also reaches 128
  at the supported setting. Composition reaches an unseen pair if the corrected 4 x 7 product
  reaches at least 39 of 256 in all three trainings and three times its own dial-0 count there.
- Stop and show the counts if the plain product at dial 0 draws more than 12 of 256. The fallback
  is a lower two-digit share or a pair whose strokes overlap more, and the user chooses.
- Stop and show the user four things: the smoke run's step time, the dial-0 plain count, the
  judge's two checks, and the figure.
- Each piece prints its counts before the next starts: canvases per kind and label draws per
  kind against the design, the step time, trainings saved, pictures per row.
- Commit nothing until the user says so.
- Filing is on: picture-cadence and record-cadence run beside the walk, because its sheets, step
  times and failures are worth keeping in the pillars.

What to hand back
- The code in poe_repair/experiments/inclusive_captions/mnist/: the canvas maker, the model, the
  trainer, the sampler and the scorer. The data and checkpoints go in data/mnist/, kept out of
  git.
- 4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/:
  scores.csv with one row per model, row and seed, carrying the dial, training seed, labels and
  call. Beside it go the two-panel figure, its sidecar JSON and a README card.
- The route update: add rows to "What we know so far" for the measured step time, the training
  time, the drawing time and the verdict. Show the schedule's edit as a diff and apply it after
  the user's yes.
- The hand-off: one verdict line, with the counts that decide it, for step 10's pressure-test.
- The proof: python3 -I poe_repair/experiments/inclusive_captions/mnist/score.py prints SUPPORTED,
  NULL or INCONCLUSIVE with every setting's counts out of 256.

What you will see
Along the way, the session stops at the four points in Rules, shows what it made, and asks one
question. The toy predicts the corrected 7 x 8 product near 256 of 256 at every dial above 0, and
the plain product between 17% and 68%. Supported needs at least 128 of 256 in all three trainings
at a setting where "7" and "8" still draw their own digit within 5 points of their training line.
At the end, a closing report in chat gives what was asked, what was done with each output's path,
the verdict with its counts, what is still open with each question and its fallback, and at most
one question.

You judge it from the figure and the scorer's printed verdict; it is done when one verdict line
is ticked and the route's rows are added.
```

- `added:` the canvas, pair, label, model, sampler, row, call, judge and bar settings, each from
  its round in the route walk; the training-line shares (computed for this route); the toy's
  prediction (toy-caption-dial-corrector.json); the results folder's git-ignore line
- `dropped:` all ten singles and the other four held-out pairs (why: one held-out pair answers the
  unseen-pair question; the four run only if 4 x 7 moves); a flat 90% purity bar (why: a perfect
  model fails it at every dial above 0.1); FashionMNIST (why: in Later, only after a supported
  verdict)
- [ ] **Gate:** the judge passes both checks before any count is read
- **Leaves:** `4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/`

### 5. 🗺️ Draw the pipeline (in the training wait)

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

### 6. 📖 Read what captions teach a single-concept condition (in the training wait)

You ~45 min · AI ~20 min · Sonnet, medium · moves: Achieve row 1 (defend the design)

**Objective:** one paragraph per source on what it found about co-occurrence in training data
and composition, and whether it predicts A against B.
**Outcome:** `READ_MAP.md` from the walk.
**Serves:** the paragraph you give a supervisor on why this is new.

```text
/drip-read --combine --lean "what the training captions teach a single-concept condition"
Sources: the papers cited in
3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md (Okawa, Liang,
Jeong, Zarei) and the ladder note in ../ideas/ideas-that-tie-the-ladder-together.md where it
names them
Question each source answers: does it change which caption a multi-object image is trained
under, or only how many such images there are? The rung 11b note says nobody has measured the
share dial; say whether anyone has measured the caption-assignment dial
Box: 45 min · stop: every source has its one-paragraph answer
```

- `added:` the question that separates this axis from the share axis (from the rung 11b note)
- **Leaves:** `READ_MAP.md` in the walk's folder

### 7. 📊 The figure

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

### 8. 👀 Look at the both-animal photos (inside step 3, piece 1)

- [ ] **Do:** open the contact sheet piece 1 writes (its path prints in the walk). For each of
  the 12 photos, say whether both animals are clearly visible, whether one is mostly hidden, and
  whether anything else dominates (a person, a sofa, a car).
- **You should see:** most photos busier than the single-animal ones. The rung 11b note warns
  that busier both-animal photos move scene complexity together with the caption.
- **Done when:** you have said a count, such as "8 of 12 show both clearly", and the walk has
  recorded it.

### 9. ✅ Tick the criterion against the figure

- [ ] **Do:** open `HYPOTHESIS.md` and the figure from step 7 side by side. Read the PoE-corrected
  bars for A and B and their difference.
- **You should see:** the difference in points and whether it falls above the support line,
  inside the inconclusive band, or below the null line.
- **Done when:** one of the three lines is ticked, with the two rates out of 128 written beside
  it.

### 10. 🧠 Explain it back

- [ ] **Do:** run the retrospective the walk's close prints, then pressure-test the verdict:

```text
/drip-retro <the retro file step 3's close prints>
```

```text
/pressure-test 4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/HYPOTHESIS.md and the ticked verdict:
could the difference come from scene complexity of the 208 photos, from the empty-caption rate,
or from the sampler, rather than from caption assignment? Step 4's MNIST verdict sits beside it:
where the two disagree, is the difference what a model that already knows "a cat" predicts?
```

- **Done when:** you can say the verdict, the two rates, the MNIST verdict and the strongest
  alternative explanation without notes.

### 11. 🗂️ Register the pages built

- [ ] **Do:** for the HLD from step 5 and the figure from step 7, write each page's section in the
  scene library, then run `scene-cards.py --apply`, `link-concepts.py --refresh-marks ... --apply`
  and `build-artifact-index.py --apply`, per `~/.claude/ARTIFACT_REUSE_FORMAT.md`.
- **Done when:** the register's `Linked from` column reads something other than **none** for
  both pages.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

None at this depth. Explaining it back is step 10. If you raise the depth later, the quiz goes
after step 2, prepared by `/quiz-prep-pulse` and built by `quiz-cadence`.

## Later

- **FashionMNIST, one later column:** step 4's design on two clothing classes, run only if step
  4's verdict is supported. Its filled, textured shapes sit closer to animals than pen strokes do.

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

- **Checkpoints at 250, 500 and 1,000 steps:** shows whether LoRA A's erosion happens early, as
  layer 5's toy predicts.
- **The interaction residual per LoRA:** `gs·(ε_J − ε_A − ε_B + ε_uncond)` measured under each
  LoRA with the project's residual code. A smaller residual under B would be the paper's link.
- **The source's three explanations** (wider overlap, corrected meaning, friendlier score
  geometry): the follow-up the source names, once the pilot shows an effect.
- **Neither-animal background images** and **bounding-box overlap splits**: need more both-animal
  photos than COCO's 208.
- **Two separately trained MNIST models against one shared model:** with no shared weights,
  "7" cannot borrow anything the network learnt for "8" (the setting of Du et al. 2023). On the
  toy, two separate networks drew 0 crisp pairs of 300 at every shared-picture fraction from 0 to
  0.5 (`toy-trained-experts.json`), where one conditional model drew 118 at dial 0.1. Runs only
  after step 4's verdict.
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
- [The rung 11b design](../../3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/evidence-poe-fails-at-scale.md#what-this-means-for-rung-11b)
- [The ladder guide](../../3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/PROMPTS.md)
- [Layer 5, the overlap dial](../../3-understanding-compositionality/artifacts/drips/the-toy/simple-distributions-rebuild/ladder/05-the-overlap-dial.md)
- [What PoE composition is](../../context/world/poe-composition.md)
- [Measured throughput](../../environment/hpc/throughput.md)

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

Paste step 4's prompt: the MNIST stage runs now on the laptop. Step 3's 128-seed scoring waits
for `/datasets` to return.
