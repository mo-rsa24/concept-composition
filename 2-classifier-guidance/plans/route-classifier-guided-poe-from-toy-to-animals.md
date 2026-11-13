# 🎯 Can a classifier's push stand in for the LoRA: from the exact toy to SDXL animals

<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** the paper's comparison table and sheet on held-out pairs (plain product,
> best-of-N at equal compute, the guided product, the joint prompt, the rank-32 LoRA), with one
> finding in `report/` per rung, or the rung where the climb stopped and its diagnosis ·
> **Time:** ~8 h to the close of rung A1, ~13 h budget at 2× for learning; later rungs are
> estimated at each rung's close · **Stop point:** each rung's bar, written before it runs; this
> route's box ends at A1's close, inside 10 sittings · **Outcome board:**
> [the classifier-guided PoE board](../report/classifier-guided-poe/board.md) · **Background:**
> [the references file](route-classifier-guided-poe-from-toy-to-animals-references.md)

Built to: route-file-template v3

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [The ladder](#the-ladder)
- [The two lanes](#the-two-lanes)
- [Schedule](#schedule)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-know-so-far) ➡️

**Goal:** the paper can say whether a small classifier's gradient, added to the product sampler at
each step with SDXL frozen, can stand in for the trained LoRA.

**Objective (the whole chain):** each rung of a fifteen-rung ladder passes or misses a bar written
before it runs, from the exact toy (cosine with the true correction, crisp pairs of 300) to
held-out SDXL pairs (two separate animals out of the held-out seeds, beside best-of-N at equal
compute, the joint prompt and the LoRA).

**Outcome (the whole chain):** the paper's comparison table and sheet on held-out pairs, and one
finding in `report/` per rung, or the rung where the climb stopped with its diagnosis. This route
schedules the climb to the close of rung A1; every later rung is re-routed from the retrospective
of the rung before.

**Outcome board:** [the classifier-guided PoE board](../report/classifier-guided-poe/board.md),
one row per rung plus the sampler repair (R0), the paper's table (P1) and your explanation (X1), per
`~/.claude/OUTCOME_BOARD_FORMAT.md`; each walk updates it as its step closes. What each row should
look like when it works is in
[the board's expected outcome per row](../report/classifier-guided-poe/board.md#expected-outcome-per-row),
whose numbers step 2 fills in.

**Background:** what was found on disk, the plan weight and the rationale are in
[the references file](route-classifier-guided-poe-from-toy-to-animals-references.md).

**Where files live.** This project is the folder `2-classifier-guidance/` of the portfolio repo
`/home/molef/PhD/poe_repair_min`: its `plans/`, `report/classifier-guided-poe/`, `artifacts/` and
`scripts/`. The guided sampler, the composers and the presence and dog heads are in the shared
package, under `poe_repair/methods/`, `poe_repair/composers/` and
`poe_repair/experiments/presence_head/`. Paths in this route are written from the repo root. The
toy card and the toy code belong to `3-understanding-compositionality/`.

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

Every walk adds a row here when it settles a fact, gets an answer, or hits a limit, then proposes
edits to the later prompts the fact changes and makes them after your yes.

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| Classifier guidance steers SDXL only unrescaled; pushes rescaled to the prediction's norm were about 44,000 times too large | found | dog-01, W&B `z2k1bdmi`, plan 18 on branch `exec/can-a-classifier-supply-the-missing-term` | 5, every SDXL rung |
| dog-01 failed its accuracy check (balanced accuracy at least 0.75 on 7 of the first 20 steps); a clean dog on 2 of 4 seeds, a dog of some kind on all 4 | found | plan 18, commit `f7981bc0` | 2 |
| Plan 15's sampler still rescales the push to the product prediction's own norm | found | `poe_repair/methods/_classifier_guided.py:112` | 5 |
| Plans 15, 17 and 18 and the classifier code are in this repo, moved from the core repo with a fresh start; their history stays there | found | this repo's `README.md`, 2026-11-13 | 3, 4, 5 |
| One-species classifier pushed on cat × dog: two animals in 0 of 16 tiles; the fused animal turned into a cat on 3 of 4 seeds | found | plan 18, commit `6be7d0c5` | 2, A3 |
| Plain-product best-of-16 on cat × dog composed 0 of 128 particles; with the adapter at strength 0.5, best of 16 is 8 of 8 | found | `report/00-INDEX.md` lines 155 and 156 | 2, A4, C1, C2 |
| GroundingDINO counts animals but cannot name them; CLIP on the detector's padded crop names them, 64 of 64 at median margin 0.990, untested on overlapping boxes | found | plan 15 quick context, `3-understanding-compositionality/scripts/census/crop_classifier_cat_vs_dog.py` | A3 onward |
| Exact toy at pair weight 0.01: exact product 300 of 300 crisp pairs, reverse diffusion 0 of 300 | found | the toy card | 2, 6 |
| Summed noised scores differ from the true noised product's score, so a discriminator learns log q_joint minus log q_sampler; A1 reports cosine against both targets | designed | the toy card, "Where the samplers lose the pair"; route round 2 | 2, 6 |
| A straight-line projection in DINOv2 space failed as a cross-pair tool (clouds as wide as the gaps) | found | `1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md` | D4 has no bar |
| Seeds 9 to 16 are never trained on; the cat × dog joint prompt is wrong on 4 of them | found | project memory, `report/` eval-reference finding | A4 |
| v58 pool: 182 cells over 48 pairs; 23 of 53 held-out pairs share no concept with training | found | `1-lora-correction/report/paper-review/held-out-set.md` | C1, C2 |
| Every sampler pushes with plan 17's Algorithm 2 form, unrescaled, capped at a fraction κ of the product's own step | taken | route round 1 | 5 onward |
| Rung A1's classifier is a discriminator, joint pictures against the product's own draws | taken | route round 1 | 2, 6 |
| The SDXL classifier reads noisy latents; DINOv2 on the decoded estimate is an A4 ablation if needed | taken | route round 1 | A3, A4 |
| Plan 18's product sheet is A3's one-species reference column | taken | route round 1 | A3 |
| Diagnosis reads D1 to D4 run beside every classifier rung's sweep and never hold it back | taken | route round 2 | 2, 4, 6 onward |
| This repo's `plans/` is the scope; the core repo's root running order gets one row pointing here | taken | 2026-11-13 | 3 |
| Discriminator guidance (Kim et al.) is not in the reading register | found | `plans/standing/literature/reading-register.md` | 1 |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have written | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 the method read | own | nothing; the read map's "What A1 takes from this" | the best discriminator q1/(q1+q2) on two 1-D Gaussians, and its log-odds slope | D* at one point, by hand | the read's prediction rounds | three questions after the read |
| 2 the criterion | gloss | the compile with a bar per rung | the ladder, with A1's 0 against 300 marked | the toy's 0.01 row | none | none |
| 3 R0, the sampler | gloss | the repaired push, capped at κ times the product's step | where the cap bites on one step's vectors | one step's push size before and after the cap | none | none |
| 4 A1, the exact toy | build | the toy discriminator's loss and log-odds gradient, typed by you | cosine with each target across reverse-diffusion steps | one draw's push at one step against the true correction | D1 minimal pairs on toy pictures | a full page after A1 |
| **Outcome** | | **A1's figure set and finding, and the board's A1 row done** | | | | **A1's bar printed pass or miss, and the retrospective walked** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-ladder) ➡️

▶ **Recommended:** `1 drip-read --lean → 2 frame-hypothesis → 3 hypothesis-to-scope →
4 populate-plans → 5 drip-execute-plan (R0) ‖ background: canary on the cluster → 6
drip-execute-plan (A1) ‖ background: quiz-prep-pulse, quiz-cadence → 7 drip-retro → re-route`
(spliced from workflow 20, the PhD experiment loop, and the research spine's read layer)

<details><summary>Alternatives</summary>

`frame-hypothesis → drip-execute-plan --request (A1)`: better when A1 must start this week, and a
scope can wait until A1 says the idea lives.

`drip-idea --hypothesis → hypothesis-to-scope → populate-plans → drip-execute-plan`: better when
the claim itself still needs arguing before any bar is written.

</details>

## The ladder

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

Two axes are climbed separately, so each rung changes one thing.

**Phase A: the setting climbs, the architecture is held.** The classifier is a discriminator
alone, joint pictures against the product's own draws, with a constant strength λ and a fixed cap
κ. A1 the exact toy, A2 the toy's trained caption model, A2b MNIST digits placed two to a canvas,
A2c animal photos at 64×64 with pasted pairs (both keep the dial for how many training pictures
hold both things, with exact labels), A3 SDXL cat × dog trained and tested on the same seeds, A4
cat × dog on held-out seeds 9 to 16. Research question 1 is answered here.

**Phase B: the setting is held at A4, one part is added per rung.** A part is kept only if it
beats the rung before it. The heads share one trunk trained once, and which head's gradient is
used is a sampling-time choice. B1 presence, B2 instance count, B3 spatial mask, B4 hybrid and
quality (research question 2). B5 the weak, strong, fading schedule, B6 gradient agreement, B7
per-head early stopping (research question 3).

**Phase C: the best configuration is carried up.** C1 trained over the v58 pool, tested on
held-out seeds; C2 held-out pairs. Both add the rank-32 pooled LoRA as a column.

**Every SDXL rung compares** the plain product, best-of-N reranking by the same classifier at
equal compute, the guided product and the joint prompt, called by eye at full resolution and by
an evaluator the classifier was not trained on (the GroundingDINO instance count of 2 or more,
and the CLIP crop classifier for species).

**Every classifier rung diagnoses its classifier beside the guided sweep, never holding it back.**
D1 the score gap on minimal pairs (a fused animal against two separate ones, one dog against two,
a two-animal joint render against the same render with one animal removed) at every noise step,
against a shuffled-label floor. D2 the push from that head alone on 4 fixed hybrid states,
decoded and called by eye. D3 the cosine of the push with the correction per step (the true
correction on the toy, the joint prompt's prediction minus the product's on SDXL). D4 an LDA
picture of the head's features labelled single, hybrid, two bodies, two clear, with no bar. A
sweep that composes while D1 fails is reported as composing for the wrong reason.

**Steps that change more than one thing.** A2c to A3 changes the model, the data and the
evaluator at once, which the step to a pretrained model cannot avoid. Pasted pairs carry seams,
which D1's third pair catches when its removal is pasted the same way. Plan 18's one-species sheet
is a reference column on A3, never the baseline. The discriminator is trained once on unguided
draws; retraining it on guided draws is its own later rung.

**What A1 compares against.** The discriminator estimates log q_joint minus log q_sampler, where
q_sampler is what the summed-score sampler actually draws at each step. The toy card shows that
differs from the true noised product. A1 reports the cosine against both targets.

**A missed bar stops the climb.** The rung's walk opens its diagnosis ladder (diagnose, fix,
recheck, half the rung's box, at most two levels) and the next rung is not routed until it closes.

**Planning weight.** This repo is the scope: its `2-classifier-guidance/plans/MASTER_PLAN.md` holds this ladder, the
diagnosis reads and a bar per rung, provisional past A1. Plan files are written one rung ahead,
R0 and A1 first, each with its review file in `2-classifier-guidance/plans/review/`. Plans 15, 17 and 18 sit beside
them in `plans/`.

## The two lanes

Navigation: ⬅️ [Previous](#the-ladder) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

The AI lane builds the rungs. Your lane reads the method, types the toy discriminator yourself,
and calls the pictures. The lanes meet at A1's close, where your typed discriminator and the AI's
run on the same draws.

| Stage | The AI lane builds | Your lane learns | Quiz (built in the background) | Confidence bar | You rebuild | Compare on |
|---|---|---|---|---|---|---|
| R0 | the capped push, the canary | what the cap does to one step | none | none | none | the canary's checksums |
| A1 | exact draws, the guided sampler, the per-step cosine read | the read (step 1), the criterion (step 2) | `quiz-prep-pulse` then `quiz-cadence` on the A1 plan | every needed-now question right and 80% of the rest (estimate) | the toy discriminator's loss and its log-odds gradient | the same 300 toy draws at pair weight 0.01 |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Assumed windows: three 90-minute sittings a week, because none were given. Completion to A1's
close is about 8 hours of your attention; the budget at 2× for the two teaching walks (steps 1
and 6) is about 13 hours, so about 10 sittings, about three and a half weeks.

| Block | Run | Checkpoint at the end of the block |
|---|---|---|
| sitting 1 | step 1 | the read's spine and its first half |
| sitting 2 | step 1, then step 2 begins | the read map's closing section written; the claim stated |
| sitting 3 | step 2, step 3, step 4 | the compile, this repo's master plan, the R0 and A1 plans with review files |
| sitting 4 | step 5 ‖ background: the canary on the cluster | the push capped, the canary checksums identical |
| sittings 5 to 9 | step 6 | A1's pieces built and tested, about one per sitting; the bar printed pass or miss |
| sitting 10 | step 7 ‖ background: the A1 quiz built during sitting 9 | the retrospective walked; the A2 re-route prompt printed |

A1's walk is the step most likely to overrun. If sitting 7 ends with fewer than three pieces
tested, sitting 8 opens with `faster`, which costs the build tier on the remaining pieces.

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. 📄 Read the discriminator-guidance paper, lean

You ~75 min (estimate) · AI ~10 min read-ahead · Opus, high · moves: needed by step 6 for the
discriminator's training recipe

**Objective:** the read map's closing section names the discriminator's inputs, loss, noise
levels and the sampler formula, each with its page.

**Outcome:** the read map, with "What A1 takes from this".

**Serves:** needed by step 2 for the claim and by step 6 for the recipe.

```text
/drip-read --lean 2211.17091

Background
This lean read is step 1 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.
The route asks whether a small trained classifier, whose gradient is added to the product of
experts at each sampling step, makes SDXL draw two separate animals instead of one fused animal,
and whether that can stand in for the trained LoRA. The paper should be Kim et al., "Refining
Generative Process with Discriminator Guidance in Score-based Diffusion Models" (ICML 2023). Its
method is the route's first rung: a discriminator trained on real data against the model's own
samples, whose log-odds gradient corrects the score.

Objectives
1. Read the paper in its own order, one paragraph per round.
2. Say how the discriminator is trained: inputs, noise levels, time input, features and loss.
3. Say how its gradient enters the sampler, with the weight and schedule the paper uses.
4. Derive why the best discriminator's log-odds gradient is the score of the density ratio.
5. Name each result the paper reports that the exact toy can check, with its number.

What to read
- The paper, whole.
-
  3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md,
  the toy it meets first.
- 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md, the ladder and the
  known facts.

What is already known
- Classifier guidance steers SDXL here only with the gradient unrescaled (run dog-01,
  2-classifier-guidance/plans/18-a-dog-classifier-trained-the-papers-way.md).
- The route's discriminator learns log q_joint minus log q_sampler, and on the toy q_sampler is
  not the true noised product (the toy card, "Where the samplers lose the pair").
- The discriminator's negatives come once from the unguided sampler; retraining on guided draws
  is a later rung.

When something is missing
- If 2211.17091 is not that paper, stop and say so; the reader gives the right id.
- If the recipe needs pretrained features the toy lacks, say what would replace them.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- Read only. Write no code and change no plan.
- Every number keeps its unit and what it measures.
- Filing: tied, because the scope's plans will hold what this read decides.

What to hand back
- The read map, closing on a section headed "What A1 takes from this": the training recipe, the
  sampler change and the checks the toy can run, each with its page.
- One row per settled fact for the route's What we know so far table.

What you will see
A spine of one sentence per section, then one paragraph per round, a prediction asked before the
derivation, and the closing section. It is done when that section names the discriminator's
inputs, loss, noise levels and the sampler formula, each with its page.

Controls: the derivation round, prove (the best discriminator is q1 / (q1 + q2)); the training
round, example (one toy draw labelled real or sampled, at one noise level)
Side routes: the derivation round, /picture-speak --derive D* = q1/(q1+q2) to the log-ratio score
Learning path: classifier guidance, no journey yet, expect a gap line; density ratio, no journey
yet, the shelf's little book of generative AI foundations p162
```

- `added:` the arXiv id and title, from memory, so the prompt checks the id first · `added:` the
  q_sampler caveat (route round 2) · `dropped:` nothing
- [ ] **Gate:** the read map's closing section exists before step 2.
- **Leaves:** the read map, which step 2 and step 6 read.
- **Learning path:** classifier guidance and density ratio have no journey yet, so expect `gap`
  lines; the shelf's [little book of generative AI foundations, p. 162](/home/molef/goal-setting/learning/mathematics/textbooks/machine-learning/the-little-book-of-generative-ai-foundations.pdf#page=162)
  teaches the discriminator.

### 2. 🧪 Frame the hypothesis over the whole ladder

You ~45 min (estimate) · AI ~5 min · Opus, high · moves: writes the bar every rung row is judged on

**Objective:** every rung from A1 to C2 has a bar, firm or provisional, with a number and a stop
rule.

**Outcome:** the frame-hypothesis compile.

**Serves:** needed by step 3 for the scope and by every rung for its bar.

```text
/frame-hypothesis A trained discriminator's log-odds gradient, added to the product-of-experts
sampler at each step with the experts frozen, makes the product draw two separate objects where it
draws one fused object, and on SDXL it composes held-out animal pairs about as often as the
rank-32 pooled LoRA and more often than best-of-N reranking at equal compute.

Background
This is step 2 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. Step 1
read the method's paper; its read map's "What A1 takes from this" section is the recipe. The
route's ladder has three phases: the setting climbs with the architecture held (A1 exact toy, A2
caption toy, A2b MNIST, A2c animals at 64 px, A3 SDXL same seeds, A4 held-out seeds 9 to 16),
then parts are added one per rung at A4 (B1 to B7), then the best configuration goes to the v58
pool and held-out pairs beside the LoRA (C1, C2). The paper needs to say whether this
sampling-time correction can stand in for the LoRA.

Objectives
1. Write the support, null and inconclusive outcomes for the whole claim, each with a number.
2. Write A1's bar firmly, with what would surprise us.
3. Write a provisional bar for every later rung, marked as settled at the close of the rung before.
4. Say which outcome on each rung stops the climb and goes to diagnosis.

What to read
- 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md, the ladder, the
  diagnosis reads and
  the known facts, whole.
- The step 1 read map's closing section.
-
  3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md,
  whole.
- report/00-INDEX.md rows on scope 06, and 1-lora-correction/report/paper-review/held-out-set.md.

What is already known
- Exact toy at pair weight 0.01: exact product 300 of 300 crisp pairs, reverse diffusion 0 of 300.
- The discriminator estimates log q_joint minus log q_sampler; on the toy that differs from the
  true noised product, so A1 reports the cosine against both targets.
- Plain-product best-of-16 on cat x dog composed 0 of 128 particles; with the adapter at strength
  0.5, best of 16 is 8 of 8 (report/00-INDEX.md, lines 155 and 156).
- One-species classifier pushed on cat x dog: two animals in 0 of 16 tiles (plan 18, commit
  6be7d0c5).
- The bar is two separate creatures in frame, called on the ladder single, hybrid, two bodies or
  heads, two clear; picture damage is reported separately.
- The cat x dog joint prompt is wrong on 4 of seeds 9 to 16.

When something is missing
- If a rung's number cannot be set before its predecessor runs, mark it provisional and say what
  will set it.
- If the step 1 read map does not exist yet, frame from the paper's abstract and say so.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- Every number carries its unit and what it counts.
- Write what would surprise us for each rung.
- Change no plan and run nothing.

What to hand back
- The compile, with the claim, the three outcomes per rung and the stop rules, for step 3.
- The number for each row's line in 2-classifier-guidance/report/classifier-guided-poe/board.md,
  section "Expected
  outcome per row", written into that table's Number column.
- One row per settled bar for the route's What we know so far table.

What you will see
One round per part of the criterion: the claim, then support, null and inconclusive, then A1, then
the later rungs. It is done when every rung from A1 to C2 has a bar, firm or provisional, with a
number and a stop rule.

Controls: the A1 round, example (the toy's 0.01 row, 0 against 300); the support round, prove
(why beating best-of-N at equal compute is the claim and not just composing)
Learning path: density ratio, no journey yet, the shelf's little book of generative AI foundations
p162
```

- `added:` the toy's 0.01 row as A1's anchor (the toy card) · `added:` a stop rule per rung (your
  rule that a missed bar stops the climb) · `dropped:` nothing
- [ ] **Gate:** you agree with A1's bar as written, before step 3.
- **Leaves:** the compile, which step 3 reads.

### 3. 🗂️ Turn the hypothesis into this repo's master plan

You ~40 min (estimate) · AI ~10 min · Opus, high · moves: needed by step 4

**Objective:** a master plan that holds the ladder, the axes, the diagnosis reads and a bar
per rung.

**Outcome:** `2-classifier-guidance/plans/MASTER_PLAN.md` in this repo.

**Serves:** needed by step 4 for the plan files.

```text
/hypothesis-to-scope <the step 2 compile> as this repo's 2-classifier-guidance/plans/MASTER_PLAN.md,
in
/home/molef/PhD/poe_repair_min

Background
This is step 3 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. Step 2
wrote the claim, the support, null and inconclusive outcomes, and a bar per rung. This step turns
that into this repo's master plan, which holds the ladder. Plan files are written one rung ahead by
step 4, so this step writes no plan file.

Objectives
1. Read this repo's README.md and CLAUDE.md, which say what lives here and what the core repo
   holds.
2. Write the master plan: mission, objectives, goals, expected outcome, definition of done.
3. Name the axes: the setting (toy, caption toy, MNIST, animals at 64 px, SDXL same seeds, SDXL
   held-out seeds, the v58 pool, held-out pairs), the architecture parts (B1 to B7), the
   sampling-time controls, and the evaluation; say which rungs carry the science.
4. Carry the ladder's three phases, the diagnosis reads D1 to D4 and each rung's bar, provisional
   past A1, marked as settled at the close of the rung before.
5. Propose one row for the core repo's root MASTER_PLAN.md running order pointing at this repo,
   for my yes.

What to read
- The step 2 compile, whole.
- 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md, the ladder and the
  known facts.
- The core repo's root MASTER_PLAN.md, its running order.
- ~/.claude/EXPERIMENT_CONVENTIONS.md, for the review-file split.
- environment/overview.md, for the facts the SDXL rungs depend on.

What is already known
- Every sampler pushes in plan 17's Algorithm 2 form, unrescaled, capped at a fraction kappa of
  the product's own step.
- Plans 15, 17 and 18 and the classifier code are in this repo; the shared code is the core
  package poe_repair, installed with pip install -e.
- The SDXL classifier reads noisy latents.
- Diagnosis reads run beside every classifier rung's sweep and never hold it back.

When something is missing
- If an axis value cannot be named before its rung, write it as open with the rung that sets it.
- If the root running order has no natural place, ask me where it goes.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- Write only this repo's plans/MASTER_PLAN.md; edit the core repo's running order only after my
  yes.
- No private labels: each rung is named by what it changes.

What to hand back
- The master plan path, and the proposed root running-order rows.
- One row per settled fact for the route's What we know so far table.

What you will see
The reframed mission, then the axes, then the ladder with its bars, each shown before it is
written. It is done when the master plan names every rung, its bar and its stop rule.
```

- `added:` the axes list (your CLAUDE.md, "Name the axes before you name the experiments") ·
  `dropped:` the experiment planner and atlas of workflow 20 (why: the ladder settled the design)
- [ ] **Gate:** you approve the root running-order rows.
- **Leaves:** the scope's master plan, which step 4 reads.

### 4. 📝 Write the R0 and A1 plans and their review files

You ~20 min (estimate) · AI ~15 min · Opus, high · moves: R0 and A1 from open to todo

**Objective:** two plan files and two review files that pass `verify-plan`.

**Outcome:** the R0 and A1 plan files under `plans/`, each with its review file in `2-classifier-guidance/plans/review/`.

**Serves:** needed by steps 5 and 6.

```text
/populate-plans /home/molef/PhD/poe_repair_min/2-classifier-guidance/plans/ only the first two rungs

Background
This is step 4 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. This
repo's master plan holds the ladder. Plans are written one rung ahead, so only rung R0 (repair
the sampler) and rung A1 (the exact toy) get plan files now.

Objectives
1. Write the R0 plan: change poe_repair/methods/_classifier_guided.py so the push is
   plan 17's form, unrescaled, capped at kappa times the product's own step; rerun the strength-0
   canary on the cluster.
2. Write the A1 plan: exact draws of pairs and of the summed-score sampler at pair weight 0.01;
   the discriminator trained on them with the step 1 recipe; the guided sampler with the cap; the
   cosine per step against both targets; D1 to D4 on toy pictures; crisp pairs of 300.
3. Write each plan's review file with the bar from the master plan, for my tick.
4. Run verify-plan on both and fix what it reports.

What to read
- This repo's 2-classifier-guidance/plans/MASTER_PLAN.md, whole.
- The step 1 read map's closing section.
-
  3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py,
  the toy's code.
- 2-classifier-guidance/plans/15-can-a-classifier-supply-the-missing-term.md, its canary task, and
  2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md, its sampler.
- environment/overview.md and environment/hpc/nodes.md, for the canary's node.

What is already known
- Plan 15's canary passed on mscluster108: three renders with checksum c291b259.
- The co3 environment cannot drive the Blackwell cards (mscluster110 to 112); pin 106, 108 or 109.
- The toy's exact product at pair weight 0.01 gives 300 of 300 crisp pairs; reverse diffusion 0.

When something is missing
- If the step 1 recipe needs features the toy lacks, the A1 plan names the replacement.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- Write only the two plans, their review files and the master plan's plan table.
- The outcome board 2-classifier-guidance/report/classifier-guided-poe/board.md: R0 and A1 move to
  todo, with the step
  named, and the board is redrawn with --step 4.

What to hand back
- The four file paths and verify-plan's output for both plans.

What you will see
Each plan's task list shown before it is written, then the review files, then verify-plan. It is
done when both plans pass verify-plan and the board shows R0 and A1 as todo.
```

- `added:` the canary's node facts (environment/overview.md) · `dropped:` plans for A2 onward
  (why: written one rung ahead)
- [ ] **Gate:** you tick or rewrite A1's bar in its review file.
- **Leaves:** the R0 and A1 plans.

### 5. 🔧 Walk R0: cap the push, rerun the canary

You ~45 min (estimate) · AI ~60 min · GPU wait ~10 min · Opus, high · moves: R0

**Objective:** the push capped, and the strength-0 canary byte-identical to
the plain product.

**Outcome:** a commit with the repaired sampler, and the canary's `canary.json`.

**Serves:** the board's R0 row, needed by every SDXL rung.

```text
/drip-execute-plan <the R0 plan step 4 wrote>

Background
This is step 5 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. Plan
15's sampler rescales the push to the product prediction's own norm, which made earlier pushes
about 44,000 times too large. This walk repairs the push.

Objectives
1. Make the push plan 17's form, unrescaled, capped at kappa times the product's own step.
2. Rerun the strength-0 canary on a pinned Quadro and confirm identical checksums.

What to read
- The R0 plan and its review file.
- poe_repair/methods/_classifier_guided.py, and plans/17-classifier-guidance-from-
  the-paper.md, its sampler.
- The core repo's environment/overview.md and runbook/00-INDEX.md, the launch recipe.

What is already known
- Plan 15's canary passed on mscluster108 with checksum c291b259.
- Every cluster action goes through scripts/cluster.sh, which refuses uncommitted tracked changes.
- Other sessions edit this repo; re-read a shared file before reporting it.

When something is missing
- If this session has no cluster access, print the scripts/cluster.sh command for me to run.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- Commit before any cluster run.
- The outcome board 2-classifier-guidance/report/classifier-guided-poe/board.md: R0's two outcome
  lines ticked with
  links, and the board redrawn with --step 5.
- Filing: tied, because this is a repair whose facts go to the plan and the board.

What to hand back
- The sampler commit, the canary's checksums, and route rows for each.

What you will see
The repaired push with one step's sizes before and after the cap, then the canary.
It is done when the canary prints one checksum for all its renders.

Controls: the cap piece, example (one step's push size before and after); the canary piece, debug
Debug kit: the canary piece, python 2-classifier-guidance/scripts/presence_head_canary.py in this
repo (reads the cached
starting noise, writes /datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/)
```

- `added:` the cluster fallback (your brief's rule) · `dropped:` nothing
- [ ] **Gate:** the canary's checksums are identical.
- **Leaves:** the repaired sampler, which every SDXL rung uses.

### 6. 🔬 Walk A1: the discriminator on the exact toy

You ~225 min to finish, ~450 min budget at 2× (estimate) · AI ~60 min · Opus, high · moves: A1

**Objective:** A1's bar from its review file, printed pass or miss.

**Outcome:** A1's figures (cosine per step against both targets, crisp pairs of 300 guided and
unguided, D1 to D4), its card and its finding in `report/`.

**Serves:** the board's A1 row, and research question 1 on the toy.

```text
/drip-execute-plan <the A1 plan step 4 wrote>

Background
This is step 6 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. On the
exact toy every density is closed form, so the true correction is known. At pair weight 0.01 the
exact product draws 300 of 300 crisp pairs and reverse diffusion with summed scores draws 0. A1
asks whether a discriminator trained on pair draws against the summed-score sampler's own draws
supplies a push that closes that gap.

Objectives
1. Draw exact pairs and the summed-score sampler's draws at pair weight 0.01, at every step.
2. Train the discriminator with the step 1 recipe; I type its loss and its log-odds gradient.
3. Guide the sampler with the capped push and count crisp pairs of 300, guided and unguided.
4. Read the cosine per step against both targets: the sampler's ratio and the true product.
5. Run D1 to D4 on toy pictures beside the sweep.

What to read
- The A1 plan and its review file.
- The step 1 read map's closing section.
-
  3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py
  and the toy card.

What is already known
- Exact toy at pair weight 0.01: exact 300 of 300, reverse diffusion 0 of 300.
- The discriminator estimates log q_joint minus log q_sampler, which differs from the true noised
  product on this toy.
- The push is unrescaled and capped at kappa times the product's own step.
- Diagnosis reads run beside the sweep and never hold it back.

When something is missing
- If training does not separate the two classes, the walk's diagnosis ladder opens.
- If the toy code lacks a per-step draw hook, add one and say where.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- I type the discriminator's loss and gradient myself.
- The outcome board 2-classifier-guidance/report/classifier-guided-poe/board.md: A1's outcome lines
  ticked with links,
  and the board redrawn with --step 6.
- Filing: on, because this is the build being learned by hand.

What to hand back
- A1's figures and card under artifacts/results/, its finding in report/, the bar's pass or miss
  line, and route rows for each settled fact.

What you will see
One piece per round, each ending on a test I run. It is done when A1's bar prints pass or miss
with its figure beside it.

Controls: the training piece, build and break (type the loss, then flip a label); the cosine
piece, example (one draw's push against the true correction at one step); the closing piece, prove
Side routes: the cosine piece, /picture-speak --consequence the cap switched off
Debug kit: the training piece, the A1 training command named by step 4 (reads the exact draws,
writes the discriminator checkpoint)
Learning path: density ratio, no journey yet, expect a gap line
```

- `added:` both cosine targets (route round 2) · `dropped:` nothing
- [ ] **Gate:** A1's bar printed; a miss opens the diagnosis ladder and stops the climb.
- **Leaves:** A1's retrospective file, which step 7 walks.

### 7. 🔁 Walk A1's retrospective, then re-route for A2

You ~30 min (estimate) · AI ~5 min · Opus, high · moves: X1 (A1's part)

**Objective:** each A1 result predicted before it is shown, and A2's re-route prompt printed.

**Outcome:** the walked retrospective and the re-route command.

**Serves:** the board's X1 row, and the route's next rungs.

```text
/drip-retro <the retrospective file A1's close printed>

Background
This is step 7 of the route in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. A1 has
closed on its bar. This walk makes me predict each result before it is shown, then prints the
re-route that schedules rung A2 from what A1 learned.

Objectives
1. Walk each chunk of the retrospective, my prediction first.
2. Add one route row per fact A1 settled.
3. Print the re-route command for A2.

What to read
- A1's retrospective file, its card and its finding.
- 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md, the ladder.

What is already known
- A1's bar and its pass or miss line, from step 6.

When something is missing
- If A1 missed its bar and its diagnosis ladder is still open, walk that first.

Rules
- Paths not in this repo are in the core repo; the route's "Where files live" says which.
- The outcome board: X1's A1 line ticked.

What to hand back
- The walked retrospective, the route rows and the re-route command.

What you will see
One chunk per round, prediction first. It is done when the re-route command is printed.
```

- [ ] **Gate:** none; the re-route is the next step.
- **Leaves:** `/drip-skill-route --from 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md`
  at the A2 round.

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 8. ✍️ Draw the best discriminator by hand, after step 1

- [ ] **Do:** on the iPad, draw two 1-D Gaussians q1 and q2, then D*(x) = q1/(q1+q2) under them,
  then the slope of log D*/(1−D*).
- **You should see:** the slope equals the slope of log q1 − log q2, which is the correction A1
  measures.
- **Done when:** you can say why a classifier's log odds is a density ratio without notes.

### 9. ⚖️ Tick A1's bar in its review file, after step 4

- [ ] **Do:** open the A1 review file step 4 wrote, read the support, null and inconclusive lines,
  tick them or rewrite them.
- **Done when:** the bar is ticked before step 6 runs anything.

### 10. 🖥️ Run the canary yourself if the session has no cluster access, in step 5

- [ ] **Do:** from the laptop checkout, after committing, run
  `scripts/cluster.sh run "sbatch <the canary sbatch the R0 plan names>"`, then
  `scripts/cluster.sh jobs` until it leaves the queue.
- **You should see:** `canary.json` under
  `/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/` with one checksum for every
  render and largest pixel difference 0.
- **Done when:** the checksums match.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| After stage | Prepared by | Taken as | A slip goes to | A gap goes to |
|---|---|---|---|---|
| 1 the read | the read's own prediction rounds | three questions in the read's close | the A1 quiz | the density-ratio gap line, learned at gloss in 5 min |
| 4 A1 | `/quiz-prep-pulse` on the A1 plan, started in sitting 9 | page by `quiz-cadence` | the A2 quiz | its node in the coverage file, learned in 30 min, then retaken |

## Next step

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents)

**Start with:** step 1, the lean read of 2211.17091. Say `re-route <what changed>` for a different
chain, `calendar` to put the schedule on Google Calendar, or `todoist` to file the runbook as
ordered subtasks. The background is in
[the references file](route-classifier-guided-poe-from-toy-to-animals-references.md).
