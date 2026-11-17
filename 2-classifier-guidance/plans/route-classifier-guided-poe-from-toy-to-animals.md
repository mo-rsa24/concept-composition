# 🧪 Can a classifier's push fix the product: from the released weights to SDXL cat × dog
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** one figure set per stage (released weights, exact toy, our photos, SDXL),
> each with its card and its pass or miss line, ending on SDXL cat × dog with a pair-aware
> classifier beside plain product, the joint prompt and the species-only control · **Time:** ~20 h
> to finish to the SDXL stage, ~38 h budget · **Stop point:** each stage's bar printed pass or
> miss, inside 19 two-hour sittings · **Outcome board:**
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
each step with the diffusion model frozen, can fix what the product of experts<sub>[1](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#31-product-of-experts-and-the-and-operator "What PoE loses when it composes, §3.1: the product of experts derived from Bayes' rule plus the one assumption that the two captions are independent given the image"), [2](../../context/world/poe-composition.md#what-poe-composition-is "Context: what PoE composition is, the model run once per concept with the two predictions combined at every denoising step")</sub> gets wrong, and so
stand in for the [trained LoRA](../../context/world/lora-corrector.md#what-the-lora-corrector-is).

**The claim, in plain notation.** Bayes gives `log p(x | cat, dog) = log p(x) + log p(cat, dog | x)`
plus a constant. The product samples as if `p(cat, dog | x) = p(cat | x) · p(dog | x)`, which is
where `ε_cat + ε_dog − ε_null` comes from. It drops `log p(cat, dog | x) − log p(cat | x) −
log p(dog | x)`, whose gradient is the interaction term<sub>[1](../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#35-the-interaction-term "What PoE loses when it composes, §3.5: the interaction term derived from Bayes as the gradient of the log ratio of the joint caption probability to the two single ones"), [2](../../context/world/interaction-term.md#what-the-interaction-term-is "Context: what the interaction term is, the per-step gap between the joint prompt's prediction and the product's")</sub>. A classifier's gradient enters the
prediction as `ε_fixed = ε_product − s · σ_t · ∇ log c(x_t)`, with `σ_t = √(1 − ᾱ_t)`. A classifier
trained on cat and dog separately only re-weights the product, because `ε_cat − ε_null` already is
`−σ_t · ∇ log p(cat | x_t)`. So the classifier has to be pair-aware: trained to tell two separate
animals from the [fused animal](../../context/world/chimera.md#what-a-chimera-is) the product draws.

**Objective (the whole chain):** each stage passes or misses a bar written before it runs, and at
every stage below SDXL the pair-aware classifier is set against a separate-label one that the maths
says cannot help.

**Outcome (the whole chain):** four figure sets with cards and findings: classifier guidance working
on the authors' released weights, the exact toy with both classifiers, our own photos with the
released model and with a model we train, and SDXL cat × dog on training seeds. Later stages (held-out
seeds, the head ablations, the v58 pool against the LoRA, the paper's table) are re-routed from the
SDXL stage's retrospective.

**Outcome board:** [the classifier-guided PoE board](../report/classifier-guided-poe/board.md), one
row per stage plus the sampler repair, the paper's table and your explanation, per
`~/.claude/OUTCOME_BOARD_FORMAT.md`; each walk updates it as its step closes.

**Background:** what was found on disk, the plan weight, your depth for this route and the rationale
are in [the references file](route-classifier-guided-poe-from-toy-to-animals-references.md).

**Where files live.** This project is the folder `2-classifier-guidance/` of the portfolio repo
`/home/molef/PhD/concept-composition`: its `plans/`, `report/classifier-guided-poe/`, `artifacts/`
and `scripts/`. The guided samplers, the composers and the presence and dog heads are in the shared
package, under `poe_repair/methods/`, `poe_repair/composers/` and
`poe_repair/experiments/presence_head/`. The toy belongs to `3-understanding-compositionality/`.
Paths in this route are written from the repo root. Reproductions of other people's code live in
`/home/molef/PhD/replications/`.

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

Every walk adds a row here when it settles a fact, gets an answer, or hits a limit, then proposes
edits to the later prompts the fact changes and makes them after your yes. A row a later check
contradicts is rewritten. Later prompts treat these rows as settled facts.

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| Classifier guidance steers SDXL only unrescaled; pushes rescaled to the prediction's norm were about 44,000 times too large | found | dog-01, W&B `z2k1bdmi`, [plan 18](18-a-dog-classifier-trained-the-papers-way.md) | 5, 10 |
| [Plan 17](17-classifier-guidance-from-the-paper.md)'s guided sampler `run_cfg_classifier_guided` uses the paper's form, unrescaled, and at strength 0 matched the empty-prompt render byte for byte on four seeds | found | plan 17 tasks 2.1 and 2.2, 2026-11-04 on mscluster108 | 5 |
| The product sampler `run_poe_classifier_guided` pushes unrescaled (strength times sqrt(1 - alpha_bar_t) times the gradient), capped at κ times the product prediction's norm; the rescaled `weight` form is removed and a nonzero `weight` raises. At strength 0 it renders cat × dog seed 9 byte-identical to plain product, with and without an untrained head (md5 `c291b259647d30a450e87b234423fe99` on all three, 0 pixels differing) | found | commits `445cef9f` and `d1b8b61f`, job 89341 on mscluster108, `/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/canary.json` | 10 |
| κ, the cap on the push, is `PUSH_CAP_FRACTION` 0.1, our value since Dhariwal and Nichol cap nothing; on plan 17's dog sheet 0.1 never fires at strengths 10 and 30 and fires on some step of every render at 100 and 300 | designed | `poe_repair/methods/_classifier_guided.py`, plan 17's `push.json` | 9, 10 |
| dog-01 failed its accuracy check (balanced accuracy at least 0.75 on 7 of the first 20 steps); a clean dog on 2 of 4 seeds, a dog of some kind on all 4 | found | plan 18, commit `f7981bc0` | 10 |
| One-species classifier pushed on cat × dog: two animals in 0 of 16 tiles; the fused animal turned into a cat on 3 of 4 seeds | found | plan 18, commit `6be7d0c5` | 10 |
| A classifier trained on cat and dog separately only re-weights the product; the push that can fix it must come from a classifier trained on the pair | designed | the Bayes step in [Quick context](#quick-context-where-you-are); plan 18's 0 of 16 fits it | 7, 8, 9, 10 |
| Plan 15's head is two sigmoid outputs trained with a rule that refuses to call a fused animal "both", which makes it pair-aware even though its outputs are per species | found | plan 15's piece 1 close line, its walk map | 10 |
| Plan 15's labelling run finished on 2026-11-04 (elapsed 298.8 s); its walk map still reads "in flight" | found | `/datasets/mmolefe/poe_repair_min/outputs/presence_head/logs/label_set.log` | 10 |
| The recommended prompts inside plans 15, 17 and 18 point at `1-lora-correction/plans/06-...` paths that no longer exist | found | `ls` on 2026-11-16 | 10 |
| Plain-product best-of-16 on cat × dog composed 0 of 128 particles; with the adapter at strength 0.5, best of 16 is 8 of 8 | found | `report/00-INDEX.md` lines 155 and 156 | 10, later |
| GroundingDINO counts animals but cannot name them; CLIP on the detector's padded crop names them, 64 of 64 at median margin 0.990, untested on overlapping boxes | found | plan 15 quick context, `3-understanding-compositionality/scripts/census/crop_classifier_cat_vs_dog.py` | 8, 10 |
| Exact toy at pair weight 0.01: exact product 300 of 300 crisp pairs, reverse diffusion 0 of 300 | found | the [toy card](../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md) | 7 |
| On the toy the product itself composes and the sampler fails, so a discriminator learns log q_joint minus log q_sampler, which differs from the true noised product; the toy reports cosine against both | designed | the toy card, "[Where the samplers lose the pair](../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md#where-the-samplers-lose-the-pair)" | 7 |
| The authors' code is github.com/openai/guided-diffusion; its README lists a class-conditional 64 px model and classifier and an unconditional 256 px model with a 256 px classifier | stated | the guided-diffusion README, checked at step 4's open | 4, 8 |
| The released 64 px model has no unconditional mode, so a product with a null expert needs the 256 px unconditional model, or a model we train with a null label | designed | follows from the row above | 8 |
| The [Reduce, Reuse, Recycle copy](../../environment/reduce-reuse-recycle-code.md#the-clevr-script) of the guided-diffusion U-Net lacks the upstream line `h = x.type(self.dtype)`; use the upstream repository | found | `environment/reduce-reuse-recycle-code.md` line 47 | 4, 8 |
| The inclusive-captions photo pool holds 550 photographs (200 cat-only, 200 dog-only, 150 both); its manifest is on the cluster; the pilot that uses it is another session's live work | found | `4-inclusive-captions/artifacts/drips/inclusive-captions-pilot/EXECUTE_WALK.md`, the [pool card](../../4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md) | 8 |
| `biggpu` allows one job per user, so long runs start with `nohup` outside Slurm; `bigbatch` caps at 6 jobs per user; large outputs go to `/datasets` only | found | `environment/hpc/nodes.md`, project memory | 4, 5, 8, 10 |
| Seeds 9 to 16 are never trained on; the cat × dog joint prompt is wrong on 4 of them | found | project memory, `report/` eval-reference finding | later |
| v58 pool: 182 cells over 48 pairs; 23 of 53 held-out pairs share no concept with training | found | `1-lora-correction/report/paper-review/held-out-set.md` | later |
| Classifier guidance has a learning journey, gated on its picture set: deep-learning/diffusion-models/13-classifier-guidance-by-hand/ (GATE.md) | found | JOURNEYS.md row 31, 2026-11-16 | 2, 7 |
| A straight-line projection in DINOv2 space failed as a cross-pair tool (clouds as wide as the gaps) | found | `1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md` | 7 |
| Discriminator guidance (Kim et al.) is not in the [reading register](../../plans/standing/literature/reading-register.md) | found | `plans/standing/literature/reading-register.md` | 3 |
| The classifier push enters after the combine forms the product's prediction and before the DDIM step, fed by a separate-label or a pair-aware classifier: [the classifier guidance diagram](https://claude.ai/artifact/FBbwapTjp1dbEmzpLspRBT) | found | `poe_repair/methods/_classifier_guided.py:120-171` | 9, 10 |
| The released guided-diffusion weights sit at `/datasets/mmolefe/poe_repair_min/weights/guided-diffusion/` (`256x256_diffusion_uncond.pt`, `256x256_classifier.pt`, `64x64_diffusion.pt`, `64x64_classifier.pt`, sizes matching the server); the clones at `/home/molef/PhD/replications/guided-diffusion` and `/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion`, upstream commit `22e0df8` | found | [the reproduce card](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md), 2026-11-17 | 7, 8 |
| Sampling the released model needs neither mpi4py nor blobfile: a wrapper that imports `script_util` and loads weights with `torch.load` runs `classifier_sample.py`'s own `cond_fn` in co3 | found | `2-classifier-guidance/scripts/guided_diffusion_reproduce.py` | 7, 8 |
| On the released weights, 250 respaced steps of ancestral sampling: scale 0 is byte-identical to plain sampling on 4 of 4 seeds; at scale 10 the released classifier calls 16, 15 and 15 of 16 samples corgi, tabby and golden retriever, against 2, 2 and 5 at scale 1; a render takes 22 s plain and 37 s guided on a Quadro RTX 8000 | found | [the reproduce card](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md), jobs 89232 and 89235 | 7, 8 |
| By a strict eye at 256 px, scale 10 gives 9, 11 and 10 of 16 clear corgi, tabby and golden retriever, against the classifier's 16, 15 and 15; all 22 disagreements over 96 tiles are the classifier calling a fox face, a beagle, a Labrador or garbled fur the asked-for class | found, eye read `inferred` | `your_call` in [the per-tile calls](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/classifier_calls.csv), instruction 13 | 8 |
| The push at scale 10 moves the step's mean by under 0.1% of its norm, largest in the first fifth of the run; the classifier's raw gradient falls once the class is reached and keeps rising at scale 1 | found | `push_per_step_c263_s10_seed0.json` in [the reproduce card](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/README.md) | 5, 7 |
| mscluster111 and mscluster112 (RTX PRO 6000 Blackwell) both work under `co3_bw`: torch 2.11.0+cu128 sees the card, a 1000 × 1000 matmul sums to 1e9, and `/datasets` is visible over SSH; mscluster110 held another user's 67.8 GB job; mscluster107 card 1 was free; mscluster109 did not answer SSH. The environment notes still list 111 and 112 as faulted | found | live probe 2026-11-17 over `scripts/cluster.sh sh` | the rung compute plans |
| On a Blackwell a plain product render takes 5.7 to 5.9 s and a joint render 5.0 to 5.2 s, against 51.5 s per product render on a Quadro RTX 8000, about 9 times faster; no guided SDXL render has been timed on any card | found | `1-lora-correction/report/paper-review/prze-10-compute.md`, plan 15's render cost | the rung compute plans |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have written | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 classifier guidance on the released weights | own | nothing typed; the guided mean read line by line in their `gaussian_diffusion.py` | `ε − σ_t ∇ log p(y \| x_t)` on one step's vectors, and the missing term from Bayes | the corgi push at scale 1 and 10, one step's numbers | the submerge journey's derivation problems | three questions after stage 1 |
| 2 the exact toy | build | the toy's guided step and the discriminator's loss and log-odds gradient, typed by you | the best discriminator q1/(q1+q2) on two 1-D Gaussians, and its log-odds slope | one draw's push against the true correction at one step | the separate-label push shown to equal a heavier product, by hand | a full page after stage 2 |
| 3 our photos, both ways | own | nothing typed; the label-dropout change read as a diff | where the null expert comes from in each arm | the product's prediction for one 64 px image, with and without the push | none | three questions after stage 3 |
| 4 SDXL cat × dog | gloss | nothing typed | the stage ladder with each stage's count | plan 15's best weight against plain product | none | the retrospective's predictions |
| **Outcome** | | **four figure sets with cards, findings and pass or miss lines** | | | | **you explain why a separate-label classifier cannot fix the product, without notes** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-ladder) ➡️

▶ **Recommended:** `1 hld-diagram ‖ 2 submerge (background) → 3 drip-read --lean ‖ background: 4's
downloads → 4 drip-execute-plan --reproduce ‖ background: 6 prompt-storyboard → render-diagrams →
5 drip-execute-plan --request (sampler repair) ‖ background: the canary → 7 drip-execute-plan
--request (toy) ‖ background: quiz-prep-pulse, quiz-cadence → 8 drip-execute-plan --request (photos,
both arms) ‖ background: the 64 px training → 9 frame-hypothesis → 10 drip-execute-plan (plan 15) →
11 drip-retro, pressure-test → re-route` (laid on the research spine: read, reproduce, rebuild
small, state the test, test it scale by scale, explain; proposed for the photo stage, where no
curated flow trains a model of our own)

<details><summary>Alternatives</summary>

`3 drip-read --lean → 7 toy → 9 frame-hypothesis → 10 plan 15`: better when the SDXL answer is
needed within two weeks and the released-weights and photo stages can be argued from the toy alone.

`drip-idea --hypothesis → hypothesis-to-scope → populate-plans → drip-execute-plan`: better when a
supervisor needs the full scope written before any run.

</details>

## The ladder

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

**Each stage changes one thing.** Stage 1 checks that classifier guidance works at all, on the
authors' own weights. Stage 2 moves to a toy where the true correction is known. Stage 3 moves to real
photos of cats and dogs at small size, once with the authors' frozen model and once with a model we
train. Stage 4 moves to SDXL. Stage 3 to stage 4 changes the model, the size and the evaluator at
once, which the step to a pretrained text-to-image model cannot avoid.

**Every stage below SDXL runs two classifiers.** A separate-label classifier (cat present, dog
present), which the maths says only re-weights the product, and a pair-aware one (two separate
animals against the product's own draws), which is the claim. Stage 4 uses plan 15's pair-aware head,
with plan 17 and 18's species-only pushes as the control.

**Stage 3 runs both arms, because they answer different questions.** The released 256 px
unconditional model with classifiers trained on our photos asks whether the push works on a model
trained on 1.28 million images it never saw ours in. A 64 px model trained by us on our 550 photos,
with a null label 10% of the time so the product can be formed, asks it on a model whose data we
control. That model is trained twice: once without the 150 both-animal photos, so the product never
saw a pair, and once on all 550.

**Every classifier stage diagnoses its classifier beside the guided sweep, never holding it back:**
the score gap on minimal pairs (a fused animal against two separate ones) at every noise step, and
the cosine of the push with the correction (the true correction on the toy, the joint prompt's
prediction minus the product's on SDXL). A sweep that composes while the minimal-pair gap fails is
reported as composing for the wrong reason.

**A missed bar stops the climb.** The stage's walk opens its diagnosis ladder (diagnose, fix,
recheck, half the stage's box, at most two levels) and the next stage is not started until it
closes.


**The scale-up ladder for fixing the product.** Each rung is the proof of concept for the next and
changes one thing: rung 1 the exact toy (A1); rung 2 SDXL cat × dog judged on held-out seeds 9 to 12
with a small discriminator (A3q); rung 3 the pair kind, animal and animal, animal and object, object
and object (K1 to K3); rung 4 more seeds (A4), more pairs (C1), unseen pairs (C2). The rungs, their
rows and the condition for going on are in [the board's ladder](../report/classifier-guided-poe/board.md#the-scale-up-ladder).
The photos (A2r, A2c) and the head ablations (B1 to B7) run beside it and gate nothing. Every
rung's compute plan is written before launch, on the quickest card that can run it.

`pending: the rung step rounds of the 2026-11-17 reopening (each rung's planning skill, compute step and bar)`

## The two lanes

Navigation: ⬅️ [Previous](#the-ladder) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

The AI builds each stage. You read the method, own the maths by hand, type the toy's guided step and
call the pictures. The lanes meet at stage 2's close, where your typed discriminator and the AI's run
on the same draws.

| Stage | The AI lane builds | Your lane learns | Quiz (built in the background) | Confidence bar | You rebuild | Compare on |
|---|---|---|---|---|---|---|
| 1 | the guided-diffusion setup, the corgi and cat and dog sheets, the sampler repair | the submerge journey on classifier guidance, the Kim et al. read | `quiz-prep-pulse` then `quiz-cadence` on the journey's plans | every needed-now question right and 80% of the rest (estimate) | the guided mean on paper, from Bayes | the released classifier's call on one sample at scale 1 and 10 |
| 2 | exact draws, both classifiers, the guided sweep, the per-step cosine | the best discriminator and its log-odds, by hand | `quiz-prep-pulse` then `quiz-cadence` on the stage 2 plan | the same | the toy discriminator's loss and its log-odds gradient | the same 300 toy draws at pair weight 0.01 |
| 3 | the photo set, both arms, both classifiers, the sheets | why the null expert matters | three questions | the same | none | none |
| 4 | plan 15's head and sweep on the repaired sampler, the control columns | the stage ladder, predicted before each reveal | the retrospective's predictions | none | none | none |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Two-hour sittings, numbered, no dates yet. Only stage 1 is scheduled in detail; the rest is an
estimate re-cut at each stage's close.

| Block | Run | You, while it runs | Checkpoint at the end of the block |
|---|---|---|---|
| sitting 1 | step 1, step 2's open, step 3 begins ‖ background: step 4's clone and checkpoint downloads, step 2's journey build | the read | the design drawn, the journey's diagram set approved, the read's spine and first section |
| sitting 2 | step 3 to its close ‖ background: step 6's storyboard and renders | the read | the read map's "What the first stage takes from this" written; instruction 12 drawn |
| sitting 3 | step 4, targets 1 and 2 | the journey's first plan while samples render | strength 0 identical; the corgi sheet called (instruction 13) |
| sitting 4 | step 4, target 3; step 5 ‖ background: the canary | the journey's second plan | the cat and dog sheet called; the SDXL sampler capped and the canary identical |
| sitting 5 | stage 1 quiz, then step 7 opens | none | quiz passed at the bar; step 7's bars confirmed |
| sittings 6 to 9 | step 7 (estimate) | none | the toy's pieces built and tested, about one per sitting; the bar printed |
| sittings 10 to 14 | step 8 (estimate) ‖ background: the 64 px training, about 12 to 24 hours (estimate) | the stage 2 quiz while the model trains | both arms' sheets called (instruction 14) |
| sitting 15 | step 9 | none | the SDXL criterion written and ticked (instruction 15) |
| sittings 16 to 18 | step 10 (estimate) | none | plan 15's sweep called at full resolution (instruction 16) |
| sitting 19 | step 11 | none | the retrospectives walked; the re-route printed |

Step 7 is the step most likely to overrun. If sitting 8 ends with fewer than three pieces tested,
sitting 9 opens with `faster`, which costs the build tier on the remaining pieces.

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. 🗺️ Draw where the classifier's push enters the product

You ~10 min (estimate) · AI ~15 min · Sonnet, medium · moves: stage 1's drawing

**Objective:** the existing sampler design gains one box for the classifier's push, beside the
methods already drawn.
**Outcome:** the "FKC on our PoE sampler" design republished at its own URL with the new box.
**Serves:** every later step's `draw` control, which extends this page instead of starting one.
**Sees:** [FKC on our PoE sampler](https://claude.ai/artifact/RZynLjqAMebuiV3nTFC5CN) · **Ticket:**
none · **Source page:** none

```text
/hld-diagram /home/molef/PhD/replications/published-methods-on-our-pairs

Background
The design "FKC on our PoE sampler" already shows where each published method touches the noise
prediction of the product sampler, at https://claude.ai/artifact/RZynLjqAMebuiV3nTFC5CN.
This route adds classifier guidance to the same sampler, so it is drawn on that page.

Objectives
1. Add one box for the classifier's push: its gradient enters as minus s times sigma_t times
   the gradient of log c(x_t), after the product's prediction is formed.
2. Show the two kinds of classifier as two inputs to that box: separate-label and pair-aware.
3. Republish to the same URL and save the SVG beside the earlier versions.

What to read
- The registry entry for "FKC on our PoE sampler" in
  ~/.claude/skills/hld-diagram/references/registry.json.
- 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md, the claim in
  Quick context.
- poe_repair/methods/_classifier_guided.py, the guided samplers as they are.

What is already known
- The push is added unrescaled; the rescaled form was about 44,000 times too large (plan 18).
- A separate-label classifier only re-weights the product (the route's Quick context).

When something is missing
- If the spec for that page is not on disk, ask whether to draw a new page; the fallback is a new
  page that links to the old one.

Rules
- Draw only; change no code.
- Stop and show the new box before republishing.

What to hand back
- The republished URL, the saved SVG path, and one route row naming the URL.

What you will see
The new box, shown once for a yes, then the URL. Done when the page shows the push box with both
classifier inputs.
```

- `added:` the existing page's URL (from the registry) · `dropped:` nothing
- [x] **Gate:** the push box is on the page · ✓ verified 2026-11-17: [the classifier guidance diagram](https://claude.ai/artifact/FBbwapTjp1dbEmzpLspRBT) carries it, the "found" row in What we know so far
- **Leaves:** the page URL, which steps 4, 7, 8 and 10 extend through their `draw` control.

### 2. 📚 Start the classifier-guidance journey

You ~10 min for the open (submerge, 4 runs, median 9.3) plus ~60 min of the journey's plans in the
gaps of sittings 3 and 4 · AI ~10 min, then the journey build in the background · Opus, high ·
moves: stage 1's tier

**Objective:** a journey that teaches classifier guidance to the own tier, with its diagram set
approved.
**Outcome:** a registered journey under `~/goal-setting/learning/deep-learning/` and its plan tree.
**Serves:** your lane for stages 1 and 2; needed by step 7 for the maths you type.
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/submerge I want to own classifier guidance in diffusion models by hand, well enough to say why a
classifier trained on cat and dog separately cannot fix a product of experts and a pair-aware one
can

Background
The project at /home/molef/PhD/concept-composition/2-classifier-guidance tests whether a
classifier's gradient, added to the product-of-experts sampler, fixes composition on SDXL. No
learning journey teaches classifier guidance yet; journey_lookup.py reports it as a gap. The
reader works at the own tier: full steps by hand, a worked example, problems, a quiz.

Objectives
1. Derive how the gradient of log p(y | x_t) enters the noise prediction, from the score of the
   noised posterior, in one move per line.
2. Derive why the product of experts equals separate-label classifier guidance, from Bayes.
3. Derive why the best discriminator's log-odds gradient is the score of the density ratio.
4. Show the dropped term, log p(cat, dog | x) minus the two single terms, on a small table.

What to read
- Dhariwal and Nichol 2021, arXiv 2105.05233, section 4.
- Kim et al. 2023, arXiv 2211.17091, section 3.
- The route's Quick context, in
  2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.
- /home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md,
  the product-of-experts rule from Bayes, which this journey leans on.

What is already known
- Density ratio has a shelf page: The little book of generative AI foundations, p162.
- The product-of-experts journey above is 0 of 6 ticked; link it, do not re-teach it.

When something is missing
- If a derivation needs a result no source states, mark it as the journey's own and say so.

Rules
- Stop at the diagram set and wait for approval before the plan tree.
- The journey is registered in JOURNEYS.md.

What to hand back
- The journey path, its approved diagram set, its plan tree, and one route row naming the path.

What you will see
The diagram set for approval, then the plan tree. Done when the journey is registered and its
first plan opens on the derivation of the guided prediction.
```

- `added:` objectives 2 and 4, the separate-against-pair-aware contrast (from the route's Quick
  context) · `dropped:` nothing
- [x] **Gate:** the diagram set approved · ✓ verified 2026-11-17: `~/goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/GATE.md` records the ten pictures approved on 2026-11-16
- **Leaves:** the journey path, which step 7's learning path names.

### 3. 📄 Finish the Kim et al. read, lean

You ~75 min to finish, ~150 min budget at 2× (estimate; the one measured lean session took 38 min)
· AI ~10 min read-ahead · Opus, high · moves: stage 2's recipe

**Objective:** the read map's closing section names the discriminator's inputs, loss, noise
levels and the sampler formula, each with its page.
**Outcome:** the read map, closed on "What the first stage takes from this".
**Serves:** needed by step 7 for the discriminator recipe and by step 9 for the claim.
**Sees:** none · **Ticket:** none · **Source page:** arXiv 2211.17091

```text
/drip-read --lean 2211.17091

Background
This lean read is open at the abstract in
2-classifier-guidance/artifacts/drips/read-refining-generative-process-discriminator/READ_MAP.md;
resume it there. The paper is Kim et al., "Refining Generative Process with Discriminator Guidance
in Score-based Diffusion Models" (ICML 2023). Its method is the route's pair-aware classifier: a
discriminator trained on real data against the model's own samples, whose log-odds gradient
corrects the score. The route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.

Objectives
1. Read the paper in its own order, one paragraph per round, from the abstract.
2. Say how the discriminator is trained: inputs, noise levels, time input, features and loss.
3. Say how its gradient enters the sampler, with the weight and schedule the paper uses.
4. Derive why the best discriminator's log-odds gradient is the score of the density ratio.
5. Name each result the paper reports that the exact toy can check, with its number.

What to read
- The paper, whole.
- 3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md,
  the toy it meets first.
- The route's Quick context and What we know so far.

What is already known
- Classifier guidance steers SDXL here only with the gradient unrescaled (plan 18).
- The discriminator learns log q_joint minus log q_sampler, and on the toy q_sampler is not the
  true noised product (the toy card, "Where the samplers lose the pair").
- A separate-label classifier only re-weights the product (the route's Quick context).

When something is missing
- If the recipe needs pretrained features the toy lacks, say what would replace them.

Rules
- Read only. Write no code and change no plan.
- Every number keeps its unit and what it measures.
- Filing: tied, because the scope's plans will hold what this read decides.

What to hand back
- The read map, closing on "What the first stage takes from this": the training recipe, the
  sampler change and the checks the toy can run, each with its page.
- One row per settled fact for the route's What we know so far table.

What you will see
One paragraph per round, a prediction asked before the derivation, then the closing section.
Done when that section names the inputs, loss, noise levels and sampler formula, each with its
page.

Controls: the derivation round, prove (the best discriminator is q1 / (q1 + q2)); the training
round, example (one toy draw labelled real or sampled, at one noise level)
Side routes: the derivation round, /picture-speak --derive D* = q1/(q1+q2) to the log-ratio score
Learning path: classifier guidance, the step 2 journey; density ratio, no journey yet, expect a gap
line, shelf page p162
```

- `added:` the resume position (from the read map) · `dropped:` the arXiv id check (why: the read
  map already confirmed the paper)
- [x] **Gate:** the read map's closing section exists before step 7 · ✓ verified 2026-11-17: "What A1 takes from this" in `2-classifier-guidance/artifacts/drips/read-refining-generative-process-discriminator/READ_MAP.md`
- **Leaves:** the read map, which steps 7 and 9 read.

### 4. 🔁 Reproduce classifier guidance on the released weights

You ~135 min to finish, ~270 min budget at 2× (3 targets at 45 min, estimate; the two measured
reproduce sessions took 12 min each) · AI ~120 min of sampling, ~30 min of downloads (estimate) ·
Sonnet, medium · moves: stage 1

**Objective:** classifier guidance on the authors' released 256 px unconditional model and
classifier draws the asked-for class, and does nothing at strength 0.
**Outcome:** three sheets and a card under
`2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/`.
**Serves:** the board's stage 1 row (classifier guidance reproduced), and the code path steps 7 and
8 port.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** arXiv 2105.05233, section 4

```text
/drip-execute-plan --reproduce 2105.05233

Background
Dhariwal and Nichol, "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021), add a noisy
classifier's gradient to a frozen diffusion model's prediction. Their code and checkpoints are at
github.com/openai/guided-diffusion. This project wants to add the same push to SDXL's product of
experts, so it first checks the push works on the authors' own weights. The project is
/home/molef/PhD/concept-composition, folder 2-classifier-guidance; the route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.

Objectives
1. Clone the repository read-only into /home/molef/PhD/replications/guided-diffusion and its
   cluster twin, and download the 256 px unconditional model, the 256 px classifier, the 64 px
   model and the 64 px classifier.
2. Check that classifier scale 0 gives a render byte-identical to plain sampling, on 4 seeds.
3. Reproduce the paper's corgi figure: the unconditional model pushed to "Pembroke Welsh corgi"
   at classifier scale 1.0 and 10.0, 16 seeds each.
4. Run the same at scale 1.0 and 10.0 for one cat class (tabby) and one dog class (golden
   retriever), 16 seeds each.
5. Write the card with the three sheets, the released classifier's call on every sample, and
   where the push enters their sampler (`condition_mean` for ancestral sampling,
   `condition_score` for DDIM).

What to read
- The paper, section 4 and its corgi figure.
- The repository README, which lists the checkpoints and the sampling commands.
- environment/overview.md and environment/hpc/nodes.md, for the interpreter, the GPUs and the
  disk rule.
- environment/reduce-reuse-recycle-code.md line 47, on a broken copy of this U-Net.

What is already known
- The README lists a class-conditional 64 px model and classifier and an unconditional 256 px
  model with a 256 px classifier; confirm this at the open.
- The Reduce, Reuse, Recycle copy of this U-Net lacks the upstream line
  `h = x.type(self.dtype)`, so use the upstream repository only.
- Python on the cluster is /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
- biggpu allows one job per user, so long sampling starts with nohup on a free card.
- Large files go under /datasets/mmolefe/poe_repair_min/ only; /home-mscluster once filled up and
  killed checkpointing.
- From the laptop, cluster commands go through scripts/cluster.sh, which refuses uncommitted
  tracked changes.

When something is missing
- If mpi4py will not install in co3, stub the repository's dist_util for one GPU and say so.
- If the cluster has no place for the clone, ask the reader; the fallback is
  /home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion.
- If 250 sampling steps do not fit the box, drop to 50 respaced steps and say so on the card.

Rules
- Work on a new branch, exec/classifier-guidance-ladder, cut from main.
- Do not edit the upstream repository; wrappers live in 2-classifier-guidance/scripts/.
- Stop after the identity check and after each sheet, and show it before the next target.
- Filing: on, so picture-cadence files each sheet and record-cadence writes the recipe.

What to hand back
- The sheets and card at the path above, with the per-sample classifier calls in a CSV beside
  them.
- The outcome board 2-classifier-guidance/report/classifier-guided-poe/board.md: the stage 1 row's
  outcome lines ticked with links, redrawn with --step 4.
- Route rows: the checkpoint paths, the clone paths, the step count used, each target's result.

What you will see
The target list at the open, then one target per round, each with its sheet. At the end, a closing
report: what was asked, what was done with paths, pass or miss per target with the counts, what is
open, and at most one question.

Controls: the open, spell-it-out; the identity target, prove (the checksums side by side); the
corgi target, example (one step's raw gradient size and applied push), plumb (the sampling script
on the cluster), draw (once the model, the classifier and the sampler are named); every target,
debug; the close, test
Side routes: the corgi target, /picture-speak --take-apart the guided mean, mu plus s times Sigma
times the classifier gradient; the cat and dog target, /picture-speak --consequence classifier
scale 0, 1 and 10 on one sample
Debug kit: the identity target, the repository's classifier_sample.py at scale 0 (launch entry
"guided-diffusion scale 0", reads /datasets/mmolefe/poe_repair_min/weights/guided-diffusion,
writes /datasets/mmolefe/poe_repair_min/outputs/guided_diffusion_reproduce); the corgi target, the
same at scale 10 (launch entry "guided-diffusion corgi scale 10", same folders)

Judged by you from the sheets: done when scale 0 is byte-identical on 4 of 4 seeds, and at scale
10 the released classifier calls at least 14 of 16 samples the asked-for class for each of corgi,
tabby and golden retriever, with scale 1 calling fewer.
```

- `added:` the cat and dog classes, the 64 px downloads for step 8, the branch, the cluster facts
  (from `environment/`) · `dropped:` FID and the paper's tables (why: 50,000 samples per number,
  which buys nothing the sheets do not show for this route)
- **Targets and bars,** level 1 (the released weights) throughout, set before any run. Arxiv
  2105.05233 at upstream commit `22e0df8`; 250 respaced steps of ancestral sampling, one sample
  per seed so seed i starts from the same noise at every scale. Calls are the released 256 px
  classifier's top-1 at t = 0 on the saved render.

  | Target | Bar: reproduced | Not reproduced | Inconclusive |
  |---|---|---|---|
  | T1 scale 0 against plain sampling, seeds 0 to 3 | sha256 of the uint8 render equal on 4 of 4 | any seed differs while the control matches | plain sampling run twice on seed 0 differs (the GPU, not the push) |
  | T2 [Figure 3](https://arxiv.org/pdf/2105.05233#page=8), Pembroke Welsh corgi (class 263) | at scale 10 at least 14 of 16 called 263, and fewer at scale 1 | 13 or fewer at scale 10, or scale 1 calls as many | the eye count in instruction 13 disagrees with the classifier by 4 or more of 16 |
  | T3 tabby cat (281) | as T2 | as T2 | as T2 |
  | T4 golden retriever (207) | as T2 | as T2 | as T2 |

  Not chosen: Figures 1, 2 and 4 to 27 and Tables 1 to 16 (FID, IS, precision and recall from
  50,000 samples, architecture ablations, galleries, LSUN). Launch: the wrapper
  `2-classifier-guidance/scripts/guided_diffusion_reproduce.py` through
  `2-classifier-guidance/scripts/guided_diffusion_reproduce.sbatch identity` or
  `... guide "<classes>" "1.0 10.0"`, weights at `/datasets/mmolefe/poe_repair_min/weights/guided-diffusion`,
  clones at `/home/molef/PhD/replications/guided-diffusion` and
  `/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion`. The wrapper never
  imports `dist_util`, so co3 needs neither mpi4py nor blobfile.
- [x] **T1** scale 0 identical to plain sampling · ✓ verified 2026-11-16: reproduced, sha256 of the uint8 render equal on 4 of 4 seeds, and plain sampling run twice on seed 0 equal (job 89232, mscluster108, Quadro RTX 8000, 22 s per plain render, 37 s at scale 0) · [the identity sheet](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-identity-scale-0-against-plain.png), [the checksums](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/identity_checksums.csv)

  <a href="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-identity-scale-0-against-plain.png"><img src="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-identity-scale-0-against-plain.png" width="200" alt="Four renders from plain sampling above four renders at classifier scale 0 for seeds 0 to 3 (a frog, a white bird, gazelles, a baguette in foil), each pair with the same checksum and marked identical"></a>

- [x] **T2** corgi sheet called · ✓ verified 2026-11-16: reproduced, the released classifier calls 16 of 16 Pembroke at scale 10 (mean p 0.854) and 2 of 16 at scale 1 (mean p 0.067); by eye 13 of 16 clear corgis at scale 10, seeds 2, 6 and 11 corgi-coloured with a fox or dingo face (job 89235) · [the corgi sheet](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-pembroke-welsh-corgi-scale-1-and-10.png), [the calls](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/guide/classifier_calls_263.csv)

  <a href="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-pembroke-welsh-corgi-scale-1-and-10.png"><img src="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-pembroke-welsh-corgi-scale-1-and-10.png" width="200" alt="Sixteen renders at classifier scale 1 on the left, mostly other animals and objects framed red, beside sixteen at scale 10 on the right, all orange and white corgi-like dogs framed green"></a>

- [x] **T3** tabby sheet called · ✓ verified 2026-11-17: reproduced, 15 of 16 called tabby at scale 10 (mean p 0.737) and 2 of 16 at scale 1 (mean p 0.066); the one miss, seed 8, is an ornate object the classifier calls pickelhaube; by eye 12 of 16 clear tabbies at scale 10 (nohup run on mscluster107 card 1) · [the tabby sheet](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-tabby-cat-scale-1-and-10.png), [the calls](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/guide/classifier_calls_281_207.csv)

  <a href="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-tabby-cat-scale-1-and-10.png"><img src="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-tabby-cat-scale-1-and-10.png" width="200" alt="Sixteen renders at classifier scale 1 on the left, mostly other animals and objects framed red, beside sixteen at scale 10 on the right, striped tabby cats framed green except one ornate object at seed 8"></a>

- [x] **T4** golden retriever sheet called · ✓ verified 2026-11-17: reproduced, 15 of 16 called golden retriever at scale 10 (mean p 0.907) and 5 of 16 at scale 1 (mean p 0.195); the miss, seed 4, is called Labrador retriever; by eye 14 of 16 clear goldens at scale 10 (same run) · [the golden retriever sheet](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-golden-retriever-scale-1-and-10.png), [the calls](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/guide/classifier_calls_281_207.csv)

  <a href="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-golden-retriever-scale-1-and-10.png"><img src="../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-golden-retriever-scale-1-and-10.png" width="200" alt="Sixteen renders at classifier scale 1 on the left, a mix of dogs, animals and objects, beside sixteen at scale 10 on the right, golden retriever faces and bodies framed green except a Labrador-like dog at seed 4"></a>

- [x] **Gate:** instruction 13 called · ✓ 2026-11-17, by Claude at your request, veto open: see instruction 13
- **Leaves:** the card, the checkpoint paths, and the clone, which steps 7 and 8 port from.

### 5. 🔧 Repair the SDXL product sampler's push and rerun the canary

You ~90 min, ~135 min budget at 1.5× (2 pieces at 45 min, estimate) · AI ~30 min plus the canary
on the cluster · Sonnet, medium · moves: the sampler repair

**Objective:** the product sampler pushes unrescaled, capped at a fraction κ of the product's own
step, and at strength 0 matches plain product byte for byte.
**Outcome:** the repaired sampler and the canary's identical checksums.
**Serves:** the board's sampler repair row; needed by step 10.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** plan 17 task 2.1

```text
/drip-execute-plan --request "Repair the classifier push in the SDXL product sampler"

Background
The SDXL product sampler in poe_repair/methods/_classifier_guided.py still rescales the classifier
push to the size of the product's own prediction (line 112). That form was about 44,000 times too
large. Plan 17's run_cfg_classifier_guided already uses the paper's unrescaled form and passed its
strength-0 identity check. The project is /home/molef/PhD/concept-composition; the route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.

Objectives
1. Replace the rescaled push in the product sampler with epsilon minus s times sigma_t times the
   classifier gradient, the form plan 17 uses.
2. Cap the push at a fraction kappa of the product's own step, with kappa a named constant.
3. Log the raw gradient size and the applied push at every step.
4. Rerun the canary: strength 0 against plain product on cat × dog seed 9.

What to read
- poe_repair/methods/_classifier_guided.py.
- 2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md, tasks 2.1 and 2.2.
- 2-classifier-guidance/scripts/presence_head_canary.py.
- environment/overview.md, for the fp16 upcast rule.

What is already known
- Plan 17's form passed byte identity on four seeds on 2026-11-04.
- A fp32 against fp16 combine differs by about 3 grey levels over 50 steps, so the identity check
  compares against a reference made in the same loop and precision.
- biggpu allows one job per user; the canary runs as an sbatch job on an idle biggpu node
  (presence_head_canary.sbatch), and with nohup on a free card only when no node is idle.

When something is missing
- If kappa has no value from the paper, use 0.1 and say so; step 9's criterion sets the final one.

Rules
- Work on branch exec/classifier-guidance-ladder; cut it from main if step 4 has not.
- Launch with scripts/cluster.sh run after committing.
- Stop and show the diff before the canary runs.
- Filing: tied, because this is a repair whose record is the commit.

What to hand back
- The commit, the canary's checksums, and route rows for kappa and the identity result.
- The outcome board: the sampler repair row's lines ticked, redrawn with --step 5.

What you will see
The diff, then the canary's two checksums. A closing report with pass or miss.

Controls: the cap piece, example (one step's push before and after the cap), debug; the canary
piece, prove
Side routes: the cap piece, /picture-speak --consequence the cap switched off on one step
Debug kit: the canary piece, presence_head_canary.py (launch entry "canary strength 0"; seed 9 has
no cached latent on the cluster, so its noise comes from manual_seed(9), as in plan 17; writes
/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary)

Judged by the canary: done when the strength-0 checksum equals plain product's.
```

- `added:` the precision note (from project memory) · `dropped:` nothing
- [x] **T1** the product sampler pushes unrescaled, capped at `PUSH_CAP_FRACTION` 0.1 of the product prediction's norm, logging the raw gradient, the push before and after the cap, and whether it fired, every step · ✓ verified 2026-11-17: commit `445cef9f`; on the mscluster login node a push 0.871 the size of ε_PoE comes out at 0.100 with direction cosine 1.000000, a push of 0.051 passes untouched, and a nonzero `weight` raises
- [x] **T2** the canary at strength 0 against plain product, cat × dog seed 9 · ✓ verified 2026-11-17: plain, strength 0 and strength 0 with an untrained head share md5 `c291b259647d30a450e87b234423fe99`, 0 pixels differing (job 89341, mscluster108, Quadro RTX 8000, 1 min 41 s) · `/datasets/mmolefe/poe_repair_min/outputs/presence_head/canary/canary.json`
- [x] **Gate:** the checksums identical · ✓ verified 2026-11-17, by T2
- **Leaves:** the repaired sampler, which step 10 uses.

### 6. 🎨 Storyboard the claim and the ladder, and render it

You ~16 min (prompt-storyboard, 5 runs) plus ~11 min (render-diagrams, 17 runs) · AI ~25 min in
the background · Sonnet, medium · moves: stage 1's drawing

**Objective:** two to four rendered pictures: what the product drops, where the push enters, and
the four stages with their two classifiers.
**Outcome:** the pictures in `2-classifier-guidance/plans/diagram-prompts.md`, rendered.
**Serves:** the pictures step 7, step 8 and the paper reuse.
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/prompt-storyboard --subject the classifier-guided product of experts

Background
The route 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md states a
claim in its Quick context: the product of experts drops a term, a separate-label classifier only
re-weights the product, and a pair-aware classifier can supply the dropped term. The pictures are
for the reader and later for the paper.

Objectives
1. One picture of what the product keeps and what it drops, using cat, dog and a fused animal.
2. One picture of where the push enters the prediction, with both kinds of classifier.
3. One picture of the four stages with both classifiers at each.
4. Write the prompts into 2-classifier-guidance/plans/diagram-prompts.md in the shape of
   ~/.claude/DIAGRAM_PROMPTS_FORMAT.md, so render-diagrams can drain them.

What to read
- The route's Quick context and The ladder.

What is already known
- The pictures speak in what a render shows, never in metric names.

When something is missing
- If diagram-prompts.md does not exist, create it with the format's header.

Rules
- Prompts only; render nothing here.

What to hand back
- The diagram-prompts.md path and the count of prompts written.

What you will see
The prompts, then the file path. Done when the file holds them in the format's shape.
```

Then, in the background:

```text
/render-diagrams 2-classifier-guidance/plans/diagram-prompts.md
```

- `added:` writing the prompts into the file (proposed hand-off: prompt-storyboard does not write
  diagram-prompts.md unless asked) · `dropped:` nothing
- [ ] **Gate:** none; the renders land in the background.
- **Leaves:** the rendered pictures, embedded where the file's slots name them.

### 7. 🧪 Port the push to the exact toy, separate-label against pair-aware

You ~225 min to finish, ~450 min budget at 2× (5 pieces at 45 min, estimate) · AI ~60 min ·
Opus, high · moves: stage 2

**Objective:** on the exact toy at pair weight 0.01, the pair-aware push composes and the
separate-label push does not.
**Outcome:** the toy's figure set (crisp pairs of 300 for unguided, separate-label and pair-aware
pushes; cosine with both targets per step; the minimal-pair gap), its card and its finding.
**Serves:** the board's exact-toy row.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** Kim et al. section 3 via the
step 3 read map

```text
/drip-execute-plan --request "Port classifier guidance to the exact toy and test both classifiers"

Background
The exact toy in 3-understanding-compositionality has closed-form experts: mixtures of Gaussians
over 16 × 32 pictures of red circles and blue squares. At pair weight 0.01 the exact product draws
300 of 300 crisp pairs and reverse diffusion with summed scores draws 0. This walk ports the
authors' classifier-guidance code to that toy and tests two classifiers. The route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.

Objectives
1. Port the push from guided-diffusion's condition_score into the toy's sampler, and check that
   scale 0 reproduces the unguided draws exactly.
2. Train a separate-label classifier, circle present and square present, on noised toy pictures.
3. Train a pair-aware discriminator: exact pair draws against the summed-score sampler's own
   draws, at every noise level, with the step 3 recipe.
4. Guide the sampler with each push and count crisp pairs and overlays of 300.
5. Read each push's cosine per step against the true correction and against the sampler's own
   ratio, and run the minimal-pair gap beside the sweep.

What to read
- 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py.
- 3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md.
- The step 3 read map's closing section.
- The step 4 card, for where the push enters their sampler.

What is already known
- Exact toy at pair weight 0.01: exact product 300 of 300, reverse diffusion 0 of 300.
- On this toy the product composes and the sampler fails, so the discriminator learns log q_joint
  minus log q_sampler, which differs from the true noised product; report cosine against both.
- A separate-label classifier only re-weights the product (the route's Quick context).
- The push is unrescaled and capped at kappa times the product's own step (from step 5).

When something is missing
- If training does not separate the two classes, the diagnosis ladder opens.
- If the toy code lacks a per-step draw hook, add one and say where.

Rules
- I type the guided step, the discriminator's loss and its log-odds gradient myself.
- Stop at each piece's test and show its number.
- Filing: on, because this is the build being learned by hand.

What to hand back
- The figures and card under
  3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/,
  the finding in report/, the bar's pass or miss line, and route rows for each settled fact.
- The outcome board: the exact-toy row's lines ticked, redrawn with --step 7.

What you will see
The bars confirmed at the open, then one piece per round, each ending on a test I run. A closing
report with pass or miss and the three counts.

Controls: the port piece, prove (scale 0 against unguided, draw by draw); the training pieces,
build and break (type the loss, then flip a label); the cosine piece, example (one draw's push
against the true correction at one step); every piece that adds a command, debug; the close, test
Side routes: the separate-label piece, /picture-speak --consequence the product with and without
the separate-label push; the cosine piece, /picture-speak --derive D* = q1/(q1+q2) to the log-ratio
score
Debug kit: the training pieces, the toy classifier trainer the walk names (reads the exact draws,
writes the two classifier checkpoints under the card's folder)
Learning path: classifier guidance, the step 2 journey; density ratio, no journey yet, shelf p162;
product of experts,
/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md

Judged by the counts, with you confirming the bars at the open: done when the pair-aware push
draws at least 150 of 300 crisp pairs with overlays no higher than unguided, and the separate-label
push draws at most 15 of 300.
```

- `added:` the separate-label classifier as the control (from the corrected claim) · `dropped:` the
  linear-discriminant picture of the head's features (why: on a toy with known densities the cosine
  says more)
- [ ] **Gate:** the bar printed; a miss opens the diagnosis ladder and stops the climb.
- **Leaves:** the toy card and a retrospective, which steps 8, 9 and 11 read.

### 8. 🧪 Our photos, both ways: the released model, and a model we train

You ~270 min to finish, ~540 min budget at 2× (6 pieces at 45 min, estimate) · AI ~3 h plus the
64 px training, about 12 to 24 hours each run (estimate) · Sonnet, medium · moves: stage 3

**Objective:** on real cat and dog photos, the pair-aware push draws more seeds with two separate
animals than the separate-label push, in at least one of the two arms.
**Outcome:** both arms' sheets, the per-seed calls, the card and the finding.
**Serves:** the board's photo rows.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** guided-diffusion's
classifier_train.py and image_train.py

```text
/drip-execute-plan --request "Run classifier guidance on our cat and dog photos with the released
model and with a 64 px model we train"

Background
Steps 4 and 7 showed classifier guidance on the authors' weights and on the exact toy. This walk
uses their code on our own photos of cats, dogs, and cats with dogs, in two arms. The first keeps
their released 256 px unconditional model frozen and trains only classifiers on our photos. The
second trains their 64 px model on our photos with a null label, so a product of a cat expert and
a dog expert can be formed. The route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.

Objectives
1. Build the photo set from the inclusive-captions pool at 256 px and 64 px, with labels cat-only,
   dog-only and both, and print the counts.
2. Start the 64 px training in the background twice: once on cat-only and dog-only photos, once on
   all photos, each with the label dropped to null 10% of the time.
3. For the released model, train with classifier_train.py a separate-label classifier (cat
   present, dog present) and a pair-aware one (both-animal photos against the product's own
   draws).
4. Sample the released model on seeds 1 to 16: plain product of the two separate-label pushes,
   then with the pair-aware push added.
5. When the 64 px runs land, sample each on seeds 1 to 16: plain product, with the separate-label
   push, with the pair-aware push.
6. Build one sheet per arm, call every tile, and write the card.

What to read
- The pool card, which says how many photos hold each animal:
  4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md
- The step 4 card and the clone at /home/molef/PhD/replications/guided-diffusion.
- The step 7 card and its retrospective.
- environment/hpc/nodes.md and environment/hpc/throughput.md, before estimating the training time.

What is already known
- The pool holds 550 photographs: 200 cat-only, 200 dog-only and 150 both; its manifest is
  /datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/manifest.json.
- The released 64 px model is class-conditional with no unconditional mode, which is why arm 1
  uses the 256 px unconditional model.
- guided-diffusion trains class-conditional models but has no label dropout; adding it is a small
  change to the training loop.
- GroundingDINO counts animals but cannot name them; CLIP on the detector's padded crop names
  them, untested on overlapping boxes.
- Composition calls overcall on downscaled sheets, so clear pairs are called from full-resolution
  tiles, and a hybrid is its own category.
- biggpu allows one job per user, so the two trainings run with nohup, or one goes to bigbatch.

When something is missing
- If the manifest counts differ from 550, use the manifest and say so.
- If the 64 px training does not fit 3 days on one card, ask the reader whether to shrink the
  model or the steps; the fallback is half the channels.

Rules
- Read the pool only; the inclusive-captions pilot is another session's live work.
- Large outputs go under /datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/.
- Stop after the photo-set counts, after each training launch, and at each sheet.
- Filing: on, so picture-cadence files each sheet.

What to hand back
- The sheets, the per-tile calls as a CSV, the card under
  2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/, and the
  finding in report/.
- The outcome board: the two photo rows' lines ticked, redrawn with --step 8.
- Route rows: the training run ids and durations, and each arm's counts.

What you will see
The counts at the open, the launch lines with their logs, then one sheet per arm called tile by
tile. A closing report with pass or miss per arm.

Controls: the photo-set piece, example (one photo per label at both sizes); each launch, plumb and
debug; each sheet piece, prove (the tiles at full resolution); the close, test
Side routes: the label-dropout piece, /picture-speak --with-code the training loop's label line;
the sampling piece, /picture-speak --consequence the null expert switched off
Debug kit: the training pieces, image_train.py with label dropout (launch entry "64 px cat dog
no-both", reads the photo set at 64 px, writes the run folder under
/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos/), and classifier_train.py
(launch entry "pair-aware classifier 256", reads the both photos and the product's draws, writes
the classifier checkpoint beside it)

Judged by you from the full-resolution tiles: done when, in at least one arm, the pair-aware push
gives two separate animals on at least 4 more of 16 seeds than the separate-label push, with
pictures no more damaged than plain product's.
```

- `added:` the two training sets with and without the both photos, the null label (from the
  maths: a product needs a null expert) · `dropped:` pasted pairs (why: real both-animal photos
  exist in the pool)
- [ ] **Gate:** instruction 14 called.
- **Leaves:** the card, the finding and the retrospective, which step 9 reads.

### 9. ⚖️ State the SDXL test before it runs

You ~45 min (estimate) · AI ~10 min · Opus, high · moves: stage 4's bar

**Objective:** the SDXL stage's support, null and inconclusive lines, with numbers, in a review file.
**Outcome:** `2-classifier-guidance/plans/review/15-can-a-classifier-supply-the-missing-term.md`
with its bar ticked.
**Serves:** needed by step 10 for its bar.
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/frame-hypothesis a pair-aware classifier's gradient, added to SDXL's cat and dog product at each
step, draws two separate animals on more seeds than plain product, and a species-only classifier
does not

Background
Steps 4, 7 and 8 of the route
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md tested the claim on
the released weights, the exact toy and our photos. This writes the SDXL test before any SDXL run,
for plan 15 at 2-classifier-guidance/plans/15-can-a-classifier-supply-the-missing-term.md.

Objectives
1. State what supports the claim, what is a null and what is inconclusive, as counts out of 8.
2. Name the control columns: plain product, the joint prompt, and the species-only pushes from
   plans 17 and 18.
3. Write the review file for plan 15.

What to read
- The cards and retrospectives from steps 7 and 8.
- Plan 15's stop point and plans 17 and 18's results.

What is already known
- Plain product composes 0 of 8 on cat × dog; the joint prompt 8 of 8 (plan 15's stop point).
- The species-only dog push gave two animals in 0 of 16 tiles (plan 18).
- Results from steps 7 and 8, as their route rows record them.

When something is missing
- If step 8 missed on both arms, ask the reader whether SDXL still runs; the fallback is to stop
  the climb and walk step 11.

Rules
- Write no code.

What to hand back
- The review file path and one route row with the bar.

What you will see
The criterion one part per round. Done when the review file holds the three lines with numbers.
```

- [ ] **Gate:** instruction 15 ticked.
- **Leaves:** the review file, which step 10 reads.

### 10. 🧪 SDXL cat × dog on training seeds: plan 15 on the repaired sampler

You ~225 min to finish, ~450 min budget at 2× (estimate) · AI ~4 h on the cluster (estimate) ·
Sonnet, medium · moves: stage 4

**Objective:** plan 15's bar from its review file, printed pass or miss.
**Outcome:** plan 15's sweep sheet with control columns, card and finding.
**Serves:** the board's SDXL training-seeds row.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** none

```text
/drip-execute-plan 2-classifier-guidance/plans/15-can-a-classifier-supply-the-missing-term.md

Background
Plan 15 trains a two-output presence head on SDXL's frozen middle block, with labels that refuse to
call a fused animal "both", and sweeps its push on cat × dog. Its walk map
2-classifier-guidance/artifacts/drips/can-a-classifier-supply-the-missing-term/EXECUTE_WALK.md
still reads "in flight"; the labelling run finished on 2026-11-04. The route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md.

Objectives
1. Bring the walk map up to date: tasks 1.2 and 1.3 finished on 2026-11-04.
2. Run tasks 3.2 and 3.3 on the repaired sampler from step 5.
3. Run the sweep, tasks 4.1 to 4.3, with two control columns: plan 17's species head pushing cat
   and dog, and plan 18's classifier recipe for cat, dog and both.
4. Read the bar from the step 9 review file.

What to read
- Plan 15, its walk map and its review file.
- Plans 17 and 18, tasks 4.1.
- The step 5 commit and the cards from steps 7 and 8.

What is already known
- The sampler pushes unrescaled and capped (from step 5).
- The recommended prompts inside plans 15, 17 and 18 point at removed paths; use the paths here.
- GroundingDINO counts animals; CLIP on the padded crop names them.
- Composition is called from full-resolution tiles, with a hybrid as its own category.
- biggpu allows one job per user; the bigbatch cap is 6 jobs; outputs go under /datasets.

When something is missing
- If the cluster worktree poe_repair_min-classifier-guided is on the old branch, make a new
  worktree on exec/classifier-guidance-ladder and launch with CLUSTER_DIR set to it.

Rules
- Stop before each launch with the exact command.
- Filing: on.

What to hand back
- The sweep sheet, card and finding, the bar's pass or miss line, route rows, and the board's
  SDXL row ticked with --step 10.

What you will see
One task per round. A closing report with the best weight's count out of 8 beside plain product,
the controls and the joint prompt.

Controls: each launch, plumb and debug; the sweep, prove (the tiles at full resolution); the close,
test
Side routes: the sweep, /picture-speak --consequence the push switched off at each weight

Judged by you from full-resolution tiles against the step 9 review file.
```

- [ ] **Gate:** instruction 16 called.
- **Leaves:** plan 15's retrospective, which step 11 walks.

### 11. 🔁 Walk the retrospectives, attack the claim, re-route

You ~45 min (estimate) · AI ~10 min · Opus, high · moves: your explanation

**Objective:** each stage's result predicted before it is shown, the main claim attacked, and the
re-route for the later stages printed.
**Outcome:** the walked retrospectives and the re-route command.
**Serves:** the board's explanation row and the route's later stages.
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/drip-retro <the retrospective file plan 15's close printed>

Background
The route 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md has
reached SDXL on training seeds. This walk asks for my prediction before each result, across the
four stages, then prints the re-route for held-out seeds, the head ablations and the v58 pool.

Objectives
1. Walk each chunk, my prediction first.
2. Add one route row per fact settled.
3. Print the re-route command.

What to read
- The retrospectives and cards from steps 4, 7, 8 and 10.

What is already known
- Each stage's pass or miss line, from its route row.

When something is missing
- If a stage's diagnosis ladder is still open, walk that first.

Rules
- The outcome board: the explanation row ticked.

What to hand back
- The walked retrospective, the route rows and the re-route command.

What you will see
One chunk per round, prediction first. Done when the re-route command is printed.
```

Then:

```text
/pressure-test the claim that a pair-aware classifier's push fixes the product of experts where a
species-only push cannot, from the cards in 2-classifier-guidance/artifacts/results/
```

- [ ] **Gate:** none; the re-route is the next step.
- **Leaves:** `/skill-route re-route` from this file at the held-out-seeds stage.

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 12. ✍️ Derive the guided prediction and the dropped term by hand, after step 3

- [ ] **Do:** on the iPad, write `log p(x | cat, dog) = log p(x) + log p(cat, dog | x) + const`,
  then replace `p(cat, dog | x)` with `p(cat | x) p(dog | x)` and circle what disappears. Then
  take the gradient of `log p(x_t | y)` and multiply by `−σ_t` to get the guided prediction.
- [ ] **Do:** draw two 1-D Gaussians q1 and q2, then D*(x) = q1/(q1+q2) under them, then the slope
  of log D*/(1−D*).
- **You should see:** the circled term is the interaction term; the guided prediction is
  `ε − σ_t ∇ log p(y | x_t)`; the slope of the log odds equals the slope of log q1 − log q2.
- **Done when:** you can say, without notes, why a cat classifier plus a dog classifier gives the
  same push as a heavier product.

### 13. 👁️ Call the released-weights sheets, in step 4

- [x] **Do:** open each sheet in
  `2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/`
  and say, per tile, whether it shows the asked-for animal · ✓ called 2026-11-17 by Claude at your
  request, tile by tile at 256 px, veto open: 0 and 9 of 16 corgi, 0 and 11 tabby, 3 and 10 golden
  retriever at scales 1 and 10, against the classifier's 2 and 16, 2 and 15, 5 and 15; all 22
  disagreements are classifier yes, eye no ([the disagreement sheet](../artifacts/results/does-classifier-guidance-work-on-the-released-weights/eye-against-classifier-disagreements.png))
- **You should see:** at scale 10 almost every tile is a corgi, a tabby or a golden retriever; at
  scale 1 fewer; at scale 0 the same picture as unguided.
- **Done when:** your counts sit beside the released classifier's in the CSV.

### 14. 👁️ Call the photo sheets at full resolution, in step 8

- [ ] **Do:** open each tile at full size, not on the sheet, and call it: single animal, hybrid,
  two bodies, or two clear animals.
- **You should see:** plain product mostly single or hybrid; the pair-aware column moving up that
  ladder if the claim holds.
- **Done when:** every tile in both arms is called in the CSV.

### 15. ⚖️ Tick the SDXL bar in its review file, in step 9

- [ ] **Do:** open `2-classifier-guidance/plans/review/15-can-a-classifier-supply-the-missing-term.md`,
  read the support, null and inconclusive lines, tick them or rewrite them.
- **Done when:** the bar is ticked before step 10 runs anything.

### 16. 👁️ Call plan 15's sweep at full resolution, in step 10

- [ ] **Do:** the same four-level call as instruction 14, on every tile of the sweep and the
  control columns.
- **Done when:** the calls are in the sweep's CSV.

### 17. 🖥️ Launch from the laptop if a session has no cluster access

- [ ] **Do:** commit, then run `scripts/cluster.sh run "<the command the step printed>"`, then
  `scripts/cluster.sh jobs` until it leaves the queue. For a `nohup` run, use
  `scripts/cluster.sh sh "pgrep -af 'sample|train'"` to see it.
- **You should see:** the job id, then the log's last lines through `scripts/cluster.sh sh "tail
  -5 <log>"`.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| After stage | Prepared by | Taken as | A slip goes to | A gap goes to |
|---|---|---|---|---|
| 1 | `/quiz-prep-pulse` on the step 2 journey's first two plans | page by `quiz-cadence`, three questions | the stage 2 quiz | its node in the coverage file, learned in 30 min, then retaken |
| 2 | `/quiz-prep-pulse` on the stage 2 plan the step 7 walk wrote | page by `quiz-cadence`, a full page | the stage 3 quiz | the same |
| 3 | none | `/quiz-me` on the step 8 card, three questions | step 11 | the same |
| 4 | none | the step 11 retrospective's predictions | the re-route | the same |

## Next step

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents)

**Start with:** step 1, then step 2's open, then step 3. Say `re-route <what changed>` for a
different chain, `calendar` to put the sittings on Google Calendar, or `todoist` to file the runbook
as ordered subtasks. The background is in
[the references file](route-classifier-guided-poe-from-toy-to-animals-references.md).
