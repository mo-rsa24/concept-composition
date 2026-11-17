# 🧪 Can a classifier's push fix the product: from the released weights to SDXL cat × dog
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** one figure set per rung of the scale-up ladder (exact toy, cat × dog on
> held-out seeds, three pair kinds, then more seeds, the v58 pool and unseen pairs beside the
> rank-32 LoRA), each with its card and its pass or miss line, plus the photo check beside it ·
> **Time:** ~18 h to finish the ladder from step 7, ~32 h budget at 2×, about 15 GPU-hours on a
> Blackwell · **Stop point:** each rung's bar printed pass or miss, inside 22 two-hour sittings ·
> **Outcome board:**
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

**Outcome (the whole chain):** [classifier guidance](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/01-the-guided-noise-prediction.md) working on the authors' released weights, then
a figure set per rung of the scale-up ladder: the exact toy with both classifiers, a small
discriminator on cat × dog judged on held-out seeds, the three pair kinds, and rung 4's more seeds,
v58 pool and unseen pairs beside the rank-32 LoRA. Our photos run beside the ladder as the
controlled-data check. The paper's table, the head ablations and plan 15 are re-routed from step
11.

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
| Exact toy at pair weight 0.01: [exact product](../../3-understanding-compositionality/artifacts/notes/how-the-poe-toy-experiment-works/how-the-experiment-works.md#2-the-exact-product) 300 of 300 crisp pairs, reverse diffusion 0 of 300 | found | the [toy card](../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md) | 7 |
| On the toy the product itself composes and the sampler fails, so a discriminator learns log q_joint minus log q_sampler, which differs from the true noised product; the toy reports cosine against both | designed | the toy card, "[Where the samplers lose the pair](../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/README.md#where-the-samplers-lose-the-pair)" | 7 |
| Exact toy, pair weight 0.01, seed 7: a pair-aware discriminator's push gives 172 of 300 crisp pairs at w = 1 and 300 at w = 2, against 0 unguided; a separate-label push<sub>[1](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/02-the-product-as-two-classifiers.md#23-pushing-harder-with-the-same-two-classifiers "Classifier guidance by hand, chapter 2, section 2.3: more of the same two separate-label pushes on the product base, where the fused animal's share rises and the pair's falls"), [2](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/04-the-separate-label-push-built.md "Classifier guidance by hand, plan 04: the separate-label push built and measured on the four-picture toy at strengths 0, 1, 3 and 10")</sub> gives at most 10 over scales 1 to 10; the closed-form best separate-label classifier reaches 34 at scale 10 | found | [A1's page](../report/classifier-guided-poe/a1-exact-toy-classifier-push.md), plan 19 | 8, 9, 10 |
| On the toy the pair-aware push<sub>[1](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/04-the-dropped-term.md#44-the-pair-aware-push "Classifier guidance by hand, chapter 4, section 4.4: the pair-aware push, a discriminator's log-odds trained on joint-caption draws against the product's own draws, applied with a weight on the product"), [2](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/08-the-pair-aware-push-built.md "Classifier guidance by hand, plan 08: the pair-aware push built and measured on the four-picture toy at weights 0, 0.5, 1 and 2")</sub> points along the sampler's own ratio (cosine 0.85 to 0.96, steps 190 to 90), not along the true correction (at most 0.21), and stays under a tenth of the summed score's size, so a cap of 0.1 never binds; the true correction capped at 0.1 composes 4 of 300 | found | plan 19's `diagnosis.json` | 9, 10 |
| On the toy, inclusive captions with no push need a pair weight between 0.3 and 0.5 to match the push at 0.01 (115 and 246 crisp of 300) | found | plan 19's `inclusive-captions-comparison.json` | 8 |
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
| mscluster111 and mscluster112 (RTX PRO 6000 Blackwell) both work under `co3_bw`: torch 2.11.0+cu128 sees the card, a 1000 × 1000 matmul sums to 1e9, and `/datasets` is visible over SSH; mscluster110 held another user's 67.8 GB job; mscluster107 card 1 was free; mscluster109 did not answer SSH | found | live probe 2026-11-17 over `scripts/cluster.sh sh`, recorded in `environment/hpc/nodes.md` | the rung compute plans |
| On a Blackwell a plain product render takes 5.7 to 5.9 s and a joint render 5.0 to 5.2 s, against 51.5 s per product render on a Quadro RTX 8000, about 9 times faster; no guided SDXL render has been timed on any card | found | `1-lora-correction/report/paper-review/prze-10-compute.md`, plan 15's render cost | the rung compute plans |
| Rung 2's bar, written before any render, counts out of 4 seeds at strengths 10, 30 and 100: support is two separate intact animals on at least 2 of 4 held-out seeds (9 to 12) at some strength, more than best-of-N, with a random push matched in size step by step composing at most 1 of 4; the disturbance reading is that random push at 2 of 4 or more; the picker reading is the same count with best-of-N equal or higher; inconclusive is at least 2 of 4 composed on seeds 9 to 12 with at most 1 intact; memorised is at most 1 of 4 on seeds 9 to 12 at every strength and at least 2 of 4 on seeds 1 to 4; null is at most 1 of 4 on both sets at every strength, called only when strength 100 visibly moved the picture on at least 2 of 8 seeds and the cap fired on fewer than half the steps, else the run is too weak to read | designed | [the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md), 2026-11-17 | 10 |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have written | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 classifier guidance on the released weights | own | nothing typed; the guided mean read line by line in their `gaussian_diffusion.py` | `ε − σ_t ∇ log p(y \| x_t)` on one step's vectors, and the missing term from Bayes | the corgi push at scale 1 and 10, one step's numbers | the submerge journey's derivation problems | three questions after stage 1 |
| 2 the exact toy | build | the toy's guided step and the discriminator's loss and log-odds gradient, typed by you | the best discriminator q1/(q1+q2) on two 1-D Gaussians, and its log-odds slope | one draw's push against the true correction at one step | the separate-label push shown to equal a heavier product, by hand | a full page after stage 2 |
| 3 our photos, both ways | own | nothing typed; the label-dropout change read as a diff | where the null expert comes from in each arm | the product's prediction for one 64 px image, with and without the push | none | three questions after stage 3 |
| 4 SDXL, rungs 2 to 4 | gloss | nothing typed | the ladder with each rung's count | rung 2's guided sheet against plain product and best-of-N on one seed | none | the retrospective's predictions |
| **Outcome** | | **four figure sets with cards, findings and pass or miss lines** | | | | **you explain why a separate-label classifier cannot fix the product, without notes** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-ladder) ➡️

▶ **Recommended:** `1 hld-diagram ‖ 2 submerge (background) → 3 drip-read --lean ‖ background: 4's
downloads → 4 drip-execute-plan --reproduce ‖ background: 6 prompt-storyboard → render-diagrams →
5 drip-execute-plan --request (sampler repair) ‖ background: the canary → 7 drip-execute-plan
--request (toy) ‖ background: quiz-prep-pulse, quiz-cadence → 9 frame-hypothesis (rung 2's readings) →
10 drip-execute-plan --request (rung 2) → 10b experiment-planner (rung 3) → 10c drip-execute-plan →
10d experiment-planner (rung 4) → 10e drip-execute-plan → 11 drip-retro, pressure-test → re-route`,
with `8 drip-execute-plan --request (photos, both arms) ‖ background: the 64 px training` beside the
ladder from sitting 5 (laid on the research spine: read, reproduce, rebuild
small, state the test, test it scale by scale, explain; proposed for the photo stage, where no
curated flow trains a model of our own)

<details><summary>Alternatives</summary>

`7 toy → 9 → 10 rung 2 → 10d → 10e`, skipping rung 3: better when the paper needs the v58 pool and
held-out pairs first and the pair kinds can wait for the re-route.

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
animals against the product's own draws), which is the claim. On SDXL the pair-aware classifier is a
small discriminator trained on checked renders of the two things apart, with lone-subject renders
and the product's own renders as negatives; plan 18's species-only push is the earlier result it
answers.

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

**The chain over the ladder.** `7 rung 1 toy → 9 rung 2's bar → 10 rung 2 (A3q) → rung 3 plan →
rung 3 → rung 4 plan → 4a, 4b, 4c → 11 retrospective`, with step 8's photos beside it. Rungs 1 and 2
are one comparison each, so a request walk writes its own plan; rung 2's bar is written in one pass
before anything renders; rungs 3 and 4 have many cells, so an axes planner (`experiment-planner`)
writes the axes and the run table before the walk.

**Every rung opens with the same compute piece.** The walk reads `environment/hpc/throughput.md`
and `environment/hpc/nodes.md`, probes the cards live, times a small batch on the chosen card, and
writes GPU-hours, wall clock, card and fallback card, and the seeds and pairs that buys into the
rung's plan. It ticks the row's compute line on the board and stops for a yes before launching.
Card order: mscluster111 or 112 (Blackwell, `co3_bw`), the other Blackwell, then mscluster107
card 1 (RTX 8000, `co3`); every column of one rung's sheet renders on one card.

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
| 4 | rungs 2 to 4: each rung's compute piece, discriminators, renders and sheets | the ladder, predicted before each reveal; each tile called by eye | the retrospective's predictions | none | none | none |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Two-hour sittings, numbered, no dates and no GPU cap; each rung records what it spent. Steps 1 to 5
are done. Each rung is re-cut at the close of the one before it.

| Block | Run | You, while it runs | Checkpoint at the end of the block |
|---|---|---|---|
| sitting 1 | step 7 opens: the compute piece, the axes block, the port ‖ background: step 6's storyboard and renders | none | scale 0 identical to unguided, draw by draw |
| sittings 2 to 4 | step 7's pieces | the stage 1 quiz while a sweep runs | rung 1's bar printed |
| sitting 5 | step 9, then step 10's compute piece and training-set sheet ‖ background: step 8's two 64 px trainings start | none | the seven readings counted; the positives called |
| sittings 6 and 7 | step 10, about 0.3 GPU-hours on a Blackwell | the stage 2 quiz | rung 2's reading printed |
| sitting 8 | step 10b | none | rung 3's axes, run table and compute block |
| sittings 9 to 12 | step 10c, about 2.7 GPU-hours over two Blackwells | none | each kind's reading printed |
| sitting 13 | step 10d | none | rung 4's three plans and bars |
| sittings 14 to 17 | step 10e: 4a in sitting 14, 4b in 15 and 16, 4c in 17; about 12 GPU-hours | none | each sub-rung's reading printed |
| sitting 18 | step 11 and the pressure test | none | the re-route printed |
| sittings P1 to P4 | step 8, any time after sitting 5, once the 64 px runs land | none | both photo arms called (instruction 14) |

**Totals.** About 18 hours of your time to finish the ladder from step 7 and about 32 hours budgeted
at 2×, in 18 sittings, plus 4 for the photos: 22 in all. About 15 GPU-hours on a Blackwell for the
ladder, most of it in rung 4, plus the photo stage's 64 px trainings (12 to 24 hours each,
estimate). A rung that misses opens its diagnosis ladder at half its box and moves every later
sitting.

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

- [x] **Gate:** instruction 13 called · ✓ 2026-11-17, called by Claude at your request and accepted by you: see instruction 13
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
0. Compute piece: time one discriminator epoch and one 300-draw guided sweep on the
   laptop CPU; write minutes, the machine, the fallback (mscluster112, co3_bw) and the
   draws it buys into the plan; tick the exact-toy row's compute line; stop for my yes.
0b. Write the axes block before any run: classifier (none, separate-label,
   pair-aware) varies; pair weight 0.01, 300 draws, noise schedule and the s grid are
   held. Print the draw count each classifier actually ran.
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
- One axis per comparison: at each s only the classifier differs.
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
draws at least 150 of 300 crisp pairs at some s with overlays no higher than unguided, and the
separate-label push draws at most 15 of 300 at every s on the grid.
```

- `added:` the separate-label classifier as the control (from the corrected claim) · `dropped:` the
  linear-discriminant picture of the head's features (why: on a toy with known densities the cosine
  says more)
- [x] **Gate:** the bar printed; a miss opens the diagnosis ladder and stops the climb. ✓ verified 2026-11-17: pass on both halves, pair-aware 172 of 300 crisp at w = 1, separate-label at most 10 · [A1's page](../report/classifier-guided-poe/a1-exact-toy-classifier-push.md)
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
   finished draws, noised forward).
4. Sample the released model on seeds 1 to 16: plain product of the two separate-label pushes,
   then with the pair-aware push added.
5. When the 64 px runs land, sample each on seeds 1 to 16: plain product, with the separate-label
   push, with the pair-aware push.
6. Build one sheet per arm, call every tile, and write the card.

What to read
- The pool card, which says how many photos hold each animal:
  4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/README.md
- The step 4 card and the clone at /home/molef/PhD/replications/guided-diffusion.
- The toy's finding: 2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md,
  and its code: 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/toy_classifier_push.py.
- The toy walk's retrospective:
  2-classifier-guidance/artifacts/notes/session-merges/2026-11-17-classifier-push-on-the-exact-toy-retro.md.
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
- On the exact toy the pair-aware push gave 172 of 300 crisp pairs at weight 1 and 300 at weight
  2, and the separate-label push at most 10, so sample the pair-aware push at weights 1 and 2 (A1's page).
- On the toy the separate-label push still made a few pairs, because the both picture carried
  both labels; the same can happen here with the 150 both photos (A1's page).
- On the toy the discriminator's fake pile was the sampler's finished draws noised forward to a
  random step, with the loss weighted per step, and its push stayed under a tenth of the
  product's own step (the toy's code and A1's page).

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
  maths: a product needs a null expert); the toy's three facts and its finding, code and
  retrospective as sources (from step 7) · `dropped:` pasted pairs (why: real both-animal photos
  exist in the pool); the step 7 card, replaced by A1's page
- [ ] **Gate:** instruction 14 called.
- **Leaves:** the card, the finding and the retrospective, which step 11 reads.

### 9. ⚖️ State rung 2's seven readings before anything renders

You ~30 min (estimate) · AI ~10 min · Opus, high · moves: rung 2's bar

**Objective:** A3q's seven readings with their counts out of 4 seeds, in a review file, before any
SDXL render.
**Outcome:** `2-classifier-guidance/plans/review/a3q-cat-x-dog-quick-discriminator-test.md` with its
bar ticked.
**Serves:** step 10's bar, and the board's rung 2 row.
**Sees:** [A3q's page](../report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md) ·
**Ticket:** none · **Source page:** none

```text
/frame-hypothesis a small pair-aware discriminator, trained on 10 to 15 checked SDXL renders of
a cat and a dog apart, steers the cat × dog product to two separate animals on held-out seeds 9
to 12 more often than plain product and than best-of-N picked by the same discriminator

Background
Rung 2 of the scale-up ladder in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. Its four
readings are on A3q's page; this pass writes their counts before step 10 renders anything.
Rung 1, the exact toy, passed on 2026-11-17.

Objectives
1. Write support, memorised, inconclusive and null as counts out of 4 seeds, by eye at full
   resolution, starting from the proposed counts below.
2. Name the columns: plain product, best-of-N at equal compute picked by the discriminator,
   guided at 3 strengths; seeds 1 to 4 beside 9 to 12.
3. Write the review file and tick its bar line.

What to read
- 2-classifier-guidance/report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md.
- 2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md, the toy's finding.
- 2-classifier-guidance/artifacts/notes/session-merges/2026-11-17-classifier-push-on-the-exact-toy-retro.md.

What is already known
- Proposed counts: support is guided at least 2 of 4 on seeds 9 to 12 at some s, more than
  best-of-N, damage no worse than plain product; memorised is at least 2 of 4 on seeds 1 to 4
  and at most 1 of 4 on 9 to 12; inconclusive is at least 2 of 4 composed with at least half
  of them damaged; null is at most 1 of 4 on both seed sets at every s.
- Plain product best-of-16 composed 0 of 128 particles on cat × dog.
- Seed 10's joint-prompt render draws three dogs, so no joint column is read on seed 10.
- On the exact toy the pair-aware push gave 172 of 300 crisp pairs at weight 1 and 300 at
  weight 2, so the 3 strengths should reach past 1 (A1's page).
- On the toy the push stayed under a tenth of the summed score's size, so the SDXL sampler's
  cap of 0.1 should not bind; log whether it fires (A1's page).
- On the toy the product's own target composed and the push repaired the sampler; on SDXL
  the gap may also be the model's, so a null here does not contradict the toy (A1's page).

When something is missing
- If a count cannot be fixed without seeing renders, write the reading without it and ask;
  nothing renders until all four have numbers.

Rules
- Write no code.

What to hand back
- The review file path and one route row with the four counts.

What you will see
The criterion one part per round. Done when the review file holds the seven readings with
numbers.
```

- [x] **Gate:** the review file's bar ticked. · ✓ 2026-11-17: [the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md), seven readings with counts
- **Leaves:** the review file, which step 10 reads.

### 10. 🧪 Rung 2: a small discriminator steers cat × dog on held-out seeds

You ~135 min to finish, ~270 min budget at 2× (4 pieces and the compute piece at 30 min,
estimate) · AI ~30 min, about 0.3 GPU-hours on a Blackwell (inferred until the compute piece
times it) · Sonnet, medium · moves: rung 2

**Objective:** one of A3q's seven readings printed with its counts.
**Outcome:** the sheet (seeds 1 to 4 beside 9 to 12; plain product, best-of-N, guided and a
size-matched random push at 3 strengths), the per-tile calls, the card and the finding.
**Serves:** the board's rung 2 row, A3q.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** none

```text
/drip-execute-plan --request "Rung 2: a small pair-aware discriminator steers SDXL's cat × dog
product on held-out seeds"

Background
Rung 2 of the scale-up ladder in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md. The fastest
real test on SDXL: a small discriminator trained on a few checked renders, its push judged on
seeds it never saw, beside plain product and best-of-N picked by the same discriminator.

Objectives
0. Compute piece: probe mscluster112, then 111, then mscluster107 card 1; assert torch sees
   the card under co3_bw; time 4 guided and 4 plain renders; write GPU-hours, wall clock,
   card, fallback and N for best-of-N into the plan; tick A3q's compute line; stop for my yes.
1. Build the training set: 10 to 15 renders of cat and dog apart from seeds 1 to 8 (joint
   prompt plus "a cat on the left and a dog on the right" and "a cat and a dog sitting
   apart"), lone cat and lone dog, and the product's renders from seeds 1 to 8. Show me the
   sheet; I call each tile.
2. Train the discriminator on noisy latents: cat and dog apart against the product's renders
   and the lone animals, all noised the same way.
3. Render seeds 1 to 4 and 9 to 12: plain product, best-of-N, guided at s = 10, 30 and 100,
   and a random push scaled to the guided push's norm at every step, at the same 3 strengths.
4. One sheet, seeds 1 to 4 beside 9 to 12; I call every tile at full resolution.

What to read
- A3q's page: report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md.
- The step 9 review file, for the seven readings' counts.
- The step 5 commit (the capped, unrescaled push) and the step 7 card.

What is already known
- Plain product best-of-16 composed 0 of 128 particles on cat × dog.
- The joint prompt shows one of each animal on 6 of 17 cached seeds (9, 11, 12, 16 and two
  below 9); seed 10's joint render draws three dogs.
- Plan 18's push turned the fused animal into one cat on 3 of 4 seeds, which is why lone
  animals are negatives here.
- mscluster111 and 112 passed a CUDA probe under co3_bw on 2026-11-17.
- Every column of the sheet renders on one card.

When something is missing
- Fewer than 10 checked positives: add renders from seeds 5 to 8 with the two wordings before
  training; never seeds 9 to 16.
- The guided sampler fails under co3_bw: move to mscluster107 card 1 under co3, all columns.

Rules
- One axis: only how the discriminator is used changes between columns.
- Stop before each launch with the exact command.
- Outputs under /datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/.
- Filing: on, so each sheet is filed.

What to hand back
- The sheet, per-tile calls as CSV, the card under
  2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/, the
  reading it landed on, route rows, A3q's lines ticked, board redrawn with --step 10.

What you will see
The compute block and my yes, the training-set sheet, then the result sheet called tile by
tile. Done when one of the seven readings is printed with its counts.

Controls: the compute piece, plumb; the training-set piece, prove (each tile at full
resolution); the result sheet, prove; the close, test
Side routes: the result sheet, /picture-speak --consequence the push switched off at each s

Judged by you from full-resolution tiles against the step 9 review file.
```

- [ ] **Gate:** every tile called; the reading printed.
- **Leaves:** the card and the retrospective, which the rung 3 planning step reads. Plan 15
  (`2-classifier-guidance/plans/15-can-a-classifier-supply-the-missing-term.md`) is re-routed
  from here.

### 10b. 📐 Plan rung 3: axes, run table and compute for three pair kinds

You ~60 min (estimate) · AI ~15 min · Opus, high · moves: rung 3's plan

**Objective:** a plan file for rung 3 whose axes, run table, compute block and per-kind bar are
written before any render.
**Outcome:** the plan file and its review file under `2-classifier-guidance/plans/`.
**Serves:** the board's rung 3 rows, K1, K2, K3 (their compute lines).
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/experiment-planner rung 3 of the scale-up ladder: rung 2's discriminator recipe on three pair
kinds

Background
Rung 2 (route step 10) tested a small pair-aware discriminator's push on cat × dog. Rung 3
changes only the pair kind. The route is
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md; the rows are
K1, K2, K3 on 2-classifier-guidance/report/classifier-guided-poe/board.md.

Objectives
1. Write the axes before any experiment: pair kind varies; 3 pairs per kind replicate;
   rung 2's recipe, columns, strengths, seeds, steps and cap are held.
2. Write the run table: one row per pair × column × seed × strength, with the card each
   pair runs on.
3. Write the compute block from rung 2's measured rates: GPU-hours, wall clock, cards
   (mscluster111 and 112 under co3_bw, mscluster107 card 1 as fallback), the cells it buys.
4. Write the per-kind bar and the rule for going on.

What to read
- Rung 2's card and retrospective, and its measured rates.
- environment/hpc/throughput.md and environment/hpc/nodes.md.
- data/prompt-sets/attend-and-excite/a.e_prompts.json and
  artifacts/_shared/cross_pair_pool_configs/pair_ledger.md.

What is already known
- Pairs, drawn from the guard-clean "and" prompts with colour words stripped, cat and dog
  excluded, 3 per kind with random.Random(0): K1 a cat and a bear, a bird and a bear, a bird
  and a turtle (of 44); K2 a cat and a backpack, a monkey and a suitcase, a lion and a clock
  (of 90); K3 a suitcase and a balloon, a bow and a bowl, a bowl and a bench (of 66).
- Colour words are stripped so only the kind changes; every object prompt in the set
  carries a colour and no animal prompt does.
- Bar per kind, 12 held-out tiles (3 pairs × seeds 9 to 12): passes when guided shows two
  separate things in frame on at least 4 more of 12 than plain product, more than best-of-N,
  damage no worse; "nothing to fix" when plain product is already at least 9 of 12; null
  when guided is within 1 of plain product at every strength.
- Every column of one pair renders on one card.

When something is missing
- If a pair yields fewer than 10 checked positives from seeds 1 to 8, draw the next pair by
  the same rule; never use seeds 9 to 16.

Rules
- No run starts from this step; it writes the plan only.

What to hand back
- The plan file, its review file, and K1 to K3's compute lines ticked on the board.

What you will see
The axes, then the run table, then the compute block, one per round.
```

- [ ] **Gate:** the review file's bar ticked.
- **Leaves:** the rung 3 plan, which step 10c walks.

### 10c. 🧪 Rung 3: the push on three pair kinds

You ~225 min to finish, ~450 min budget at 2× (5 pieces at 45 min, estimate) · AI ~1.5 h wall
clock, about 2.7 GPU-hours over two Blackwells (inferred from rung 2) · Sonnet, medium · moves:
rung 3

**Objective:** each kind's bar printed pass, nothing to fix, or null.
**Outcome:** one sheet per kind, per-tile calls, the card and the finding.
**Serves:** K1, K2, K3.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** none

```text
/drip-execute-plan <the rung 3 plan file step 10b wrote>

Background
Rung 3 of the scale-up ladder in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md: rung 2's recipe,
unchanged, on 3 animal pairs, 3 animal and object pairs and 3 object pairs.

Objectives
1. Run the plan's compute piece and stop for my yes.
2. Per pair: build and show the training-set sheet; I call each tile.
3. Train each pair's discriminator; render its columns on one card.
4. One sheet per kind, seeds 1 to 4 beside 9 to 12; I call every tile at full resolution.

What to read
- The rung 3 plan and its review file; rung 2's card.

What is already known
- The pairs, the bar and the cards are in the plan.
- Two things in frame counts, fused and destroyed are called apart, and a hybrid is its own
  category.

When something is missing
- A card fails its probe: the next card in the plan's order, all of that pair's columns.

Rules
- Stop before each launch with the exact command.
- Filing: on.

What to hand back
- The sheets, per-tile calls as CSV, the card under
  2-classifier-guidance/artifacts/results/does-the-push-work-on-each-pair-kind/, the finding,
  route rows, K1 to K3 ticked, board redrawn with --step 10c.

What you will see
One pair per round, then one kind's sheet per round. Done when each kind's reading is printed.

Judged by you from full-resolution tiles against the plan's review file.
```

- [ ] **Gate:** every kind's reading printed; rung 4 starts when at least one kind passes.
- **Leaves:** the card and retrospective, which step 10d reads.

### 10d. 📐 Plan rung 4: more seeds, more pairs, unseen pairs

You ~60 min (estimate) · AI ~15 min · Opus, high · moves: rung 4's plan

**Objective:** a plan for three sub-rungs, each changing one thing, with axes, run table, compute
block and bar written before any render.
**Outcome:** the plan file and its review file under `2-classifier-guidance/plans/`.
**Serves:** A4, C1, C2 (their compute lines).
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/experiment-planner rung 4 of the scale-up ladder: the push scaled to more seeds, the v58 pool
and unseen pairs

Background
Rungs 1 to 3 of 2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md
tested the push on the toy, on cat × dog, and on three pair kinds. Rung 4 scales it in three
sub-rungs, each changing one thing. Rows A4, C1, C2 on
2-classifier-guidance/report/classifier-guided-poe/board.md.

Objectives
1. 4a, more seeds: rung 2's cat × dog discriminator, unchanged, on seeds 9 to 16; s fixed at
   rung 2's best; seeds 13 to 16 are read apart because s was chosen on 9 to 12.
2. 4b, more pairs: one discriminator trained over the v58 pool (182 cells over 48 pairs),
   its positives the pool's joint-prompt targets already admitted by eye; held-out seeds 9
   to 16; columns plain product, best-of-N, guided, joint prompt and the rank-32 LoRA. It
   is also scored on cat × dog seeds 9 to 16 beside 4a's discriminator, which separates the
   effect of pooling from the effect of more pairs.
3. 4c, unseen pairs: 4b's discriminator, unchanged, on the 53 held-out pairs at seeds 9 to
   16, the 23 pairs sharing no concept with training reported apart.
4. Per sub-rung: the axes, the run table, the compute block (GPU-hours, wall clock, cards
   mscluster111 and 112 under co3_bw, mscluster107 card 1 as fallback, cells it buys) and
   the bar.
5. The scoring: an automatic two-things count (GroundingDINO count, CLIP on the padded
   crop naming each) on every tile, and by eye on one fixed seed per pair for guided and
   plain product, which measures how far the automatic count can be trusted.

What to read
- The cards and retrospectives of rungs 2 and 3, with their measured rates.
- 1-lora-correction/report/paper-review/held-out-set.md, for the held-out pairs.
- 1-lora-correction/report/paper-review/prze-10-compute.md, for the Blackwell rates and
  which renders already exist.
- environment/hpc/throughput.md and environment/hpc/nodes.md.

What is already known
- Blackwell rates: plain product 5.7 to 5.9 s, joint 5.0 to 5.2 s, the adapter render 12.1
  to 14.3 s; a guided render's time comes from rung 2.
- Proposed compute: 4a about 0.15 GPU-hours; 4b about 6 (384 cells × about 45 s for five
  columns, plus training); 4c about 5.5 (424 cells); about 3 hours of wall clock each on
  two Blackwells.
- The cat × dog joint prompt is wrong on seeds 10, 13, 14 and 15, so its joint column is not
  read there.
- The 182 v58 training targets are joint-prompt renders admitted by eye over several rounds;
  not every seed was checked, so 4b's positives get one more look on a sheet.
- GroundingDINO counts things but cannot name them; CLIP on the padded crop names them,
  untested on overlapping boxes.
- Every column of one pair renders on one card; renders made earlier on another card or
  sampler are not reused.

When something is missing
- If rung 3 passed in fewer than three kinds, 4b and 4c report the missing kinds apart and
  their bar is read on the passing kinds only.

Rules
- No run starts from this step; it writes the plan only.

What to hand back
- The plan file, its review file, A4, C1 and C2's compute lines ticked on the board.

What you will see
One sub-rung per round: its axes, run table, compute block and bar.
```

- [ ] **Gate:** the review file's three bars ticked.
- **Leaves:** the rung 4 plan, which step 10e walks.

### 10e. 🧪 Rung 4: more seeds, the v58 pool, unseen pairs

You ~270 min to finish, ~540 min budget at 2× (6 pieces at 45 min, estimate) · AI about 12
GPU-hours over two Blackwells, about 7 hours of wall clock (inferred) · Sonnet, medium · moves:
rung 4

**Objective:** each sub-rung's bar printed pass or miss, in order, a miss stopping the climb.
**Outcome:** the sheets, the per-pair counts, the cards and the findings, ready for the paper's
table.
**Serves:** A4, C1, C2, and the paper's table row P1 through step 11.
**Sees:** the step 1 design · **Ticket:** none · **Source page:** none

```text
/drip-execute-plan <the rung 4 plan file step 10d wrote>

Background
Rung 4 of the scale-up ladder in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md: 4a more seeds,
4b the v58 pool, 4c the held-out pairs, in that order.

Objectives
1. 4a: compute piece, my yes, the run, the sheet called at full resolution, the bar.
2. 4b: compute piece, my yes, train the pooled discriminator, the run, the automatic count
   and my calls on the fixed seed per pair, the bar.
3. 4c: compute piece, my yes, the run, the same reads, the bar, the 23 no-shared-concept
   pairs apart.

What to read
- The rung 4 plan and its review file; the rung 2 and 3 cards.

What is already known
- The cards, the bars and the scoring are in the plan.
- biggpu allows one Slurm job per user; a second run goes on the other Blackwell with nohup;
  outputs go under /datasets only.

When something is missing
- A sub-rung misses: its diagnosis ladder opens and the next sub-rung waits.

Rules
- Stop before each launch with the exact command.
- Filing: on.

What to hand back
- Per sub-rung: the sheet, per-tile calls and automatic counts as CSV, the card under
  2-classifier-guidance/artifacts/results/does-the-push-scale/, the finding, route rows, the
  row ticked, board redrawn with --step 10e.

What you will see
One sub-rung per two or three rounds. Done when the three readings are printed.

Judged by you from full-resolution tiles and the automatic count, against the plan's review
file.
```

- [ ] **Gate:** each sub-rung's reading printed in order.
- **Leaves:** three cards and retrospectives, which step 11 walks.

### 11. 🔁 Walk the ladder's retrospectives, attack the claim, re-route to the paper

You ~60 min (estimate) · AI ~10 min · Opus, high · moves: your explanation, and the paper's table

**Objective:** each rung's result predicted before it is shown, the main claim attacked, and the
re-route to the paper's table and the head ablations printed.
**Outcome:** the walked retrospectives and the re-route command.
**Serves:** X1, and P1 and B1 to B7 through the re-route.
**Sees:** none · **Ticket:** none · **Source page:** none

```text
/drip-retro <the retrospective file rung 4's close printed>

Background
The scale-up ladder in
2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md has closed at
rung 4, or stopped at an earlier rung. This walk asks for my prediction before each result, rung
by rung, then prints the re-route for the paper's table (P1), the head ablations (B1 to B7) and
plan 15.

Objectives
1. Walk each rung's chunk, my prediction first.
2. Add one route row per fact settled.
3. Print the re-route command.

What to read
- The retrospectives and cards of steps 7, 8, 10, 10c and 10e.

What is already known
- Each rung's reading, from its route row.

When something is missing
- If a rung's diagnosis ladder is still open, walk that first.

Rules
- The outcome board: X1 ticked.

What to hand back
- The walked retrospective, the route rows and the re-route command.

What you will see
One chunk per round, prediction first. Done when the re-route command is printed.
```

Then:

```text
/pressure-test the claim that a small pair-aware discriminator's push fixes the product of
experts on held-out seeds and pairs, from the cards in 2-classifier-guidance/artifacts/results/
```

- [ ] **Gate:** none; the re-route is the next step.
- **Leaves:** `/skill-route re-route` from this file at the paper's table.

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
  request, tile by tile at 256 px, accepted by you: 0 and 9 of 16 corgi, 0 and 11 tabby, 3 and 10 golden
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

### 15. ⚖️ Tick rung 2's seven readings in their review file, in step 9

- [ ] **Do:** open `2-classifier-guidance/plans/review/a3q-cat-x-dog-quick-discriminator-test.md`,
  read the seven readings and their counts out of 4 seeds, tick them or rewrite them. Do the same
  for the rung 3 and rung 4 review files when steps 10b and 10d write them.
- **Done when:** each bar is ticked before its rung renders anything.

### 16. 👁️ Call each rung's sheets at full resolution, in steps 10, 10c and 10e

- [ ] **Do:** call every training-set tile and every result tile three ways, as
  [the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md#words-this-file-uses)
  defines them: the animals (one single, one fused, two separate), the picture (intact or
  damaged), and the species (a cat and a dog, two of one kind, unclear). On rung 4b and 4c, call
  the one fixed seed per pair for guided and plain product, beside the automatic count.
- **Done when:** the calls are in each rung's CSV.

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
| rungs 2 to 4 | none | the step 11 retrospective's predictions | the re-route | the same |

## Next step

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents)

**Start with:** step 1, then step 2's open, then step 3. Say `re-route <what changed>` for a
different chain, `calendar` to put the sittings on Google Calendar, or `todoist` to file the runbook
as ordered subtasks. The background is in
[the references file](route-classifier-guided-poe-from-toy-to-animals-references.md).

## Cross-references

- The mention of **the route's Quick context, as the source of the dropped-term claim** in [the plan "The dropped term, on four rows"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/07-the-dropped-term-table.md#description).
- The mention of **the route**, whose stage 2 cites the account, in [the plan "The account you could defend"](../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/plans/09-the-account-you-could-defend.md#where-this-connects).
