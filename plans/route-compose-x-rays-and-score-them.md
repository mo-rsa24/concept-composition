# 🎯 Composed chest films for one disease pair, and a classifier scorecard that says they are worth making
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** a set of composed chest films for one disease pair, and an AUROC table across
> the rare-cell dial (100 / 25 / 5 / 1 / 0% of the real both-disease films shown to the generator),
> with joint conditioning and prevalence recalibration as the two baselines · **Time:** ~44 h to
> finish the whole ladder (estimate), ~4.5 h in the one sitting scheduled here · **Stop point:**
> rung 1 ends on the operator table and the pick, inside 4.5 h

Built to: route-file-template v2

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [The two lanes](#the-two-lanes)
- [Schedule](#schedule)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Later](#later)
- [Why this chain](#why-this-chain)
- [References](#references)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-know-so-far) ➡️

**Goal:** know which kind of composition fits silicotuberculosis chest films, and be able to defend
that choice to a supervisor and a reviewer.

**Objective (the whole chain):** an AUROC table for a classifier trained with composed films added,
read across five settings of the share of real both-disease films the generator saw, against joint
conditioning and against prevalence recalibration.

**Outcome (the whole chain):** the composed films themselves, the scorecard, and the figure that
puts a number beside the films it came from.

**Found on disk:**

- [the kinds-of-composition combined reading](../3-understanding-compositionality/artifacts/drips/combine-kinds-of-composition/COMBINE_MAP.md),
  four ideas compared, nothing tested with numbers
- [the composition-against-editing return](../artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md),
  which already triages CompDiff and nine neighbours and already designs the six-arm experiment
- [the silicotuberculosis claim walk](../artifacts/ideas/silicotb-conversation/IDEA_MAP.md), whose
  runs 3 and 5 measure the overlap regimes on VinDr-CXR
- the journey `deep-learning/diffusion-models/05-poe-composition-diffusion/`, which teaches
  classifier-free guidance and PoE composition
- nothing in flight on the cluster

**Plan weight:** none up front, because the first rung is a reading whose own map is the plan, and
every rung after it is re-planned from what the rung before learned.

**Who you are for this route:** rebuild tier on the composition operators. You type the sampler and
the composition rule yourself at toy scale while Claude's lane builds the production version.

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| `drip-request-plan --reproduce` does not exist. The reproduce-shaped step is `drip-execute-plan --reproduce`, fed by `unbuild` | found | `mode_index.py --skill drip-execute-plan`, 2026-11-04 | step 2 |
| Liu-style AND on one conditional chest-film model is already published: Factored Classifier-Free Guidance, ε = ε(∅) + Σ ω(ε(c)−ε(∅)), used for MIMIC-CXR counterfactuals with a ResNet-18's AUROC as the measure | taken | [the composition-against-editing return](../artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md), 2026-10-17 | steps 1, 2 |
| No medical work found that multiplies separately trained experts, uses negation, or uses MCMC samplers on chest films | taken | same return | step 1's novelty row |
| Composition can only beat joint conditioning where the both-disease cell is too rare for a joint model to fit. That share is the dial, and editing has no dial | designed | same return | steps 3, 4 |
| Negation with a "contains" label is unbounded: p(drain\|x)^(−w) grows without limit, so use the positive "no drain" label wherever one exists | designed | same return | step 4 |
| The overlap regimes are measurable per pair on VinDr-CXR: aortic enlargement with cardiomegaly 1% overlapping at every IoU floor over 1,973 films; lung opacity with pulmonary fibrosis 76% any-overlap and 46% at IoU 0.5 over 666 films, with the double-label artefact on only 26% | found | [the silicotuberculosis walk](../artifacts/ideas/silicotb-conversation/IDEA_MAP.md) runs 3 and 5, 2026-11-04 | step 1's pair choice, step 4 |
| Pleural effusion with pleural thickening reads 94% same-region only because one radiologist drew one box and attached both labels on 737 of its 839 films. It is not a measured overlap | found | same walk, run 5 | step 4: do not use that pair |
| Data lives only on the cluster under `/datasets/mmolefe/`; every count and run goes through `scripts/cluster.sh sh "<cmd>"` with `CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, and figures come back only with `scripts/cluster.sh pull` | taken | project `CLAUDE.md` and the silicotuberculosis walk's limits | every step that runs |
| Whether any of the six papers released weights or code is unchecked | open | nothing yet | step 2's target choice |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have written | You will draw, from a small example | A worked example | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 which operator | own | the operator table, by hand, one row per paper | the three places composition can sit (score, conditioner, mask) on one film, with the two disease boxes drawn on it | FCFG's ε = ε(∅) + Σ ω(ε(c)−ε(∅)) worked on two findings with numbers you pick | 4 from the journey's CFG plan | 3 questions |
| 2 reproduce one figure | own | the paper's own pipeline run once, its number printed beside theirs | nothing new | their reported number against yours, with the tolerance | 2 | 3 questions |
| 3 the criterion | own | `HYPOTHESIS.md` | the dial curve you predict, before any run | the count needed per cell at AUROC 0.75 | 2 | 3 questions |
| 4 compose and score | build | `compose_pair()` and `score_dial()`, typed by you at toy scale, compared against the production run | the sampler's loop on the whiteboard | one composed film traced step by step | 6 to 10 | full page |
| **Outcome** | | **the composed films and the AUROC-against-dial table** | | | | **the table has five dial settings, three arms, and a confidence interval on every cell** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

▶ **Recommended:**
`drip-read --combine --lean` → `unbuild` → `drip-execute-plan --reproduce` → `drip-idea --hypothesis`
→ `drip-execute-plan --request` → `pair-figure` → `drip-retro`
‖ background: `paper-scout` on code and weights, `quiz-prep-pulse` one stage ahead, `video-cadence`
over the journey's CFG plan.

Spliced from the paper-and-reproduce row of skill-route's own planning table and the six-arm
experiment designed in [the composition-against-editing return](../artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md).
The `--lean` into `--combine` order is skill-route's own proposal, so it is labelled: no skill's text
chains them.

<details><summary>Alternatives</summary>

`drip-read` in full on FCFG alone → `unbuild` → `--reproduce`: better when the reproduction matters
more than the options table, because one paper read in the authors' own order reproduces faster than
six read across one idea.

`drip-idea --code` on the composition pipeline first: better when you already know the operator and
only the system is unclear.

</details>

## The two lanes

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

| Stage | The AI lane builds | Your lane learns | Quiz | Confidence bar | You rebuild | Compare on |
|---|---|---|---|---|---|---|
| 1 | the operator table and the novelty row | the combined read, one idea across six papers | 3 questions on where composition can sit | every needed-now question right | the FCFG formula on two findings, by hand | your worked numbers against the paper's formula |
| 2 | the paper's pipeline run, their figure's number reproduced | what their code does at each stage | 3 questions | same | nothing yet | their reported number |
| 3 | `HYPOTHESIS.md` | what would falsify the dial claim | 3 questions | same | the count calculation by hand | the Hanley-McNeil interval |
| 4 | the production composition run and the classifier arms | the sampler and the composition rule | full page | every needed-now right, 80% of the rest | `compose_pair()` from an empty file at toy scale | the same two experts and the same seed |

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

One long sitting this week, taken as 4.5 hours. Only rung 1 is scheduled. Everything after it is
estimated and re-routed at rung 1's close, because what the reading settles changes the reproduction
target.

| Block | Run | You, while it runs | Checkpoint at the end of the block |
|---|---|---|---|
| 0:00 to 0:20 | step 0: dispatch the code-and-weights scout, in the background | read the position line and the two prior maps | the scout is detached and you know what the combined read is for |
| 0:20 to 2:00 | step 1, papers 1 to 3 (FCFG, CompDiff, XReal) | ▶ CME296 lecture 4, 1:27:42 to 1:34:34 (7 min) while the read-ahead runs | three rows of the operator table, each saying where composition sits and what it measures |
| 2:00 to 2:20 | break | | |
| 2:20 to 4:00 | step 1, papers 4 to 6 (RadEdit, decentralized, clinically aware) | 📐 the journey's CFG plan, section 1 | all six rows, plus the novelty row |
| 4:00 to 4:30 | step 1's `compile`, then the gate | | the pick named, with its reason, its null and the pair it will run on |

**Totals.** Rung 1: ~4 h of your attention (estimate: six papers at about 25 minutes each on one
idea, plus half again for tangents, plus 30 minutes to compile), ~50 min of AI time of which
~40 min runs beside your reading. The whole ladder: ~44 h to finish (estimate), ~22 h of it rungs
2 and 4.

**What was cut to fit, and what it costs.** The 2× learning multiplier would put rung 1 at 8 hours,
which is two sittings. The cut is depth per paper, not papers: each of the six is read on one idea
only (where its composition sits and what it measures), not section by section. The cost is that
you finish rung 1 able to compare the six and unable to reproduce any of them from the reading
alone, which is what rung 2 is for.

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. 🔧 Find out which of the six papers released code or weights

You ~2 min · AI ~25 min, detached · Sonnet, medium · moves: step 2's target choice

**Objective:** for each of the six papers, the repository URL or the statement that none exists,
and whether weights are downloadable without an application.
**Outcome:** a six-row table in the scout's return file.
**Serves:** needed by step 2, which cannot pick a reproduction target without it.

```text
/paper-scout for each of these six papers, find the code repository and the model weights, and say
for each whether the repository exists, whether it has a runnable training or sampling entry point,
and whether weights can be downloaded without an application or a licence form:
arXiv 2506.14399 (Factored Classifier-Free Guidance), 2603.16551 (CompDiff), 2403.09240 (XReal),
2312.12865 (RadEdit), 2606.10153 (Compositional Generative Modeling from Decentralized Data),
2603.15525 (Clinically Aware Synthetic Image Generation).
Chain: composed chest films and a classifier scorecard · this step delivers the reproducibility
column of the operator table.
Settled facts: FCFG is the only one of the six that composes a chest film and then measures a
classifier's AUROC (from the composition-against-editing return, 2026-10-17). CompDiff retrains
classifiers on synthetic films but composes inside a learned conditioner, not over scores (same
source).
Outputs: artifacts/ideas/silicotb-conversation/routes/02-code-and-weights-for-the-six.return.md,
one row per paper: repo URL or none, entry point or none, weights reachable yes / application / no.
Target: six rows, none left blank. A paper whose repository cannot be found is written as "searched
<where>, none found", never left empty.
Proof: the return file has six rows and every cell filled.
Hand-off: step 2 picks its reproduction target from the rows where an entry point exists.
```

- `added:` the six arXiv ids and the three columns (from this route's reading set) · `dropped:`
  nothing
- [ ] **Gate:** dispatch it before step 1 starts, so it returns while you read
- **Leaves:** `artifacts/ideas/silicotb-conversation/routes/02-code-and-weights-for-the-six.return.md`

### 1. 🔧 Read one idea across all six papers: where composition sits, and what it measures

You ~3 h 40 min · AI ~50 min · Opus, high · moves: Achieve row 1

**Objective:** a six-row table, one row per paper, naming where that paper's composition happens
(over scores, inside a learned conditioner, or by a mask), what it composes (labels, attributes,
regions), what it measures to claim success, and whether its construction can express a region-level
overlap at all.
**Outcome:** `COMBINE_MAP.md` for the new idea, with the table and the pick.
**Serves:** the operator choice the whole chain rests on; step 4 cannot be written without it.

```text
/drip-read --combine --lean the idea is: where in the model does composition happen, and what does
each paper measure to claim it worked. Read it across six sources, in this order, and give each one
a row rather than a summary:
  1. arXiv 2506.14399, Factored Classifier-Free Guidance
  2. arXiv 2603.16551, CompDiff
  3. arXiv 2403.09240, XReal
  4. arXiv 2312.12865, RadEdit
  5. arXiv 2606.10153, Compositional Generative Modeling from Decentralized Data
  6. arXiv 2603.15525, Clinically Aware Synthetic Image Generation
Chain: composed chest films for one disease pair and an AUROC-against-dial scorecard · this step
delivers the operator choice and the reason for it.
Learning path: product of experts, rungs 1 to 8 of
  /home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/
  (none ticked; start at 03-plain-independence.md)
Learning path: classifier-free guidance, rung 1,
  05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md, with CME296 lecture 4
  at 1:27:42 to 1:34:34.
Starts from: 3-understanding-compositionality/artifacts/drips/combine-kinds-of-composition/COMBINE_MAP.md, which already compares
four kinds of composition and marks nothing tested; and
artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md,
which is read whole, not sampled.
Settled facts: FCFG runs Liu-style AND inside one conditional model as
ε = ε(∅) + Σ ω(ε(c)−ε(∅)) and measures the intervention with a ResNet-18's AUROC on MIMIC-CXR.
CompDiff composes inside a learned conditioner over age, sex and race, including held-out
intersections, and retrains classifiers on the synthetic films. No medical work found so far
multiplies separately trained experts, uses negation, or uses an MCMC sampler. Negation with a
"contains" label is unbounded, so a positive "no B" label is used wherever one exists.
The overlap regimes are measurable on VinDr-CXR: lung opacity with pulmonary fibrosis is 76%
any-overlap and 46% at IoU 0.5 over 666 films, and pleural effusion with pleural thickening is not
usable because 737 of its 839 films carry one box labelled twice by one radiologist.
When something is missing: a paper behind a fetch block is read from its HTML mirror, and if that
fails the row says "unread, fetch blocked" rather than inferring the method from the abstract.
Outputs: artifacts/drips/where-composition-sits/COMBINE_MAP.md, with the six-row table, a
differences section, and a final section naming the operator picked for a silicotuberculosis-shaped
pair, the reason, the null that would make it the wrong pick, and the VinDr pair it will run on.
Target: six rows, every cell filled, and the pick naming one construction and one pair. The pick is
judged by whether it can express the overlap regime the chosen pair actually shows.
Progress: one row per round, each round ending with that paper's place in the table.
Proof: the map's table has six rows with no empty cells, and its pick section names a construction,
a pair, a reason and a null.
Hand-off: step 2 reproduces a figure from whichever paper the pick rests on; step 3 turns the null
into a falsification criterion.
Controls:
  paper 1 example (FCFG's formula worked on two findings with numbers)
  paper 1 prove (that its AUROC measures the intervention and not the classifier's prior)
  paper 3 plumb (XReal's spatial control, since the regimes are spatial)
  after paper 6 test (the stage quiz)
Side routes:
  paper 1 /picture-speak --take-apart "ε = ε(∅) + Σ ω(ε(c) − ε(∅))"
Limits: no cluster run in this step. Any count goes through
scripts/cluster.sh sh "<cmd>" with
CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.
```

- `added:` the two prior maps as starting points, the VinDr pair numbers, the control and side-route
  lines (from the silicotuberculosis walk and `mode_index.py`) · `dropped:` "reproduce what can be
  reproduced" (why: it is step 2, and a reading step that also reproduces finishes neither)
- [ ] **Gate:** the pick names one construction and one VinDr pair before step 2 is pasted
- **Leaves:** `artifacts/drips/where-composition-sits/COMBINE_MAP.md`
- **Learning path for product of experts**, earliest unticked first:

  | # | | Open | Textbook | Watch |
  |---|---|---|---|---|
  | 1 | ⬜ | [independence on a table](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/03-plain-independence.md) | [reading a table](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/textbook/01-reading-a-table.md) | |
  | 2 | ⬜ | [independence inside a slice](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/04-conditional-independence.md) | same page | |
  | 3 | ⬜ | [a joint as a product of conditionals](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/07-chain-rule-of-probability.md) | same page | [chain rule, 22:03 to 29:43](https://youtu.be/rNEujZmD2Tg?t=1323) |
  | 4 | ⬜ | [factorisation as arrows](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/08-factorisation-as-a-graph.md) | [a joint as a graph](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/textbook/03-a-joint-as-a-graph.md) | [Bayesian networks, 4:48 to 11:00](https://youtu.be/BX6hdDH3lqQ?t=288) |
  | 8 | ⬜ | [the PoE rule from the fork](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md) | [the product-of-experts fork](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/textbook/06-the-product-of-experts-fork.md) | [chains and forks, 18:34 to 26:42](https://youtu.be/BX6hdDH3lqQ?t=1114) |

  Rungs 5 to 7 are cut here; run the lookup on rung 4's plan to see them.

- **Learning path for classifier-free guidance**: rung 1 only,
  [CFG as base plus omega times a difference](/home/molef/goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md),
  with [CME296 lecture 4, 1:27:42 to 1:34:34](https://www.youtube.com/watch?v=WUUq6TVAu8U&t=5262s).
  No textbook page; listed for a `video-cadence` sweep.
- **AUROC has no journey.** The shelf page is *Introduction to probability for data science*, p. 605,
  ROC and precision-recall curves. Expect a `gap` line and read that page before step 3.

### 2. 🔧 Clone the picked paper's code, then reproduce the one figure that can carry a bar

You ~2 h 30 min · AI ~2 h · Opus, high · moves: Achieve row 2 · **re-route before pasting**

**Objective:** the paper's own reported number reproduced within 30%, with the curve rising and
falling where theirs does.
**Outcome:** the reproduced figure and the printed comparison line.
**Serves:** proof that the picked construction runs here before anything is built on it.

```text
/unbuild <the repository step 0 found for the picked paper>
then, once UNBUILD.md exists:
/drip-execute-plan --reproduce <journey path unbuild chose> reproduce <the one figure that prints
numbers>, within 30% of the paper's values, and with the curve rising and falling where theirs does.
Chain: composed chest films and a classifier scorecard · this step delivers the evidence that the
picked construction runs on our hardware.
Starts from: artifacts/drips/where-composition-sits/COMBINE_MAP.md (the pick and its reason),
artifacts/ideas/silicotb-conversation/routes/02-code-and-weights-for-the-six.return.md (the entry
point and the weights).
Target: each reported number within 30%, which is about 70% closeness. Exact correspondence is not
the bar.
Proof: a printed line with their number, yours and the ratio, saved beside the figure.
Hand-off: the environment facts UNBUILD.md records go to step 4's prompt.
Limits: weights are not downloaded onto a cluster node without asking first.
```

- `added:` the tolerance and the proof line (from skill-route's replication rule) · `dropped:`
  "reproduce each of them" (why: six reproductions is 30 hours and only one of them is the pick)
- [ ] **Gate:** a figure that prints numbers exists. A teaser figure cannot carry a tolerance, and
  if none of the six prints comparable numbers, say so and reproduce the nearest thing that does
- **Leaves:** `UNBUILD.md` and the reproduced figure

### 3. 🔧 Write the falsification criterion for the dial, before any production run

You ~1 h · AI ~30 min · Opus, high · moves: Achieve row 3

```text
/drip-idea --hypothesis the claim is: composition recovers the both-disease cell at shares of real
both-disease films where joint conditioning does not, and editing stays flat because it never uses
those films.
Chain: composed chest films and a classifier scorecard · this step delivers the criterion every
later number is read against.
Starts from: read whole
  artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md
and artifacts/drips/where-composition-sits/COMBINE_MAP.md.
Settled facts: the dial is the share of real both-disease films the generators see, at 100, 25, 5, 1
and 0%. The test films are held out of every arm. The toy predicts joint conditioning degrades as
the share falls, composition holds, and editing stays flat. The null is that composition tracks
joint conditioning at every share. At 25 positives against 25 negatives and AUROC 0.75 the
Hanley-McNeil 95% interval is about ±0.14, and at 100 against 100 about ±0.07, so the count per cell
is part of the criterion.
Outputs: artifacts/ideas/silicotb-conversation/HYPOTHESIS.md, in the shape frame-hypothesis
produces.
Target: the supporting result, the null and the inconclusive band each carry a number, and the
counts needed per cell are computed rather than asserted.
Proof: the file names a measure, a target, a tolerance and who judges.
Hand-off: step 4's bar is this file's supporting result, verbatim.
```

- [ ] **Gate:** the counts per cell are computed against the real VinDr counts for the picked pair
- **Leaves:** `artifacts/ideas/silicotb-conversation/HYPOTHESIS.md`

### 4. 🔧 Compose the films and score the classifier across the dial

You ~8 h · AI ~14 h, most of it runs · Opus, high · moves: Achieve row 4 · **re-route before pasting**

```text
/drip-execute-plan --request compose chest films for the picked pair with the picked construction,
at five settings of the share of real both-disease films the generator sees (100, 25, 5, 1, 0%),
and train one classifier per arm: plain training, prevalence recalibration, joint conditioning, and
composition. Report AUROC on real both-disease films against single-disease negatives, held out of
every arm, with a confidence interval per cell.
Chain: composed chest films and a classifier scorecard · this step delivers the outcome itself.
Starts from: HYPOTHESIS.md (the bar), COMBINE_MAP.md (the construction), UNBUILD.md (how the
picked code runs).
Settled facts: every count and run goes through scripts/cluster.sh sh "<cmd>" with
CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain, and figures come
back only with scripts/cluster.sh pull. Large artifacts go to /datasets only. Before any classifier
is trained, lesion size per arm and nearest-neighbour distance to the training films are measured,
because a gain no larger than upweighting the real films is not a gain.
Target: the bar in HYPOTHESIS.md, judged by you.
Proof: the scorecard file, with counts per cell printed beside every AUROC.
Controls: the piece that writes the sampler build and break; the piece that launches plumb; the
closing piece test.
Hand-off: pair-figure puts the number beside the films.
```

- [ ] **Gate:** the measurement of lesion size and nearest-neighbour distance lands before any
  classifier trains
- **Leaves:** the scorecard, read by step 5

### 5. 🔧 Put the number beside the films, then explain it back

You ~2 h · AI ~1 h · Sonnet, medium · moves: the Outcome row

```text
/pair-figure the claim is <the scorecard's headline>, shown as the AUROC-against-dial curve beside
the composed films at the dial setting where the two arms diverge most.
```

```text
/drip-retro <the walk's retrospective file that step 4's close prints>
```

- **Leaves:** the figure and the retrospective

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 6. ✍️ Draw the three places composition can sit, on one film

- [ ] **Do:** on the iPad, draw one chest film with two disease boxes on it (use the lung opacity
  with pulmonary fibrosis pair, 76% any-overlap). Draw three arrows into it, one per place
  composition can happen: over the scores at sampling time, inside the conditioner before sampling,
  and by a mask after sampling. Label each with the paper that does it.
- **You should see:** three arrows entering the picture at three different stages, and the reason
  only one of them can place a lesion in a named region.
- **Done when:** you can say, without notes, which of the three can express the overlap regime the
  pair shows.

### 7. ✍️ Work the FCFG formula on two findings, by hand

- [ ] **Do:** take ε = ε(∅) + Σ ω(ε(c) − ε(∅)) with two conditions, pick your own numbers for
  ε(∅), ε(c₁) and ε(c₂) and for ω, and compute ε. Then set ω to 0 and to 3 and say what happens to
  each condition's contribution.
- **You should see:** the composed noise prediction moving along the sum of two differences, and
  both conditions pulling at once rather than in turn.
- **Done when:** you can predict the direction of the change before computing it.

### 8. ✍️ Read the AUROC page before step 3

- [ ] **Do:** *Introduction to probability for data science*, p. 605, ROC and precision-recall
  curves. Then compute the Hanley-McNeil interval for 25 against 25 at AUROC 0.75 by hand.
- **You should see:** about ±0.14, which is why the count per cell is part of the criterion.
- **Done when:** your number matches the one in step 3's settled facts.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

| After stage | Prepared by | Taken as | A slip goes to | A gap goes to |
|---|---|---|---|---|
| 1 | `/quiz-prep-pulse` on the stage-1 quiz | the page `quiz-cadence` builds from step 1's `test` control | the stage-2 quiz | its node in the coverage file, learned in 30 min, then retaken |
| 2 | same | `/quiz-me` on `UNBUILD.md` | the stage-3 quiz | same |
| 3 | same | `/quiz-me` on `HYPOTHESIS.md` | the stage-4 quiz | same |
| 4 | same | the full page, from step 4's `test` control | the retrospective | same |

The confidence bar before you rebuild from scratch at stage 4: every needed-now question right and
at least 80% of the rest (an estimate, change it if it is wrong). Claude's lane does not wait for
the bar.

## Later

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

- **MCMC samplers for composed scores.** Nothing medical uses them, so they are a novelty claim
  rather than a prerequisite. The journey teaches them at
  `deep-learning/diffusion-models/05-poe-composition-diffusion/`.
- **Negation as an operator.** Unbounded with "contains" labels, and the positive label works, so it
  is out of this chain.
- **The twenty-films-by-hand check on the VinDr overlap.** Owed to the silicotuberculosis walk's
  claim 1.10, not to this route.
- **The NIOH licence request.** Route 01 of the silicotuberculosis walk owns it.

## Why this chain

Navigation: ⬅️ [Previous](#later) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

<details><summary>Why the reading comes before the reproduction</summary>

Five of the six papers can be reproduced in principle and only one of them is the construction this
project will use. Reproducing before picking spends the reproduction budget on a method that gets
dropped. The reading's own gate is the pick, so step 2 has one target rather than six.

</details>

<details><summary>Why the baselines are in the chain and not optional</summary>

The composition-against-editing return records that prevalence recalibration alone reaches 0.73
AUROC on the misaligned films, and that joint conditioning can be trained directly wherever labels
exist. A scorecard showing composition works, without those two arms, says nothing about whether
composition was needed.

</details>

## References

Navigation: ⬅️ [Previous](#why-this-chain) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [the composition-against-editing return](../artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md),
  the six-arm design and the ten-paper triage
- [the kinds-of-composition combined reading](../3-understanding-compositionality/artifacts/drips/combine-kinds-of-composition/COMBINE_MAP.md)
- [the silicotuberculosis claim walk](../artifacts/ideas/silicotb-conversation/IDEA_MAP.md), runs 3
  and 5 for the VinDr overlap
- [the PoE composition journey](/home/molef/goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/)

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

Paste step 0, then step 1. Re-route at step 1's close with
`/skill-route re-route <what the pick settled>`.
