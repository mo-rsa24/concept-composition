# 📒 Report: does captioning both animal photos under each name help PoE compose? No verdict is recorded: the card calls this "a sheet and not a verdict"

**The claim**

Two SDXL fine-tunes were trained on the same photos and differ in how the photos holding a cat and a dog
are captioned. On the six seeds looked at, both still draw one animal in every
[product](../../../context/world/poe-composition.md#what-poe-composition-is) picture of "a cat" and "a
dog". Run A's animals are often fused cat-dogs and run B's are mostly a clean single cat or dog. The
planned test, a [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) on 128 seeds
per run, has not been run, so [the card](../../artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/README.md)
records "a sheet and not a verdict".

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. What the two fine-tunes draw](#1-what-the-two-fine-tunes-draw)
- [2. The planned test](#2-the-planned-test)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-what-the-two-fine-tunes-draw) ➡️

**The criterion was written in [the review file](../../plans/11-do-inclusive-captions-help-poe-compose/review/01-does-captioning-widen-the-cat-condition.md#the-question-written-before-the-run)
before either run trained.** It compares run B's compose rate with run A's on the same 128 seeds, where
the band is two standard errors of their difference.

| Outcome | What it takes |
|---|---|
| support | run B's rate exceeds run A's by the band or more |
| no support | run B's rate is at or below run A's |
| inconclusive | run B's rate is above run A's by less than the band |
| the fourth outcome | all three compose rates are below 0.10, and run B's cat column looks worse by eye on 5 or more of the 8 sheet seeds |

## 1. What the two fine-tunes draw

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-the-planned-test) ➡️

<a href="../../artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/poe-cat-x-dog-plain-run-a-run-b-six-seeds.png"><img src="../../artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/poe-cat-x-dog-plain-run-a-run-b-six-seeds.png" width="200" alt="Three rows of six SDXL product renders of a cat and a dog: plain SDXL, run A with fused cat-dogs, run B with mostly clean single animals; one animal in every picture"></a>

**0 of 6 seeds show two animals in any row.** Columns are seeds 0, 1, 3, 5, 6 and 7, with the same
starting noise down each column. The rows are plain SDXL, run A and run B `verified` (looked at; matches
the card).

| Row | What was trained | How a both-animal photo is captioned |
|---|---|---|
| plain SDXL | nothing | none |
| run A | a LoRA on 550 photos: 200 cat, 200 dog, 150 both | "a cat and a dog" 90%, blank 10% |
| run B | a LoRA on the same photos | "cat" 40%, "dog" 40%, "a cat and a dog" 10%, blank 10% |

**On the eight watching seeds, no checkpoint draws two animals in the product.** Read by eye from the
[pilot walk](../../artifacts/drips/inclusive-captions-pilot/EXECUTE_WALK.md) `stated`.

| Checkpoint | Product: two animals in frame | Joint prompt |
|---|---|---|
| run A, step 0 | 0 of 8 | one cat beside one dog on 8 of 8 |
| run A, step 500 | 0 of 8 | drifts: two cats on 2 to 3 seeds, a third animal on 2 |
| run B, step 500 | 0 of 8 | one cat and one dog where run A drew a third cat on seed 7 |
| run A, step 2,000 | 0 of 8, three read as a fused head | one cat beside one dog on 8 of 8 |
| run B, step 2,000 | 0 of 8, clean single animals | not recorded in the walk's entry for this checkpoint |

## 2. The planned test

Navigation: ⬅️ [Previous](#1-what-the-two-fine-tunes-draw) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

> 🖼️ **Figure wanted**: compose rate on y (0 to 1), one bar each for base SDXL, run A and run B on x, each
> bar split into one cat and one dog, a hybrid, and a single animal, with the criterion's band drawn
> across it. Numbers from a sidecar json, 128 seeds per run. Render into
> `artifacts/results/do-inclusive-captions-help-poe-compose/`, update that grouping's README card, then
> embed here.

**No compose rate exists for either run.** Run A's first 128 scoring renders were discarded at 52 seeds
after a contrast bug was found in both renderers, and run B's 128 were never started `stated` (the card).

## Where the sources disagree

Navigation: ⬅️ [Previous](#2-the-planned-test) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| the photos trained on | the [master plan](../../plans/11-do-inclusive-captions-help-poe-compose/MASTER_PLAN.md#glossary) and the review file: 491 photos with 91 both-animal, all COCO | the card, the picture and [plan 01 task 2.6](../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md#2--finish-the-data-build): 550 photos with 150 both-animal, 38 from COCO and 112 generated |
| how often run B's "cat" draws hold a dog | the review file, the pilot walk and the route: 16.9% | plan 01 task 2.7, on the 550 photos: 25.1% |
| whether the runs were trained | plan 01 leaves tasks 5.1 to 5.3 unticked | the card and the walk describe both runs through step 2,000 |
| which checkpoint the sheet shows | the card does not name one | the render script's usage example points at `step_02000` `inferred` |

> [!IMPORTANT]
> The criterion was written for the 491-photo pool and its Runs table still says 91 both-animal photos.
> Whether it applies unchanged to the 550-photo pool is the owner's call.

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Six seeds called by eye cannot give a rate.** The sheet shows what the two runs look like on those
seeds. It cannot say whether run B composes more or less often than run A.

**A cleaner animal may have another cause.** The walk records that trained products look softer than
plain ones by eye, so a run that looks better could be less degraded and no better composed.

**The contrast of sheets made before the renderer fix cannot be compared.** Those sheets were rendered half
a stop light. The squeeze is the same on every sheet, so calls about composition across sheets stand
`stated` (the card).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the files named above.

**A LoRA**

Instead of rewriting a table of 100 numbers (10 by 10), store one column of 10 and one row of 10 and add
their product to the table. That is 20 numbers to learn, and "low-rank" names this saving. A LoRA does
this to a model's weights, so fine-tuning leaves the original model in place. Read [the low-rank update](../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md#1--the-drip)
or the [LoRA note](../../../../../Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this).

**The band, two standard errors of a difference**

Toss two fair coins 128 times each. Each share of heads wobbles, and the gap between them wobbles by a
standard error of sqrt(0.5 × 0.5 / 128 + 0.5 × 0.5 / 128) = 0.0625. Two of those are 0.125, or 12.5
percentage points, so a smaller gap could be chance. At rates near 0.1 the same sum gives 7.5 points,
the two values the review file quotes. See [one standard error, worked through](../../../../../goal-setting/learning/mathematics/statistics/sampling/plans/18-one-standard-error-through-a-slope.md#1--without-the-slope-the-shares-standard-error-handed-on).

**How often a "cat" draw holds a dog**

If a jar has 9 plain sweets and 1 striped one, a draw is striped 1 time in 10. In run B, the 200 cat-only
photos give 200 × 0.9 = 180 "cat" draws, and the 150 both-animal photos give 150 × 0.4 = 60. The share
holding a dog is 60 / (180 + 60) = 25%, close to the measured 25.1%.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the sheet and what it shows | [results card](../../artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/README.md) and the picture, read 2026-11-21 | `verified` |
| eight-seed checkpoints | [pilot walk](../../artifacts/drips/inclusive-captions-pilot/EXECUTE_WALK.md) | `stated` |
| the criterion | [review file](../../plans/11-do-inclusive-captions-help-poe-compose/review/01-does-captioning-widen-the-cat-condition.md) | `stated` |
| the plan | [plan 01, the pilot](../../plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md) | `stated` |
| what a compose rate is | [compose rate](../../../context/world/compose-rate.md#what-a-compose-rate-is) | `stated` |
| what the LoRA is in this project | [the LoRA corrector](../../../context/world/lora-corrector.md#what-the-lora-corrector-is) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Run the 128-seed scoring pass for run A and run B. The route records that it waits on `/datasets`.
- [ ] Decide whether the criterion written for 491 photos applies to the 550 photos trained.
- [ ] Test the crop classifier on frames holding two animals. The review file lists this as the one untested piece of the scorer.
- [ ] Tick plan 01's tasks 5.1 to 5.3, or record why they stay open.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
