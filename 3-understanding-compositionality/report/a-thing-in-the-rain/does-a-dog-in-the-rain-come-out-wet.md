# 📒 Report: does a dog in the rain come out wet

**The test recorded for this sheet did not hold: the joint prompt drew a wet dog on 2 of 5 seeds, and product of experts also drew one on 2 of 5.** The guess written before the render was that the joint prompt, one prompt naming both things, would wet the dog on at least 3 of 5 seeds. [Product of experts](../../../context/world/poe-composition.md#what-poe-composition-is) (PoE), which adds the pull of "a dog" and the pull of "rain" at every drawing step, was expected to do so on at most 1 of 5.

## Table of contents

- [What each way of asking drew](#what-each-way-of-asking-drew)
- [Where the prompts push on seed 10](#where-the-prompts-push-on-seed-10)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## What each way of asking drew

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-prompts-push-on-seed-10) ➡️

<a href="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png" width="200" alt="Five seeds down the page and five columns across: a dog alone, rain alone, the joint prompt, product of experts and SuperDiff."></a>

**Asked for a wet dog directly, SDXL draws one on all 5 seeds, and both ways of combining a dog with rain reach 2.** Seeds 1, 2, 4, 9 and 10 run down the sheet. The first four columns share one starting noise per row and use 50 DDIM steps at guidance 7.5. SuperDiff, a sampler that reweights the two prompts as it draws, starts from its own noise and runs 200 steps `stated` ([the card](../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/README.md)).

| How it was asked | The reader's call over 5 seeds | Source |
|---|---|---|
| "a wet dog" | wet 5 | [direct prompt calls](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/asked-directly-calls.json) `verified` |
| "a dog in the rain" | wet 4, half 1 | the same file `verified` |
| joint prompt "a dog and rain" | wet 2, half 1 | [the walk, piece 1](../../artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md#the-cut) `stated` |
| PoE of "a dog" and "rain" | wet 2. On seed 10 a person with an umbrella replaces the dog | the walk `stated`; seed 10 seen on the sheet `verified` |
| SuperDiff at 200 steps | no call recorded | |

The guess did not say how a half call counts. The result is the same under either reading, because PoE's 2 already exceeds the limit of 1.

| Reading of a half call | Joint prompt, needs at least 3 | PoE, needs at most 1 | The guess holds |
|---|---|---|---|
| half counts as dry | 2 | 2 | no |
| half counts as wet | 3 | 2 | no |

## Where the prompts push on seed 10

Navigation: ⬅️ [Previous](#what-each-way-of-asking-drew) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

| Recording that ended on the sheet's person | Recording that kept the dog |
|---|---|
| <a href="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-pushes-in-the-plane.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-pushes-in-the-plane.png" width="200" alt="Six panels at steps 0, 5, 10, 20, 35 and 49 with arrows for a dog, rain, their sum and the joint prompt."></a> | <a href="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-batch4-pushes-in-the-plane.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-batch4-pushes-in-the-plane.png" width="200" alt="The same six panels for the recording that used one batch of four predictions."></a> |
| <a href="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-pushes-over-the-run.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-pushes-over-the-run.png" width="200" alt="Three line plots against denoising step: angle between the pushes, share of the joint push outside their plane, and SuperDiff weight on a dog."></a> | <a href="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-batch4-pushes-over-the-run.png"><img src="../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain-seed10-batch4-pushes-over-the-run.png" width="200" alt="Two line plots against denoising step: angle between the pushes and share of the joint push outside their plane."></a> |
| PoE's three predictions in one batch, the joint prompt in its own `verified` ([current script, lines 71 to 79](../../scripts/concept_pair_pushes.py)) | all four predictions in one batch `verified` (first version of the script, commit `841263f3`, lines 70 to 83) |

**At all six steps drawn, 32% to 70% of the joint prompt's push lies outside the flat plane of the dog and rain pushes.** No reweighting of the two pushes can reach that part, so PoE cannot produce it `stated` ([script notes, lines 9 to 14](../../scripts/concept_pair_pushes.py)).

In the top row, each panel is one denoising step with its own scale. The dog push lies along x at its true length, and the rain push sits at its true angle. In the bottom row, the x-axis is the denoising step from 0 (pure noise) to 49 (final) and one point is one step. The y-axes are the angle between the two pushes in degrees and the share of the joint push's length outside their plane, in percent. The third panel on the left is SuperDiff's weight on "a dog", with its 200 steps rescaled onto the 50-step axis.

**At steps 0, 5 and 10 the two recordings match to within 0.1 in every length, and by step 20 they have separated.** Length is the size of a prompt's noise prediction minus the empty prompt's, before guidance is applied. These values are read off the panels `verified`. The numbers behind them are in a `pushes.json` on the cluster, which is absent from this laptop.

| Step | Recording | "a dog" length | "rain" length | Joint length | Angle between pushes | Joint push outside the plane |
|---|---|---|---|---|---|---|
| 20 | ended on the person | 2.8 | 9.3 | 3.3 | 86° | 54% |
| 20 | kept the dog | 5.2 | 5.6 | 5.5 | 102° | 32% |
| 49 | ended on the person | 2.5 | 3.7 | 3.1 | 77° | 70% |
| 49 | kept the dog | 2.7 | 3.7 | 3.3 | 77° | 68% |

## Where the sources disagree

Navigation: ⬅️ [Previous](#where-the-prompts-push-on-seed-10) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded points conflict. The code explains the first, and the second does not change the result.**

| What | One source | The other source | What explains it |
|---|---|---|---|
| whether a push recording reproduces the sheet's PoE picture | the script's header says "the final picture is the sheet's PoE tile" | [the walk](../../artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md#out-in-the-world) records one recording that kept the dog (job 65971) and one that matched the sheet's person with a pixel gap of 6.5 of 255 (job 65972) | the batch of predictions changes the fp16 arithmetic ([script, lines 71 to 73](../../scripts/concept_pair_pushes.py)). The walk attributes the remaining 6.5 to the graphics card `stated`. Which file is job 65972 is `inferred`: the walk names only `seed10-batch4` for job 65971 |
| how a half call counts in this test | the test written before the render does not say | the camel test written on 2026-10-20 counts a half as dry | unresolved, and both readings fail (table above) |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The sheet shows what was drawn and leaves open why.** The counts cover five seeds, and the walk did not record which seeds were called wet for the joint prompt or PoE.

**The SuperDiff column is a mixed comparison.** It differs from the other columns in starting noise, sampler and step count, so its tile cannot be compared pixel by pixel with its row.

**The push plots cover one seed.** The walk records that seed 10 sits on a switch between a dog and a person, where the batch shape and the graphics card can each flip the final picture.

The same question for an animal rarely photographed wet is on the [camel page](does-a-camel-in-the-rain-come-out-wet.md). Adding "wet" as a third prompt is on the [third-concept page](does-an-effect-need-support.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the pictures and scripts above.

**A push**

Say the model's prediction for an empty prompt is (2, 1) and for "a dog" is (5, 1). The push of "a dog" is the second minus the first, (3, 0): the direction that prompt moves the drawing. Its length is 3. On seed 10 at step 0, the "a dog" push has length 2.6. [Classifier-free guidance](../../../../../goal-setting/learning/deep-learning/diffusion-models/05-poe-composition-diffusion/plans/02-cfg-as-base-plus-omega-difference.md) multiplies a push by a weight, here 7.5, and adds it to the empty prediction.

**Product of experts as added pushes**

With a dog push of (3, 0) and a rain push of (0, 4), PoE moves along their sum, (3, 4). In the project's code the step is empty + 7.5 × dog push + 7.5 × rain push ([the combining rule](../../../poe_repair/_sdxl/metrics.py)). The [product of experts note](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this) gives the probability behind it.

**The share outside the plane**

Two pushes (1, 0, 0) and (0, 1, 0) span a flat plane. A joint push of (3, 4, 12) has length 13. Its [projection](../../../../../Obsidian/Mathematics/Linear%20Algebra/Projection.md#where-to-learn-this) onto the plane is (3, 4, 0), which leaves (0, 0, 12) outside, so the share is 12 / 13 = 92%. The share is the outside length divided by the whole length. On seed 10 at step 49 it is 70% on the recording that ended on the person.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the sheet and how it was rendered | [the card](../../artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/README.md) and [the sheet script](../../../scripts/shared/concept_pair_sheet.py), jobs 65969 and 65970, 2026-10-09 | `stated` |
| the test, the dog sheet's totals and the two push jobs | [the dog and rain walk](../../artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md), read 2026-11-22 | `stated` |
| the direct prompt calls | [direct prompt calls](../../artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/asked-directly-calls.json), read 2026-11-22 | `verified` |
| the push measurements | [the push script](../../scripts/concept_pair_pushes.py) and the four plots, read 2026-11-22 | `verified` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Call the dog sheet seed by seed, which is step 7 of the [route on whether an effect needs support](../../plans/10-what-poe-loses-when-it-composes/route-causal-does-an-effect-need-support.md). The third-concept test needs these calls to choose its dog seeds on this sheet.
- [ ] Choose how a half call counts in this test. Both readings give the same result today.
- [ ] Read piece 2 of the walk, the push recordings, against a test. The walk lists its test as open.
- [ ] Run pieces 3 to 5 of the walk: the text embeddings, the sheet for "raining", "wet" and "water", and seed 10 across four models.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
