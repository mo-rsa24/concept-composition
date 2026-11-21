# 📒 Report: does an effect need support in the animal's own pictures

**No verdict has been recorded, because none of the 128 renders in this experiment has been called.** The idea under test is that [product of experts](../../../context/world/poe-composition.md#what-poe-composition-is) (PoE) can only show an effect, such as wet fur, if the animal's own training pictures already hold it. That is what "support" means here. Adding "wet" as a third prompt should then bring wetness back for a dog, often photographed wet, and leave a camel dry.

## Table of contents

- [The two sheets](#the-two-sheets)
- [What would count](#what-would-count)
- [How this differs from the cluster sheets](#how-this-differs-from-the-cluster-sheets)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The two sheets

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-would-count) ➡️

| Camel, 8 seeds by 8 columns | Dog, 8 seeds by 8 columns |
|---|---|
| <a href="../../artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/camel-sheet.jpg"><img src="../../artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/camel-sheet.jpg" width="200" alt="Sixty-four camel renders, seeds 1 to 8 down the page and eight prompt combinations across."></a> | <a href="../../artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/dog-sheet.jpg"><img src="../../artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/dog-sheet.jpg" width="200" alt="Sixty-four dog renders, seeds 1 to 8 down the page and the same eight prompt combinations across."></a> |
| Every column in a row starts from the same noise `verified` ([render script, lines 82 and 83](../../scripts/local_sdxl_poe.py)) | The same layout and settings `verified` (all 128 settings files read) |

**Each row holds the same eight ways of asking, from the animal alone to PoE of three prompts.** The numbers match the column labels printed on the sheets `verified`.

| Column | What is drawn | Its job in the test |
|---|---|---|
| 1 | "a camel" | the animal alone |
| 2 | joint prompt "a camel and rain" | shows whether one prompt reaches the effect |
| 3 | "a camel in the rain" | the scene asked directly |
| 4 | "a wet camel" | shows whether SDXL can draw the effect at all |
| 5 | PoE of "a camel" and "rain" | two prompts, where the effect may be lost |
| 6 | PoE of "a camel" and "wet" | the effect word without its cause |
| 7 | PoE of "a camel", "rain" and "wet" | the third concept under test |
| 8 | PoE of "a camel", "wet" and "wet" | column 7's total weight with no rain |

## What would count

Navigation: ⬅️ [Previous](#the-two-sheets) | 📋 [TOC](#table-of-contents) | [Next](#how-this-differs-from-the-cluster-sheets) ➡️

**A seed counts only when language can reach the effect and two-prompt PoE lost it.** That means column 2 or column 4 shows the effect on the animal, and column 5 does not. A half call counts as not showing it. The [plan's pass marks](../../plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md#the-test-written-before-any-render) read as follows.

| Result on the counted seeds | Recorded reading |
|---|---|
| dog recovered by column 7 on at least half, camel on at most 1 | supported |
| camel recovered on 3 or more | falsified |
| camel recovered on exactly 2 | inconclusive for the camel |
| dog recovered on fewer than half | the third concept is not a remedy |
| either animal has fewer than 2 counted seeds | inconclusive for that animal |

**Column 8 asks whether rain added anything that a second "wet" does not.** It is judged on the seeds where column 6 lacks the effect and column 7 has it. If column 8 shows the effect on at least half of them, "push is enough". If it shows it on at most 1 of at least 3 such seeds, "a cause is needed". Anything between is inconclusive, and so is an animal with fewer than 2 such seeds.

> [!IMPORTANT]
> Two combinations of results fit more than one row of the pass marks, and the choice is the owner's. Neither has occurred, because no tile has been called.

| Combination | One reading | The other reading |
|---|---|---|
| the camel has 1 counted seed and it is recovered, and the dog passes | supported, since the camel is at most 1 | inconclusive for the camel, since it has fewer than 2 counted seeds |
| the camel is recovered on 3 or more, and the dog on fewer than half | falsified | the third concept is not a remedy |

## How this differs from the cluster sheets

Navigation: ⬅️ [Previous](#what-would-count) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**These laptop renders differ from the dog and camel sheets in more than their starting noise, so a tile here cannot be matched to a tile there.** Steps (50 DDIM), guidance (7.5), size (1024 by 1024) and the fp16 UNet are the same `verified`.

| Setting | Cluster sheets | This experiment | Code |
|---|---|---|---|
| starting noise | the cached latent of cat and dog at that seed | fresh noise from the seed | [sheet script, line 118](../../../scripts/shared/concept_pair_sheet.py); [render script, lines 82 and 83](../../scripts/local_sdxl_poe.py) |
| seeds | 1, 2, 4, 9, 10 | 1 to 8 | the two cards and the settings files |
| empty prompt | the text encoders run on "" | zeros in place of the encoder's output `inferred` | [cluster helper, line 24](../../../poe_repair/composers/_helpers.py); render script, lines 60 to 63 |
| predictions per UNet call | 2 for one prompt, 3 for PoE | 1 | [cluster sampler, lines 158 and 217](../../../poe_repair/methods/_sampling.py); render script, lines 69 to 72 and 90 |
| graphics card | RTX 3090 for these columns | RTX 4070 Laptop | the card and [the plan](../../plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md#steps) |

The empty-prompt row is `inferred` at one step. The script's comment says SDXL zeroes the empty prompt, and diffusers 0.31.0 does so when `force_zeros_for_empty_prompt` is on, which is its default. The model's own config file is absent from this laptop, so its value was not read. The number of predictions per call matters because [batch shape changes the UNet's fp16 output](../../../artifacts/notes/batch-shape-nondeterminism/README.md).

## Where the sources disagree

Navigation: ⬅️ [Previous](#how-this-differs-from-the-cluster-sheets) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Three recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| whether the three-prompt renders exist | [the write-up, section 7.1](../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#71-what-is-done) lists PoE with "wet" and three-prompt PoE as not rendered | 128 tiles and their settings files are on disk, and the plan ticks the render on 2026-10-22 |
| when the column 8 pass marks were written | the plan's heading says "written before any render" | [the walk](../../../5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/EXECUTE_WALK.md#decisions) adds column 8 after the reader saw camel seeds 1 to 6 with "tile 7 wet where 5 and 6 are dry". Its marks were written before column 8 itself rendered |
| whether the settings match the cluster's | the plan and the script call them the same settings as the earlier sheets | the code differs in the empty prompt and the batch (table above) |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Until the tiles are called, these sheets show only what was drawn.** The one recorded observation is the reader's remark in the walk about camel seeds 1 to 6, with no per-seed calls behind it.

**Column 7 against column 5 is a mixed comparison.** Column 7 adds the word "wet" and also a third unit of guidance weight. Column 8 exists to separate the two.

**A PoE miss on SDXL could come from the sampler.** The [exact toy](../the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md) is where training content is varied directly. Eight seeds per animal is also few, and the plan sends a borderline verdict to a larger rerun on the cluster.

The renders this experiment builds on are on the [dog page](does-a-dog-in-the-rain-come-out-wet.md) and the [camel page](does-a-camel-in-the-rain-come-out-wet.md).

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the plan and the render script.

**Support**

Say 90% of a camel model's pictures are dry and 10% are wet, and a rain model likes dry-looking pictures with score 0.2 and wet-looking ones with score 0.8. Multiply each pair: dry 0.9 × 0.2 = 0.18 and wet 0.1 × 0.8 = 0.08. Dividing by their total, 0.26, gives a wet share of 0.08 / 0.26 = 31%. If the camel's wet share were 0, the product would be 0 whatever rain scores. Support is the part of the pictures where a model's probability is above zero, and a [product of experts](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Product%20Of%20Experts.md#where-to-learn-this) can only reweight inside it. The [note on when neither training set holds both](../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/When%20Neither%20Training%20Set%20Holds%20A%20And%20B.md#the-four-quadrant-answer-key) covers the cases.

**A third expert**

With pushes of (3, 0) for the animal and (0, 4) for rain, PoE moves along (3, 4). A third push of (1, 1) for "wet" makes it (4, 5). In the [render script, line 90](../../scripts/local_sdxl_poe.py), the step is the empty prediction plus 7.5 times the sum of every prompt's push, so column 7 carries three pushes where column 5 carries two.

**A counted seed**

Say four seeds are called like this. Seed A: joint wet, two-prompt PoE dry. Seed B: joint wet, PoE wet. Seed C: joint dry, "a wet ..." dry, PoE dry. Seed D: "a wet ..." wet, PoE dry. Only A and D count: B has no lost effect to recover, and C never reached the effect through language. If the third prompt brings wetness back on A only, it recovered 1 of 2 counted seeds. The [tally function](../../scripts/local_sdxl_poe.py) applies this rule at line 137.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the columns, the pass marks and the render ticks | [the third-concept plan](../../plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md), read 2026-11-22 | `stated` |
| the working record | [the third-concept walk](../../../5-text-embedding-third-expert/artifacts/drips/today-the-third-concept/EXECUTE_WALK.md), read 2026-11-22 | `stated` |
| how the renders were made | [the render script](../../scripts/local_sdxl_poe.py) and all 128 settings files, read 2026-11-22 | `verified` |
| the sheets are built from these tiles | each tile resized to its sheet cell and compared by average absolute pixel difference on a 0 to 255 scale: at most 3.56 for the right tile, and at least 27.30 when the cell is compared with the same column's tile from the next seed | `verified` |
| 122 full-size tiles held out of the drain | 61 in [the camel tiles](../../artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/camel) and 61 in [the dog tiles](../../artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/dog), each a cell of a sheet placed above. They are git-ignored, so this is the only copy, and the wet or dry calls still to be made need them | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Call all 128 tiles and write them to `calls.json`. The walk stops at this step, and neither `calls.json` nor `tally.json` exists.
- [ ] Run the tally and write one verdict line against the pass marks.
- [ ] Choose how the two ambiguous combinations of the pass marks read.
- [ ] Write the card for this folder. The [grouping index](../../artifacts/results/a-thing-in-the-rain/README.md) says "no card yet".
- [ ] Decide whether the cluster rerun in step 8 of the [causal route](../../plans/10-what-poe-loses-when-it-composes/route-causal-does-an-effect-need-support.md) should use the same empty prompt as these tiles or as the earlier sheets.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
