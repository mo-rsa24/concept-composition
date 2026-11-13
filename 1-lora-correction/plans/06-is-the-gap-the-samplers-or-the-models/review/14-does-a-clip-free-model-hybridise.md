# 🧪 Review: does the same product still fuse a cat and a dog on a model with no CLIP in it?

The verdict file for
[hypothesis-14](../plans/hypothesis/14-does-a-clip-free-model-hybridise.md). The design is there;
what the runs cost, what came back and what it licenses are here.

## Recommended prompt (when the run lands)

```
/analyze-run @plans/06-is-the-gap-the-samplers-or-the-models/review/14-does-a-clip-free-model-hybridise.md — <the three counts out of eight>
```

## Position in the plan tree

**Step 77 of 77.** Pairs with
[hypothesis-14](../plans/hypothesis/14-does-a-clip-free-model-hybridise.md). The one order is the
`## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)
- [Cross-references](#cross-references)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The product**: the empty prompt's noise prediction plus the guidance scale times the sum, over
  "a cat" and "a dog", of each prompt's difference from the empty prediction. One extra denoiser
  pass per prompt plus one for the empty prompt.
- **The joint prompt**: the single sentence "a cat and a dog" put through the model the ordinary
  way, with no product at all. It is the reference column, and a model that cannot draw the pair
  from the joint prompt cannot be asked about the product.
- **Two clear animals**: a cat that reads as a cat and a dog that reads as a dog, both whole, in
  one picture, judged at full resolution.
- **One hybrid animal**: a single animal carrying features of both, such as a cat's face on a dog's
  body. Its own category, not a failed version of either other one.
- **One single animal**: one animal of one species, with no feature of the other. A different
  failure from a hybrid, and worth separating because the product dropping a prompt entirely and
  the product averaging two prompts are different mechanisms.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim**, one-directionally. Swapping SDXL for PixArt-alpha changes five things at once
(the text encoder, the denoiser, the sampler, the step count, the guidance scale), so this is a
mixed comparison. A fusion still licenses removing CLIP as a necessary cause, because no other
difference can create a fusion that CLIP was required for. Two clear animals licenses nothing about
cause, and is recorded as the arithmetic composing here while failing there.

What would make the run worthless is a composition path that does not reproduce the released
pipeline's plain output, since then the product column would measure the plumbing. The identity
check at task 2.1 is the gate, and it runs before any product render.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The weights fetch, `PixArt-alpha/PixArt-XL-2-1024-MS` onto `/datasets` | setup | 2026-11-04, from mscluster84 | 21.8 GB | `/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS` | ✅ 21 GB on disk |
| The single-prompt identity check, "a cat" at seed 1 through the pipeline and through the script | gates the claim | 2026-11-04, mscluster84 | 2 renders, 6 minutes including a 2m45 load | the two images and the largest absolute pixel difference between them | ✅ 0 grey levels, byte-identical |
| The four columns on eight seeds | Tests the claim | 2026-11-04, mscluster84 | 32 renders at 1024², 20 steps each, 22 minutes | tiles and sidecars under `1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/`, sheet `pixart-cat-dog-sheet.png` | ✅ 0 of 8 product, 7 of 8 joint |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] **Does the product of "a cat" and "a dog" on PixArt-alpha hold two clear animals, or one?**
      Eight seeds, 1 to 8, at PixArt-alpha's own defaults of 20 steps and guidance 4.5, with the
      joint prompt "a cat and a dog" on the same eight seeds as the reference column. Every call is
      made by eye from the full-resolution tile, in the three categories named above, because the
      repo's instance-count scorer counts animals without naming them and reads a hybrid and two
      animals of one species identically.

      **Support** if 2 or fewer of 8 product renders hold two clear animals while the joint-prompt
      column holds two clear animals on 6 or more of 8. CLIP is then removed as a necessary cause
      of the fusion.

      **Null** if 6 or more of 8 product renders hold two clear animals. The arithmetic composes on
      a CLIP-free model, so the SDXL failure is not caused by the arithmetic alone, and the next
      question is which of the five differences carries it.

      **Inconclusive** at 3 to 5 of 8, or whenever the joint-prompt column itself holds two clear
      animals on fewer than 6 of 8, which would mean the model cannot draw the pair at all.

      The SDXL side of the comparison is not re-run. It is quoted from
      [the corrector finding](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md):
      joint prompt 8 of 8, plain product 0 of 8, held-out seeds 9 to 16, counted by the validated
      instance count rather than by eye.

      **Verdict: SUPPORT.** The product holds two clear animals on **0 of 8** seeds: six tiles are
      one hybrid animal and two are one single animal, both of those cats. The joint-prompt column
      holds a clear cat and a clear dog on **7 of 8**, above the 6-of-8 the bar demanded, so the
      model can draw the pair at this guidance scale and the weak-guidance escape is closed. The
      eighth joint tile, seed 2, holds two cats and counts as neither. CLIP is removed as a
      necessary cause of the fusion. Counts and per-tile sentences in
      [the calls](../../../artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/calls.json), sheet and
      card in [the same folder](../../../artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/README.md).

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] **Does the script reproduce the released pipeline's plain output?** The largest absolute
      difference between the two "a cat" renders, in 0-to-255 pixel values, and the fraction of
      pixels that differ at all. A largest difference of 0 means the two paths are the same
      computation. Above about 2 grey levels is a plumbing fault, and the product renders do not
      run until it is fixed rather than tolerated.

      **Answered: 0.** Largest absolute difference 0 of 255 grey levels, 0.0 of pixels differing,
      and the two PNGs are byte-identical (md5 `d9c08c28048b8da98124be0246099b07`). The summed
      guidance path and `PixArtAlphaPipeline` as released are the same computation at one condition,
      so a hybrid in the product column is the arithmetic rather than the plumbing. Record in
      `/datasets/mmolefe/poe_repair_min/outputs/clip_free_hybridise/identity-check.json`.
- [x] **Which seeds did what?** One record per product tile in `calls.json`, with its seed, its
      category and one sentence on what is in the picture, so a disagreement later is about a named
      tile rather than about a count.

      **Answered.** Hybrids on seeds 1, 4, 5, 6, 7 and 8; a single cat on seeds 2 and 3. The joint
      column's one failure is seed 2, which draws two cats.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Post-hoc: does the product ever drop a prompt rather than average it?** Two of the eight tiles are
a single clean cat with no canine feature, which is a different failure from a hybrid. Whether "a
dog" is being dropped or being out-voted is not answerable from eight pictures and is not planned.

**Post-hoc: the two single-animal tiles are both cats, never dogs.** With two tiles this is what a
coin would do, so it is recorded and not read.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

**The five-way difference.** The honest limit of this run, stated in the run kind above. Only the
fusion direction is read as evidence.

**The guidance scale.** PixArt-alpha runs at 4.5 where SDXL runs at 7.5. A lower scale weakens
every prompt's push, so a fusion here could be a weak-guidance artefact rather than the same
failure. The joint-prompt column is the control for this: the same scale drawing two animals from
one sentence shows the scale is high enough to render the pair.

**The eight seeds are not the SDXL eight.** SDXL's counts come from held-out seeds 9 to 16, which
were held out of a training set this model has nothing to do with. PixArt-alpha trained on neither
set, so seeds 1 to 8 are as arbitrary here as any other eight, and the sidecar names them.

**The by-eye call.** One reader's judgment on eight pictures. The sheet is published so the call
can be disagreed with per seed, and instruction 7.2 is where that disagreement rewrites the counts.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The sheet, as a figure, with the SDXL counts in its caption rather than in the body text.
- One sentence in the paper's discussion of what causes the failure, naming the model, its text
  encoder and the counts, and saying plainly that the comparison is mixed.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- **Whether any of the other four differences matters.** Only a null here makes that question
  live, and it would be answered by matching the sampler and step count across the two models
  rather than by a third model.
- **Whether the fusion survives on a third text encoder.** Not planned. A second CLIP-free model
  would strengthen the removal but is not needed for a one-directional claim.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

The finding is [does a clip-free model hybridise](../../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md),
and the scope's index carries its row. The question this opens, and does not answer, is which of the
five differences carries the failure; the cheapest next cut is SDXL re-run at PixArt's own 20 steps,
DPM-Solver++ and guidance 4.5, which removes four of the five in one run.

## Cross-references

Navigation: ⬅️ [Next step](#next-step) | 📋 [TOC](#table-of-contents)

- [The corrector finding](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md),
  which owns the SDXL counts quoted here.
- [What product-of-experts composition is](../../../../context/world/poe-composition.md) and
  [the compose rate](../../../../context/world/compose-rate.md), for what the arithmetic is and what a
  rate counts.
- [The compose-score validation set](../../../artifacts/results/can-we-trust-the-compose-score/README.md),
  for why the instance-count scorer cannot make these calls.
