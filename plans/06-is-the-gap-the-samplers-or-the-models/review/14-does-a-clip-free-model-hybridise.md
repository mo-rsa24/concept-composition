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
`## Running order` table in the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

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
| The weights fetch, `PixArt-alpha/PixArt-XL-2-1024-MS` onto `/datasets` | setup | 2026-11-04, from mscluster84 | 21.8 GB | `/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS` | ◑ in flight |
| The single-prompt identity check, "a cat" at seed 1 through the pipeline and through the script | gates the claim | not yet | 2 renders | the two images and the largest absolute pixel difference between them | ⚠️ not started |
| The four columns on eight seeds | Tests the claim | not yet | 32 renders at 1024², 20 steps each | tiles and sidecars under `artifacts/results/does-a-clip-free-model-hybridise/`, sheet `pixart-cat-dog-sheet.png` | ⚠️ not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] **Does the product of "a cat" and "a dog" on PixArt-alpha hold two clear animals, or one?**
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

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] **Does the script reproduce the released pipeline's plain output?** The largest absolute
      difference between the two "a cat" renders, in 0-to-255 pixel values, and the fraction of
      pixels that differ at all. A largest difference of 0 means the two paths are the same
      computation. Above about 2 grey levels is a plumbing fault, and the product renders do not
      run until it is fixed rather than tolerated.
- [ ] **Which seeds did what?** One record per product tile in `calls.json`, with its seed, its
      category and one sentence on what is in the picture, so a disagreement later is about a named
      tile rather than about a count.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Nothing yet. Questions that occur to anyone after seeing the sheet go here, marked as post-hoc,
and they do not move the bar above.

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

Nothing waits on this file. On support, the finding lands under
`report/is-the-gap-the-samplers-or-the-models/` and the scope's own index gains a row.

## Cross-references

Navigation: ⬅️ [Next step](#next-step) | 📋 [TOC](#table-of-contents)

- [The corrector finding](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md),
  which owns the SDXL counts quoted here.
- [What product-of-experts composition is](../../../context/world/poe-composition.md) and
  [the compose rate](../../../context/world/compose-rate.md), for what the arithmetic is and what a
  rate counts.
- [The compose-score validation set](../../../artifacts/results/can-we-trust-the-compose-score/README.md),
  for why the instance-count scorer cannot make these calls.
