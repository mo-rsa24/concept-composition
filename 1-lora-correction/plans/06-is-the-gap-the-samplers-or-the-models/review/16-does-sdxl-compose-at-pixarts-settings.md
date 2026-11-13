# 🧪 Review: does SDXL still fuse when it is sampled the way PixArt-alpha is?

The verdict file for
[hypothesis-16](../plans/hypothesis/16-does-sdxl-compose-at-pixarts-settings.md). The design is
there; what the runs cost, what came back and what it licenses are here.

## Recommended prompt (when the run lands)

```
/analyze-run @plans/06-is-the-gap-the-samplers-or-the-models/review/16-does-sdxl-compose-at-pixarts-settings.md — <the three counts out of eight>
```

## Position in the plan tree

**Step 79 of 79.** Pairs with
[hypothesis-16](../plans/hypothesis/16-does-sdxl-compose-at-pixarts-settings.md). The one order is
the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

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
  "a cat" and "a dog", of each prompt's difference from the empty prediction.
- **The joint prompt**: "a cat and a dog" as one sentence, no product. The reference column.
- **SDXL's own settings**: 50 steps of DDIM at guidance 7.5, which every published SDXL number in
  this project came from.
- **PixArt's settings**: 20 steps of DPM-Solver++ order 2 at guidance 4.5, that model's released
  defaults.
- **Two clear animals**: a cat that reads as a cat and a dog that reads as a dog, both whole, in
  one picture, judged at full resolution. This is the product column's category.
- **Two separate animals**: two whole animals of any species, which is what the instance-count
  scorer measures and what every published SDXL count in this project was measured with. This is
  the joint column's category, because on held-out seeds 9 to 16 the joint prompt draws two animals
  of the same species on about half the seeds, so a species-aware reading of that column would sit
  below any reachable bar before the run started. The question the column has to answer is whether
  the configuration renders two separate subjects at all.
- **One hybrid animal**: a single animal carrying features of both. Its own category.
- **One single animal**: one animal of one species, with no feature of the other.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim**, on one axis. Same model, same weights, same cached starting noise, same eight
held-out seeds; the sampler, the step count and the guidance scale move together to PixArt's
values. The three are one choice and are not separated here, so a null names the group rather than
a member of it.

What would make the run worthless is a code path that differs from the one the published counts
came from. The control run at task 2.1 is the gate, and it runs before the matched settings.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The control, DDIM 50 steps at 7.5, seeds 9 to 16, four columns | gates the claim | 2026-11-04, mscluster84 | 32 renders, 25 minutes | tiles under `1-lora-correction/artifacts/results/does-sdxl-compose-at-pixarts-settings/control/` | ✅ product 0 of 8, joint two animals 8 of 8 |
| The matched run, DPM-Solver++ 20 steps at 4.5, same seeds, four columns | Tests the claim | 2026-11-04, mscluster84 | 32 renders, 13 minutes | tiles and sheet under `1-lora-correction/artifacts/results/does-sdxl-compose-at-pixarts-settings/` | ✅ product 0 of 8, joint two animals 8 of 8 |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] **Does the product of "a cat" and "a dog" on SDXL hold two clear animals when SDXL is sampled
      at 20 steps of DPM-Solver++ and guidance 4.5?** Held-out seeds 9 to 16, the same cached
      starting latents every other SDXL result uses, with the joint prompt on the same seeds and
      the same settings as the reference column. Every call by eye from the full-resolution tile,
      because the instance-count scorer counts animals without naming them.

      **Support** if 2 or fewer of 8 product renders hold two clear animals while the joint-prompt
      column holds two separate animals on 6 or more of 8. The sampler, the step count and the guidance scale are
      removed as the carrier of the fusion, leaving the text encoder and the denoiser.

      **Null** if 6 or more of 8 product renders hold two clear animals. One of those three
      settings carries the failure, which would mean the published 0 of 8 describes a
      configuration rather than the arithmetic.

      **Inconclusive** at 3 to 5 of 8, or whenever the joint column holds two separate animals on
      fewer than 6 of 8.

      The comparison is against this project's own published SDXL counts: joint prompt 8 of 8,
      plain product 0 of 8, from
      [the corrector finding](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md),
      and against PixArt's 0 of 8 product and 7 of 8 joint, from
      [does a clip-free model hybridise](../../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md).

      **Verdict: SUPPORT.** At DPM-Solver++ order 2, 20 steps and guidance 4.5 the product holds
      two clear animals on **0 of 8** seeds: seven tiles are one hybrid animal and the eighth, seed
      16, holds two bodies whose species are both unclear and is counted as neither. The joint
      column holds two separate animals on **8 of 8**, above the bar, and a clear cat beside a
      clear dog on 5 of 8 (failing at seeds 14 and 15, two cats, and seed 16, two dogs). The sampler, the step count and the guidance scale are removed as the
      carrier of the fusion. Counts and per-tile sentences in
      [the calls](../../../artifacts/results/does-sdxl-compose-at-pixarts-settings/calls.json),
      card in [the same folder](../../../artifacts/results/does-sdxl-compose-at-pixarts-settings/README.md).

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] **Does the new script reproduce the published renders at SDXL's own settings?** The control
      run's product column must read 0 of 8 and its joint column two separate animals on 8 of 8.

      **Answered: yes.** DDIM, 50 steps, guidance 7.5, seeds 9 to 16, run 2026-11-04 on mscluster84:
      the product column holds one animal on 8 of 8, including seed 16, which is one body carrying
      two heads rather than two animals. The joint column holds two separate animals on 8 of 8, and
      a clear cat beside a clear dog on 5 of 8 (seeds 9, 11, 12, 15, 16); seeds 10 and 13 draw two
      dogs and seed 14 two cats. Tiles and sheet under
      `1-lora-correction/artifacts/results/does-sdxl-compose-at-pixarts-settings/control/`.
- [x] **Which seeds did what?** One record per product tile in `calls.json`, with its seed, its
      category and one sentence on what is in the picture.

      **Answered.** Hybrids on seeds 9 to 15 at both settings; seed 16 is one body with two heads
      at the control's settings and two bodies of unclear species at the matched settings.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Post-hoc: seed 16 moves from one body with two heads to two bodies.** The only seed where the two
settings differ in kind rather than in style. One tile is not a result, and nothing is claimed from
it.

**Post-hoc: 20 steps does not look thinner than 50 here.** The single-prompt columns are as sharp
at the matched settings, so the hybrid calls are not being made on vague renders.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

**Three settings moved together.** Deliberate, stated in the run kind. A null names the group.

**The lower guidance scale.** At 4.5 every prompt pushes less hard, so a fusion could be a
weak-guidance artefact. The joint column at the same 4.5 is the control for it.

**Twenty steps is fewer steps.** A thinner render could read as a hybrid by being vague rather than
by fusing. The single-prompt columns at the same 20 steps show what a clean cat and a clean dog look
like at this budget, which is what the hybrid call is made against.

**The joint prompt is unreliable on these seeds.** It draws two animals of the same species on 3 of
8, so the column is read as two separate animals rather than as a cat and a dog. That is what the
published SDXL counts measured, and it is the right control for whether the configuration renders
two subjects, but it does not show the model can tell these two species apart at these settings.

**The by-eye call.** One reader's judgment on eight pictures. The sheet is published so the call can
be disagreed with per seed, and instruction 7.2 is where that rewrites the counts.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The sheet, as a figure, with the SDXL-own-settings and PixArt counts in its caption.
- One sentence in the paper's discussion of what causes the failure, saying whether the sampling
  configuration is in or out.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- **Which of the three settings matters, if any does.** Only a null here makes that live, and it
  would be answered by moving one at a time.
- **Whether the denoiser or the text encoder carries it.** Untouched by this run either way; a
  second CLIP-free model with a different encoder family is the next cut on that axis.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Nothing waits on this file. The finding lands under
`1-lora-correction/report/is-the-gap-the-samplers-or-the-models/` either way, because both outcomes change what the
paper may claim.

## Cross-references

Navigation: ⬅️ [Next step](#next-step) | 📋 [TOC](#table-of-contents)

- [Does a clip-free model hybridise](../../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md),
  the run this one narrows.
- [Does a corrector alone produce two animals](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md),
  which owns the published SDXL counts the control must reproduce.
- [What product-of-experts composition is](../../../../context/world/poe-composition.md) and
  [the compose rate](../../../../context/world/compose-rate.md).
