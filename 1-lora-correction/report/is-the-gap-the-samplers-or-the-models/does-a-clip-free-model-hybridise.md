# Does a model with no CLIP in it still fuse a cat and a dog?   🟢 support · verified 2026-11-04

**The claim**

Yes. Summed-guidance [product-of-experts](../../../context/world/poe-composition.md) composition on PixArt-alpha, whose only text encoder is
Flan-T5-XXL and which carries no CLIP in any role, holds two clear animals on **0 of 8** seeds: six
renders are one hybrid animal and two are one single cat. The joint prompt "a cat and a dog" on the
same eight seeds and the same settings draws a clear cat beside a clear dog on **7 of 8**.

CLIP is removed as a necessary cause of the fusion. An account resting on CLIP's pooled caption
representation being unable to hold two separate things cannot be the whole story, because the same
arithmetic fuses on a model that has never seen a CLIP embedding.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. The product fuses on every seed](#1-the-product-fuses-on-every-seed)
- [2. The joint prompt at the same guidance draws the pair](#2-the-joint-prompt-at-the-same-guidance-draws-the-pair)
- [3. The composition path is the released pipeline](#3-the-composition-path-is-the-released-pipeline)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-the-product-fuses-on-every-seed) ➡️

Written into
[the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/14-does-a-clip-free-model-hybridise.md)
before anything ran: support at 2 or fewer of 8 product renders holding two clear animals while the
joint column holds two on 6 or more of 8; null at 6 or more of 8 product renders composing;
inconclusive between 3 and 5 of 8, or whenever the joint column itself falls below 6 of 8, which
would mean the model cannot draw the pair at all.

What would have made the run worthless is a composition path that does not reproduce the released
pipeline's own single-prompt output, since the product column would then measure the plumbing.

## 1. The product fuses on every seed

Navigation: ⬅️ [What would have counted](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-the-joint-prompt-at-the-same-guidance-draws-the-pair) ➡️

Zero of eight. Six tiles carry cat ears and whiskers on a canine muzzle, or a cat's face under
folded dog ears; two (seeds 2 and 3) are one clean cat with no canine feature at all. Every call was
made by eye from the full-resolution 1024² tile, because the repo's instance-count scorer counts
animals without naming them and reads a hybrid and two cats identically.

<a href="../../artifacts/results/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png"><img src="../../artifacts/results/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png" width="200" alt="Eight seeds down, four columns across: cat alone, dog alone, the product, the joint prompt"></a>

Per-tile categories with one sentence each are in
[the calls](../../artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/calls.json).

## 2. The joint prompt at the same guidance draws the pair

Navigation: ⬅️ [1. The product fuses on every seed](#1-the-product-fuses-on-every-seed) | 📋 [TOC](#table-of-contents) | [Next](#3-the-composition-path-is-the-released-pipeline) ➡️

Seven of eight. This is the control for the guidance scale: PixArt runs at 4.5 where SDXL runs at
7.5, so a fusion here could have been a weak-guidance artefact rather than the failure under study.
The same 4.5 drawing a clean pair from one sentence shows the scale is high enough to render two
animals. The one failure, seed 2, draws two cats rather than a cat and a dog, and counts as neither.

## 3. The composition path is the released pipeline

Navigation: ⬅️ [2. The joint prompt at the same guidance draws the pair](#2-the-joint-prompt-at-the-same-guidance-draws-the-pair) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

Before any product render, "a cat" at seed 1 was rendered twice: once through `PixArtAlphaPipeline`
from the raw prompt string, once through the composition script with one condition. Largest absolute
difference **0 of 255 grey levels**, 0.0 of pixels differing, and the two PNGs byte-identical (md5
`d9c08c28048b8da98124be0246099b07`). At one condition the script's sum has a single term, so it
evaluates the pipeline's own expression in the same order and nothing is reassociated.

## What this cannot tell you

Navigation: ⬅️ [3. The composition path is the released pipeline](#3-the-composition-path-is-the-released-pipeline) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The comparison is mixed.** Swapping SDXL for PixArt-alpha changes five things at once: the text
encoder, the denoiser (a UNet for a diffusion transformer), the sampler, the step count and the
guidance scale. Only the fusion direction licenses a claim, because no other difference can
manufacture a fusion that CLIP was required for. Two clear animals would have licensed nothing.

**Which of the five carries the failure is untouched.** This removes one candidate; it does not
name the cause.

**Whether CLIP is sufficient for the fusion is untouched too.** Removing a necessary cause and
showing a sufficient one are different claims.

**The seeds are not SDXL's seeds.** SDXL's counts come from held-out seeds 9 to 16, held out of a
training set this model has nothing to do with. PixArt trained on neither set, so seeds 1 to 8 are
as arbitrary here as any other eight.

**The calls are one reader's.** The sheet and the per-tile sentences are published so a call can be
disagreed with per seed.

## Where this came from

Navigation: ⬅️ [What this cannot tell you](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

`1-lora-correction/scripts/pixart_multi_poe.py`, run 2026-11-04 on mscluster84, an RTX 3090, at 20 steps, guidance
4.5, DPM-Solver++ order 2, 1024², fp16, which are PixArt-alpha's own pipeline defaults rather than
SDXL's 50 steps at 7.5. Weights at
`/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS`; tiles and sidecars at
`/datasets/mmolefe/poe_repair_min/outputs/clip_free_hybridise/`. The card is
[does a clip-free model hybridise](../../artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/README.md).

## Depends on

Navigation: ⬅️ [Where this came from](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [Does a corrector alone produce two animals](does-a-corrector-alone-produce-two-animals.md),
  which owns the SDXL counts quoted here: joint prompt 8 of 8, plain product 0 of 8, held-out seeds
  9 to 16, counted by the validated instance count.
- [What product-of-experts composition is](../../../context/world/poe-composition.md) and
  [the compose rate](../../../context/world/compose-rate.md).

## Still open

Navigation: ⬅️ [Depends on](#depends-on) | 📋 [TOC](#table-of-contents)

- **Which of the five differences carries the failure.** The cheapest cut is SDXL re-run at
  PixArt's own 20 steps, DPM-Solver++ and guidance 4.5, which removes four of the five in one run
  and needs no download.
- **Whether the fusion survives a text encoder that is neither CLIP nor T5.** Sana conditions on
  Gemma-2, a decoder-only language model; a fusion there would remove the text encoder as a family
  rather than removing one encoder. Not planned.
- **Whether the two single-animal tiles are the product dropping a prompt or out-voting it.** Not
  answerable from eight pictures.
