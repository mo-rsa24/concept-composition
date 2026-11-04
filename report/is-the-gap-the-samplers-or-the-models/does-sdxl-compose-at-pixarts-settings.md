# Does SDXL still fuse when it is sampled the way PixArt-alpha is?   🟢 support · verified 2026-11-04

**The claim**

Yes. Changing SDXL's sampler, step count and guidance scale to PixArt-alpha's released defaults
(DPM-Solver++ order 2, 20 steps, guidance 4.5) leaves the summed-guidance [product](../../context/world/poe-composition.md) of "a cat" and
"a dog" holding two clear animals on **0 of 8** held-out seeds, the same as at SDXL's own 50 DDIM
steps and guidance 7.5.

The sampling configuration is removed as the carrier of the fusion. With CLIP already removed by
[does a clip-free model hybridise](does-a-clip-free-model-hybridise.md), the five differences
between SDXL and PixArt-alpha are down to the denoiser architecture and a text encoder that has
already been shown not to be necessary.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. The control reproduces the published counts](#1-the-control-reproduces-the-published-counts)
- [2. At PixArt's settings the product still fuses](#2-at-pixarts-settings-the-product-still-fuses)
- [3. The reference column, read two ways](#3-the-reference-column-read-two-ways)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-the-control-reproduces-the-published-counts) ➡️

Written into
[the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/16-does-sdxl-compose-at-pixarts-settings.md)
before the matched run: support at 2 or fewer of 8 product renders holding two clear animals while
the joint column holds two separate animals on 6 or more of 8; null at 6 or more of 8 product
renders composing; inconclusive between 3 and 5 of 8.

The joint column's instrument was settled before the matched run and is recorded in `DRIFT_LOG.md`:
it is read as two separate animals of any species, which is what every published SDXL count in this
project was measured with, because on these seeds the joint prompt itself draws two animals of the
same species on 3 of 8. The product column's bar, which is the claim under test, was never changed.

## 1. The control reproduces the published counts

Navigation: ⬅️ [What would have counted](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-at-pixarts-settings-the-product-still-fuses) ➡️

The new script at DDIM, 50 steps and guidance 7.5 on seeds 9 to 16: product 0 of 8, joint two
separate animals 8 of 8, matching
[the corrector finding](does-a-corrector-alone-produce-two-animals.md)'s numbers, which came from a
different script. Without this the matched run would be measuring the code rather than the settings.

<a href="../../artifacts/results/does-sdxl-compose-at-pixarts-settings/control/sdxl-control-sheet.png"><img src="../../artifacts/results/does-sdxl-compose-at-pixarts-settings/control/sdxl-control-sheet.png" width="200" alt="Seeds 9 to 16 down, four columns across, at DDIM 50 steps and guidance 7.5"></a>

## 2. At PixArt's settings the product still fuses

Navigation: ⬅️ [1. The control reproduces the published counts](#1-the-control-reproduces-the-published-counts) | 📋 [TOC](#table-of-contents) | [Next](#3-the-reference-column-read-two-ways) ➡️

Zero of eight. Seven tiles are one animal carrying both sets of features, cat ears and whiskers over
a canine muzzle. The eighth, seed 16, holds two bodies, and neither reads clearly as a cat or
clearly as a dog, so it is counted as neither category rather than rounded into the hybrid count.

<a href="../../artifacts/results/does-sdxl-compose-at-pixarts-settings/sdxl-matched-sheet.png"><img src="../../artifacts/results/does-sdxl-compose-at-pixarts-settings/sdxl-matched-sheet.png" width="200" alt="The same eight seeds at DPM-Solver++ order 2, 20 steps and guidance 4.5"></a>

Per-tile sentences for both runs are in
[the calls](../../artifacts/results/does-sdxl-compose-at-pixarts-settings/calls.json).

## 3. The reference column, read two ways

Navigation: ⬅️ [2. At PixArt's settings the product still fuses](#2-at-pixarts-settings-the-product-still-fuses) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

At the matched settings the joint prompt holds two separate animals on 8 of 8, which closes the
weak-guidance escape: at guidance 4.5 this model still renders two subjects from one sentence. Read
for species it holds a clear cat beside a clear dog on 6 of 8, failing on seeds 14 and 15, which
draw two cats. The control's figures are 8 of 8 and 5 of 8.

## What this cannot tell you

Navigation: ⬅️ [3. The reference column, read two ways](#3-the-reference-column-read-two-ways) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Three settings moved together.** The run names the group, not a member of it. Whether 20 steps
alone, or guidance 4.5 alone, changes anything is untested.

**The joint prompt is unreliable on these seeds.** Two to three of eight draw two animals of the
same species, so the reference column shows that two subjects can be rendered, not that this model
separates these two species reliably here.

**Seed 16 is one tile.** The only seed whose failure changes kind between the two settings, and
nothing is claimed from it.

**The calls are one reader's.** Both sheets are published so a call can be disagreed with per seed.

## Where this came from

Navigation: ⬅️ [What this cannot tell you](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

`scripts/sdxl_matched_multi_poe.py`, run 2026-11-04 on mscluster84, an RTX 3090, over held-out seeds
9 to 16 of cat × dog from the repo's cached starting latents. The two pinned SDXL paths,
`scripts/concept_multi_poe.py` and `scripts/and_vector_sheet.py`, were not modified. Tiles and
sidecars at `/datasets/mmolefe/poe_repair_min/outputs/sdxl_matched/`; the card is
[does sdxl compose at pixart's settings](../../artifacts/results/does-sdxl-compose-at-pixarts-settings/README.md).

## Depends on

Navigation: ⬅️ [Where this came from](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [Does a corrector alone produce two animals](does-a-corrector-alone-produce-two-animals.md),
  which owns the published SDXL counts the control reproduces.
- [Does a clip-free model hybridise](does-a-clip-free-model-hybridise.md), the run this one narrows.
- [What product-of-experts composition is](../../context/world/poe-composition.md).

## Still open

Navigation: ⬅️ [Depends on](#depends-on) | 📋 [TOC](#table-of-contents)

- **The denoiser architecture.** The last of the five differences with nothing said about it. A
  second CLIP-free model on a different architecture would speak to it.
- **Which of the three settings does anything at all.** Only interesting now as a sampler question,
  not as a cause of the fusion.
- **Why the joint prompt draws two animals of one species on these seeds.** It weakens every
  reference column measured here and is not this run's to answer.
