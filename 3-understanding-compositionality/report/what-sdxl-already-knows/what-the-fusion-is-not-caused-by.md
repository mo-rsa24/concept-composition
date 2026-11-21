# 📒 Report: what the fused cat-dog is not caused by

**The fusion survives removing CLIP and survives changing SDXL's sampler, step count and guidance together: the product of "a cat" and "a dog" holds two clear animals on 0 of 8 seeds in all three setups.** One summary grid shows this. The verdicts belong to two findings in the correction project, which this page links and does not restate.

## Table of contents

- [Three setups, one result](#three-setups-one-result)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Three setups, one result

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

<a href="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png"><img src="../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png" width="200" alt="Three setups side by side, four seeds each: the sentence a cat and a dog draws two animals and the sum of the two prompts draws one fused animal"></a>

**In all three setups one sentence draws two animals and the sum of two prompts draws one.** Each row is one seed. Within a setup, the left column is the sentence "a cat and a dog" and the right column is the sum of "a cat" and "a dog" `verified` (read on the picture). The grid shows 4 of the 8 seeds per setup.

| Setup | Model's text encoder | Sampling | Sum: two clear animals | Sentence: two animals | Finding |
|---|---|---|---|---|---|
| SDXL, usual settings | includes CLIP | DDIM, 50 steps, guidance 7.5 | 0 of 8 | 8 of 8 separate animals; 5 of 8 a clear cat and dog | [control in the settings finding](../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md#1-the-control-reproduces-the-published-counts) |
| SDXL, PixArt's settings | includes CLIP | DPM-Solver++ order 2, 20 steps, guidance 4.5 | 0 of 8 | 8 of 8 separate animals; 5 of 8 a clear cat and dog | [the settings finding](../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md) |
| PixArt-alpha | Flan-T5-XXL, no CLIP | its own defaults, 20 steps, guidance 4.5 | 0 of 8 | 7 of 8 a clear cat and dog | [the CLIP-free finding](../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md) |

All counts are as those two findings record them `stated`. Both findings carry a support verdict against a bar written before their runs.

## Where the sources disagree

Navigation: ⬅️ [Previous](#three-setups-one-result) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two descriptions of this grid need a qualifier.**

| What | One source | The other source |
|---|---|---|
| what the sentence draws | the picture's headline says every setup "draws two animals" | the findings count two animals of any species. For a clear cat beside a clear dog, SDXL reaches 5 of 8. Seed 13 in the first setup reads as two dogs on the grid `inferred` |
| where the runs are recorded | [the card](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/README.md) says the runs behind each panel are not recorded beside it | the two findings record the runs and scripts. No file links a panel of this grid to a tile, and no script in the repository carries its title |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**Each change moves several things at once.** Sampler, step count and guidance change together between the first two setups, and PixArt differs from SDXL in its denoiser as well as its text encoder.

**The two SDXL setups and PixArt use different seeds on the grid** (9, 12, 13, 15 against 1, 4, 5, 8), so a row across all three is not the same starting noise.

**A cause that is not necessary can still contribute.** The grid shows the fusion happens without CLIP. It does not show that CLIP plays no part in SDXL.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. How the sum of two prompts works is taught on [the cats and dogs page](does-poe-of-cats-and-dogs-match-the-joint.md#how-hard-a-product-pushes).

**Ruling out a necessary cause**

Suppose 8 of 8 cakes burn in a gas oven, and you suspect the gas. Bake 8 more in an electric oven and 8 of 8 burn again. Gas is therefore not needed for burning, though it may still add to it. Here the suspect is CLIP: the fusion appears on 8 of 8 seeds in a model with no CLIP, so CLIP is not needed for it.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the picture | [the summary grid](../../artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png), opened 2026-11-22 | `verified` |
| counts for both SDXL setups | [does SDXL still fuse at PixArt's settings](../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md) | `stated` |
| counts for PixArt | [does a model with no CLIP still fuse](../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md) | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Record which tiles and script built this grid. The card states its subject from the file name alone.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
