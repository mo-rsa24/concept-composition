# 📒 Report: how the SDXL autoencoder is built

**This page explains how the model works, so it carries no verdict.** SDXL's variational autoencoder
(VAE) compresses a 1024 by 1024 image into a 4 by 128 by 128 latent, the small code the denoising
network works on, and decodes that code back to pixels. Two scenes walk through it on this laptop:
[the shape of the VAE](../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-scene.html)
and [inside the VAE](../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-deep-learning-scene.html).
A [published VAE page](https://claude.ai/artifact/PWPgjgzaRtjZnBMnVXx3BU) also exists.

## Table of contents

- [The autoencoder end to end](#the-autoencoder-end-to-end)
- [Where the parameters sit](#where-the-parameters-sit)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The autoencoder end to end

Navigation: 📋 [TOC](#table-of-contents) | [Next](#where-the-parameters-sit) ➡️

<a href="../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-stills/sdxl-vae-high-level-design-explainer__system-light.png"><img src="../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-stills/sdxl-vae-high-level-design-explainer__system-light.png" width="200" alt="The VAE as one chain: image, encoder, mean and log-variance, sample z, the latent, decoder."></a>

**The encoder halves the image's width and height three times, so each latent position stands for 8 by
8 pixels.** The picture reads left to right, and its right edge is cut off in this still `verified`.

| Step | Shape after it | Mark |
|---|---|---|
| image in | 3 channels by 1024 by 1024 | `verified` |
| encoder, with 128, 256, 512 and 512 channels across its four stages | 8 by 128 by 128 | `verified` |
| split into a mean and a log-variance | 4 by 128 by 128 each | `verified` |
| draw one sample, z = mean + spread × noise, then multiply by 0.13025 | 4 by 128 by 128 | `verified` |
| decoder, after dividing by 0.13025 | 3 by 1024 by 1024 | `verified` |

Why the 0.13025 is there has [its own page](why-the-latent-is-scaled.md).

## Where the parameters sit

Navigation: ⬅️ [Previous](#the-autoencoder-end-to-end) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**The decoder holds 59.16% of the 83,653,863 parameters.** It has 14 residual blocks to the encoder's
10, because each of its stages carries three where the encoder's carry two `verified`. Each share is
the part's parameter count divided by the total, from the `nodes` table in
[the VAE tree data](../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/tree-data.json).

| Part | Parameters | Share of the total |
|---|---|---|
| encoder | 34,163,592 | 40.84% |
| decoder | 49,490,179 | 59.16% |
| the two small convolutions beside the latent | 72 + 20 | under 0.001% |

The decoder's stages hold 15,345,408 more parameters than the encoder's (38,972,544 against
23,627,136). The whole decoder holds 15,326,587 more than the whole encoder, which is the "extra 15M"
the scene names `verified`.

## Where the sources disagree

Navigation: ⬅️ [Previous](#where-the-parameters-sit) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| the scaling factor | `model/autoencoder_kl.py` line 83 has the class default 0.18215 | the traced SDXL config in `tree-data.json` has 0.13025, the value both scenes draw |
| which scene is published | [the scaling card](../../artifacts/results/how-sdxl-works/why-the-latent-is-scaled/README.md) gives one link for "the VAE page" | this folder holds two VAE scenes with different titles, and no card says which one the link serves |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**These counts describe the model's structure, and say nothing about how well it reconstructs an
image.** The trace's provenance reads "shape-traced, CPU, random-init from cached config, batch 1", so
no trained weight was used.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The project numbers come from the sources above. No vault or textbook
link is recorded for these concepts yet.

**Downsampling by halves**

A photo 8 tiles wide, halved three times, is 4, then 2, then 1 tile wide, so one final tile covers
2 × 2 × 2 = 8 of the originals. "Down" is the shrinking and "sampling" is keeping fewer positions.
SDXL's encoder does this to 1024, giving 512, 256, then 128.

**Drawing a latent from a mean and a spread**

Suppose a dice-rolling machine reports a mean of 2 and a spread of 0.5, and the random noise comes up
as 1. The draw is 2 + 0.5 × 1 = 2.5. The formula is z = μ + σ × ε, read as the mean, plus the spread,
times a random number. The VAE makes one such draw at each of its 4 × 128 × 128 latent values.

**Counting a convolution's parameters**

A 3 by 3 filter from 1 input channel to 2 output channels has 3 × 3 × 1 × 2 = 18 weights and 2 offsets,
20 in all. The encoder's first convolution runs from 3 channels to 128: 3 × 3 × 3 × 128 + 128 = 3,584,
the count the tree data records for `encoder.conv_in`.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| config, shapes and parameter counts | [the VAE tree data](../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/tree-data.json), 242 modules, read 2026-11-22 | `verified` |
| class default and code lines | the copies in `artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/model/`, read 2026-11-22 | `verified` |
| what the folder holds | [the VAE card](../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/README.md) | `stated` |
| the published link | [the scaling card](../../artifacts/results/how-sdxl-works/why-the-latent-is-scaled/README.md); the link itself was not opened | `stated` |
| 16 pictures held out of the drain | the other light and dark stills and one SVG copy of the system still, in this result's folder. They repeat the pictures placed above | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Record which of the two VAE scenes the published link serves.
- [ ] Write card entries for `vae-stills/` and the design picture, which the grouping card names only as a folder.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
