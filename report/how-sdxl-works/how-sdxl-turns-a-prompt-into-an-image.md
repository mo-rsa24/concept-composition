# 📒 Report: how SDXL turns a prompt into an image

**This page explains how the model works, so it carries no verdict.** SDXL is the image model every
project here builds on. A prompt becomes numbers, those numbers steer a denoising network over 50
steps, and a decoder turns the result into a 1024 by 1024 picture. To click through it, open
[the SDXL scene](../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/sdxl-scene.html)
on this laptop or [the published SDXL map](https://claude.ai/artifact/KPpMEVBfkJnq3mwgGSbyPs).

## Table of contents

- [The whole system on one sheet](#the-whole-system-on-one-sheet)
- [From the prompt to 77 integers](#from-the-prompt-to-77-integers)
- [How the prompt reaches the image](#how-the-prompt-reaches-the-image)
- [How byte-pair merges shrink text](#how-byte-pair-merges-shrink-text)
- [What joins the timestep](#what-joins-the-timestep)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## The whole system on one sheet

Navigation: 📋 [TOC](#table-of-contents) | [Next](#from-the-prompt-to-77-integers) ➡️

<a href="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/sdxl-stills/sdxl-ldm-high-level-design-explainer__system-light.png"><img src="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/sdxl-stills/sdxl-ldm-high-level-design-explainer__system-light.png" width="200" alt="The SDXL system map in four bands: pixel space, latent space, what is added to the timestep, and cross-attention."></a>

**The prompt reaches the denoising network (the UNet) by two routes.** The word-by-word route enters
through 70 cross-attention layers as a table of 77 token positions by 2048 numbers. The summary route
joins the timestep as one vector of 1280 numbers `verified` (the picture and
[the shape trace](../../artifacts/scenes/sdxl-vae-architecture-map/cross_trace.json)). The still is
cut off at its right edge `verified`.

| Part | Size | Mark |
|---|---|---|
| first text encoder, CLIP ViT-L | 12 layers, 768 numbers per token | `verified` |
| second text encoder, OpenCLIP ViT-bigG | 32 layers, 1280 numbers per token | `verified` |
| latent, the compressed image the UNet works on | 4 channels by 128 by 128 positions | `verified` |
| UNet | 2,567,463,684 parameters, 70 cross-attention layers | `verified` |
| [LoRA adapter](../../context/world/lora-corrector.md) in the traced model | rank 8, 4,956,160 trainable parameters, 0.19% of the UNet's | `verified` |
| sampling loop | 50 DDIM steps `stated` (the picture), first timestep 981 `verified` | |

## From the prompt to 77 integers

Navigation: ⬅️ [Previous](#the-whole-system-on-one-sheet) | 📋 [TOC](#table-of-contents) | [Next](#how-the-prompt-reaches-the-image) ➡️

| What goes in | What comes out |
|---|---|
| <a href="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/tokenizer-input.png"><img src="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/tokenizer-input.png" width="200" alt="Two input strings: the prompt a cat and the empty null token."></a> | <a href="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/tokenizer-lane.png"><img src="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/tokenizer-lane.png" width="200" alt="A lane of six token boxes over their integer ids, starting 49406, 320, 2368, 49407."></a> |
| The tokenizer receives the prompt and an empty prompt, and pads both to the same length `verified`. | "a cat" becomes 49406, 320, 2368, 49407, and 49407 repeats to fill the 77 positions `verified` for the first 12 ids, the ones the trace keeps. |

## How the prompt reaches the image

Navigation: ⬅️ [Previous](#from-the-prompt-to-77-integers) | 📋 [TOC](#table-of-contents) | [Next](#how-byte-pair-merges-shrink-text) ➡️

<a href="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/unet-crossattn.png"><img src="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/unet-crossattn.png" width="200" alt="One head of a cross-attention layer: the latent gives queries, the prompt gives keys and values, and a softmax weights the values."></a>

**Each of the 4096 image positions in this layer reads all 77 token positions, through 10 heads of 64
numbers each.** Those heads together make the layer's width of 640 `verified` (the trace, layer
`down_blocks.1.attentions.0.transformer_blocks.0.attn2`). The attention weights drawn are illustrative
because the trace kept shapes and left out values `stated` (the card).

## How byte-pair merges shrink text

Navigation: ⬅️ [Previous](#how-the-prompt-reaches-the-image) | 📋 [TOC](#table-of-contents) | [Next](#what-joins-the-timestep) ➡️

<a href="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/bpe-6.png"><img src="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/bpe-6.png" width="200" alt="The sixth merge frame: theat and er join into theater, leaving 24 tokens from 44 characters."></a>

**Six merges take a 44-character sentence down to 24 tokens.** This last frame joins `theat` and `er`
in 2 places. Rerunning the merge rule from `make_bpe_carousel.py` on its sentence, "the theater is
larger than the other theater", gives 44, 38, 33, 30, 28, 26, then 24 tokens `verified`. The script
computes its merges on this one sentence `stated` (the card).

## What joins the timestep

Navigation: ⬅️ [Previous](#how-byte-pair-merges-shrink-text) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

<a href="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/sdxl-stills/sdxl-ldm-high-level-design-explainer__tsum-inside-light.png"><img src="../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/sdxl-stills/sdxl-ldm-high-level-design-explainer__tsum-inside-light.png" width="200" alt="Inside the sum: size and crop numbers get Fourier features, join the pooled text at width 2816, and a small network projects them to 1280."></a>

**The image size, crop and pooled text are packed into one 1280-wide vector and added to the timestep's
embedding.** Six size and crop numbers get 256 features each, which makes 1536. The 1280 pooled text
numbers join them for 2816, and a two-layer network brings that back to 1280 `verified`
(`model/unet_2d_condition.py` lines 775 to 799, and the trace's first projection layer holds 3,605,760
parameters, which is 2816 × 1280 weights plus 1280 offsets).

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-joins-the-timestep) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Four drawn details differ from the code or the trace, and each is shown here unresolved.**

| What | The picture | The code or trace |
|---|---|---|
| how many tokens "a cat" has | `tokenizer-lane.png` says 3 positions carry the prompt "plus its start and end markers", and shades 4 boxes | the trace's `n_nonpad_tokens` is 3, the count of ids other than 49407: start marker, `a`, `cat`. The words are 2 tokens, and with both markers the prompt fills 4 positions |
| how many tokens the empty prompt has | `tokenizer-input.png` says "1 token of 77" | the trace's ids open 49406, 49407, a start marker and an end marker |
| how many tokenizers there are | the system still draws one `CLIPTokenizer` box feeding both encoders | `model/pipeline_xl.py` line 274 builds a list of two, `tokenizer` and `tokenizer_2` |
| how the timestep is embedded | the sum still runs `t` into the same "Fourier features, 256 per value" box as the six size and crop numbers | `model/unet_2d_condition.py` line 259 gives `t` its own features, 320 wide, and line 799 adds the two embeddings after projection. The trace's `time_embedding.linear_1` holds 410,880 parameters, which is 320 × 1280 weights plus 1280 offsets |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The trace measures shapes on random weights, so it says nothing about what the trained model has
learned.** Its own provenance reads "shape-traced, CPU, random-init from cached config", on diffusers
0.29.2. It ran a product-of-experts batch with three prompts, and the pictures draw only the prompt and
the empty prompt `stated` (the card).

**Rank 8 describes the traced adapter only.** The pooled adapter runs in this project are configured
at rank 32, per the `r32` in their folder names under `artifacts/_shared/cross_pair_pool_configs/all_groups/`
`stated`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The project numbers come from the sources above. No vault or textbook
link is recorded for these concepts yet.

**A byte-pair merge**

Take "banana" as letters: b a n a n a. The neighbouring pair "an" appears 2 times, so "byte pair" names
two neighbours and "merge" means they become one token: b an an a, 4 tokens from 6. Each merge removes
as many tokens as the pair had places, as long as those places do not overlap. On the project's sentence
the first merge joins `t` and `h` in 6 places, so 44 tokens become 38.

**Cross-attention**

Three coffee reviews score 10, 20 and 30, and you trust them 0.7, 0.2 and 0.1. Your reading is
0.7 × 10 + 0.2 × 20 + 0.1 × 30 = 14. "Attention" is the trust, and "cross" means the one asking (an
image position) and the ones answering (the prompt's tokens) come from different sources. The query
sets the trust, the keys are what it is compared with, and the values are the scores. Here one image
position spreads its trust over 77 tokens in each of 10 heads.

**Concatenating widths**

Joining the lists [1, 2] and [3] gives [1, 2, 3], so the widths add: 2 + 1 = 3. Nothing is summed
inside. SDXL does this to its two text encoders, 768 + 1280 = 2048, and to the pooled text and the size
numbers, 1280 + 1536 = 2816.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| shapes, layer counts, parameter counts, token ids | [the shape trace](../../artifacts/scenes/sdxl-vae-architecture-map/cross_trace.json), read 2026-11-22 | `verified` |
| code lines | the copies in `artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/model/`, read 2026-11-22 | `verified` |
| how each picture was drawn | [the pictures card](../../artifacts/results/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image/pictures/README.md) | `stated` |
| the published link | [the scaling card](../../artifacts/results/how-sdxl-works/why-the-latent-is-scaled/README.md); the link itself was not opened | `stated` |
| 32 pictures held out of the drain | the other light and dark stills, the earlier frames of the placed byte-pair picture, and one SVG copy of the system still, in this result's folder. They repeat the pictures placed above | `stated` |

> [!NOTE]
> The SDXL scene embeds nine teaching slides from the Kempner Institute and Harvard workshop "Building
> a Transformer from Scratch", and they travel with the scene if it is shared. They are kept off this
> page as material fetched from the web: `pictures/kempner/`. Five reference figures in `pictures/` have
> no recorded source and were not opened: `latent-diffusion-figure-with-glyph-legend.png`,
> `latent-diffusion-figure-with-forward-diffusion-strip.jpeg`, `sd-pipeline-vae-and-unet-on-one-sheet.png`,
> `clip-contrastive-training-in-three-steps.png` and `transformer-architecture.png` (not shown: not
> cleared against the subjects the pair ledger keeps off sheets).

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Decide whether the four drawn details in the disagreement table should be redrawn.
- [ ] Decide whether the nine workshop slides stay in the published SDXL map.
- [ ] Write card entries for `sdxl-stills/` and the design picture, which the grouping card names only as a folder and a link.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
