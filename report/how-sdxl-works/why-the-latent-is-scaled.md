# 📒 Report: why SDXL multiplies its latent by 0.13025

**This page explains how the model works, so it carries no verdict.** SDXL's autoencoder produces a
code with a spread near 7.7, while the noise schedule assumes a spread near 1. One constant, 0.13025,
sits between them. To work through the argument, open
[the scaling scene](../../artifacts/results/how-sdxl-works/why-the-latent-is-scaled/why-the-latent-is-scaled.html)
on this laptop or [its published page](https://claude.ai/artifact/7yTQUKjEawymqWRnk9gvjn).

## Table of contents

- [Where the constant sits](#where-the-constant-sits)
- [What the schedule expects](#what-the-schedule-expects)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Where the constant sits

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-schedule-expects) ➡️

<a href="../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-stills/sdxl-vae-high-level-design-explainer__system-light.png"><img src="../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/vae-stills/sdxl-vae-high-level-design-explainer__system-light.png" width="200" alt="The VAE chain, with the sample step multiplying by 0.13025 and the decoder step dividing by it."></a>

**The latent is multiplied by 0.13025 after encoding and divided by it before decoding.** The picture
shows both steps on the chain `verified`, the traced config holds the value `verified`, and
line 410 of `how-sdxl-turns-a-prompt-into-an-image/model/pipeline_xl.py` does the division `verified`. How the rest of the
chain works is on [the autoencoder page](how-the-sdxl-vae-is-built.md).

> 🖼️ **Figure wanted**: a still of the scaling scene's frame comparing the latent's values at × 0.13025
> and × 1 against the unit normal. Render into `artifacts/results/how-sdxl-works/why-the-latent-is-scaled/`,
> update that card, then embed here.

## What the schedule expects

Navigation: ⬅️ [Previous](#where-the-constant-sits) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**On SDXL's schedule, signal and noise are equally strong at step 354 of 1000.** On a cosine schedule
the same crossing comes at step 496, so the step a page quotes depends on which schedule it computes
from `verified`.

| Schedule | First step where the signal-to-noise ratio is below 1 |
|---|---|
| scaled-linear, beta from 0.00085 to 0.012 over 1000 steps (SDXL's) | 354 |
| cosine, offset 0.008 | 496 |

Both are recomputed for this page. For scaled-linear, beta at step i (counting from 0) is
(√0.00085 + (i / 999)(√0.012 − √0.00085))², and ᾱ is the running product of (1 − beta). For cosine,
ᾱ at step i is g(i + 1) / g(0), with g(t) = cos²(((t / 1000 + 0.008) / 1.008)(π / 2)). The ratio is
ᾱ / (1 − ᾱ). The scaled-linear betas match the fallback in
[the interaction-term cache code](../../poe_repair/experiments/interaction_term/cache.py) at line 99.

## Where the sources disagree

Navigation: ⬅️ [Previous](#what-the-schedule-expects) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**Two recorded facts conflict, and each is shown here unresolved.**

| What | One source | The other source |
|---|---|---|
| whether training pulls the code toward a spread of 1 | the scaling scene's `build_page.py` line 122: "Nothing in the autoencoder's training pushed its code towards standard deviation 1" | `vae-maths.json` line 42 says the KL term "pulls the latent space toward a standard normal", and `build_page.py` line 341 names the KL term as "why the code lands near 7.7 rather than near 1" |
| the factor in the code | line 83 of `how-the-sdxl-vae-is-built/model/autoencoder_kl.py` has the class default 0.18215 | the traced SDXL config has 0.13025 |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

> [!WARNING]
> The spread of 7.68 is the reciprocal of the factor and was never measured here. The scene's own
> notes mark it inferred. Its curves meet at 0.13025 because 7.68 × 0.13025 = 1.00032, so that meeting
> follows from the assumption and gives no independent evidence that the factor is right.

**The picture of a cat in the scene stands in for the latent.** The real latent is a 4 by 128 by 128
code that is not a picture, so the scene's squares show how much survives each step and do not show
what a latent looks like `stated` (the card and the scene's notes). The scene draws no image generated
at a wrong factor.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers are invented. The project numbers come from the sources above. No vault or textbook
link is recorded for these concepts yet.

**Rescaling a spread**

Two dice-game scores sit at -4 and +4, so their average distance from the middle, the standard
deviation, is 4. Multiply both by 0.25 and they sit at -1 and +1, with a standard deviation of 1. The
factor multiplies the spread directly. Here 7.68 × 0.13025 = 1.00032.

**Signal-to-noise ratio**

Mix a drink that is 0.8 juice and 0.2 water, and the ratio of juice to water is 0.8 / 0.2 = 4. "Signal"
is the juice, "noise" is the water, and the ratio is signal over noise. In diffusion the signal share
is ᾱ and the noise share is 1 − ᾱ, so the ratio is ᾱ / (1 − ᾱ). It equals 1 when ᾱ is 0.5, which SDXL's
schedule passes at step 354.

**A noise schedule**

Suppose three steps each remove a share of the signal, called beta: 0.1, 0.2 and 0.4. What survives is
0.9 × 0.8 × 0.6 = 0.432, and that running product is ᾱ. A schedule is the list of betas. "Scaled-linear"
means the square roots of the betas rise in equal steps, and SDXL takes 1000 of them.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| the factor in the config | [the VAE tree data](../../artifacts/results/how-sdxl-works/how-the-sdxl-vae-is-built/tree-data.json), read 2026-11-22 | `verified` |
| the crossings at 354 and 496 | recomputed 2026-11-22 from the formulas above; they match [the scaling card](../../artifacts/results/how-sdxl-works/why-the-latent-is-scaled/README.md) | `verified` |
| the spread of 7.68 | the scene's notes, as 1 / 0.13025 = 7.6775 rounded | `inferred` |
| the published link | the scaling card; the link itself was not opened | `stated` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Measure the raw latent's standard deviation on real images, so 7.68 stops resting on the factor it is used to explain.
- [ ] Settle the scene's wording on whether the KL term pulls the code toward a spread of 1.
- [ ] Render the still the figure slot above asks for.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
