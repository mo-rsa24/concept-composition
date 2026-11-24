# Reading table: can the score path share a space with text and noisy canvases?

This table holds one row for each of 12 papers read from the full PDF with appendices for step X1 of Loop 10: the 7 named in [the loop's brief](../../prompts/session-10-the-score-path-as-a-third-space.md) and 5 found by [the search for score-path encoders](search-for-score-path-encoders.md). Four more papers that Loops 4, 5 and 9 already read are cited from their notes. Each row links to its note, and every claim in a note carries its page.

<a href="three-spaces-and-the-closest-papers.png"><img src="three-spaces-and-the-closest-papers.png" width="620" alt="Three inputs, a caption, a noisy canvas and a score path, each with an arrow toward one shared space. The caption reaches it through a frozen text encoder and the canvas through a trained image encoder. The score path's arrow is dashed, labelled 'no paper trains this leg'. Four boxes of papers point at the score path: hand-built caption scores from noise error, contrastive losses on noise predictions, the summed score path as a quality signal, and a classifier on a reconstruction residual. ImageBind and CLIP goes 3D point at the shared space."></a>

The picture places each paper by what it reads. Its source is [the diagram file](three-spaces-and-the-closest-papers.mmd).

## Table of contents

- [Words this uses](#words-this-uses)
- [What the reading changes for this loop](#what-the-reading-changes-for-this-loop)
- [What these papers could change in the two papers](#what-these-papers-could-change-in-the-two-papers)
- [The table](#the-table)
- [Papers read by other loops and cited here](#papers-read-by-other-loops-and-cited-here)
- [Closer papers these cite, not read](#closer-papers-these-cite-not-read)
- [Where sources disagree](#where-sources-disagree)
- [How the rows were checked](#how-the-rows-were-checked)

## Words this uses

**Score path.** The stack of $d_t(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)$ over sampling steps 0 to 15 for one canvas and one seed: what condition $c$ adds to the unconditional noise prediction at each step.

**Paper 1, the separate-experts paper.** The paper that composes two models trained on disjoint data, with a trained checker guiding the product, as numbered in [the problem statements](../../problem-statements.md), the targets and the scoreboard.

**Paper 2, the CO3 paper.** The paper that improves composition inside one text-to-image model, adding a checker's push to the product or to CO3.

## What the reading changes for this loop

**No paper trains an encoder whose input is a score path.** The search on 2026-11-24 found none in 55 distinct arXiv papers and five web queries, and each of the five closest papers uses only one part of the idea ([the search record](search-for-score-path-encoders.md)).

**The seed is the first thing a score-path encoder could learn, and three papers say how to keep it out.** Clark and Jaini score every caption on one shared noisy input, and that sharing alone cut the samples needed about 100 times ([p. 5](https://arxiv.org/pdf/2303.15233#page=5)). Li et al. show the error gap between two prompts stays steady on a shared draw while each prompt's raw error swings ([Fig. 2, p. 4](https://arxiv.org/pdf/2303.16203#page=4)). DCR feeds one $x_t$ and one $t$ to every condition in its contrastive batch ([Algorithm 1, p. 13](https://arxiv.org/pdf/2603.04803#page=13)). NoiseCLR pairs the same direction across different images as positives, so its loss ignores the image and the draw ([its note](noiseclr.md)). Together these suggest negatives that share a seed and positives that cross seeds (inferred). On SDXL the same pair at two seeds gave a cosine of about 0.002 between interaction terms, a measurement held only in a project memory note of 2026-08-11 with no finding under `report/` that owns it.

**The length of a score path already carries quality, before any encoder.** The sum $\sum_t \lVert d_t \rVert^2$ over the early steps ranks seeds of one prompt by quality, and keeping 4 of 50 seeds by it lifts GenEval two-object on SD1.5 from 0.354 to 0.518 ([p. 8](https://arxiv.org/pdf/2505.23343#page=8)). An encoder fed raw paths can sort by length before it learns the digits, so X4 needs a control that removes length (inferred).

**The conditional-minus-unconditional difference is the right input, and the raw error is not.** DiffusionITM finds the raw denoising error "depends almost entirely on the image", and only the difference from the unconditional error picks an image ([p. 5](https://arxiv.org/pdf/2305.16397#page=5)).

**The high-noise steps this loop reads are the least measured and the weakest.** Li et al.'s single-step accuracy on Pets peaks near middle noise, about 46% near t = 500, read from Fig. 3 ([p. 4](https://arxiv.org/pdf/2303.16203#page=4)). Clark and Jaini weight low noise most ([p. 4](https://arxiv.org/pdf/2303.15233#page=4)). REPA's timestep evidence stops at t = 0.5 ([p. 10](https://arxiv.org/pdf/2410.06940#page=10)). Whether steps 0 to 15 of the digit sampler are the high-noise end is X5's first fact to check.

**A sum of two scores is the product's score only at t = 0.** For every $t > 0$ the sum of the noised scores differs from the noised product's score ([RRR, p. 4](https://arxiv.org/pdf/2302.11552#page=4)). Part of any joint-minus-product gap at a noisy step would exist for an exact product too, so the "gap predicts the term" test is cleanest at low noise (inferred).

**Realism may be learned before content.** DIRE's ResNet-50 tells a model's own samples from other images at 99.9% accuracy from the residual after inversion and reconstruction ([Table 2, p. 6](https://arxiv.org/pdf/2303.09295#page=6)). Product and joint paths may differ in how on-manifold they are, so the tests need a control matched for that (inferred).

**ImageBind gives the training shape.** It trains each new encoder only against a frozen image encoder, and the new encoders reach text with no text pairs ([its note](imagebind.md)). Here that would mean freezing W4's text-and-canvas pair and training only a score-path encoder against the canvas of the same seed and step (inferred). Its claim that summed embeddings compose rests on example retrievals, so no published success rate exists for "sum lands on joint".

## What these papers could change in the two papers

Each row is a reader's inference from the paper's page, and none has been run. The owner chose on 2026-11-24 to carry these into step X3, where each is cut into its assumptions and marked, before anything is designed or launched.

| Suggestion | For | What it would cost | Source |
|---|---|---|---|
| Pick seeds by the summed squared score-path length over the early steps, on the product (from $\epsilon_{\text{PoE}} - \epsilon_\varnothing$) or on CO3 | both: a no-training baseline for the separate-experts paper, an add-on for the CO3 paper | no training; computed during sampling. Counting dropped on SDXL (0.409 to 0.325), so it can hurt one category while helping another | [Sampling Path Tells More, p. 8](https://arxiv.org/pdf/2505.23343#page=8) |
| Judge a composed render by scoring the right pair caption against its swapped caption with the diffusion model itself | both | tens to about 100 noise samples per caption pair; 18 seconds per image at 512 pixels for Li et al. | [Clark and Jaini, p. 9](https://arxiv.org/pdf/2303.15233#page=9); [DiffusionITM, p. 18](https://arxiv.org/pdf/2305.16397#page=18); [Li et al., p. 5](https://arxiv.org/pdf/2303.16203#page=5) |
| Compute that judge from unguided predictions, because guidance made Li et al.'s classifier worse at every guidance weight above 0 | the CO3 paper | none | [Li et al., Fig. 8, p. 18](https://arxiv.org/pdf/2303.16203#page=18) |
| Add a Langevin corrector to CO3: repeat the reverse step at the same noise level with noise scaled by $\sqrt 2$, and do not clip intermediate samples | the CO3 paper | extra score evaluations per step. On a score model, U-HMC lifts 5-cube CLEVR accuracy from 57.4% to 62.3% at equal evaluations; 72.7% needs an energy-parameterized model | [RRR, Table 2, p. 6; p. 14 to 15](https://arxiv.org/pdf/2302.11552#page=6) |
| Use SuperDiff's AND mode as a baseline that should lose on two separate creatures, because it aims at one object that looks like both | the CO3 paper | one more baseline run; an SDXL port exists, with no SDXL results reported | [SuperDiff, p. 40](https://arxiv.org/pdf/2412.17762#page=40) |
| Guard any loss that pushes a sample away from a hybrid with DiffusionITM's clip of the negative term against the positive; its unclipped run collapsed within one epoch | the separate-experts paper | none | [DiffusionITM, p. 18](https://arxiv.org/pdf/2305.16397#page=18) |
| Expect composition of models trained on different datasets to be the hard case: Composable Diffusion reports "limited success", and RRR's product assumes the experts' supports overlap | the separate-experts paper | none; a risk to state | [Composable Diffusion, p. 13](https://arxiv.org/pdf/2206.01714#page=13); [RRR, p. 18](https://arxiv.org/pdf/2302.11552#page=18) |
| Act only in a window of steps, as NoiseCLR's edits do from $0.5T$, or in $[0.9T, 0.8T]$ for coarse edits | the CO3 paper | none | [NoiseCLR note](noiseclr.md) |
| Flag which expert's manifold a composed sample left with a DIRE-style reconstruction error under each expert | the separate-experts paper | one inversion per expert per sample | [DIRE note](dire.md) |
