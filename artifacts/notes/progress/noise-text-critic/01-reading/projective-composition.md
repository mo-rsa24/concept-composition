# Mechanisms of Projective Composition of Diffusion Models

arXiv 2502.04549v3, read from the full PDF (40 pages, appendix included). The PDF prints no venue.

## What it does

The paper defines a correct composition as a distribution that matches expert $i$ when both are viewed through a feature extractor $\Pi_i$ chosen for that expert (Definition 4.1, p. 5). It proves when the usual sum of scores reaches that target and when reverse diffusion can sample it (Theorem 5.3, p. 6; Theorem 6.1, p. 8; Lemma 7.1, p. 9). The experiments use CLEVR object scenes, with two single-image SDXL examples (p. 1, p. 10).

## What the model or score reads

The theory reads exact scores of the noised distributions $p_i^t$ (Eq. 8, p. 6). The CLEVR models are a re-implementation of the EDM2 U-Net with 124M trainable weights. They read 128 by 128 images and are conditioned on one attribute, either 2D location or colour (p. 15). SDXL appears with text prompts, using the empty prompt as the background (p. 20). CLIP image and text embeddings appear only in a check on mean vectors, with no gradient and no noise (p. 10, pp. 20 to 21).

## What it trained on

Three CLEVR sets were generated: 50,000 empty scenes, 1,550,000 single-object scenes, and 2,500,000 scenes holding 1 to 5 objects, with 500,000 per object count (p. 15). Each model is conditioned on one attribute and the other attributes vary at random (p. 15). Batch size is 2048, training takes about 16 hours on 32 A100 GPUs, and inference uses a 32-step Heun sampler with 65 model calls (p. 15). One SDXL model is fine-tuned on 30 photos of the author's dog for Figure 1 (p. 20). No text and image critic is trained.

## How its score or gradient enters

$$
\nabla_x \log \mathcal{C}[\vec p](x) = \nabla_x \log p_b(x) + \sum_i \big(\nabla_x \log p_i(x) - \nabla_x \log p_b(x)\big)
$$

This is Eq. 6 (p. 6). $p_b$ is a reference background distribution, $p_i$ is expert $i$, and $\mathcal{C}$ is the composed distribution $\frac{1}{Z} p_b \prod_i p_i / p_b$ (Eq. 5, p. 6). Each expert has coefficient 1 and the background has net coefficient $1-k$ for $k$ experts. No normalisation or cap is applied.

Theorem 5.3 evaluates this sum on the noised distributions at every step (Eq. 8, p. 6). The CLEVR runs use no classifier-free guidance (p. 15). SDXL Figure 1 uses DDPM with 30 steps and guidance weight 2. SDXL Figure 7 uses 100 DDPM predictor steps, 3 Langevin iterations per step and no guidance (p. 20). No classifier or text and image gradient enters any sampler.

## When adding scores is valid

**The exact condition (Definition 5.2 and Theorem 5.3, p. 6)**

The coordinates must split into separate blocks $M_b, M_1, \dots, M_k$ such that:

1. Under expert $i$, block $M_i$ is independent of all other coordinates.
2. Under the background, all blocks are mutually independent.
3. Outside $M_i$, expert $i$ has the same distribution as the background.

In plain words, each expert is "draw a background, then overwrite my own block" (Eq. 7, p. 6). When the three conditions hold, reverse diffusion with the summed noised scores produces samples whose block $M_i$ is distributed as in expert $i$ and whose remaining block is distributed as in the background (Eq. 9, pp. 6 to 7).

**Feature spaces (Theorem 6.1, p. 8)**

It is enough that some invertible, differentiable map $A$ makes the three conditions hold in feature space. The composed distribution at $t=0$ is then correct, and nobody needs to know $A$. This theorem carries no sampling guarantee (p. 9). Reverse diffusion is guaranteed only when $A$ is orthogonal (Lemma 7.1, p. 9). For other maps it can fail even though the target is correct (Lemma 7.2, p. 9). In CLEVR, yellow composes with blue, cyan and magenta and fails with brown, gray, green and red (Figure 5, p. 5).

**A necessary check (Lemma 8.1, p. 10)**

If the conditions hold, the mean differences from the background are orthogonal: $(\mu_i - \mu_b)^T(\mu_j - \mu_b) = 0$. Score differences are also orthogonal (Remark 5.5, p. 7; pp. 21 to 22). Orthogonality alone does not imply the conditions (p. 22).

**Where it fails**

An unconditional background can break a composition that an empty background allows (Figure 3, p. 3; p. 7). Concepts that control the same thing are predicted to fail: "photo of a dog" plus "photo of a horse" fails on SDXL, and "photo of a dog" plus "photo, with red hat" works (Figure 7, p. 10). Figure 8 puts dog, horse and cat in one high-similarity group and says concepts from the same group may not compose (p. 11). Cat and dog, this project's base pair, sit inside that group (inferred).

**Lemma 5.7 is not used here**

The paper offers an approximate version with a KL bound (Lemma 5.7, p. 7). This project's memory records a counterexample with KL 2.88 against a bound of 0.89 (`bradley-lemma-5-7-counterexample`). This read did not recheck that counterexample, so nothing below relies on the lemma.

**Could the loop's right-minus-wrong push satisfy the conditions? (this whole subsection is inferred)**

For the swap pairs in this loop, the conditions fail whenever the base draws a mix of the right and wrong pictures, and no invertible feature space repairs that. The push may still work in practice, but Theorems 5.3 and 6.1 give it no guarantee.

Write $s = -\epsilon/\sigma_t$ for a score. The loop's update becomes:

$$
\hat s_t = s_{\text{base}} + \lambda\,\nabla_{x_t}\big[S(x_t, R) - S(x_t, W)\big]
$$

Take the best case for $S$: a noise-aware critic at the optimum of a contrastive loss with text negatives, $S(x_t, y) = \log p_t(x_t \mid y) - \log p_t(x_t) + c(x_t)$. That optimum is a standard contrastive-learning result and does not come from this paper. The $c$ and $p_t(x_t)$ terms cancel in the difference, so the bracket's gradient is $s_R - s_W$. With $\lambda = 1$:

$$
\hat s_t = s_W + (s_{\text{base}} - s_W) + (s_R - s_W)
$$

This is Eq. 6 with the wrong sentence as background and two experts, the base and the right sentence. The wrong sentence must be the background because it is the only term with coefficient $-1$. The theorem then demands four things:

1. $\lambda = 1$, because Eq. 6 has unit coefficients.
2. $S$ equals the log ratio of the noised densities at every step.
3. $p_R$ differs from $p_W$ only on a block $M_R$, and $p_{\text{base}}$ differs from $p_W$ only on a separate block $M_{\text{base}}$.
4. Since $M_R$ lies outside $M_{\text{base}}$, the base must already match the wrong sentence on $M_R$: $p_{\text{base}}(x|_{M_R}) = p_W(x|_{M_R})$.

Requirement 4 is the one that breaks. Suppose the base draws both arrangements, $p_{\text{base}} = w\,p_R + (1-w)\,p_W$ with $0 < w < 1$. Taking the marginal on $M_R$ gives $w\,p_R(x|_{M_R}) + (1-w)\,p_W(x|_{M_R}) = p_W(x|_{M_R})$, so $p_R(x|_{M_R}) = p_W(x|_{M_R})$. Conditions 1 to 3 then give $p_R = p_R(x|_{M_R})\,p_W(x|_{M_R^c}) = p_W(x|_{M_R})\,p_W(x|_{M_R^c}) = p_W$. The conditions can hold only if the two sentences describe the same distribution. An invertible map keeps a mixture a mixture, so the same argument applies in every feature space of Theorem 6.1.

Lemma 8.1 shows the same failure. For an equal mixture, $\mu_{\text{base}} - \mu_W = \tfrac{1}{2}(\mu_R - \mu_W)$, so the two mean differences have cosine $+1$ where the lemma requires 0.

One narrow case survives: the base already draws the wrong arrangement on every coordinate the right sentence controls, and everything else factorises. The push is then a local edit from wrong to right. A product base gives no reason to expect this.

The and-score has a clean reading under the same best-case critic:

$$
S(cd) - S(c) - S(d) + S(\varnothing) = \log \frac{p_t(x_t \mid cd)\,p_t(x_t \mid \varnothing)}{p_t(x_t \mid c)\,p_t(x_t \mid d)}
$$

Its gradient in $x_t$ is zero exactly when the joint prompt's noised distribution equals the composition of the two single prompts with the empty prompt as background. The and-score therefore estimates what the sum of scores leaves out.

## Which of the five properties it bears on

- **Counting:** tested on CLEVR. Table 1 hand-counts objects at requested locations for 1 to 6 requested objects, 100 images per method (p. 17). Table 2 reports the largest count that succeeds (p. 39).
- **Attribute binding:** not tested. Colour composition checks whether two colours appear, with each model conditioned on one colour and no colour tied to a named subject (Figure 5, p. 5; p. 15).
- **Spatial relation:** not tested. Conditions are absolute 2D locations (p. 15).
- **Position and size:** position is tested through location conditioning (Tables 1 and 2, p. 17, p. 39). Size is not tested, since unconditioned attributes vary at random (p. 15).
- **Prompt match:** not tested. SDXL appears as single example images with no prompt-match metric (Figure 1, p. 1; Figure 7, p. 10).

## Unseen concepts

This paper does not test held-out concepts. It tests held-out object counts: compositions holding more objects than any single model saw in training (Figure 2, p. 1; Table 2, p. 39).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| Largest object count $K_{\max}$ that succeeds for every smaller count, models trained on 1 object (higher is better) | 4, summed scores with empty background | 3, summed scores with the background switched partway through sampling; 1 with unconditional background | CLEVR | EDM2 U-Net | p. 39, Table 2 |
| $K_{\max}$, models trained on 1 to 3 objects (higher is better) | 8, summed scores with learned background | 9, one model conditioned on all locations | CLEVR | EDM2 U-Net | p. 39, Table 2 |
| Images out of 100 with all 6 requested objects at their locations (higher is better) | 97, empty background | 98, unconditional background on cluttered scenes | CLEVR | EDM2 U-Net | p. 17, Table 1 |

The Single-object Bayes rows of Table 1 are unusable as printed. The row for 2 requested objects sums to 109 images and lists 32 images with 3 correct objects, and the row for 6 sums to 43 images (p. 17). Tables 1 and 2 also use different success criteria: Table 2 takes its definition from Bradley (2025), which this PDF does not restate (p. 39).

## What to copy, what to avoid

- Copy Lemma 8.1 as a cheap check before any render: compute the cosine between $\mu_R - \mu_W$ and $\mu_{\text{base}} - \mu_W$ in the critic's image embedding. A value near $+1$ says the push is choosing between arrangements the base already mixes (inferred).
- Report the and-score as an estimate of what the sum of scores omits, and check whether it falls toward zero on pairs that already compose (inferred).
- Include $\lambda = 1$ among the tested strengths, since it is the only strength Eq. 6 covers (inferred).
- Expect counting and position to be easier than binding. The paper's working cases separate objects in space, and its colour failures come from shared channels (p. 5, p. 9). This matches the concern that CLIP treats text much like a bag of words (inferred).
- Avoid citing Theorem 5.3 or 6.1 as support for the swap-pair push, and avoid citing Lemma 5.7 as a robustness bound (inferred).
- A high critic score alone proves nothing about composition. The earlier CLIP checker raised its own score by 1 to 13.5 logits over a random push without moving the animals, and this theory gives no reason for a sentence score to escape that (inferred).

## Closer work it cites

- GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models, arXiv 2112.10741 (reference on p. 12, cited on p. 3). This paper cites it only for out-of-distribution text prompts. GLIDE trains a CLIP model on noised images and steers sampling with its gradient, a description that comes from outside this PDF.
