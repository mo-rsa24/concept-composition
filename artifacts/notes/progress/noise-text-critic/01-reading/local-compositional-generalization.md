# Local Mechanisms of Compositional Generalization in Conditional Diffusion

arXiv 2509.16447 (v3), read from the full PDF (36 pages, appendix included).

## What it does

The paper proves that a conditional distribution factorizes over disjoint sets of pixels exactly when its score at each pixel depends only on that pixel's set and that set's condition (p. 6). It tests this on CLEVR models asked to draw more objects than they saw in training, and a model forced to use local scores gains that ability (p. 3, p. 8). It then inspects a pretrained SDXL and finds word influence spread evenly over pixels, with some category structure in the U-Net's hidden features (p. 9).

## What the model or score reads

**CLEVR models.** An EDM2 U-Net (`edm2-img64-xs` size, base width 128, 124M weights) reads noisy 128 by 128 images (p. 22, p. 23). The condition is a 16 by 16 grid counting object centres per cell, or a list of up to 10 embedded xy locations, or an 8-entry count of objects per colour (p. 22).

**Local model (Experiments 2L and 3L).** Each 16 by 16 grid cell is denoised from a patch of 5 by 5 cells (neighbourhood radius k = 2), conditioned only on the locations inside the patch and on the patch's absolute position (p. 29).

**SDXL analysis.** `stabilityai/stable-diffusion-xl-base-1.0` is read in VAE latent space for pixel gradients and word ablation, with each word's token embedding replaced by the padding embedding (p. 31). The feature analysis reads post-FFN outputs of each `BasicTransformerBlock`, for 24 prompts in 4 categories with 10 seeds each (p. 9, p. 32).

## What it trained on

CLEVR datasets were rendered with 1 to K objects, K from 1 to 6, at 500,000 images per object count, so a 1 to 3 object model saw 1,500,000 images (p. 22). Training uses the EDM2 loss with batch size 2048 over 128 × 2^20 samples, repeating samples as needed (p. 23). The local model samples one patch with an active condition and one without per image (p. 29). SDXL and the Matryoshka Diffusion Model (trained on a 50M Flickr subset) are used pretrained (p. 31, p. 32).

## How its score or gradient enters

No external gradient enters a sampler. CLEVR inference uses raw conditional scores with no classifier-free guidance (p. 23). The local conditional score is defined as

$$
s_t[x \mid c_{\mathcal J}](i) := \nabla \log p_t\big[x_{N_i^t} \mid c_{L_i^t(\mathcal J)}\big](i) \quad \forall i
$$

where $N_i^t$ is the set of pixels pixel $i$ depends on and $L_i^t(\mathcal J)$ is the set of active conditions it depends on (p. 5). The local model applies this by denoising each patch and copying its centre cell back into the image (p. 29).

## When adding scores is valid

**The exact condition (Definition 2, p. 5; Lemma 1, p. 6; proof p. 15 to 16).** Concepts compose exactly when the conditional distribution factorizes over disjoint sets $M_j$:

$$
p(x \mid c_{\mathcal J}) := p(x_{M_{\mathcal J}^\complement} \mid \emptyset) \prod_{j \in \mathcal J} p(x_{M_j} \mid c_j)
$$

Lemma 1 says this holds if and only if the score at a pixel in $M_j$ reads only $x_{M_j}$ and condition $j$. The factorization survives Gaussian noise at every time $t$ (p. 15). From the per-pixel cases in that proof, the joint score equals $s(x\mid\emptyset) + \sum_j [s(x\mid c_j) - s(x\mid\emptyset)]$, so the unconditional score is counted once (inferred).

**The approximate condition (Definition 3 and Lemma 3, p. 16).** If each set is nearly independent of the rest, with KL errors $\varepsilon_j, \tilde\varepsilon_j, \varepsilon_b$, the local score's distribution is within KL $\sum_{j\in\mathcal J}\varepsilon_j + \sum_{j\notin\mathcal J}\tilde\varepsilon_j + \varepsilon_b$ of the true one.

**Noise (Lemmas 4 and 5, p. 16 to 17).** The paper concludes that distributions become more compositional at higher noise. Lemma 4 bounds $\mathrm{KL}(N_t[p](x)\,\|\,N_t[p](x\mid y))$ for a fixed conditioning value $y$. Definition 3 conditions on the other pixels with the KL arguments in the opposite order, and at time $t$ those pixels are noised too, so Lemma 4 does not directly bound the errors in Definition 3 (inferred).

**Feature space (Definition 4 and Corollary 1, p. 6, p. 20).** The same equivalence holds after an invertible transform $A$. Noising does not commute with $A$, so sampling needs a time-dependent transform $A_t$ that depends on the unknown distribution, and the paper says the feasibility of learning it is unclear (p. 21).

**A necessary check (Lemma 2, p. 7; proof p. 21).** If the feature-space score is local, then at the highest noise level the condition-minus-unconditional differences are exactly orthogonal:

$$
d_i^\top d_j = 0 \quad \forall i \neq j, \qquad d_i := \mathbb E[s_A^{t_{\max}}(\cdot \mid c_i)] - \mathbb E[s_A^{t_{\max}}(\cdot \mid \emptyset)]
$$

Passing it does not prove locality.

**Where it fails.** The paper calls pixel-space factorization unlikely for text prompts, because a word such as "watercolor" affects all pixels (p. 6). In SDXL the per-word influence curves are nearly identical at different pixel locations (p. 9, p. 32).

## Which of the five properties it bears on

**Counting.** Measured directly. $K_{\max}$ is the largest object count up to which a model "sometimes succeeds", meaning exactly $K$ objects in at least 50% of samples and at least $K-2$ in at least 90%, scored by a ResNet-18 counter on 1024 samples (p. 23). The counter counts all objects, so $K_{\max}$ can overstate correct placement for models that draw extra objects (p. 26).

**Attribute binding.** Not tested. The colour condition is a count per colour, and the location and colour attributes are never conditioned together (p. 22, p. 23).

**Spatial relation.** Not measured as a relation. One SDXL prompt places a horse in the middle, a dog on the left and a cat on the right, and the paper reports that SDXL rarely gets all three right, with no rate given (p. 8, p. 9).

**Position and size.** Position is the CLEVR condition and is measured through $K_{\max}$ (p. 3, p. 23). Size is not tested.

**Prompt match.** Not tested as a score. The out-of-distribution Matryoshka prompts are judged by eye (p. 35).

## Unseen concepts

The held-out quantity is object count: models trained on 1 to 3 objects are tested on up to 12 conditioned locations (p. 2, p. 26). With the grid condition on all objects, this reaches 7 objects, and the same grid labelling one object reaches 2 (p. 3). No new object types or colours are held out. For SDXL the training set is unknown, so the paper cannot say which compositions are unseen (p. 9). For the Matryoshka model, prompts such as "a cow jumping over a candlestick" had no conceptual match in a keyword search of the captions, and results are qualitative (p. 32, p. 35).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| $K_{\max}$, objects drawn reliably (higher is better), trained on 1 to 3 objects | 7 (all objects labelled, grid) | 3 (list condition); 2 (grid, one label) | CLEVR 128 px | EDM2 U-Net | p. 3, Table 1 |
| $K_{\max}$ after enforcing local scores, trained on 1 to 3 objects | 9 (Exp. 2L and 3L) | 2 (Exp. 2), 3 (Exp. 3) | CLEVR | Local patch EDM2 | p. 3, Table 1 |
| $K_{\max}$, trained on one object only | 5 (Exp. 2L) | 1 (Exp. 1, 2) | CLEVR | Local patch EDM2 | p. 3, Table 1 |
| $K_{\max}$, colour condition, trained on 1 to 5 objects, 25% success threshold, 64 samples counted by hand | 7 | none | CLEVR | EDM2 U-Net | p. 29, p. 30, Table 4 |
| Cosine of concept differences in hidden features (0 is orthogonal, the value Lemma 2 requires) | 0.41 to 0.47 within category, 0.27 to 0.33 between | none | 24 prompts | SDXL | p. 9 |
| Within over between cosine ratio (higher is more separated) | 1.4 to 1.6 in features | 1.42 pixel, 1.31 VAE latent | 24 prompts | SDXL | p. 9 |
| Same ratio | 1.49 to 1.68 | none | 12 prompts | Matryoshka (Flickr) | p. 9, p. 35 |

## What to copy, what to avoid

- Copy Lemma 2 as a cheap check before training: measure the cosine between $\epsilon(\text{cat}) - \epsilon(\emptyset)$ and $\epsilon(\text{dog}) - \epsilon(\emptyset)$ in U-Net features at early steps. Our U-Net checker reads the pair at balanced accuracy 0.63 to 0.79 at steps 0 to 15, so those features hold the pair at the time this check applies (inferred).
- Read the between-category cosines of 0.27 to 0.33 as a failed strict test. Lemma 2 requires 0, and the paper's evidence is a ratio (p. 7, p. 9) (inferred).
- Expect cat and dog to be the hard case. Animals form one high-similarity block in the SDXL heatmap, and "A horse, dog, and cat." is printed among the less successful compositions (p. 10, read from the figure). If concepts in one block share feature sets, the disjoint-set condition fails for two animals (inferred).
- Avoid a single whole-image sentence score for position. SDXL's word influence is spatially uniform (p. 9), which fits our CLIP checker raising its score by 1 to 13.5 logits over a random push without moving the animals (inferred).
- Try scoring a sentence per region if position must move. Restricting each patch to its local conditions took $K_{\max}$ from 2 to 9 objects for models trained on 1 to 3 objects (p. 3) (inferred).

## Closer work it cites

None.
