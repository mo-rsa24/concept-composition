# Harnessing the Spatial-Temporal Attention of Diffusion Models for High-Fidelity Text-to-Image Synthesis

arXiv 2304.03869, read from the full PDF (20 pages, appendix included). No venue is printed on the PDF.

## What it does

A text-only transformer predicts a centre point for each noun phrase, and a circle around that point becomes the object's region (p. 4). Inside each region, Stable Diffusion's cross-attention output is blended between the full caption and a short local prompt such as "A photo of a red car" (pp. 3, 5). The blend weights, one per object per denoising step, are optimised per prompt against a CLIP similarity on the finished image (pp. 4, 5).

## What the classifier or loss reads

**The layout predictor reads only the caption.** It is a RoBERTa-base transformer that takes the text description $D$ as input (pp. 4, 5). At the position where the caption mentions object $O_i$, it outputs a 5-component Gaussian mixture over that object's centre coordinate $C_i = [X_i, Y_i]$ (pp. 4, 5). It sees no image, latent, attention map or diffusion timestep (p. 4, inferred from its stated input).

**The region is a fixed circle.** At inference a centre is drawn from the mixture, and the region is a circle of fixed radius $r$ around it (p. 4). The experiments use $r = 0.2$ (p. 12). The binary mask $M_i$ is 1 inside the region (p. 5).

**The optimisation loss reads the decoded final image.** A global term compares the whole image $I(\lambda)$ with the caption. A local term compares each object's crop $I_{O_i}(\lambda)$, resized to 224 by 224 pixels, with its local prompt (p. 5). Both use pre-trained CLIP ViT-B/32 (pp. 11, 12). No trained checker reads cross-attention maps, and the PDF does not say which U-Net attention layers receive the blend.

## What it trained on

**Layout predictor.** RoBERTa-base is fine-tuned for 100 epochs at batch size 64. The learning rate falls from $10^{-6}$ to $10^{-8}$ for the encoder and from $4\times10^{-5}$ to $10^{-8}$ for the mixture output layer (pp. 11, 12).

**Absolute position loss.** On MS-COCO captions with extracted object labels and centre coordinates, the predictor minimises the negative log-likelihood of each true centre under its mixture (p. 4, Eq. 2).

**Relative position loss.** On GPT-synthetic, GPT-3 writes a sentence from sampled objects, colours and relations (left of, right of, above, below). A margin loss requires the rightmost mixture mean of the left object to sit left of the leftmost mixture mean of the right object (p. 4, Eq. 3; p. 11).

$$\ell_{rel}(R_{ij} = \text{left}; f_i, f_j) = \max\{\max_k \mu_{ik}(0) - \min_k \mu_{jk}(0),\ -\delta\}$$

The total is $\mathcal{L}_{layout}(\theta) = \mathcal{L}_{abs}(\theta) + \xi \mathcal{L}_{rel}(\theta)$ (p. 4, Eq. 5). GPT-synthetic covers the 80 MS-COCO object categories (p. 11). The PDF does not print the training-set sizes, the value of $\xi$ or the value of $\delta$.

## How its gradient enters

The paper optimises attention combination weights $\lambda_{it}$. It prints no update to the latent. The layout predictor supplies the masks and takes no part in the sampling-time loss (pp. 4, 5).

The attention output at step $t$ (p. 5, Eq. 8):

$$O(t) = \sum_{i=1}^{N} \lambda_{it} M_i \odot \text{Attention}(Q, K_{L_i}, V_{L_i}) + \Big(1 - \sum_{i=1}^{N} \lambda_{it} M_i\Big) \odot \text{Attention}(Q, K_D, V_D)$$

The loss on those weights (p. 5, Eq. 9):

$$\mathcal{L}_{attend}(\lambda) = -\text{CLIP}(I(\lambda), D) - \gamma \sum_{i=1}^{N} \text{CLIP}(I_{O_i}(\lambda), L_i)$$

$M_i$ is object $i$'s binary mask, $L_i$ its local prompt, $D$ the full caption, $\odot$ element-wise multiplication and $\gamma = 5$ (pp. 5, 12).

**Which diffusion steps.** The layout predictor's own loss is applied at no diffusion step, because it is trained offline on text. Its mask is applied at each of the 50 PLMS steps (pp. 4, 5, inferred from Eq. 8). The CLIP loss is computed once per optimisation pass on the finished image, and it updates a separate weight for every step (p. 5).

**Scale and limits.** The optimiser is Adam with learning rate 0.05, starting from $\lambda_{it} = 1/N$ (p. 12). Each weight is projected to the range $[-1, 2]$ after each step (p. 11). The number of optimisation passes is not printed. One generation takes about 10 minutes (p. 8). The model is Stable Diffusion v1.4 at 512 by 512 pixels with classifier-free guidance 7.5 (p. 12).

## Unseen concepts

This paper does not test held-out concepts.

No noun or category is stated as held out from the layout predictor. Its synthetic data covers all 80 MS-COCO categories (p. 11). The "uncommon" test set contains 100 descriptions and changes only two things: objects may come from different super-categories, and samples are kept only if unlikely in real life (p. 14). It therefore tests new combinations of the same categories (inferred). Object recall also counts only objects belonging to MS-COCO categories (p. 6).

The PDF refers to a GPT-synthetic "test set" but does not describe the split (pp. 5, 8). MS-COCO trains the absolute objective and is also an evaluation set, and VSR is built from a subset of MS-COCO (pp. 5, 11). The PDF does not say whether evaluated captions were excluded from layout training.

## Word leakage and confounds

The layout predictor reads the words directly, so it can learn a position habit for each noun from COCO statistics (inferred). The paper does not address this. The CLIP loss also compares the image with the object's words by design (p. 5).

Paint-with-Words receives the same predicted regions, so that comparison holds the layout fixed (p. 5). The other three baselines lack this region input, so their gaps mix the effect of the layout with the effect of the attention method (p. 6, inferred).

The paper reports that CLIP similarity is too close across methods to show which is better on multi-object text (p. 11). Its own method optimises CLIP similarity and still scores below plain Stable Diffusion on global CLIP for GPT-synthetic: 0.2944 against 0.3006, on CLIP's cosine similarity scale, higher better (p. 11, Table 5).

## Numbers

All rows use Stable Diffusion v1.4. Percentages and scores are higher-is-better.

| Metric | This method | Best baseline | Dataset | Page and table |
|---|---:|---|---|---|
| Object recall (% of mentioned COCO-category objects DETR finds) | 47.2 | 45.9, Paint-with-Words with the same regions | GPT-synthetic, 500 descriptions | p. 6, Table 2 |
| Spatial relation precision (% of relations correct among relations whose objects were found) | 66.7 | 59.3, Structure Diffusion | GPT-synthetic | p. 6, Table 2 |
| Object recall | 65.1 | 63.8, Paint-with-Words | VSR | p. 6, Table 2 |
| Spatial relation precision | 75.0 | 72.2, Composable Diffusion | VSR | p. 6, Table 2 |
| Object recall | 69.6 | 61.7, Structure Diffusion | MS-COCO | p. 6, Table 2 |
| Human object score (0 to 100) | 84 | 65, Paint-with-Words | GPT-synthetic, 25 descriptions | p. 6, Table 1 |
| Human spatial score (0 to 100) | 86 | 60, plain Stable Diffusion | GPT-synthetic, 25 descriptions | p. 6, Table 1 |
| Human object score (0 to 100) | 75 | 75, Paint-with-Words (tie) | MS-COCO, 25 descriptions | p. 6, Table 1 |
| Pairs where the baseline was rated better than this method (%) | n/a | 52, Paint-with-Words | MS-COCO, 25 descriptions | p. 6, Table 1 |
| Object recall on uncommon pairs | 42.4 | 41.2, Paint-with-Words | 100 descriptions | p. 14, Table 9 |
| Spatial relation precision on uncommon pairs | 59.6 | 54.7, Paint-with-Words | 100 descriptions | p. 14, Table 9 |

Ablations on GPT-synthetic, against 47.2 object recall and 66.7 spatial precision for the full method (p. 8, Table 3): removing spatial control gives 39.6 and 51.7, removing temporal control gives 43.8 and 61.4, and removing weight optimisation gives 41.0 and 55.6. Replacing predicted centres with true box centres on VSR gives 66.3 and 78.1, against 65.1 and 75.0 for predicted centres (p. 13, Table 8).

## What to copy, what to avoid

- Copy the order of operations for a position checker: assign each subject a region first, then steer attention into it. Our symmetric attention-map separation drew two animals on 0 of 8 seeds, and this paper's largest ablation loss comes from removing region control (inferred).
- Test a region-based attention edit as a separate baseline from a gradient checker, because this paper's lever edits the attention output directly (inferred).
- Keep regions away from the image edge. The paper's failure cases have predicted centres near the edge, and moving the region inward recovers the missing object (p. 14) (inferred for our design).
- Avoid a decoded-image CLIP loss as the layout signal. Our checker through the decoded estimate changed texture and left layout alone, and this paper finds CLIP similarity unable to rank multi-object results (p. 11) (inferred).
- Avoid citing this paper for transfer across concepts. Its predictor is untested on held-out nouns, and a text-only predictor could learn word-specific positions (inferred).
- Avoid its per-prompt optimisation loop as the sampling method. At about 10 minutes per image it does not fit a step-by-step checker capped at 0.4 of PoE's prediction norm (p. 8) (inferred).

## Closer work it cites

None. Its layout predictor is adapted from LayoutTransformer (pp. 4, 10), which the citation describes as scene layout generation. Nothing in this PDF presents it as a classifier for an abstract property or as a test on unseen concepts.
