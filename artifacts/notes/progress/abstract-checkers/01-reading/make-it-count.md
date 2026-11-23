# Make It Count: Text-to-Image Generation with an Accurate Number of Objects

arXiv 2406.10210 (v1), read from the full PDF (17 pages, appendix included). The PDF prints "Preprint. Under review." (p. 1).

## What it does
CountGen runs SDXL to timestep 500, finds each object instance by clustering SDXL's own attention features, and counts the clusters (p. 3 to 4). If the count is wrong, it removes the smallest instances or calls a small trained U-Net (ReLayout) that adds one instance mask at a time (p. 5). It then generates the image with a cross-attention loss and self-attention masking that hold objects to the corrected layout (p. 6 to 7).

## What the classifier or loss reads
**Counter.** The counter is a clustering procedure with no trained classifier. It reads noisy SDXL UNet features during generation and never decodes an image (p. 2 to 4).

**Exact input.** Cross-attention maps for the object's tokens are thresholded with Otsu to give a foreground mask (p. 4). Self-attention features from up-block layer $l^{up}_{52}$ at timestep $t = 500$ inside that mask are clustered with DBSCAN using cosine distance and $\epsilon$ in $[0.1, 0.2]$ (p. 3, p. 4, p. 13). The number of clusters is the count. The feature resolution is not printed.

**Early noisy steps.** The counter reads at the middle of sampling. The authors chose $t = 500$ as the most robust timestep, and before it "clear separation is not observed in some objects" (p. 3, p. 15, Fig. 12). This evidence is a PCA visualisation, with no measured accuracy per timestep (p. 3, p. 15).

**ReLayout.** ReLayout reads only a multi-channel binary mask with one channel per instance (p. 5, p. 6). It receives no image, no text and no attention values.

**Layout loss.** The generation loss reads aggregated cross-attention $c$ against the target binary mask $m$ (p. 6).

## What it trained on
**The counter is not trained.** The one trained network is ReLayout, which edits layouts.

**Data.** About 10,000 SDXL image pairs come from the same seed with prompts differing by one in count, such as "a photo of two cats" and "a photo of three cats" (p. 5). Pairs are kept only if the extracted masks differ by exactly one object (p. 5). Hungarian matching pairs up the instances across each pair (p. 6, p. 14 to 15).

**Network and loss.** The U-Net takes 9 input channels and outputs 10, supporting counts up to 10 (p. 6). The loss is a Dice loss plus a mask overlap penalty, with weights 1 and 0.25 (p. 6, p. 13). Training uses Adam, learning rate $8 \times 10^{-6}$ and batch size 32, with horizontal flip and channel shuffle augmentation (p. 13). The number of epochs is not printed.

## How its gradient enters
The paper prints the loss and gives no latent update equation, step size or gradient normalisation (p. 6). The code base starts from Attend-and-Excite (p. 13).

$$L(c, m) = -\sum_i w_i \left( m_i \log c_i + (1 - m_i) \log(1 - c_i) \right), \qquad w_i = \begin{cases} 10 & m_i = 1 \\ 1 & \text{otherwise} \end{cases}$$

$c_i$ is the aggregated cross-attention score at pixel $i$ and $m_i$ is the target mask (p. 6). Self-attention links from background pixels to foreground pixels are set to zero (p. 6 to 7):

$$S_t^{*(l)}[i, j] = \begin{cases} 0 & i \in B^{(l)},\ j \in F^{(l)} \\ S_t^{(l)}[i, j] & \text{otherwise} \end{cases}$$

The loss runs at $t \in [1000, 500]$ in all UNet layers. The masking runs at $t \in [1000, 900]$ in the decoder layers (p. 13). The model is `stable-diffusion-xl-base-1.0` (p. 13). The layout is read at $t = 500$ and the loss starts at $t = 1000$, so generation restarts for the second pass (inferred).

## Unseen concepts
This paper does not test held-out concepts. The classes used to train ReLayout are not listed. Fig. 5 shows ties and horses among its training examples (p. 6), and both are also in the 20-class CoCoCount test set (p. 15), so the training and test classes overlap at least there (inferred).

## Word leakage and confounds
**Word access.** The instance counter uses the object word's cross-attention to find the foreground (p. 4). ReLayout sees binary masks only, so it has no direct access to the word or appearance (p. 5, p. 6). The paper does not discuss word leakage. Mask shape and arrangement could still carry the category, since the paper itself contrasts rows of bottles with clusters of elephants (p. 5) (inferred).

**Scoring.** Accuracy comes from YOLOv9 and from three human raters per image, and neither is part of the method (p. 7, p. 16).

**Scope.** Experiments cover "a single object per prompt" and up to 10 instances (p. 9). Two different subjects in one image are outside what the paper tests.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Count accuracy by human raters, percent of images, higher is better | 54 | 38 DALL-E 3, 26 SDXL | CoCoCount, 200 prompts | SDXL | p. 8, Table 1 |
| Count accuracy by YOLOv9, higher is better | 50 | 29 Random masks + Bounded Attention, 28 SDXL | CoCoCount | SDXL | p. 8, Table 1 |
| Count accuracy by human raters, higher is better | 48 | 36 DALL-E 3, 29 SDXL | T2I-Compbench count subset, 218 prompts | SDXL | p. 7, p. 8, Table 1 |
| **Count 2**, images with exactly 2 by human raters | 30 of 34 (88%) | SDXL about 62%, Counting Guidance about 53%, DALL-E 3 about 94% (plot) | CoCoCount target 2 | SDXL | p. 15, p. 17, Fig. 17; p. 9, Fig. 8 |
| **Count 2**, accuracy by YOLOv9, read from plot, higher is better | about 86% | about 61% DALL-E 3, 50% SDXL, 44% Counting Guidance | CoCoCount target 2 | SDXL | p. 14, Fig. 10 |
| **Count 2**, repeating the object in the prompt ("a cat and a cat"), read from plot | Not this method | about 53% by humans, 41% by YOLOv9, below SDXL at about 62% and 50% | CoCoCount target 2 | SDXL | p. 7, p. 14, Figs. 9 and 10 |

At target 2, Fig. 17 shows 30 images with two objects, 2 with one and 2 with three (p. 17). Its caption does not name the dataset. The diagonal sums to 108 images, and 108 of CoCoCount's 200 prompts is the 54% in Table 1, so the matrix is CoCoCount (inferred). All count 2 prompts name one class, so these figures measure two instances of the same thing.

## What to copy, what to avoid
- Copy the separation between reading and steering. It reads the instance layout at $t = 500$, then steers from $t = 1000$ on a second pass. Our push is limited to steps 0 to 15 of 50, which is about $t = 1000$ to $700$ on a uniform schedule, so it ends before the point where this paper finds instances separable (inferred).
- Test whether word cross-attention alone can count. This paper needs self-attention features to separate instances and uses cross-attention only for the foreground. A checker reading only word maps may tell where each word landed and still miss how many separate things are there (inferred).
- Copy ReLayout's input design for an abstract checker. Binary instance masks contain no word and no appearance, which is the kind of input our checkers aim for. Keep held-out classes in its training split, since this paper does not (inferred).
- Copy a spatial target for the attention loss. A target mask gave working layouts here, while our target-free repulsion of the cat and dog maps drew two animals on 0 of 8 seeds (inferred).
- Avoid relying on its count 2 numbers for our pair task. They come from one repeated class, and two different subjects remain untested here (inferred).

## Closer work it cites
- **Teaching CLIP to Count to Ten** (Paiss et al.), no arXiv id printed, p. 11. It trains a CLIP model for counting, an abstract property across categories.
