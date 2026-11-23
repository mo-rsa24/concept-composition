# Detector Guidance for Multi-Object Text-to-Image Generation

arXiv 2306.02236, read from the full PDF (17 pages, appendix included).

## What it does

Detector Guidance (DG) trains a small object detector on Stable Diffusion's word cross-attention maps (CAMs) and runs it during sampling to give each noun its own region (p. 2, p. 5). It then rewrites later cross-attention maps so a rival noun has no influence inside another noun's region, and rescales what is left (p. 6). The method takes no gradient, so the latent is only changed through the edited attention (p. 6, p. 15).

## What the classifier or loss reads

The detector reads a stack of five cross-attention maps for one core noun, with total size 20 × 16 × 16, averaged over the first head dimension and across tokens when the noun spans several tokens (p. 14). The maps come from the middle of the U-Net at 16 × 16 resolution (p. 5). The paper does not name the five blocks. A map is defined as $M = KQ^T$, the value before the softmax (p. 4).

The listed input holds no image, no latent and no separate word embedding (p. 14, inferred from the input list). The output is a bounding box and a confidence for that noun (p. 14).

## What it trained on

The detector is a YOLOv1 with a ResNet backbone and convolution stride 1, trained on the official COCO train split (p. 7, p. 14). To make an example, the authors add Gaussian noise to a COCO image, use the image's object labels as the prompt, run the frozen diffusion model, and capture the middle CAMs (p. 5). Algorithm 3 draws the timestep from 0 to $T$ (p. 15).

The loss keeps YOLOv1's confidence and box terms and drops the classification term (p. 14). Training uses AdamW with learning rate 0.001, batch size 32, 100k steps, and random flips, brightness changes and crops (p. 14). The cost is 2 RTX3090-days, with nearly 70% of that spent computing CAMs (p. 7).

## How its gradient enters

No gradient enters. DG edits the attention maps inside each cross-attention block (p. 6, Algorithm 5 on p. 15):

$$
\text{CAM}_1 = \text{CAM}_0 \cdot (1 - \text{mask}) + KQ_{uc}^T \cdot \text{mask}
$$

$$
\text{CAM}_2 = \text{CAM}_1 \cdot \frac{\text{CAM}_0.\max(\dim=-1)}{\text{CAM}_1.\max(\dim=-1)}
$$

$$
\text{CAM}_3 = \text{CAM}_2 \cdot s + \text{CAM}_0 \cdot (1 - s), \qquad \text{output} = \text{Softmax}\!\left(\text{CAM}_3 / \sqrt{d}\right) V
$$

$\text{CAM}_0 = KQ_c^T$ is the conditional map and $KQ_{uc}^T$ is the unconditional map. The mask marks a pixel and token when that pixel belongs to an object the token conflicts with (p. 6). $s$ is a smoothing factor that ramps through 25%, 50%, 75% and 100% (p. 6). $d$ is the projection dimension (p. 4). There is no guidance scale, and the authors state the correction has no hyperparameters for different situations (p. 6).

The correction starts when $t \le 800$ on the 1000-step scale (p. 6). A footnote prints the transition as $I = 800$ while Algorithm 1 skips correction for $t = T$ down to $T - I + 1$ (p. 6). Read literally with $T = 1000$, correction would start at $t = 200$. The surrounding text and Figure 7 fit a start at $t = 800$ (p. 6, p. 9, inferred). The models are Stable Diffusion v1.4 and v2.1 (p. 7). The detector is described as built on v2.1, and the paper does not say whether v1.4 got its own detector (p. 14).

## Unseen concepts

This paper does not test held-out concepts.

The authors claim "good generalization to unseen categories" and say most objects in their demos fall outside COCO (p. 3, p. 5, p. 8). No category is held out by design and no number separates seen from unseen categories. The only detector accuracy is on the COCO validation set, which has the same categories as training (p. 14).

The MRO benchmark does run the detector on many nouns outside COCO (p. 16). By exact COCO category name, 2 of its 30 prompts have both nouns in COCO, 8 have one, and 20 have neither (inferred, using the standard 80-name COCO list, which the paper does not print). Several of the 20 are sub-types of a COCO class, such as owl under bird and golden retriever under dog, so this is not a clean split. MRO results are reported only as one pooled number (p. 7).

## Word leakage and confounds

The detector receives no word as an input and has no classification head, so it cannot be scored for naming the class (p. 14). Each map is still computed from that word's embedding, so word identity can reach the detector through the map's values (inferred). The paper does not test or discuss this.

Training prompts are the labels of objects present in the image (p. 5). The paper describes no training example where the word is absent from the image, so the detector may never learn what a map looks like for a missing object (inferred).

The COCO human evaluation covers only the 500 of 5,000 image pairs with the largest L2 distance between the with-DG and without-DG images (p. 8). That selects the pairs DG changed most, so its +11 point result does not describe a typical prompt (inferred).

## Numbers

| Metric (scale, direction) | DG | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Share of image pairs judged better aligned by humans (0 to 100%, higher is better, the rest tied) | 35.3% | SD 13.0% (+22 points) | MRO, 30 prompts × 10 samples | SD 2.1 | p. 7, Figure 4 table |
| Same human measure | 23.6% | SD 12.8% (+11 points) | COCO, 500 most-changed pairs | SD 2.1 | p. 7 to 8, Figure 4 table |
| Same human measure | 15.0% | SD 14.3% (+0.7 points) | CC, about 500 prompts | SD 2.1 | p. 7, Figure 4 table |
| Same human measure | 23.5% | SD 15.9% (+7.6 points) | CC | SD 1.4 | p. 7, Figure 4 table |
| Same human measure, DG combined with Structured Diffusion | 34.3% | SD 24.4% (+9.9 points) | CC | SD 1.4 | p. 7, Figure 4 table |
| Detector mAP at IoU 0.5 (0 to 100, higher is better) at $t$ = 0, 200, 400, 600, 800 | 56.5, 58.4, 54.8, 38.1, 18.0 | none | COCO validation | SD 2.1 | p. 14, Table 1 |
| IoU between the predicted box at step $t$ and at the last step (0 to 1, higher is steadier) | about 0.75 near $t$ = 760, about 0.44 at $t$ = 800, 0.1 to 0.3 above 800 | none | not stated | not stated | p. 9, Figure 6, read from the image |

The abstract's "8-22%" range leaves out the +0.7 point result on CC with SD 2.1 (p. 1, p. 7). The authors call CC too simple for that model (p. 7).

## What to copy, what to avoid

- Copy the input design: one word's maps in, with no class output. This is the closest existing form of the proposed word-map checker, and it shows such a model can be trained on COCO in 2 GPU-days (inferred).
- Avoid assuming the maps carry layout at the earliest steps. DG's detector falls to mAP 18.0 at $t = 800$ against 58.4 at $t = 200$, and the authors say it "essentially makes random guesses" for $t \ge 800$ (p. 8, p. 14). Our push runs on steps 0 to 15 of 50. If those steps are evenly spaced over 1000 timesteps, 10 of the 16 fall above $t = 800$ (inferred, and this assumes SD 2.1 behaviour carries to SDXL).
- Avoid reading DG as evidence that a gradient on attention maps moves layout. DG overwrites attention values directly. Our own gradient push on the cat and dog maps drew two animals on 0 of 8 seeds, and DG's result does not bear on that failure (inferred).
- Copy the MRO idea of related pairs (tiger and leopard) with 10 samples each, because fusion is hardest there and our PoE failure is fusion (p. 7, p. 16, inferred).
- Add what DG lacks: a named held-out category list, a seen-versus-unseen number, and training examples where the word is absent (inferred). Our result that presence labels on every pair made pictures worse means the negative examples need care.
- Avoid scoring only the most-changed image pairs (p. 8, inferred).

## Closer work it cites

None.
