# VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting

arXiv 2312.16580, read from the full PDF (12 pages, appendix included). The PDF carries a 2024 AAAI copyright line (p. 1).

## What it does

VLCounter counts the objects of a named class in a real photograph by predicting a density map whose sum is the count (pp. 2, 3). It builds a patch-by-text similarity map from CLIP and uses the text in three added modules that adapt CLIP's image encoder and pass filtered features to a decoder (pp. 3, 4). It generates no images and contains no diffusion model.

## What the classifier or loss reads

**Image.** A clean RGB photograph resized to 384x384 pixels (p. 5). CLIP's position embeddings are resized to accept this input (p. 10). With 16-pixel patches this gives a 24x24 patch grid (inferred).

**Encoder.** CLIP ViT-B/16 with frozen pretrained weights and learned prompts (pp. 2, 3, 5).

**Text.** The class name goes into a prompt such as "A photo of a number of {class}" (p. 3, Fig. 2). The plural form uses 11 templates (p. 11). The paper does not say how the templates are combined. The text encoder outputs one embedding $\mathcal{T}$ (p. 3), so the model has no separate map per word (inferred).

**How the text enters.** It enters at four points (pp. 3, 4, 10):

1. A projection of $\mathcal{T}$ is added to 10 learnable tokens at each of the 12 image encoder layers.
2. Cosine similarity between $\mathcal{T}$ and each patch embedding forms the similarity map $S$.
3. A learned per-position affine transform turns $S$ into the counting map $\hat{S}$, which is concatenated with the patch embeddings as decoder input.
4. $\hat{S}$ filters encoder features from layers 7, 8 and 9 before they join decoder layers 2, 3 and 4.

**Noisy inputs.** VLCounter never reads a noisy diffusion latent and never steers a generator. The paper contains no sampler or diffusion model (pp. 1 to 12).

## What it trained on

**Data.** FSC-147, with seen class names and density maps during training (pp. 2, 5). The dataset has 6,135 images from 147 categories (p. 5).

**Class split.** The paper writes the seen and unseen class sets as disjoint, $C^s \cap C^u = \emptyset$ (p. 2). It does not print how many classes each split holds.

**Training.** AdamW at learning rate $10^{-4}$ and weight decay $10^{-2}$ for 200 epochs, batch size 16, on one RTX A6000 (p. 5). Rank-contrastive thresholds are 0.8, 0.6 and 0.4 (p. 10). The sentence giving $\lambda$ and $\tau$ as $10^{-6}$ and 1 names them in the opposite order to the phrase "temperature scaling and loss-balancing", so which value belongs to which symbol is ambiguous (p. 5).

## How its gradient enters

No gradient enters a sampler. The equations below define the counter and train it on real images.

The similarity map (p. 3, Eq. 2), where $v_{ij}$ is the patch embedding at position $(i,j)$:

$$S_{ij}(\mathcal{V}, \mathcal{T}) = \frac{v_{ij}\mathcal{T}^T}{\lVert v_{ij} \rVert \lVert \mathcal{T} \rVert}$$

The learned transform (p. 4, Eq. 7), where $W$ and $B$ are learned matrices and $\otimes$ is element-wise multiplication:

$$\hat{S} = W \otimes S + B$$

The training objective (p. 5, Eqs. 4, 9 and 10), where $D$ is the density map and $\hat{S}^{pos}_r$, $\hat{S}^{neg}_r$ are patches above and below each threshold:

$$\mathcal{L}_{total} = \lVert D_{pred} - D_{gt} \rVert_2^2 + \lambda \cdot \left(-\sum_{k=1}^{K} \log \frac{\sum_{\hat{S}_i \in \hat{S}^{pos}_r} \exp(\hat{S}_i/\tau)}{\sum_{\hat{S}_j \in (\hat{S}^{pos}_r \cup \hat{S}^{neg}_r)} \exp(\hat{S}_j/\tau)}\right)$$

## Unseen concepts

**Held-out classes are tested.** The problem statement requires test classes disjoint from training classes (p. 2). Test MAE on FSC-147 is 17.05 objects per image (p. 5, Table 1).

**No seen-class comparison exists.** The paper gives no error on training classes, so the cost of moving to unseen classes cannot be read from it (inferred).

**The encoders are not naive to these classes.** CLIP's pretrained encoders were not trained under the FSC-147 split, so the held-out condition applies to the counting parts (inferred).

**Small counts are absent.** There is no result by count. The smallest ground-truth count printed in an example is 13 eggs (p. 12, Fig. 9). The paper says counting "mainly assumes the existence of multiple instances in every image" and uses plural prompts for that reason (p. 6). It gives no result near a count of two.

**Cross-dataset tests.** VLCounter is tested on CARPK and PUCPR+ without fine-tuning (p. 6). It calls cars "newly seen objects" (p. 7), and the PDF does not list FSC-147's training classes, so this cannot be checked from the paper.

## Word leakage and confounds

**The word reaches the features before the counter reads them.** The text embedding is added to prompts in every image encoder layer (p. 4), so the patch embeddings are already conditioned on the class name. The paper does not treat this as a leakage risk.

**One text path has little measured effect on test MAE.** Removing the text from the tuned prompts changes test MAE from 17.05 to 17.19 objects per image, while validation MAE moves from 18.06 to 19.07 (p. 6, Table 4). Text still reaches the model through the similarity map in that run (p. 6). No experiment removes all text paths, so the paper cannot say how much of the count depends on the word (inferred).

**Wording changes the result.** Plural prompts have test MAE 17.05 and singular prompts 19.18. RMSE moves the other way, 106.16 for plural and 105.04 for singular (p. 6, Table 5).

**Text naming a different class is untested.** No experiment supplies a wrong class name, an absent class, or an image holding two named classes.

## Numbers

MAE and RMSE are mean absolute and root mean squared counting errors in objects per image. Lower is better. The model is CLIP ViT-B/16 throughout.

| Metric | VLCounter | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| Test MAE | 17.05 | 17.78 (CLIP-Count) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 11, Table 7 |
| Test MAE against its main-table text baseline | 17.05 | 22.09 (ZSC) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 5, Table 1 |
| Test RMSE | 106.16 | 106.62 (CLIP-Count) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 11, Table 7 |
| Test MAE against few-shot | 17.05 | 14.62 (BMNet+, with exemplars) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 5, Table 1 |
| Test MAE, baseline to full model | 32.20 VLBase; 17.58 with prompt tuning only; 17.05 full | none | FSC-147 test | CLIP ViT-B/16 | p. 6, Table 3 |
| Test MAE without text in prompts | 17.19, against 17.05 with text | none | FSC-147 test | CLIP ViT-B/16 | p. 6, Table 4 |
| MAE | 6.46 | 10.44 (BMNet+) | CARPK, no fine-tuning | CLIP ViT-B/16 | p. 5, Table 2 |
| MAE | 48.94 | 62.42 (BMNet+) | PUCPR+, no fine-tuning | CLIP ViT-B/16 | p. 5, Table 2 |

The text reports PUCPR+ as a 27.54% gain over BMNet+ (p. 6). The table values give a 21.6% reduction relative to BMNet+, and 27.54% results from dividing the same difference by VLCounter's error (inferred).

## What to copy, what to avoid

- This is the nearest counter here to the proposed input, because it keeps a spatial map of how strongly each patch matches the text (p. 3). The decoder always receives image features as well (pp. 3, 4), so it does not show that a text-match map alone can count (inferred).
- Run the map-only test ourselves: train one head on word attention maps alone and one with image features, on held-out concepts (inferred).
- Do not feed text into the features the checker reads unless a wrong-word control is present. VLCounter conditions every encoder layer on the text and never tests a wrong class (p. 4) (inferred).
- Do not take plural prompts or FSC-147 counts as a design for two subjects. The method assumes many instances (p. 6) and prints no count below 13 (p. 12) (inferred).
- Our failed attention push was on 32x32 maps with a minus-correlation loss and drew two animals on 0 of 8 seeds. VLCounter's map is read by a trained decoder on clean images, so its success does not predict that a pushed attention map will change layout (inferred).
- Do not assume this works during sampling. It reads clean 384-pixel images, while our push acts on steps 0 to 15 of 50 (inferred).

## Closer work it cites

None of these is in the list of 18.

- Zero-shot Object Counting (ZSC, Xu et al.), arXiv 2303.02001 (p. 9). A class-name counter tested on FSC-147 (pp. 1, 2, 5).
- Exemplar Free Class Agnostic Counting (RepRPN-Counter), ACCV (p. 8). A counter that needs no exemplars and is tested on FSC-147 (pp. 1, 2, 5).
