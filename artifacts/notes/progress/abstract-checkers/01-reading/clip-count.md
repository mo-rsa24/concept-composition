# CLIP-Count: Towards Text-Guided Zero-Shot Object Counting

arXiv 2305.07304, read from the full PDF (12 pages, appendix included). ACM Multimedia 2023 (MM '23), printed on p. 1.

## What it does

CLIP-Count counts the objects named by a text prompt in a real photograph. It predicts a density map, and the sum of that map is the count (p. 3). CLIP's image and text encoders stay frozen, a small set of learned prompts adapts them, and a decoder reads patch features after they have attended to the text (pp. 3, 5). It generates no images and contains no diffusion model.

## What the classifier or loss reads

**Image.** A clean RGB photograph. Training images are downsampled to 224x224 pixels (p. 5). At inference the shortest side is resized to 224 pixels and a 224x224 window slides over the image with a stride of 128 pixels, averaging overlapping predictions (p. 11).

**Image encoder.** OpenAI CLIP ViT-B/16 with its weights frozen, plus 20 trainable visual prompts at each transformer layer (pp. 3, 5). The counter uses the 14x14 patch feature map of 768-dimensional vectors, projected into CLIP's 512-dimensional text space (pp. 3, 4).

**Text.** The FSC-147 class name, such as "apple", goes through the frozen CLIP text encoder with 2 learned context tokens and becomes one embedding $E_t$ (pp. 3, 4, 5).

**How the text enters.** Two blocks each apply self-attention, cross-attention from image to text, and a small MLP. They output feature maps at 14x14 and 28x28, which a convolutional decoder turns into the density map (pp. 4, 5).

**Noisy inputs.** The counter never reads a noisy diffusion latent and never steers a generator. The paper contains no sampler or diffusion model (pp. 1 to 12). Training augmentations are inherited from CounTR and are not itemised here (p. 5).

## What it trained on

**Data.** The FSC-147 training set, with the class name as the prompt and dot annotations as the density target. The exemplar boxes are not used (p. 5).

**Class split.** FSC-147 has 6,135 images across 147 classes "with non-overlapping object classes in each set" (p. 5). The paper also writes that training classes share nothing with validation or test classes (p. 3). It does not print how many classes each split holds.

**Training.** 30 epochs with the patch-text contrastive loss alone, then 200 epochs with mean squared error on the density map (p. 5). Batch size 32, AdamW at learning rate $1 \times 10^{-4}$, decayed by a factor of 0.33 after 100 epochs, about 3 hours on one RTX-3090Ti (p. 5).

## How its gradient enters

No gradient enters a sampler. The losses below train the counter on real images.

The patch-text contrastive loss (p. 4, Eq. 3), where $\mathcal{P}$ and $\mathcal{N}$ are patches with and without objects according to the max-pooled ground-truth density map, $s$ is cosine similarity and $\tau = 0.07$:

$$\mathcal{L}_{con} = -\log \frac{\sum_{i \in \mathcal{P}} \exp(s(E_p^i, E_t)/\tau)}{\sum_{i \in \mathcal{P}} \exp(s(E_p^i, E_t)/\tau) + \sum_{k \in \mathcal{N}} \exp(s(E_p^k, E_t)/\tau)}$$

The density loss (p. 5), where $y$ is the ground-truth density map, $\hat{y}$ the prediction, and $H, W$ the image size:

$$\mathcal{L}_{MSE} = \frac{\lVert y - \hat{y} \rVert^2}{H \times W}$$

## Unseen concepts

**Held-out classes are tested.** The model trains on FSC-147 training classes and is evaluated on validation and test classes that do not overlap them (pp. 3, 5). Test mean absolute error (MAE) is 17.78 objects per image (p. 6, Table 1).

**No seen-class comparison exists.** The paper gives no error on training classes, so the cost of moving to unseen classes cannot be read from it (inferred).

**The encoders are not naive to these classes.** The held-out split applies to the counting head and tuned prompts. CLIP was pretrained on large-scale image and text pairs (p. 3), so the test words and objects may already be familiar to it (inferred).

**Small counts are absent.** The paper has no result broken down by count. The smallest ground-truth count in any printed example is 9 objects (p. 7, Fig. 5a; p. 8, Figs. 6b and 7a). It gives no result near a count of two.

**Cross-dataset tests.** The FSC-147 model is tested without fine-tuning on CARPK (p. 6) and on ShanghaiTech crowds (p. 6). The paper does not say whether cars or people were removed from its training classes.

## Word leakage and confounds

**The model sees the word by design.** Text enters through the learned prompt, the contrastive loss and cross-attention (pp. 4, 5).

**No test uses text naming a different class.** The paper never asks for a class absent from the image and never swaps the prompt on an image, so it cannot show how far the count depends on the word.

**The benchmark does not require reading the word.** Three counters given no text and no exemplars have lower test MAE: CounTR 14.71, LOCA 16.22 and RCC 17.12, against 17.78 for CLIP-Count (p. 6, Table 1). A low FSC-147 error therefore does not establish text understanding (inferred).

**Word and part ambiguity cause visible failures.** The prompt "apples" also counts a drawing of an apple (predicted 11.63, true 9). "Sunglasses" counts lenses separately (predicted 284.10, true 140) (p. 8, Fig. 7).

## Numbers

MAE and RMSE are the mean absolute and root mean squared counting errors in objects per image, computed over the test images (p. 5). Lower is better. The model is CLIP ViT-B/16 throughout.

| Metric | CLIP-Count | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| Test MAE | 17.78 | 22.09 (Xu et al., the other text-specified counter) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 6, Table 1 |
| Test RMSE | 106.62 | 115.17 (Xu et al.) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 6, Table 1 |
| Test MAE against counters with no text | 17.78 | 14.71 (CounTR, no exemplars) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 6, Table 1 |
| Test MAE against 3 exemplar boxes | 17.78 | 10.97 (LOCA) | FSC-147 held-out test classes | CLIP ViT-B/16 | p. 6, Table 1 |
| Validation MAE | 18.79 | 17.43 (LOCA, no exemplars) | FSC-147 held-out validation classes | CLIP ViT-B/16 | p. 6, Table 1 |
| MAE | 11.96 | 10.44 (BMNet+, 3 exemplars); 21.38 (RCC, no exemplars) | CARPK, cross-dataset | CLIP ViT-B/16 | p. 6, Table 2 |
| MAE | 45.7 | 66.6 (RCC) | ShanghaiTech Part B, trained on FSC-147 | CLIP ViT-B/16 | p. 6, Table 3 |
| MAE | 192.6 | 217.0 (CrowdCLIP) | ShanghaiTech Part A, trained on FSC-147 | CLIP ViT-B/16 | p. 6, Table 3 |
| Test MAE by encoder treatment | 28.84 frozen, 22.51 fully fine-tuned, 18.77 frozen with visual prompts, 17.78 after adding the contrastive loss | none | FSC-147 test | CLIP ViT-B/16 | p. 7, Table 4 |

Result by small count: none reported. Test with text naming a different class: none reported.

## What to copy, what to avoid

- Copy the class-disjoint split and state it as a set condition before training (p. 3) (inferred).
- Add the control this paper lacks: score each image with the right word, a wrong word and no word. Text-free counters beat this model on FSC-147 (p. 6), so a held-out error alone cannot show a checker reads the word map (inferred).
- Do not use FSC-147 counters as evidence about telling one subject from two. Nothing printed here is below 9 objects (pp. 7, 8), and our target is a count of two (inferred).
- Do not carry its accuracy to our sampling window. It reads clean 224-pixel photographs, while our push acts on steps 0 to 15 of 50, and our CLIP checker on decoded early images reads the pair at chance (balanced accuracy 0.50 at step 0) (inferred).
- Keep a small trainable part on top of frozen features. Purely frozen CLIP gave a test MAE of 28.84 and adding prompts gave 18.77 (p. 7, Table 4). Our SDXL stays frozen, so the head has to carry the task (inferred).
- Watch for counting repeated parts. "Sunglasses" were counted per lens (p. 8), so a density-style count could score a two-headed hybrid as two subjects (inferred).

## Closer work it cites

None of these is in the list of 18.

- Teaching CLIP to Count to Ten, arXiv 2302.12066 (p. 9, ref. 27). The paper names it among CLIP-based counting models limited to image-level classification (p. 3), and its title targets small counts.
- Zero-shot Object Counting (Xu et al.), arXiv 2303.02001 (p. 9, ref. 41). A class-name counter tested on FSC-147 (pp. 2, 6).
- Learning to Count Anything: Reference-less Class-agnostic Counting with Weak Supervision (RCC), arXiv 2205.10203 (p. 9, ref. 6). It regresses a count with no text and has a lower FSC-147 test MAE than CLIP-Count (pp. 2, 6).
- CounTR: Transformer-based Generalised Visual Counting, arXiv 2208.13721 (p. 9, ref. 19). It is the strongest no-text counter in Table 1 (p. 6).
