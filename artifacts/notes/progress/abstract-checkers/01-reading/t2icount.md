# T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting

arXiv 2502.20625, read from the full PDF (10 pages; the PDF has no appendix). No venue is printed on the PDF.

## What it does

T2ICount counts the objects of a category named in text, in real photographs, by predicting a density map that describes the count (p. 3). It reads features from one denoising step of a fine-tuned Stable Diffusion v1.5 U-Net and refines them with a trained head (pp. 3, 4, 6). It is an image-analysis model and generates nothing.

## What the classifier or loss reads

**Input to the model.** A real image goes through the frozen VAE encoder to a latent $z_t$, and the class name goes through the frozen CLIP text encoder (pp. 3, 5, 6). The U-Net runs for a single denoising step at $t = 1$, chosen to stay close to the original image (p. 4).

**What the counter reads.** Four decoder feature maps $F_i$ of size $\frac{H}{2^{i-1}} \times \frac{W}{2^{i-1}}$ for $i \in \{1,2,3,4\}$, where $H, W$ are the latent dimensions (pp. 3, 4), together with the text features $c'$ (p. 4). The Hierarchical Semantic Correction Module fuses them and computes a text-image similarity map $S_i$ by cosine similarity (p. 4, Eq. 3). A counter of self-attention and three convolutional layers outputs the density map (p. 6).

**What the cross-attention maps do.** Maps at 12x12, 24x24 and 48x48 are upsampled, min-max normalised and fused with weights 0.6, 0.3 and 0.1 (pp. 5, 6). The fused map is used only to build the training target for the similarity map, by marking likely background (pp. 3, 5). The paper describes no path from the fused map into the counter (pp. 3, 5, 6). It does not say which tokens' maps are fused.

**How good those maps are.** At $t = 1$ the attention maps "show weak sensitivity to specific object categories" and capture general foreground (p. 5). The authors use them for that reason, to find background pixels (p. 5).

## What it trained on

**Data.** The FSC-147 training split: 3,659 images (p. 5). The text input is the class name (p. 5). Labels are the dataset's point annotations, turned into a ground-truth density map $D^{gt}$ (p. 5).

**What is trained.** The U-Net is fine-tuned at one tenth of the base learning rate, and the correction module and counter are trained (pp. 3, 6). The VAE encoder and text encoder stay frozen (p. 6).

**Settings.** Base learning rate $5 \times 10^{-5}$, 400 epochs, AdamW with weight decay $1 \times 10^{-4}$, batch size 16 (p. 6). Loss weights $\lambda = 2$ and $\gamma = 0.01$ (p. 6). The thresholds $\tau$ and $\theta$ in Eq. 6 have no printed values (pp. 5, 6).

## How its gradient enters

No gradient enters any generator. The paper never guides image generation. The gradients below train the counter.

The fused attention map (p. 5, Eq. 5):

$$\mathcal{A}^{cross} = \sum_i w_i \cdot \mathrm{norm}(\mathcal{A}_i^{cross}) \in \mathbb{R}^{H \times W}$$

Each position gets a positive, negative or ambiguous label (p. 5, Eq. 6):

$$p_{jk} = \begin{cases} 1, & \text{if } D^{gt}_{jk} \ge \tau \\ 0, & \text{else if } \mathcal{A}^{cross}_{jk} \le \theta \\ -1, & \text{otherwise} \end{cases}$$

The Representational Regional Coherence Loss on the similarity map $S$ (p. 5, Eqs. 7 to 9):

$$\mathcal{L}_{RRC} = \lambda \mathcal{L}_{pos} + \mathcal{L}_{neg}, \quad \mathcal{L}_{pos} = \sum_{jk} 1 - S_{jk} \;\text{ if } p_{jk} = 1, \quad \mathcal{L}_{neg} = \sum_{jk} \max(0, S_{jk}) \;\text{ if } p_{jk} = 0$$

The total training loss (p. 5, Eq. 10), where $\mathcal{L}_{reg}$ is the regression loss taken from the CUT crowd-counting paper:

$$\mathcal{L} = \mathcal{L}_{reg} + \gamma \mathcal{L}_{RRC}$$

Ambiguous positions ($p_{jk} = -1$) receive no loss (p. 5).

## Unseen concepts

**Held-out classes are tested.** FSC-147 has 147 classes split into 3,659 training, 1,286 validation and 1,190 test images "with non-overlapping classes across splits" (p. 5). The model trains on the training split only (p. 6). On the test split, whose classes it never trained on, mean absolute error is 11.76 objects per image, against 13.30 for the next zero-shot method, GeCo (p. 6, Table 1).

**There is no seen-against-unseen comparison.** The paper reports no error on the training classes, so the cost of moving to unseen classes cannot be read from it (inferred).

**The strongest result is not shown to be held out.** FSC-147-S is 196 images "manually" extracted from FSC-147, each with a less frequent category added (p. 6). The paper does not say which split those images come from, or whether the added categories occur among the training classes. The 4.69 error on FSC-147-S therefore cannot be cited as a held-out result (inferred).

**The cross-dataset test** on CARPK (car parks seen from drones, 1,448 images) uses a model trained on FSC-147 (pp. 5, 7). The paper does not say whether cars appear in its training classes.

**This is a counter on real images.** It is not a checker used to guide generation, so its held-out result says nothing direct about guidance (inferred).

## Word leakage and confounds

**The model sees the word by design.** Text enters through U-Net cross-attention and again through the text features in the correction module (pp. 3, 4).

**The confound the paper tackles is ignoring the word.** A model can score well on FSC-147 by counting the dominant object, because labels mostly name the majority class (p. 2). FSC-147-S tests this by asking for the less frequent class, with an average count of 4.6 against 49.4 for the originally annotated class (p. 6). The baseline of diffusion features plus a counter has an error of 14.66 on the ordinary test split and 24.34 on FSC-147-S (p. 8, Table 4).

**The opposite confound is untested.** The paper has no name-only control, so it cannot show how much of the count could be guessed from the class name alone (inferred).

## Numbers

MAE is mean absolute counting error in objects per image, and RMSE is root mean squared counting error in objects per image. Lower is better for both. The model is Stable Diffusion v1.5 throughout (p. 6).

| Metric | T2ICount | Best baseline | Dataset | Page and table |
|---|---|---|---|---|
| MAE | 11.76 | 13.30 (GeCo) | FSC-147 test, classes unseen in training | p. 6, Table 1 |
| RMSE | 97.86 | 103.42 (DAVE) | FSC-147 test | p. 6, Table 1 |
| MAE | 4.69 | 31.30 (CounTX) | FSC-147-S, 196 images, less frequent class | p. 7, Table 2 |
| RMSE | 8.06 | 61.34 (PseCo) | FSC-147-S | p. 7, Table 2 |
| MAE | 8.61 | 8.13 (CounTX) | CARPK, cross-dataset | p. 7, Table 3 |
| RMSE | 13.47 | 10.87 (CounTX) | CARPK | p. 7, Table 3 |
| MAE, ablation | 24.34 baseline, 9.59 with $\mathcal{L}_{RRC}$, 4.69 with the correction module added | none | FSC-147-S | p. 8, Table 4 |

T2ICount is second on CARPK MAE and third on CARPK RMSE (p. 7, Table 3). Few-shot LOCA, which is given three example boxes, has a lower FSC-147 test MAE of 10.97 (p. 6, Table 1).

The text's FSC-147-S reductions of 85.1% in MAE and 86.9% in RMSE (p. 7) use different comparison methods. From Table 2, the MAE reduction against CounTX computes to 85.0%, and the RMSE reduction matches PseCo, which is outside the three CLIP-based methods that sentence names (inferred).

## What to copy, what to avoid

- Copy the split by category before training, with no class shared between train and test (p. 5). This is the held-out design our checkers need (inferred).
- Copy the FSC-147-S idea: test on cases where the requested concept is the less dominant one, so a checker that ignores the word fails visibly (p. 6) (inferred).
- Add the control this paper lacks: a word-only or shuffled-word baseline, because our checkers could learn from the word instead of the picture (inferred).
- Do not cite this paper as evidence that word attention maps alone can count. The counter reads U-Net decoder features and text, and attention only supplies a background mask (pp. 3, 5) (inferred).
- Treat word attention as a risky sole input. At $t = 1$ on SD1.5 its maps were weak on category and good on general foreground (p. 5), and our proposed checkers read only those maps (inferred).
- Do not carry its feature quality over to our sampling window. It reads an almost clean real image at $t = 1$, while our push acts on steps 0 to 15 of 50, and our CLIP checker on decoded early images reads the pair at chance (balanced accuracy 0.50 at step 0) (inferred).
- Do not assume frozen features are enough. T2ICount fine-tunes the U-Net (p. 6), and our SDXL stays frozen (inferred).
- Consider its three-way labels when training on uncertain regions. It applies no loss to ambiguous positions (p. 5), which may help where our presence labels on every pair made pictures worse (inferred).

## Closer work it cites

These train a text-specified counter and test it on FSC-147's held-out classes (p. 3, p. 6 Table 1). None is in the list of 18.

- CLIP-Count: Towards text-guided zero-shot object counting (ACM MM 2023). No arXiv id printed (p. 9, ref. 7). It fine-tunes CLIP's image encoder for counting (p. 2).
- Open-world text-specified object counting, CounTX (BMVC 2023). No arXiv id printed (p. 9, ref. 1). It has the lowest CARPK error in this paper's Table 3 (p. 7).
- VLCounter: Text-aware visual representation for zero-shot object counting (AAAI 2024). No arXiv id printed (p. 9, ref. 8).
- Zero-shot object counting with good exemplars, VA-Count, arXiv 2407.04948 (p. 10, ref. 39). It uses Grounding DINO detections to pick exemplars (p. 3).
