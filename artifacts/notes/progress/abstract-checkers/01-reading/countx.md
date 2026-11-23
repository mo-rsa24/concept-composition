# Open-world Text-specified Object Counting

arXiv 2306.01851, read from the full PDF (20 pages, appendix included). No venue is printed on the PDF.

## What it does

CounTX counts the objects described by a text phrase in a real photograph. It predicts a density map whose sum is the count (pp. 3, 4). A fine-tuned CLIP image encoder and a frozen CLIP text encoder feed a small transformer decoder and convolutional head (pp. 4, 5). It generates no images and contains no diffusion model.

## What the classifier or loss reads

**Image.** A clean RGB photograph. Training uses a random square crop resized to 224x224 pixels. Inference slides a square window with a stride of 128 pixels and resizes each window to 224x224 (p. 7).

**Image encoder.** CLIP ViT-B-16 pretrained on LAION-2B, with 16x16 patches, 12 layers and 512-dimensional output. Only patch tokens are used, and the CLS token is discarded (pp. 4, 5). This encoder is fine-tuned on counting (p. 5).

**Text.** A class description such as "the coins", passed through the frozen CLIP text encoder to a single 512-dimensional vector (pp. 2, 5).

**How the text enters.** Two transformer decoder layers use image features as queries and the text feature as keys and values (p. 5). The output is reshaped, resized to 24x24, and upsampled by four convolutional blocks to a density map at image resolution (p. 5).

**Noisy inputs.** CounTX never reads a noisy diffusion latent and never steers a generator. The paper contains no sampler or diffusion model (pp. 1 to 20). Its image augmentation includes pixel Gaussian noise with standard deviation 0.1 (p. 13, Table 3), which is unrelated to a diffusion noise level.

## What it trained on

**Data.** Images from the FSC-147 training set with text from FSC-147-D (p. 7). FSC-147-D rewrites class names as answers to "what object should be counted?" (p. 6). Of its descriptions, 5,668 (92.4%) are the class name with "the" in front and 467 (7.6%) needed more rephrasing (p. 17).

**Class split.** The FSC-147 training set has 89 classes, and the validation and test sets have 29 classes each. The paper states that the three sets are disjoint (p. 6).

**Labels.** Dot maps are filtered with a Gaussian of standard deviation 1 and radius 4, so each map still sums to the count (pp. 7, 13). Density values are multiplied by 60, and 20% of pixel errors are randomly dropped (p. 7).

**Training.** AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$, batch size 8 and learning rate $6.25 \times 10^{-6}$, with a 10-epoch warm-up and half-cycle cosine decay. The epoch with the lowest validation MAE across 1000 epochs is kept (p. 7).

## How its gradient enters

No gradient enters a sampler. The equations below define the counter and train it on real images.

The model (p. 4, Eq. 2), where $X$ is the image, $t$ the text, $f_{\theta_{ViT}}$ the image encoder, $f_{\theta_{TT}}$ the text encoder, $f_\phi$ the feature interaction module and $f_\psi$ the decoder:

$$\hat{Y} = f_\psi\big(f_\phi(f_{\theta_{ViT}}(X), f_{\theta_{TT}}(t))\big)$$

The training loss (p. 5, Eq. 3), the mean squared error per pixel between predicted and true density maps:

$$L(\hat{Y}, Y) = \frac{1}{H \times W} \sum_{p,q} \big((\hat{Y})_{p,q} - (Y)_{p,q}\big)^2$$

## Unseen concepts

**Held-out classes are tested.** The 29 test classes are disjoint from the 89 training classes (p. 6). The test MAE is 15.73 objects per image with original class names and 15.88 with FSC-147-D descriptions (p. 7, Table 1). Model selection uses the validation classes (p. 7).

**No seen-class comparison exists.** The paper gives no error on training classes, so the cost of moving to unseen classes cannot be read from it (inferred).

**The pretrained encoders are not naive to these classes.** CLIP was pretrained on LAION-2B image and text pairs (pp. 4, 5), so held-out means held out from the counting task (inferred).

**The training range excludes a count of two.** FSC-147 images contain "at minimum 7 objects and at most 3731 objects with an average of 56 objects per image", and CounTX "has never been trained on images with under 7 objects" (p. 19).

**Small counts have examples only.** CountBench has 540 images with two to ten objects (p. 6), and CounTX runs on a subset whose size is not printed (pp. 8, 19). There is no error figure by count. The printed examples with a true count of 1 to 5 are:

| True count | Text | Predicted | Page |
|---|---|---|---|
| 2 | "the men in the portrait" | 1.9 | p. 19, Fig. 10 |
| 2 | "the blue peacock birds" | 2.83 | p. 19, Fig. 10 |
| 2 | "the cats staring at the sign" | 3.69 | p. 20, Fig. 11 |
| 3 | "the candles" | 3.96 | p. 19, Fig. 10 |
| 5 | "The women in front of a yellow background" | 4.72 | p. 19, Fig. 10 |
| 5 | "the canadian geese flying over the ocean" | 4.89 | p. 19, Fig. 10 |

No example has a true count of 1 or 4. Rounded to a whole number, one of the three two-object examples gives 2 (inferred). These images were picked by the authors, so they give no error rate (inferred).

## Word leakage and confounds

**The model sees the text by design.** The text feature supplies the keys and values of the interaction module (p. 5).

**A text-free counter has a lower MAE.** CounTR with zero exemplars takes no text and has a test MAE of 14.12, against 15.73 for CounTX with class names (p. 7, Table 1). CounTX has the lower RMSE of the two, 106.88 against 108.01. CounTR with zero exemplars counts the dominant class (p. 3), so the FSC-147 error cannot show that a model reads the text (inferred).

**The mixed-class test is qualitative.** Pairs of FSC-147 test images are stitched together, and CounTX is asked for each class in turn (p. 8, Fig. 4). The four mosaics give eight queries with true counts from 9 to 22. Absolute errors range from 0.26 to 1.34 objects (computed from Fig. 4, inferred). The paper calls these "a few examples" and gives no aggregate number.

**Text naming an absent class is untested.** In the mosaics and in Fig. 1 (p. 2), each query names a class that is present. No test asks for a class missing from the image, so the expected output of zero is never checked.

**Part and relation failures.** CounTX can place one dot per repeated part: sunglasses are predicted at 87.97 against a true 42, and butterfly stickers at 16.78 against 8 (p. 20, Fig. 11). It also counts an illustrated cat when the text asks for the real cats staring at the sign (p. 20).

## Numbers

MAE and RMSE are mean absolute and root mean squared counting errors in objects per image (p. 6). Lower is better. The model is CLIP ViT-B-16 from LAION-2B unless the row says otherwise.

| Metric | CounTX | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| Test MAE | 15.73 class names; 15.88 descriptions | 22.09 (Patch-selection, text) | FSC-147 held-out test classes | CLIP ViT-B-16 | p. 7, Table 1 |
| Test RMSE | 106.88 class names; 106.29 descriptions | 115.17 (Patch-selection) | FSC-147 held-out test classes | CLIP ViT-B-16 | p. 7, Table 1 |
| Test MAE against no text | 15.73 | 14.12 (CounTR, 0-shot) | FSC-147 held-out test classes | CLIP ViT-B-16 | p. 7, Table 1 |
| Test MAE against 3 exemplars | 15.73 | 10.79 (LOCA) | FSC-147 held-out test classes | CLIP ViT-B-16 | p. 7, Table 1 |
| Test MAE by encoder freezing | 37.37 both frozen; 15.88 image fine-tuned and text frozen | none | FSC-147 test | CLIP ViT-B-16 | p. 8, Table 2 |
| MAE | 12.15 | 35.56 (Mask-RCNN, closed-set) | Test-COCO | CLIP ViT-B-16 | p. 14, Table 4 |
| MAE, trained on FSC-147 only | 11.64 ("car"); 11.72 ("cars") | 2.12 (HLCNN, trained for cars) | CARPK | CLIP ViT-B-16 | p. 15, Table 5 |
| MAE, trained on FSC-147 and CARPK | 8.13 ("the cars") | 2.12 (HLCNN) | CARPK | CLIP ViT-B-16 | p. 15, Table 5 |
| Frozen-backbone test MAE, 3-exemplar task | 24.96 frozen CLIP ViT-B-16 | 14.93 (frozen CounTR encoder) | FSC-147 test | listed backbones | p. 16, Table 6 |

## What to copy, what to avoid

- Copy the disjoint 89, 29 and 29 class split and select the model on held-out validation classes (pp. 6, 7) (inferred).
- Do not train a two-subject checker on FSC-147 counts. Its smallest count is 7 (p. 19), so a count of two is outside the labelled range (inferred).
- Copy the stitched-image test and extend it. Query each present subject, query an absent subject, and require a near-zero answer for the absent one (p. 8) (inferred).
- Treat a two-headed hybrid as the main risk. CounTX can count repeated parts as separate objects (p. 20), so a count near two could be earned by a fused animal with two heads (inferred).
- Do not rely on frozen CLIP features for counting. Frozen CLIP was weaker than a frozen self-supervised encoder (p. 16), which fits our CLIP checker reading the pair at chance on decoded early images (balanced accuracy 0.50 at step 0) (inferred).
- Do not assume this works during sampling. Every result here comes from clean real images, while our push acts on steps 0 to 15 of 50 (inferred).

## Closer work it cites

None of these is in the list of 18.

- Teaching CLIP to Count to Ten, arXiv 2302.12066 (p. 12, ref. 26). It is the source of CountBench, whose images hold two to ten objects (p. 6).
- Zero-shot Object Counting (Xu et al.), CVPR 2023 (p. 12, ref. 34). A class-name counter tested on FSC-147 (pp. 3, 7).
- Learning to Count Anything: Reference-less Class-agnostic Counting with Weak Supervision (RCC), arXiv 2205.10203 (p. 11, ref. 10). A counter that uses no exemplars or text (pp. 3, 7).
- CounTR: Transformer-based Generalised Visual Counting, BMVC 2022 (p. 11, ref. 19). Its zero-exemplar mode has the lowest no-text test MAE in Table 1 (p. 7).
