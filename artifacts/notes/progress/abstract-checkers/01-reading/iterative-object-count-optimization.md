# Detection-Driven Object Count Optimization for Text-to-Image Diffusion Models

arXiv 2408.11721 (v2), filed under the slug `iterative-object-count-optimization`, read from the full PDF (15 pages, appendix included). No venue is printed on the PDF. The page header carries the placeholder author name "Trovato et al.". The extracted text has a stray page 16 marker holding the captions of Figs. 19 and 20, which sit on PDF page 15.

## What it does
The method optimises one new text token so that images generated with it hold the requested count (p. 3 to 4). Each iteration generates a full image, scores it with a pretrained density-map counter whose scale is set by a pretrained object detector, and updates the token embedding (p. 2, p. 4). The finished token can be reused on other prompts and classes with no further optimisation (p. 5).

## What the classifier or loss reads
**Counter.** The counter is a density map $\Phi(x,c)$ from the pretrained CLIP-Count model, summed over pixels (p. 4, Eq. 3). It takes the fully generated image $x^{(k)}$ and the class word $c$ (p. 4).

**Detector.** YOLO or DETR counts class $c$ in the same finished image. That count only sets a scale factor and is excluded from backpropagation (p. 4, Eq. 4 and 5).

**Early noisy steps.** It does not read them. The authors chose finished images because the counter was trained on clean images and is unreliable on noisy intermediates (p. 1, p. 2, p. 4). Backpropagation runs only through the final denoising step (p. 2).

## What it trained on
No counter is trained. CLIP-Count, YOLO and DETR are all used pretrained (p. 4). The only optimised object is the token embedding $e_{count}$, randomly initialised, for about 30 iterations and about one minute on SDXL-Turbo (p. 4, p. 12). The prompt is "A photo of $N$ $c$" (p. 4). The loss is the count distance plus a CLIP penalty with $\lambda = 5$, at learning rate 0.05 (p. 5, p. 12). One token is trained for one class and count, and the paper proposes dedicated tokens for different quantities (p. 7).

## How its gradient enters
The gradient updates the token embedding between whole generations and never enters the noise prediction (p. 4, Eq. 1, 2, 4, 5; p. 5, Eq. 7):

$$e_{count}^{(k+1)} \leftarrow e_{count}^{(k)} + \alpha \frac{\nabla_{e_{count}^{(k)}} \mathcal{L}(x^{(k)}, c, N)}{\lVert \mathcal{L}(x^{(k)}, c, N) \rVert^2}$$

$$\mathcal{L}_{counting}(x^{(k)}, c, N) = \left\lVert \mathrm{Count}(x^{(k)}, c) - N \right\rVert$$

$$\mathrm{Count}(x^{(k)}, c) = \sum_{i=1}^{h \cdot w} \hat{\lambda}_{scale}(x^{(k)}, c)\, \Phi_i(x^{(k)}, c), \qquad \hat{\lambda}_{scale} = \frac{\mathrm{DetectionCount}(x^{(k)}, c)}{\sum_{j=1}^{h \cdot w} \Phi_j(x^{(k)}, c)}$$

$$\mathcal{L}(x, c, N) = \mathcal{L}_{counting}(x, c, N) + \lambda\, \mathcal{L}_{semantic}(x, c)$$

$\alpha$ is the learning rate, $N$ the target count and $\Phi_i$ the density at pixel $i$. $\mathcal{L}_{semantic}$ is CLIP similarity to "A photo of $c$" (p. 5, Eq. 6). The static variant uses a fixed scale of 60 (p. 12).

Eq. 1 is printed with a plus sign and with the norm of the loss in the denominator (p. 4). Read literally, this is gradient ascent on a loss the text says is reduced, so the printed sign is probably a typo (inferred). The tables name the model SD-Turbo (p. 6), while the appendix and Fig. 15 name SDXL-Turbo (p. 12, p. 14).

## Unseen concepts
**Token reuse on other classes (partly).** A token trained on one class is applied to other classes at count 10 across three groups: fruits, birds and domestic mammals (p. 7). Mean absolute error is 6.11 objects within a group and 6.61 across groups, and the dashed SD line in Fig. 8 sits near 16.4 (p. 7, Fig. 8). This tests the token and covers count 10 only.

**Counter on classes it never trained on (one sentence).** With YOLO as the scorer, error is higher on classes unseen by CLIP-Count: MAE 8.2 against 7.72 objects (p. 7). The paper does not print which classes, how many images, or the counts involved. It measures the final generation and gives no accuracy for the counter itself.

## Word leakage and confounds
**Word access.** The counter, the detector and the CLIP penalty all receive the class word (p. 4, p. 5). The paper does not discuss word leakage.

**Adversarial solutions.** Without the CLIP penalty, InstaFlow runs reached correct counts through "adversarial changes" such as colour distortions and content shifts (p. 12).

**Scoring with the steering detector.** Each detector scores best when it also steered. YOLO-steered runs get YOLO MAE 3.04 against 4.95 for DETR-steered runs. DETR-steered runs get DETR MAE 2.51 against 3.42 for YOLO-steered runs (p. 6, Table 1, up to 9 objects). The accuracy column uses a YOLO and DETR verifier that accepts agreement from either one and allows a margin of one count for targets above two (p. 5).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Count accuracy, percent of images, higher is better, margin of one above count 2 | 45.61% (dynamic DETR) | 31% CountGen on SDXL, 26.9% SD-Turbo | Up to 9 objects | SD-Turbo | p. 6, Table 1 |
| Count accuracy, same definition | 24% (dynamic DETR) | 12.63% SD-Turbo, 9.47% Counting Guidance on SD 1.4 | Full benchmark, 3,674 samples, targets 1 to 25 | SD-Turbo | p. 5, p. 6, Table 1 |
| DETR MAE in objects, lower is better | 2.51 (dynamic DETR) | 3.53 CountGen, 3.61 SDXL, 6.99 SD-Turbo | Up to 9 objects | SD-Turbo | p. 6, Table 1 |
| YOLO MAE in objects, lower is better | 3.04 (dynamic YOLO) | 5.49 CountGen, 5.90 SDXL, 10.32 SD-Turbo | Up to 9 objects | SD-Turbo | p. 6, Table 1 |
| Token reuse MAE in objects at count 10, lower is better | 6.11 in-domain, 6.61 out-of-domain | About 16.4 SD (dashed line, read from plot) | 9 classes in 3 groups | SD-Turbo | p. 7, Fig. 8 |
| **Count 2**, YOLO MAE in objects, read from the plot, lower is better | about 1.0 | about 1.8 SD | Benchmark images at target 2 | SD-Turbo, variant not named in the caption | p. 14, Fig. 16 |

An error of about 1.0 object at target 2 is half the target (inferred). The paper prints no accuracy at count 2 and does not test two different object classes in one count.

## What to copy, what to avoid
- Copy the plain finding that a counter trained on clean images is unreliable on look-ahead images. It agrees with our CLIP checker reading the pair at chance on early decoded images (inferred).
- Avoid its outer loop for our use. It needs repeated whole generations and relies on a distilled model to afford them (p. 4), while our setup steers one frozen SDXL during sampling (inferred).
- Treat the adversarial failures on p. 12 as a warning for any gradient through a decoded image. Our image-space checker likewise changed texture and left layout alone (inferred).
- Score with a counter that did not steer. Table 1 shows each detector favouring the runs it steered (inferred).
- Report the unseen-class split properly. The 8.2 against 7.72 result shows a small gap for one counter, and our claim needs the class lists, the image counts and the counter's own accuracy (inferred).

## Closer work it cites
- **CLIP-Count: Towards Text-Guided Zero-Shot Object Counting**, arXiv 2305.07304, p. 8. It is the trained counter this method relies on, built to count classes given by text.
- **AFreeCA: Annotation-Free Counting for All**, arXiv 2403.04943, p. 8, described on p. 3. It uses diffusion models to produce supervised counting data through unsupervised object sorting and density-based labels.
