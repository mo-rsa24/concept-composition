# Are Diffusion Models Vision-And-Language Reasoners?

arXiv 2305.16397 (v3), read from the full PDF (21 pages, appendix included). NeurIPS 2023, as printed on p. 1.

## What it does
The paper scores an image-text pair with Stable Diffusion by how much the text lowers the noise-prediction error compared with no text (p. 5, Eq. 5). This subtraction fixes image retrieval, where the plain error picks whichever image is easiest to denoise regardless of the caption (p. 2, p. 5). It also fine-tunes Stable Diffusion with LoRA on MS-COCO hard negatives, and bundles 7 matching tasks plus a bias test as GDBench (p. 5 to 7).

## What it reads
The input is a clean image $x$ (encoded by SD), noised at timestep $t$ with noise $\epsilon$, and a caption $w$ (p. 4, Eq. 1). The score is the conditional squared error minus the unconditional squared error at the same noisy image (p. 5, Eq. 5). Timesteps are drawn uniformly from 0 to 1000 and guidance is set to 0 (p. 8). The main zero-shot results use 250 noise-timestep samples per pair, and other experiments use 10 (p. 8). The main model is SD 2.1, with SD 1.5 and SDXL compared (p. 8; p. 10). No internal features are read.

## What it trains
Zero-shot DiffusionITM trains nothing. HardNeg-DiffusionITM trains LoRA layers added to the U-Net's cross-attention to the text, with Stable Diffusion otherwise frozen (p. 5). The data is MS-COCO, 109K examples, with the hard negatives of Yuksekgonul et al.: text negatives made by swapping words of the same part of speech ("Men keep watch on a herd of goats" becomes "Goats keep watch on a herd of men") and image negatives taken as nearest neighbours in CLIP image space (p. 5; p. 21). The loss is (p. 5, Eq. 6 and 7):

$$
L = L_{\text{pos}} + \operatorname{clip}\!\left(L_{\text{neg}},\, |\lambda L_{\text{pos}}|\right), \qquad L_{\text{pos}} = \mathbb{E}_{x,\epsilon,t}\|\epsilon - \epsilon_\theta(x, t, w_{\text{pos}})\|^2, \qquad L_{\text{neg}} = -\mathbb{E}_{x,\epsilon,t}\|\epsilon - \epsilon_\theta(x, t, w_{\text{neg}})\|^2
$$

$L_{\text{neg}}$ is clipped to $[-L_{\text{pos}}, L_{\text{pos}}]$ so the model cannot gain without bound by predicting nonsense for negatives (p. 5 to 6). Settings: learning rate $10^{-4}$, $\lambda = 1.0$, batch size 112, checkpoint chosen after 8 epochs on hard-negative validation (p. 8). Without the clip, ARO COCO Order jumps to 63.1% (against 34.9% for the best clipped run) while Pets falls from 60.9% to 53.2%, and the model collapses to chance on all tasks within less than one epoch (p. 18).

## How noise predictions are combined or compared
Text retrieval keeps the plain error and picks the caption with the lowest expected error (p. 4, Eq. 2):

$$
\arg\min_{w}\; \mathbb{E}_{\epsilon,t}\,\|\epsilon - \epsilon_\theta(x, t, w)\|_2^2
$$

Image retrieval subtracts the unconditional error (p. 5, Eq. 5):

$$
\arg\min_{x}\; \mathbb{E}_{\epsilon,t}\left(\|\epsilon - \epsilon_\theta(x, t, w)\|_2^2 - \|\epsilon - \epsilon_\theta(x, t)\|_2^2\right)
$$

Here $\epsilon_\theta(x, t)$ is the prediction with no text. The paper says it adopts most of the Diffusion Classifier setup, which keeps the sampled $(\epsilon, t)$ constant across captions (p. 4; p. 8). It does not state outright that the conditional and unconditional terms share the same $(\epsilon, t)$, though Eq. 5 is written with one expectation over both (inferred). There is no weighting over timesteps. The bias test uses 20 samples (p. 8, footnote 6).

## When adding scores or embeddings is valid
The paper does not add the scores of two conditions. It does justify one subtraction. The Bayes denominator for $p(w \mid x)$ is a marginal over all captions, and the authors replace it with the unconditional model, arguing that "taking an integral over all possible caption strings (w) is equivalent to simply the unconditional" (p. 5). The subtraction is offered as intuition and is not proved. The mirror version fails: subtracting an "image-unconditional" error, computed on a gray image, for text retrieval gives a significant drop in performance (p. 18).

## What it says about the noise and the seed
Two points. First, the raw denoising error "depends almost entirely on the image, independent of text conditioning" (p. 5). For two similar Flickr30K images, the wrong image had the lower absolute error for the caption (p. 5, Figure 3). Second, the number of noise samples matters up to a point. On Flickr30K text retrieval, accuracy plateaus around 100 samples (p. 18; p. 20, Figure 9). Dropping from 250 samples to 10 lowers Flickr text retrieval from 69.8% to 55.3% for vanilla SD (p. 7, Table 1b; p. 8, Table 2b). The DrawBench comparison used 2 seeds per prompt (p. 15). Seed-to-seed variance of the score is not reported.

## Which timesteps carry the signal
Not measured. Timesteps are sampled uniformly from 0 to 1000 throughout (p. 8). Appendix C shows edits at strength 0.4 to 0.7 qualitatively, with no numbers (p. 16 to 17).

## Unseen concepts and composition
This paper does not test held-out concepts. HardNeg-DiffusionITM is trained on MS-COCO only and evaluated zero-shot on the other GDBench tasks, which is transfer across datasets (p. 6; p. 9).

Composition is tested on Winoground, ARO (swapped word order and relations), SVO (pairs differing in subject, verb or object) and generated CLEVR scenes (p. 6). On Winoground text retrieval, zero-shot DiffusionITM scores 37.5 against 30.25 for OpenCLIP ViT-L/14 and 26.3 for CLIP RN50x64. On Winoground image retrieval it scores 10.2 against 13.8 for CLIP RN50x64 (p. 7, Table 1). On CLEVR, the zero-shot model scores 84.5% on pair binding by colour ("a gray cylinder and a purple sphere" against the swap), where CLIP RN50x64 scores 53.0%. On spatial relations ("on the right" against "on the left") it scores 43.9%, near the 50% chance level (p. 18, Table 6). On the word-order subtasks of ARO it stays behind CLIP: COCO Order 23.5% against 52.0% for CLIP RN50x64 (p. 7, Table 1b).

SDXL scores below SD 2.1 on every task except three ARO subtasks, and its top predictions tie far more often (p. 10). The authors offer two readings: the matching score may measure editing ability more than generation ability, or the score is exposing a real weakness (p. 10).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Text retrieval accuracy (%, higher better, chance 25) | 37.5 zero-shot; 39.5 HardNeg | 30.25 (OpenCLIP ViT-L/14) | Winoground text | SD 2.1, 250 samples | p. 7, Table 1b |
| Image retrieval accuracy (%, higher better) | 10.2 zero-shot; 12.3 HardNeg | 13.8 (CLIP RN50x64); Diffusion Classifier 0.25 | Winoground image | SD 2.1 | p. 7, Table 1a |
| Image retrieval accuracy (%, higher better, chance 10) | 61.5 zero-shot; 66.1 HardNeg | 79.9 (OpenCLIP ViT-L/14); Diffusion Classifier 11.1 | Flickr30K, top-10 CLIP candidates | SD 2.1 | p. 7, Table 1a |
| CLEVR text retrieval, average over subtasks (%, higher better, chance 50) | 67.9 zero-shot; 73.3 HardNeg | 64.3 (OpenCLIP ViT-L/14) | CLEVR (self-generated) | SD 2.1 | p. 7, Table 1b |
| CLEVR pair binding by colour (%, higher better, chance 50) | 84.5 zero-shot | 53.0 (CLIP RN50x64) | CLEVR | SD 2.1 | p. 18, Table 6 |
| ARO COCO Order (%, higher better, chance 20) | 23.5 zero-shot; 34.4 HardNeg | 52.0 (CLIP RN50x64) | ARO | SD 2.1 | p. 7, Table 1b |
| Average absolute bias effect size (lower is less biased) | 0.65 (SD 2.1) | 0.79 (SD 1.5); 1.09 (CLIP RN50x64) | Janghorbani and De Melo bias set | SD 2.1 | p. 9, Table 3 |
| DrawBench blind pairwise alignment wins (count of 208) | 60 HardNeg | 33 vanilla SD; 115 ties | DrawBench, 104 prompts, 2 seeds | SD 2.1 | p. 15 |

## What it suggests for this loop
- For a score-path encoder (inferred): this paper shows that the raw error is dominated by the image and that only the part the text adds is useful for matching (p. 5). The brief's $d_t(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)$ is that part. Their score can be written through $d_t$: with $r = \epsilon - \epsilon_\theta(x_t, t, \varnothing)$, the score is $\|d_t(w)\|^2 - 2\langle r, d_t(w)\rangle$. A score-path encoder that reads $d_t$ starts from the input that worked here. It should still be given $r$ or $x_t$ if it is to match images, since the score depends on both.
- For "sum lands on joint" (inferred): the same score gives an untrained version of the test. For a canvas with digits $a$ and $b$, compare the score of the joint caption with that of $\epsilon_{\text{PoE}}$, using $d_t(a) + d_t(b)$ in place of $d_t(w)$, under shared $(\epsilon, t)$.
- For "gap predicts the term" (inferred): CLEVR results suggest that SD's text conditioning carries attribute binding (84.5% on colour swaps) but not left-right position (43.9%). If the digit model is asked about which digit sits on which side, the joint-minus-product gap may carry the position signal weakly, so the test should be scored separately on position and identity.
- For the separate-experts paper (inferred): the clipped hard-negative loss is a stable recipe for teaching a diffusion model to raise its error on a wrong caption without collapse. The unclipped run collapsed within one epoch (p. 18). The same clip is a guard for any loss that pushes a product away from a hybrid.
- For the CO3 paper (inferred): the unconditional-normalised score is a cheap judge of whether a corrected sample matches its joint caption better than the uncorrected one. It needs about 100 noise samples per pair to plateau (p. 18).

## Closer work it cites
None.
