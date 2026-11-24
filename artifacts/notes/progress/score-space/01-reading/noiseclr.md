# NoiseCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions in Diffusion Models

arXiv 2312.05390 (v1), read from the full PDF (17 pages, supplementary included). No venue is printed on the PDF.

## What it does
NoiseCLR learns a set of directions that are fed to frozen Stable Diffusion 1.5 in place of a text condition. It trains them with a contrastive loss on the change each direction makes to the noise prediction (p. 4 to 5, Eq. 4 and 5). The pull is toward the same direction applied to different images, and the push is away from different directions applied to the same image. The directions are then added to the noise prediction to edit images, one at a time or several summed together (p. 5, Eq. 7 and 8).

## What it reads
The contrastive loss reads the "feature divergence" of direction $d_k$ on image $x^n$ (p. 5, Eq. 4):

$$
\Delta\epsilon^n_k = \epsilon_\theta(x^n_t, d_k) - \epsilon_\theta(x^n_t, \phi)
$$

Here $x^n_t$ is a real image after the forward diffusion process to timestep $t$, and $\phi$ is the null text (p. 4, Figure 3; p. 4, Eq. 3). This is the brief's $d_t(c)$ with the text condition $c$ replaced by a learned vector $d_k$ (inferred). The paper does not state the size of $d_k$. It is passed where a text condition goes, so it presumably lives in the text-conditioning space (inferred). Training considers all timesteps of the diffusion model (p. 14). The paper does not say how $t$ is drawn per batch, nor whether the images in a batch share a timestep. No internal U-Net features are read.

## What it trains
Only the directions $D = \{d_1, \dots, d_K\}$ are trained; Stable Diffusion 1.5 stays frozen (p. 1, Figure 1 caption; p. 6). The data is about 100 unlabeled real images per domain, drawn from FFHQ faces, AFHQ cats, Stanford Cars and a small set of paintings (p. 6). No pairs or labels are made. Positives are the same direction's divergence on two different images, and negatives are different directions' divergences on the same image (p. 5). The loss is (p. 5, Eq. 5 and 6):

$$
L = -\log \frac{\sum_{a=1}^{|X'|}\sum_{b=1}^{|X'|} \mathbb{1}[a \ne b]\, \exp\!\big(\operatorname{sim}(\Delta\epsilon^a_j, \Delta\epsilon^b_j)/\tau\big)}{\sum_{a=1}^{|X'|}\sum_{i=1}^{|D'|} \mathbb{1}[i \ne j]\, \exp\!\big(\operatorname{sim}(\Delta\epsilon^a_j, \Delta\epsilon^a_i)/\tau\big)}, \qquad \operatorname{sim}(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|}
$$

$X' \subset X$ is a subset of images and $D' \subset D$ a subset of directions. Settings: $N = 100$ images, $K = 100$ directions for faces and art and $K = 50$ for cats and cars, $\tau = 0.5$, $|D'| = 20$ per iteration, AdamW with learning rate $10^{-3}$, batch size 6, random seed 0 (p. 6). Learning 100 face directions takes about 7 hours on one NVIDIA L40, and an edit then takes about 5 seconds (p. 6). The number of training iterations is not given.

## How noise predictions are combined or compared
During training, divergences are compared by cosine similarity inside the contrastive loss above (p. 5). Each negative pair shares the same noisy image $x^a_t$ and differs only in direction. Each positive pair uses two different images, so two different noisy latents.

For editing, a direction is added on top of the classifier-free-guided prediction for the original condition $c$ (p. 4, Eq. 3; p. 5, Eq. 7):

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \phi) + \lambda_g\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \phi)\big), \qquad
\bar\epsilon_\theta(x_t, c, d_e) = \tilde\epsilon_\theta(x_t, c) + \lambda_e\big(\epsilon_\theta(x_t, d_e) - \epsilon_\theta(x_t, \phi)\big)
$$

Several directions are summed with their own scales (p. 5, Eq. 8):

$$
\bar\epsilon_\theta(x_t, L) = \sum_{i=1}^{|L|} \lambda_i\big(\epsilon_\theta(x_t, d_i) - \epsilon_\theta(x_t, \phi)\big), \qquad \bar\epsilon_\theta(x_t, c, L) = \tilde\epsilon_\theta(x_t, c) + \bar\epsilon_\theta(x_t, L)
$$

Real images are first inverted with DDIM, and the base term becomes the null prediction (p. 6, Eq. 9). Edits are applied only over part of the sampling path. The rule of thumb is to start at $t = 0.5T$, and coarse edits such as eyeglasses need the interval $[0.9T, 0.8T]$ (p. 6; p. 14 to 15).

## When adding scores or embeddings is valid
The paper assumes that a difference $\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \phi)$ "encodes semantic information encoded by the condition c" (p. 5). Because an edit is a sum of such differences, it treats several edits as addable (p. 5). Validity rests on the directions being disentangled, which the contrastive push between directions is meant to produce (p. 5; p. 14). Nothing is proved.

The evidence is qualitative (stacked edits in Figure 6, p. 7) plus a single-edit re-scoring with a CLIP classifier (p. 9, Table 1). That table also shows where it fails. The "Child" edit lowers the CLIP probability of "Indian" by 37.7 and of "Asian" by 14.1, which the authors put down to biases inside Stable Diffusion (p. 9). Summed edits are never scored with a number. The paper also notes that learning only $K = 10$ directions gives coarser and slightly more entangled edits than $K = 100$ (p. 14).

## What it says about the noise and the seed
Not addressed. All experiments use one fixed random seed, 0, for reproducibility (p. 6). There is no measurement of how much of $\Delta\epsilon$ is noise or seed variance. The positive pairs ask the same direction's divergence to point the same way across different images, and so across different noisy latents (p. 5). The loss pushes for consistency across noise draws, but the paper reports no achieved cosine values (inferred).

## Which timesteps carry the signal
Measured only qualitatively. Over 50 sampling steps, applying an edit at steps 5 to 10 changes the image a lot, and applying it at steps 35 to 50 keeps the coarse structure (p. 16; p. 17, Figure S.13). Iteration 0 is $t = T$, pure noise (p. 17). No numbers are given.

## Unseen concepts and composition
This paper does not test held-out concepts. Face directions learned on FFHQ photos also apply to artistic paintings (p. 5, Figure 4; p. 7). Some found directions, such as "cartoon", are not present in the training images (p. 7). Directions from 100 generated images instead of real ones cover a narrower range: only age, two race directions, gender, mustache, chin shape, child and cartoon were found (p. 13).

On composition, edits are stacked within one domain (race plus mustache on a face, eye colour plus lion on a cat) and across domains (a car edit plus a face edit in one image), shown qualitatively (p. 7 to 8, Figure 6). No compositional prompts, counting or binding tests are run.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| User study, disentanglement (mean rating on 1 to 5, higher better, 50 MTurk raters) | 3.05 | 2.73 (Cycle-Diffusion) | face edits | SD 1.5 | p. 9, Table 2 |
| User study, edit quality (mean rating on 1 to 5, higher better) | 2.65 | 2.19 (Composable Diffusion) | face edits | SD 1.5 | p. 9, Table 2 |
| LPIPS to the input image, Race edit (perceptual distance, lower better) | 0.13 | 0.26 (Cycle-Diffusion) | face edits | SD 1.5 | p. 9, Table 3 |
| LPIPS to the input image, Age edit (perceptual distance, lower better) | 0.17 | 0.10 (Cycle-Diffusion) | face edits | SD 1.5 | p. 9, Table 3 |
| LPIPS, Race edit (perceptual distance, lower better) | 0.15 | 0.18 (Concept Sliders) | face edits | SD 1.5 | p. 13, Table S.4 |
| Change in CLIP classifier probability for the target attribute (points, higher better) | 29.8 Indian, 27.5 Asian, 48.9 Mustache, 32.8 Child, 11.0 Lipstick | no baseline | face edits | SD 1.5 | p. 9, Table 1 |
| Change in CLIP probability of an unedited attribute (points, near 0 better) | Child edit: −37.7 Indian, +11.7 Lipstick | no baseline | face edits | SD 1.5 | p. 9, Table 1 |

## What it suggests for this loop
- For a score-path encoder (inferred): NoiseCLR is the closest working case of contrastive training on exactly the brief's quantity, the conditional-minus-null noise prediction. Its positives are the same condition on different images, so its loss is built to ignore the image and the noise draw. A score-path encoder that uses the same (canvas, seed) pairing as positives would learn the seed. Positives should be the same caption across different canvases and seeds, the way NoiseCLR pairs them.
- On the seed problem (inferred): the brief measured a cosine of about 0.002 between $d_t$ vectors for one pair at two seeds on SDXL. NoiseCLR trains vectors whose $\Delta\epsilon$ is meant to agree across images, and reports that edits work, but it never reports the cosine it reached. Agreement across seeds may exist for some directions and not for raw prompts. Measuring the cross-seed cosine of $d_t$ on the digit canvases before training would show which case holds here.
- For "sum lands on joint" (inferred): NoiseCLR composes by summing scaled $\Delta\epsilon$ terms, the same form as $\epsilon_{\text{PoE}}$. Its own table shows a summed attribute leaking into unrelated ones (Child moves Indian by −37.7 points). Leakage of this kind is what an interaction term would look like, and the paper does not measure it for stacked edits.
- For "gap predicts the term" (inferred): NoiseCLR's directions are learned so that different directions on the same image point apart. A learned direction set in $d_t$ space could be fitted to the joint-minus-product residual to see whether the residual is spanned by a few shared directions.
- For the separate-experts paper (inferred): the contrastive objective on $\Delta\epsilon$ is a way to train an expert direction without labels or text. Two such directions summed is a PoE in which each expert was shaped to stay apart from the others. That is a test of whether disentangled experts compose with less interaction.
- For the CO3 paper (inferred): edits are applied only in a window of steps (from $0.5T$, or $[0.9T, 0.8T]$ for coarse edits). This matches the idea that a corrector should act only where its term matters, and gives a step window to try.

## Closer work it cites
- Concept Sliders: LoRA adaptors for precise control in diffusion models, Gandikota et al., arXiv 2311.12092, cited p. 12 (reference [7], p. 10). Trains LoRA edits from text prompts or paired images, which makes it a supervised counterpart of learned noise-prediction directions.
- Unsupervised compositional concepts discovery with text-to-image generative models, Liu, Du, Li, Tenenbaum and Torralba, ICCV 2023, no arXiv id printed, cited p. 3 (reference [22], p. 10). Decomposes images into a set of composable concepts in a text-to-image diffusion model, with a memory-bound limit on how many concepts can be learned (p. 3).
- Diffusion models already have a semantic latent space, Kwon, Jeong and Uh, arXiv 2210.10960, cited p. 3 (reference [20], p. 10). Edits by learning transformations on the U-Net bottleneck features, an internal space of the diffusion model (p. 3).
