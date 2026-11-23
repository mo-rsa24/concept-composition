# Correcting Diffusion Generation through Resampling

arXiv 2312.06038, read from the full PDF (24 pages, appendix included).

## What it does

The method denoises $K$ images side by side and, at a few noise levels, resamples them with a weight that estimates how far each one is from the real-image distribution (p. 3 to 4). The weight combines a trained real-versus-generated discriminator with a pretrained object detector, and the best final image is kept (p. 5 to 6). The aim is to reduce missing objects and improve image quality in text-to-image generation (p. 1).

## What the classifier or loss reads

The object detector is DETR with a ResNet-50 backbone (p. 6). It reads $f(X_t)$, the diffusion model's one-step prediction of the clean image, so it never sees the noisy image directly (p. 5). For each object it takes the maximum probability over its proposal regions (p. 14).

The unconditional discriminator reads the noisy image $X_t$ and time $t$ through the down and middle blocks of the SD v2.1-base U-Net, with a linear layer on the middle block's features (p. 5, p. 15). A conditional version also receives the caption (p. 4). Neither model reads cross-attention maps.

## What it trained on

The detector is pretrained and is not trained here (p. 1, p. 6).

Each discriminator is trained on 1M LAION Aesthetics images with aesthetic score above 6 as real and 1M images generated with 49 function evaluations (NFE) of the EDM sampler as fake (p. 14). Training noises each image at a time drawn uniformly from $[10^{-5}, 1]$ and minimises the standard discriminator loss of Eq. (6) (p. 4, p. 15). It runs 2 epochs at batch size 128 and learning rate $10^{-4}$, taking 22 GPU hours on an A6000, with the down block frozen (p. 15).

The correction also needs $\kappa_{it}$, the rate at which the detector finds object $i$ when the caption names it, counted from one initial generation round (p. 5). $\pi_{it}$ is a hyperparameter between 0 and 1, set equal for all objects and times (p. 5, p. 13).

## How its gradient enters

The proposed method puts no gradient into the sample (Table 5, p. 16). It resamples with weight (p. 4):

$$
w(X_{t+1}, X_t \mid C) = \frac{\phi_t(X_t \mid C)}{\phi_{t+1}(X_{t+1} \mid C)}
$$

$$
\phi_t(X_t \mid C) = \frac{p(X_t)}{q(X_t)} \cdot \frac{\prod_{i: O_{Ci}=1} p(O_{Ci}=1 \mid X_t)}{\prod_{i: O_{Ci}=1} q(O_{Ci}=1 \mid X_t)}
$$

$p$ is the real-image distribution and $q$ is the model's distribution (p. 3). The first ratio comes from the discriminator and the second from the detector (p. 5). $O_{Ci} = 1$ means the caption mentions object $i$ (p. 3).

With the Restart sampler, resampling happens six times, at noise levels $\sigma = 1.09, 0.59, 0.59, 0.30, 0.30, 0.06$ (p. 15, Table 4, repeat counts from the same table). The main model is SD v2.1-base (p. 6).

The gradient baseline, D-Guidance, adds this term to the score, with $\phi_t$ estimated by the same hybrid approach (p. 6):

$$
\nabla \log \frac{p(X_t \mid C)}{q(X_t \mid C)}
$$

It runs only when $\sigma(t) < 5$, because at larger noise "the object detector barely detects any object, which results in a noisy gradient" (p. 14). The paper prints no scale for this gradient in the text-to-image experiment.

## Unseen concepts

This paper does not test held-out concepts.

The GPT-Synthetic captions use objects from MS-COCO (p. 6), and the MS-COCO captions are filtered to phrases matching a COCO category (p. 14). The detector was built for those categories, so every object scored is one it knows.

## Word leakage and confounds

The detector sees images only. The caption decides which object scores count, through noun phrases matched to COCO categories (p. 5, p. 14). The final score uses a different detector (DETR with ResNet-101) from the guiding one (DETR with ResNet-50), which reduces the risk of optimising one detector's quirks (p. 6). Both are DETR models for COCO objects, so they are not fully independent (inferred). A human evaluation on 100 captions per dataset gives a separate check (p. 6 to 7).

The comparison of gradient and resampling is mixed, because two things differ at once. D-Guidance makes one image, while PF-Hybrid makes $K$ images and keeps the best (p. 6).

## Numbers

Values marked "Figure 4" were measured from a 600 dpi render of the vector figure on p. 6, calibrated against its gridlines. Object occurrence is the share of objects named in a caption that a detector finds, averaged over captions, on a 0 to 1 scale where higher is better (p. 6).

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Object occurrence (0 to 1, higher is better) | PF-Hybrid 0.689 at $K$=5, 0.733 at $K$=10, 0.744 at $K$=15 | Plain SD 0.551, D-Guidance 0.558, Object Select 0.744 at $K$=15 | GPT-Synthetic, 500 captions | SD 2.1-base | p. 6, Figure 4 |
| Object occurrence (0 to 1, higher is better) | PF-Hybrid 0.816, 0.845, 0.849 at $K$=5, 10, 15 | Plain SD 0.706, D-Guidance 0.707, Object Select 0.861 at $K$=15 | MS-COCO, 261 captions | SD 2.1-base | p. 6, Figure 4 |
| FID (lower is better) | PF-Hybrid 23.71 at $K$=15 | Plain SD 24.70, D-Guidance 24.66, Object Select 25.86 at $K$=15 | MS-COCO, 5,000 captions | SD 2.1-base | p. 6, Figure 4 |
| Object occurrence (%, higher is better), ablation | PF-Hybrid 72.96 | best-of-$K$ with no resampling 67.16 | GPT-Synthetic | SD 2.1-base | p. 8, Table 2 |
| Object occurrence (%, higher is better), ablation | PF-Hybrid 83.84 | best-of-$K$ with no resampling 80.49 | MS-COCO | SD 2.1-base | p. 8, Table 2 |
| Human-rated object occurrence (%, higher is better) | PF-Hybrid 75.79 | Reward Select 71.97, plain SD 64.48 | GPT-Synthetic, 100 captions | SD 2.1-base | p. 7, Table 1 |
| Cost with the Restart sampler (NFE and runtime in seconds, as printed) | PF-Hybrid at $K$=5: 360 NFE, 53 s | D-Guidance 106 NFE, 41 s; plain SD 66 NFE, 9 s | text-to-image | SD 2.1-base | p. 16, Table 5 |
| FID, class-conditional (lower is better) | PF 1.02 with 4 images at 165 NFE each | not stated beside this value | ImageNet-64 | EDM model | p. 8, p. 19 |

Table 2 does not print $K$. Its FID of 24.03 matches the $K$=10 point in Figure 4, and its occurrence values sit within 0.007 of that point (p. 6, p. 8, inferred).

## Check of the project claim

The claim's three figures are correct. Two of its readings need changing before the design relies on them.

| Part of the claim | What the paper shows | Result |
|---|---|---|
| Detector and discriminator gradient moves occurrence from about 0.55 to 0.56 on GPT-Synthetic | Plain SD 0.551, D-Guidance 0.558 (Figure 4, p. 6) | Agrees |
| Resampling with the same signal reaches about 0.74 | PF-Hybrid reaches 0.744 only at $K$=15. It is 0.689 at $K$=5 and 0.733 at $K$=10 (Figure 4, p. 6) | Agrees at $K$=15, and $K$ must be stated |
| The gain belongs to resampling | Keeping the best of 15 finished images by detector score alone (Object Select) also reaches 0.744 (Figure 4, p. 6). In Table 2, best-of-$K$ with no resampling reaches 67.16%, so it supplies about 12 of the roughly 18 points gained over plain SD, and resampling adds 5.8 (p. 8, inferred arithmetic) | Differs |
| The test used a clean-image detector at low noise | The detector is a clean-image model reading the one-step estimate $f(X_t)$ (p. 5). The gradient runs for $\sigma < 5$ (p. 14). On the paper's Eq. (24) schedule, $\sigma = 5$ falls at $t \approx 0.80$ of a 0-to-1 scale, where $x_t \approx 0.196\,x_0 + 0.981\,\epsilon$ (p. 14, inferred arithmetic). That is high noise. Low noise describes the resampling steps, at $\sigma \le 1.09$ (p. 15) | Differs |

The gradient's signal also includes a discriminator trained on noisy images across all times, so "clean-image" describes only the detector half (p. 15).

The test says little about gradients at the earliest steps, because the gradient was switched off for $\sigma \ge 5$. Under even spacing, that excludes 10 of our 16 push steps (steps 0 to 9 of 50, $t$ above about 800) (inferred, and this assumes SDXL shares the SD 2.1-base noise schedule).

## What to copy, what to avoid

- Copy a best-of-$K$ selection baseline into every comparison. Choosing among 15 finished images with a detector matches the resampling result at 0.744, so a checker gradient must beat seed selection at equal compute to count as steering (inferred).
- Avoid citing this paper as proof that classifier gradients fail. It tests one configuration with an unprinted scale, one mixed comparison, and no gradient at $\sigma \ge 5$ (p. 6, p. 14, inferred).
- Copy the separate scoring detector, with a different backbone from any detector used for guidance (p. 6, inferred).
- Keep the checkers aimed at what an object detector misses. The paper lists colour, location, count and size as its failure types, because its weight drops the term for object characteristics (p. 20, inferred link to our binding, counting and spatial checkers).
- Avoid a detector tied to named classes for the concept-general claim, because its reach stops at the COCO categories (p. 14, inferred).
- Consider selection or resampling as a fallback if our gradient cannot move layout. Our checkers could score candidates at a low-noise step, where this paper's signal is reliable (inferred).

## Closer work it cites

- Refining Generative Process with Discriminator Guidance in Score-Based Diffusion Models (Kim et al., 2023, reference [31], p. 9, arXiv id not printed). It trains a real-versus-generated classifier across all concepts and adds its gradient to the score. This is the D-Guidance baseline here.
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation (Xu et al., 2023, reference [63], p. 11, arXiv id not printed). It trains a scorer for human preference on generated images, and this paper uses it as a selection criterion (p. 6), so it is a learned overall-match property.
