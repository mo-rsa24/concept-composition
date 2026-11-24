# Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think

arXiv 2410.06940 (v4), read from the full PDF (43 pages, appendix included). Published as a conference paper at ICLR 2025, as printed on every page.

## What it does

REPA adds one loss term to diffusion transformer training. A small MLP projects the hidden states of an early transformer block, computed from the noisy latent, and the loss pulls each patch of that projection toward the patch features a frozen pretrained image encoder (DINOv2 by default) gives on the clean image (p. 6). The denoising loss stays, the new term is weighted by λ = 0.5, and SiT-XL/2 reaches the FID of a 7M-iteration vanilla model in 400K iterations, a speed-up the paper states as over 17.5 times (p. 1, p. 9).

## What it reads

The diffusion transformer reads the noisy latent $z_t = \alpha_t z_* + \sigma_t \epsilon$, a Stable Diffusion VAE latent of shape 32×32×4 for 256×256 ImageNet images, with the timestep and class label (p. 4, p. 6). The aligned quantity is the hidden state $h_t = f_\theta(z_t)$ after the first 8 transformer blocks (the first 4 for SiT-B), one token per patch, passed through a trainable three-layer MLP $h_\phi$ with SiLU activations (p. 6, p. 8, p. 23, p. 25). The target is $y_* = f(x_*) \in \mathbb{R}^{N \times D}$, the frozen encoder's patch features on the clean image $x_*$, with the encoder's positional embeddings interpolated to match the patch count (p. 6, p. 23). The alignment term never reads the network's output (velocity or noise prediction). The timesteps are every $t$ the denoising loss samples; the expectation over $t$ carries no weighting, and a weight tied to the noise schedule is listed as future work (p. 6, p. 43).

## What it trains

The whole diffusion transformer is trained from scratch together with the projection head (p. 6). Data is class-conditional ImageNet at 256×256 (and 512×512 in Appendix J), with latents pre-computed and no data augmentation (p. 6, p. 23, p. 40). There are no constructed pairs: each training image's own clean encoder feature is the target for its own noisy latent (p. 6). The loss, as printed (p. 4, p. 6):

$$
\mathcal{L}_{\text{REPA}}(\theta,\phi) := -\mathbb{E}_{x_*,\epsilon,t}\left[\frac{1}{N}\sum_{n=1}^{N} \mathrm{sim}\big(y_*^{[n]},\, h_\phi(h_t^{[n]})\big)\right],
\qquad
\mathcal{L} := \mathcal{L}_{\text{velocity}} + \lambda\,\mathcal{L}_{\text{REPA}}
$$

$$
\mathcal{L}_{\text{velocity}}(\theta) := \mathbb{E}_{x_*,\epsilon,t}\,\big\lVert v_\theta(x_t,t) - \dot\alpha_t x_* - \dot\sigma_t \epsilon \big\rVert^2
$$

- $n$ indexes the $N$ patches, $\mathrm{sim}$ is NT-Xent or negative cosine similarity (cosine was chosen after Table 2), $\lambda$ is the weight on alignment (p. 6, p. 8).
- The interpolant is linear, $\alpha_t = 1 - t$ and $\sigma_t = t$, so $t = 0$ is the clean latent and $t = 1$ is pure noise (p. 4, p. 23).
- Optimisation: AdamW, constant learning rate $10^{-4}$, no weight decay, batch size 256, fp16 with gradient clipping, 400K to 4M iterations, on 8 H100 GPUs at about 5.4 steps per second (p. 23).
- λ from 0.25 to 1.0 gives FID 8.6, 7.9, 7.8 and 7.8 for SiT-XL/2 at 400K iterations, so the result saturates after λ = 0.5 (p. 10, Table 5).
- A text-to-image version trains MMDiT (depth 24, width 768) for 150K iterations on the MS-COCO train split with CLIP text embeddings (p. 41).

## How noise predictions are combined or compared

Nothing in REPA combines the predictions of two conditions. The only comparison is between the projected hidden state and the clean encoder feature, patch by patch, in the loss above (p. 6). The paper writes the velocity as a sum of two conditional expectations and converts it to a score (p. 4):

$$
v(x,t) = \mathbb{E}[\dot x_t \mid x_t = x] = \dot\alpha_t\,\mathbb{E}[x_* \mid x_t = x] + \dot\sigma_t\,\mathbb{E}[\epsilon \mid x_t = x],
\qquad
s(x,t) = \sigma_t^{-1}\cdot\frac{\alpha_t v(x,t) - \dot\alpha_t x}{\dot\alpha_t \sigma_t - \alpha_t \dot\sigma_t}
$$

Sampling uses the SDE Euler-Maruyama sampler with diffusion coefficient $w_t = \sigma_t$, 250 function evaluations and a last step of 0.04 (p. 7, p. 23). Classifier-free guidance is used for the system results (scale 1.35, and 1.8 inside the guidance interval [0, 0.7] for the best number), but its equation is not printed (p. 9, p. 25). The qualitative comparison in Figure 4 reuses the same noise, sampler and step count across the two models (p. 5, p. 9).

## When adding scores or embeddings is valid

This paper does not address it. The only sum it writes is the velocity as a weighted sum of the expected clean data and expected noise (p. 4).

## What it says about the noise and the seed

The paper reports no seed repeats and no error bars; every FID is one number from 50,000 samples (p. 7). Its stated aim is that the early hidden states "predict noise-invariant, clean visual representations from noisy inputs", so the alignment target is the same for every noise draw of a given image (p. 6). How many noise draws each image gets during training is not stated beyond the expectation over $\epsilon$. In the PCA pictures of layer features, the vanilla model's feature maps look noisy at large $t$ and the REPA model's look coarse to fine (p. 42, Figure 38; qualitative only). Evaluating in tf32 instead of fp32 made a negligible difference (p. 24).

## Which timesteps carry the signal

Measured only for $t$ = 0, 0.25 and 0.5, which is the less noisy half of the path in this paper's convention (p. 2, p. 10).

- Figure 7 (p. 10), values read from the page image so approximate: ImageNet linear-probe top-1 accuracy (%) on the hidden states is about 53, 59 and 43 for vanilla SiT-XL/2 at $t$ = 0, 0.25, 0.5, and about 72, 66 and 50 with REPA. Maximum CKNNA against DINOv2-g (a 0 to 1 alignment score, higher is more aligned) is about 0.15 at all three for vanilla and about 0.49, 0.49 and 0.43 with REPA.
- The text says REPA "consistently reduces the representation gap across different noise levels" (p. 9). Both models lose about 16 to 23 points of probe accuracy between their best timestep and $t$ = 0.5 (inferred from the Figure 7 reading).
- Across layers, the vanilla probe peaks at layer 20 of 28 and then falls (p. 5). Aligning a deeper layer raises probe accuracy from 66.2% (layer 6) to 71.1% (layer 16) while FID worsens from 10.3 to 12.1 (p. 7, Table 2).
- Nothing is measured for $t > 0.5$, the high-noise half, beyond the qualitative PCA at $t$ = 0.7 and 0.9 (p. 42).

## Unseen concepts and composition

This paper does not test held-out concepts. Generation is class-conditional on the same ImageNet classes it trains on (p. 6). The text-to-image run is evaluated on MS-COCO validation captions after training on the train split, which holds out captions and images but no concepts, and it has no compositional measure (p. 41). Encoders trained without ImageNet (CLIP, SigLIP) still help, which the paper reads as ruling out leakage of ImageNet data through the encoder (p. 22, Table 6).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| FID-50K, lower is better, no guidance | 7.9 at 400K iterations | 8.3, vanilla at 7M iterations | ImageNet 256×256 | SiT-XL/2 | p. 9 Table 3, p. 25 Table 8 |
| FID-50K, lower is better, no guidance | 5.9 at 4M iterations | 8.3, vanilla at 7M iterations | ImageNet 256×256 | SiT-XL/2 | p. 9 Table 3 |
| FID-50K with guidance (scale 1.35), 800 epochs | 1.80 | 2.06, vanilla SiT-XL/2 at 1400 epochs | ImageNet 256×256 | SiT-XL/2 | p. 9 Table 4 |
| FID-50K with guidance interval, 800 epochs | 1.42 | 1.58, MDTv2-XL/2 with its own guidance schedule | ImageNet 256×256 | SiT-XL/2 | p. 9 Table 4 |
| FID-50K, lower is better, 400K iterations | 9.9 (DINOv2-L, cosine) | 18.8, vanilla | ImageNet 256×256 | SiT-L/2 | p. 7 Table 2 |
| Linear-probe top-1 accuracy (%) on hidden states, higher is better | 70.3 at 400K iterations | not reported for vanilla in that table | ImageNet val | SiT-XL/2 | p. 25 Table 8 |
| FID with guidance (scale 1.35), lower is better | 2.08 at 200 epochs | 2.62, vanilla SiT-XL/2 at 600 epochs | ImageNet 512×512 | SiT-XL/2 | p. 40 Table 11 |
| FID on MS-COCO, guidance scale 2.0, SDE 250 steps | 4.14 | 5.30, MMDiT without REPA | MS-COCO 256 | MMDiT | p. 42 Table 12 |

## What it suggests for this loop

- For a score-path encoder (inferred): REPA is a working recipe for matching something computed from noisy inputs to a fixed target from clean data: a small MLP head, patch-wise cosine similarity, and a target from a frozen encoder. The loop could align a projection of the $d_t$ stack (or the digit model's own mid-layer features) to a frozen text embedding of the canvas caption in the same way.
- For a score-path encoder (inferred): REPA aligns hidden states, which are richer than the network output. If $d_t$ alone carries too little about the digits, the digit model's internal features at the same steps are the fallback input, and REPA's Table 2 says middle layers trade generation quality for semantic content.
- For a score-path encoder (inferred): REPA measures semantic content only at $t \le 0.5$, and probe accuracy already drops by about 16 points from $t$ = 0.25 to 0.5 with REPA. The loop's steps 0 to 15 sit at the high-noise start of sampling, a range REPA never measured, so per-step semantic content there should be expected to be weaker.
- For the seed problem (inferred): REPA's target is the same for every noise draw of an image. The loop's SDXL interaction direction is seed-specific (cosine about 0.002 across seeds), so a contrastive score-path encoder would need several seeds of one canvas and caption as positives to push the seed out of the embedding.
- For the two arithmetic tests (inferred): REPA offers no evidence for or against "sum lands on joint" or "gap predicts the term", since it never adds representations.
- For the separate-experts paper (inferred): if the separate digit experts are trained from scratch, a REPA term toward a frozen encoder may shorten their training, and λ = 0.5 needed no tuning in REPA's runs.
- For the CO3 paper (inferred): no direct bearing. REPA changes training of class-conditional DiT and SiT models and is never applied to SDXL or to a pretrained text-to-image model.

## Closer work it cites

- Denoising Diffusion Autoencoders are Unified Self-supervised Learners (Xiang et al., 2023), reference on p. 16, cited p. 2 and p. 5: linear probes trained on diffusion hidden states.
- Diffusion Models Beat GANs on Image Classification (Mukhopadhyay et al., 2021), reference on p. 14: classifiers trained on diffusion features.
- Deconstructing Denoising Diffusion Models for Self-Supervised Learning (Chen et al., 2024), arXiv 2401.14404, reference on p. 12: representation learning from a denoiser, used for REPA's probing protocol (p. 21).
- Diffusion-Based Representation Learning (Abstreiter et al., 2021), arXiv 2105.14257, reference on p. 11: trains an encoder whose code conditions the denoiser.
- SODA: Bottleneck Diffusion Models for Representation Learning (Hudson et al., 2024), reference on p. 13: an encoder that learns a compact latent by guiding a diffusion model.
- SD-DiT: Unleashing the Power of Self-supervised Discrimination in Diffusion Transformer (Zhu et al., 2024), reference on p. 17: adds a self-supervised discrimination loss on diffusion transformer internals.
- DreamTeacher: Pretraining Image Backbones with Deep Generative Models (Li et al., 2023b), reference on p. 14: distils diffusion features into an image backbone.

## Where the paper disagrees with itself

- Table 3 gives SiT-L/2 + REPA at 400K iterations FID 9.7 (p. 9), while Table 8 gives the same model and iteration count FID 10.0 (p. 25). Table 7 says the Table 3 SiT-L run used DINOv2-L with NT-Xent (p. 23), Table 8's caption says all its runs used DINOv2-B with cosine similarity (p. 25), and Table 2 gives 9.7 for DINOv2-B with NT-Xent and 10.0 for DINOv2-L with NT-Xent (p. 7). The configurations cannot all be right.
- The text says cosine similarity is used "in future experiments" (p. 8), yet Table 7 lists NT-Xent for the Table 3 SiT-L run (p. 23).
- The variance-preserving interpolant is printed with $\sigma_t = \cos(\tfrac{\pi}{2}t)$, the same as $\alpha_t$, which would break $\sigma_0 = 0$ (p. 19); this reads as a typo for $\sin$ (inferred).

No text in the paper was addressed to the reader of this note.
