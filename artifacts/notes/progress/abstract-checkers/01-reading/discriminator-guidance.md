# Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models

arXiv 2211.17091v4, read from the full PDF (49 pages, appendix included). Proceedings of the 40th International Conference on Machine Learning, PMLR 202, 2023 (p. 1). Pages 30 to 49 are uncurated sample grids, so their captions were read and the grids themselves were not inspected.

## What it does

After a diffusion model has finished training, the method trains a time-dependent binary classifier to tell real images from the model's own samples at every noise level (p. 1, p. 2, p. 4). During sampling, the gradient of the classifier's log-odds is added to the frozen model's score (p. 4). The tests use class-conditional and unconditional image models on CIFAR-10, CelebA, FFHQ and ImageNet 256x256, and no text-to-image model is used (p. 6).

## What the classifier or loss reads

**Input.** The discriminator $d_\phi(x_t, t)$ reads the noisy sample $x_t$ and the time $t$ (p. 4). It does not read a decoded image estimate, word attention maps or text.

**Architecture on pixel-space models.** $(x_t, t)$ enters a frozen pretrained ADM classifier, and the latent $z_t$ is taken from that classifier's last pooling layer (p. 7). $(z_t, t)$ then enters a small trainable U-Net encoder that outputs one real/fake value (p. 7). The latent is 8x8 with 512 channels, and the output has shape $(B, 1)$ (p. 22, Table 8).

**Architecture on latent models.** DiT-G++ trains a latent classifier with the ADM architecture on 32x32x4 inputs, freezes it, and feeds its 8x8x384 output to the small encoder (p. 7, p. 22). LSGM-G++ uses no pretrained classifier and trains a U-Net encoder from scratch on 16x16x180 inputs (p. 7, p. 21, p. 22).

**Class conditioning.** The discriminator is class-conditional on conditional CIFAR-10, ADM ImageNet and DiT ImageNet, and unconditional on the other four runs (p. 7, p. 22, Table 8). The paper does not print how the label enters the small encoder.

**Choices tested.** An MLP, a ResNet18 and a transformer were tried and the U-Net encoder performed best (p. 7, footnote 1). Fine-tuning the small encoder on frozen features performed strictly better than training everything from scratch on CIFAR-10 (p. 27, Figure 28b on p. 28).

## What it trained on

**Labels.** Real training images are labelled real and samples drawn from the frozen model are labelled fake (p. 3, Algorithm 1; p. 4).

**Loss.** Time-weighted binary cross-entropy, Eq. (4) (p. 4):

$$
\mathcal{L}_\phi = \int \lambda(t)\Big(\mathbb{E}_{p_r^t(x_t)}\big[-\log d_\phi(x_t,t)\big] + \mathbb{E}_{p_{\theta_\infty}^t(x_t)}\big[-\log\big(1-d_\phi(x_t,t)\big)\big]\Big)\,dt
$$

**How the noisy fakes are made.** Only each sample's final image is stored. It is then noised again by the forward process at a random time (p. 3, p. 15, p. 16). The authors chose this to save memory and because fresh noise augments the data and reduces overfitting at large $t$ (p. 15, p. 16). The discriminator therefore never trains on the intermediate states of a real sampling trajectory.

| Run | Real images | Generated images | Epochs | Batch | Page |
|---|---|---|---|---|---|
| LSGM-G++, CIFAR-10 | 50,000 | 50,000 | 280 | 128 | p. 22 |
| EDM-G++, CIFAR-10 unconditional | 50,000 | 25,000 | 60 | 128 | p. 22 |
| EDM-G++, CIFAR-10 conditional | 50,000 | 50,000 | 250 | 128 | p. 22 |
| Soft Truncation-G++, CelebA | 10,000 | 10,000 | 150 | 128 | p. 22 |
| EDM-G++, FFHQ | 60,000 | 60,000 | 250 | 128 | p. 22 |
| ADM-G++, ImageNet 256 | 1,281,167 | 400,000 | 10 | 512 | p. 22, p. 26 |
| DiT-G++, ImageNet 256 | 1,281,167 | 1,281,167 | 7 | 512 | p. 22, p. 26 |

**Cost.** On CIFAR-10 with EDM, discriminator training is 1M network evaluations and 10 minutes, against 480 hours of score training and 1 hour of sample generation (p. 7, Table 6).

**Data size.** On CIFAR-10 the authors reduced generated images to 25k, 10k and 5k with 50k real images, and also reduced both sets together to 25k, 10k and 5k (p. 26, p. 27, Figure 26). The discriminator overfits as the generated count falls, and reducing the real images as well makes overfitting arrive sooner (p. 26).

## How its gradient enters

The correction is the gradient of the discriminator's log-odds (p. 4):

$$
c_\phi(x_t,t) := \nabla \log \frac{d_\phi(x_t,t)}{1-d_\phi(x_t,t)}
$$

Discriminator Guidance adds it to the score, Eq. (5) (p. 4):

$$
dx_t = \big[f(x_t,t) - g^2(t)\,(s_{\theta_\infty} + c_\phi)(x_t,t)\big]\,d\bar t + g(t)\,d\bar w_t
$$

With a time-dependent weight and optional classifier guidance, Eq. (6) (p. 5):

$$
dx_t = \Big[f(x_t,t) - g^2(t)\Big(\big(s_{\theta_\infty} + w_t^{DG} c_{\phi_\infty}\big)(x_t,t) + w_t^{CG}\,\nabla \log p_{\psi_\infty}(c \mid x_t,t)\Big)\Big]\,d\bar t + g(t)\,d\bar w_t
$$

$s_{\theta_\infty}$ is the frozen score, $f$ and $g$ are the SDE's drift and volatility, $w_t^{DG}$ is the discriminator weight and $w_t^{CG}$ is the classifier-guidance weight (p. 3, p. 5).

**When it is on.** The adjusted equation runs on $t \in [t_{mid}, T]$ and the unadjusted equation on $t \in [0, t_{mid}]$ (p. 21). The guidance therefore runs from the start of sampling and switches off for the lowest-noise stretch. The paper's probability-flow form carries a factor of $\tfrac12$ on $g^2$ (p. 21).

| Run | $w_t^{DG}$ | $t_{mid}$ | Sampler and evaluations |
|---|---|---|---|
| LSGM-G++, CIFAR-10 | 2 | 0.1 | RK45, 138 |
| EDM-G++, CIFAR-10 unconditional | 2 | 0.01 | Heun, 35 |
| EDM-G++, CIFAR-10 conditional | Adaptive | 0.01 | Heun, 35 |
| Soft Truncation-G++, CelebA | Adaptive | 0.01 | RK45, 131 |
| EDM-G++, FFHQ | Adaptive | 0.01 | Heun, 71 |
| ADM-G++, ImageNet 256 | Adaptive | 0.1 | DDPM, 250 |
| DiT-G++, ImageNet 256 | 1 | 0.1 | DDPM, 250 |

All rows are from Table 8 (p. 22).

**Adaptive weight.** On conditional CIFAR-10 the weight is 2 for samples whose density ratio is below 0 on odd denoising steps, and 1 otherwise (p. 24). A negative value only makes sense for the log ratio, so "density ratio" here probably means its log (inferred).

**Scale limits.** No norm cap is printed. The discriminator's output is clipped, usually to $[10^{-5}, 1-10^{-5}]$ (p. 22, Table 8). Table 8 prints $10^{-5}$ as both bounds for DiT-G++, which looks like a printing error (inferred). One ADM setting multiplies the discriminator weight by the ratio of the classifier-guidance norm to the discriminator-guidance norm (p. 26). With the Heun solver the discriminator is applied only on the first-order stage by default (p. 24).

**What the weight does.** At the optimal discriminator the adjusted score is the score of $(p_r^t)^w (p_{\theta_\infty}^t)^{1-w}$ (p. 5). Figure 5 on DiT-XL-G++ shows recall rising from about 0.64 at $w=0$ to about 0.72 at $w=3$, with FID-10k lowest near $w=1.3$ and worse past $w=2$ (p. 5, values read from the plot).

**Printed conditions on when the correction is accurate.**

1. Theorem 1 says the adjusted process equals the true data process under two assumptions: the score at time $T$ equals the prior's score, and the model's log-likelihood equals its evidence lower bound (p. 3). The authors check the second only approximately, on an EDM CIFAR-10 checkpoint (p. 14, Figure 14).
2. The correction equals the true one only at the optimal discriminator (p. 4, p. 5). For a trained discriminator, Theorem 2 bounds the KL divergence by $D_{KL}(p_r^T\|\pi) + E_{\theta_\infty,\phi}$, where $E_{\theta_\infty,\phi} = \tfrac12\int_0^T g^2(t)\,\mathbb{E}_{p_r^t}\|c_{\theta_\infty} - c_\phi\|_2^2\,dt$ (p. 4). The error that counts is the error in the gradient.
3. Theorem 2 "does not guarantee that the gain is strictly positive" (p. 4). A completely blind discriminator, $d \equiv 0.5$, gives no gradient signal and zero gain (p. 4, p. 5, Table 1).
4. A discriminator can separate real from fake perfectly and still be wrong between the two supports, which the paper calls the density-chasm problem (p. 29). The authors give this as one reason the best lower stopping scale on CIFAR-10 is strictly positive, near $\sigma = 0.0025$ (p. 29).
5. Gains shrink at low step counts because discretisation error dominates (p. 8, Figure 9).
6. Applying guidance over a larger range of noise scales keeps improving FID on CIFAR-10. The authors conclude its effect lies in "constructing the global shape of the generation" (p. 28, p. 29, Figure 30).

## Unseen concepts

This paper does not test held-out concepts.

**What the data splits are.** Every class-conditional discriminator trains on the same classes it guides at test time (p. 7, p. 22). The ImageNet discriminators use all 1,281,167 training images (p. 22). The ImageNet validation set appears only as a reference row for the metrics (p. 7, Table 5).

**The closest thing to transfer.** The frozen feature extractor for CIFAR-10 is a 32x32 classifier trained on ImageNet (p. 22), and the 64x64 ADM classifier is reused for CelebA and FFHQ (p. 22, p. 25). The trainable head is always trained on the dataset it guides, so this shows reuse of features and no transfer of a trained discriminator.

**Transfer across concepts.** One ImageNet discriminator covers all 1000 classes in training, with the class label as an input. The paper never evaluates it on a class outside that set, so it gives no evidence of transfer to unseen concepts. Correcting Diffusion cites it in a reference entry on its p. 9 and says on its p. 2 that this method "leverages a discriminator to generate samples closer to ground-truth distribution", which is also no claim of transfer.

**Image-to-image test.** For cat to dog translation the fake set contains translated source images plus the source images, and the real set is the target images (p. 8, p. 19). Both categories are in the discriminator's training data, so this is no held-out test.

## Word leakage and confounds

**Word leakage.** No text enters this method. The class-conditional runs give the discriminator the class label directly (p. 7, p. 22). The paper does not ask whether a label-conditioned discriminator learns a general notion of realism or one realism check per class.

**Features built for classes.** The frozen extractor was trained to identify classes (p. 7, p. 22), so the small head reads features that carry category identity (inferred). This is the same risk as this project's current checker on SDXL UNet features, which may learn species (inferred).

**Confounds in the headline numbers.**

- DiT-XL/2-G++ at FID 1.83 also uses its own classifier-guidance schedule, 1.25 for $t < 200$ and 3.0 otherwise (p. 26). The 1.83 against 2.27 comparison in Table 5 therefore does not isolate the discriminator (inferred).
- ADM-G++ at FID 3.18 stacks entropy-driven sampling, adaptive classifier guidance, adaptive discriminator weighting and norm-ratio scaling (p. 25, Table 10). Adding the discriminator alone at weight 1 to ADM-G moves FID from 4.59 to 4.18 (p. 25, Table 10, case a).
- The EDM and LSGM rows change only the discriminator, as the authors state (p. 7).
- EDM's baseline FID is recalculated under a random seed: 2.03 unconditional and 1.82 conditional, against 1.97 and 1.79 reported with a manual seed (p. 6, Table 3).

## Numbers

FID is the Fréchet Inception Distance between generated and real image features, unbounded above, lower is better. Recall is the coverage of the real distribution on a 0 to 1 scale, higher is better (p. 7).

| Metric | This method | Same model without it | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| FID | 1.83 | 2.27 (validation data 1.68) | ImageNet 256 | DiT-XL/2 | p. 7, Table 5 |
| Recall | 0.64 | 0.57 (validation data 0.66) | ImageNet 256 | DiT-XL/2 | p. 7, Table 5 |
| FID, full stack | 3.18 | 4.59 | ImageNet 256 | ADM-G | p. 7, Table 5 |
| FID, discriminator alone at weight 1 | 4.18 | 4.59 | ImageNet 256 | ADM-G | p. 25, Table 10 |
| FID, unconditional | 1.77 | 2.03 | CIFAR-10 | EDM | p. 6, Table 3 |
| FID, conditional | 1.64 | 1.82 | CIFAR-10 | EDM | p. 6, Table 3 |
| FID, unconditional | 1.94 | 2.10 | CIFAR-10 | LSGM | p. 6, Table 3 |
| FID | 1.34 | 1.90 (best listed, Diffusion StyleGAN2, 1.69) | CelebA 64 | Soft Truncation | p. 6, Table 4 |
| FID | 1.98 | 2.39 | FFHQ 64 | EDM | p. 6, Table 4 |
| FID, cat to dog translation at weight 8 | 61.92 | 74.02 | Cat to dog | SDEdit | p. 49, Figure 51 |

## What to copy, what to avoid

- Do not cite this paper as a checker that transfers across concepts. It has no held-out class test, and its ImageNet discriminator is given the class label (p. 7, p. 22) (inferred).
- Do not rely on Theorem 1 for PoE. The theorem needs the score to be the true score of a diffused model distribution (p. 3, p. 13), and a sum of two noise predictions has no such guarantee (inferred). Any correction on PoE has to be judged from the pictures it produces (inferred).
- Consider the log-odds form $\nabla\log\frac{C}{1-C}$ in place of our $\nabla\log C$. Since $\nabla\log C = (1-C)\,\nabla\log\frac{C}{1-C}$, our form fades as the checker grows confident and the paper's form does not (p. 4) (inferred). A stronger late push would need our cap of 0.4 of PoE's norm to stay in place (inferred).
- Do not read a checker's accuracy as proof its gradient is useful. The bound depends on gradient error (p. 4), and a discriminator can classify perfectly while being wrong between the two supports (p. 29) (inferred).
- A chance-level checker is different from this paper's blind case. The paper's blind discriminator is the constant 0.5 with zero gradient (p. 4), while a network at our measured balanced accuracy of 0.50 on early decoded images can still return a nonzero, misleading gradient (inferred).
- Copy the cheap data recipe if a real-against-generated checker is tried: save only final samples and noise them again at random times (p. 15, p. 16). This assumes the forward-noised marginal stands in for PoE's reverse-time states, which this paper does not test for a mismatched score such as PoE (inferred).
- Copy the frozen-features-plus-small-head shape, which beat training from scratch here (p. 27). Our UNet-feature checker already follows it and moved layout, while our decoded-image checker changed only texture. This paper's discriminator also reads noisy inputs directly (p. 7) (inferred).
- Keep the push in the high-noise steps. The paper finds the discriminator mainly shapes global structure (p. 29), which is consistent with our window of steps 0 to 15 of 50, though its evidence is CIFAR-10 only (inferred).
- Budget for data. Overfitting grows as generated samples fall below tens of thousands on CIFAR-10 (p. 26, p. 27), and a checker trained on fewer than 5,000 generated images would sit below the smallest size tested here (inferred).

## Closer work it cites

None. The cited methods that reweight samples with a real-versus-generated discriminator (Azadi et al. 2019, Che et al. 2020, Turner et al. 2019; p. 6) are GAN sampling methods, and this paper reports no test of them on unseen concepts.
