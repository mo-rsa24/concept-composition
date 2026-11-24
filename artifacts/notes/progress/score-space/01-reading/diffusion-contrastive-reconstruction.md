# Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation

arXiv 2603.04803 (v2, 20 Mar 2026), read from the full PDF (23 pages, appendix included). No venue is printed on the PDF.

## What it does

The paper fine-tunes a CLIP image encoder so that a frozen Stable Diffusion 2.1 model, conditioned on the encoder's projected [CLS] token, produces noise predictions that are contrastively separable: an image's own prediction should sit close to the prediction from its augmented view and to the true noise, and far from the predictions conditioned on the other images in the batch (p. 2, p. 5). The method is called Diffusion Contrastive Reconstruction (DCR). It is evaluated as an image-representation improvement: fine-detail questions (MMVP-VLM), zero-shot clustering, and LLaVA-1.5 benchmarks (p. 7, p. 8).

## What it reads

The compared quantity is the full noise prediction of the frozen SD2.1 UNet, $\hat\epsilon = \epsilon_\theta(x_t, h_\omega(f_\phi(x)), t)$, at one noisy VAE latent $x_t$ and one timestep $t$ per image (p. 5; Algorithm 1, p. 13). The condition is the CLIP vision encoder's [CLS] token passed through a two-layer MLP projector; all patch tokens are discarded (p. 7). Nothing subtracts an unconditional prediction, so the compared vector is $\epsilon_\theta(x_t, c, t)$ itself and the paper never forms the quantity this loop calls $d_t(c)$ (p. 5). The timestep is drawn uniformly, $t \sim \mathrm{Unif}(1, T)$ (p. 13). The noise predictions are the space where the contrast is computed. They are never the input to an encoder: the trained encoder reads clean images (p. 4, Fig. 3).

## What it trains

Two stages, with the diffusion model frozen throughout (p. 5). Stage 1 trains only the projector $h_\omega$ with the CLIP encoder frozen; stage 2 freezes the projector and fine-tunes the CLIP vision encoder with LoRA of rank 16 (p. 5, p. 7). Both stages use the DCR loss (Algorithm 1, p. 13).

Data: CC3M images (p. 7). Positives are made by augmenting the same image ("random crop or color jitter", p. 5); negatives are the other images in the mini-batch (p. 5). There are no class labels: the "classes" in the theory are the batch's images (p. 15, inferred from the proof's use of other batch images as different classes).

The loss (p. 5, Eq. 12):

$$
\mathcal{L}_{\text{dcr}} = -\frac{1}{2}\sum_{p \in \mathcal{P}} \log \frac{d(\hat\epsilon, p)}{\sum_{c \in \mathcal{C}} d(\hat\epsilon, c)}, \qquad d(u, v) = \exp\big(\mathrm{sim}(u, v)/\tau\big)
$$

with $\mathcal{P} = \{\hat\epsilon^+, \epsilon^{gt}\}$, $\mathcal{N} = \{\hat\epsilon^-_k\}_{k=1}^{N-1}$, $\mathcal{C} = \mathcal{P} \cup \mathcal{N}$, and $\mathrm{sim}$ the cosine similarity. The temperature $\tau$ is never given a value in the paper (inferred from a search of the full text).

Sizes: batch 16, 4,600 training steps in total, AdamW with weight decay 0.01, learning rate $10^{-4}$ in stage 1 and $10^{-5}$ in stage 2 (p. 7). The split of the 4,600 steps between the stages is not stated. One A100 80 GB GPU (p. 7); 7.7 hours and 53,544 MB of GPU memory for OpenAI CLIP ViT-L@224 (Table 8, p. 21).

The simpler alternative it replaces is a weighted sum of an InfoNCE loss on CLIP features and the usual reconstruction loss, $\mathcal{L}_{\text{joint}} = \lambda_{\text{con}}\mathcal{L}_{\text{con}} + \lambda_{\text{rec}}\mathcal{L}_{\text{rec}}$ (p. 4, Eq. 9). In that sum, 86.3% of training steps had a negative cosine between the two losses' gradients (p. 5).

## How noise predictions are combined or compared

Noise predictions are compared by cosine similarity inside the contrastive loss above (p. 5). For one image $x_i$ (Algorithm 1, p. 13):

$$
\hat\epsilon_i = \epsilon_\theta(x_t, h_\omega(f_\phi(x_i)), t), \quad \hat\epsilon_{+,i} = \epsilon_\theta(x_t, h_\omega(f_\phi(a(x_i))), t), \quad \hat\epsilon^-_{j,i} = \epsilon_\theta(x_t, h_\omega(f_\phi(x_j)), t)
$$

- $x_t = \sqrt{\bar\alpha_t}\,\tilde x_i + \sqrt{1-\bar\alpha_t}\,\epsilon^{gt}_t$ is image $i$'s noisy latent, and the same $x_t$ is fed under every condition: the anchor, the positive and all negatives share one noisy latent and one noise draw (p. 13).
- $a(\cdot)$ is the augmentation; $x_j$ are the other batch images.
- $\epsilon^{gt}_t$, the noise actually added, is a second positive (p. 5).

One timestep and one noise sample per image per step, with no weighting over timesteps (p. 13). Section 4.2 writes the condition as $h_\omega(f_\phi(\tilde x))$, CLIP applied to the VAE latent, while Algorithm 1 writes $f_\phi(x_i)$ on the image (p. 5, p. 13). The second matches CLIP's input, so the first is likely a notation slip (inferred).

## When adding scores or embeddings is valid

This paper does not address it. Nothing adds or subtracts noise predictions or embeddings. The only structural assumption is Assumption 1: for a fixed $t$, the map $T(z) = \epsilon_\theta(x_t, h_\omega(z), t)$ from CLIP feature to noise prediction is bi-Lipschitz, so distances between features are preserved within constant factors in noise space (p. 14). Theorem 1 uses it to bound the feature-space within-class scatter by the noise-space one, $S_{\text{inner}} \le \frac{1}{m^2} S^{(\epsilon)}_{\text{inner}}(t)$ (p. 6, Eq. 13). The assumption is stated and never checked on the model.

## What it says about the noise and the seed

Not addressed as a measurement. The design removes the noise draw from the comparison by reusing one $x_t$ across anchor, positive and negatives (p. 13), so any difference between the compared predictions comes from the condition alone (inferred). No seeds, repeats or variances are reported for any table.

## Which timesteps carry the signal

Not measured. Training samples $t$ uniformly over all steps (p. 13), and Theorem 1 holds at each fixed $t$ (p. 6). There is no per-timestep ablation.

## Unseen concepts and composition

This paper does not test held-out concepts. The encoder trains on CC3M and is evaluated zero-shot on other datasets (MNIST, CIFAR-10, EuroSAT, Caltech-101, DTD, ImageNet-1K) by clustering (p. 7), with no concept split between training and test.

On composition, MMVP-VLM reports per-pattern accuracy including "Quantity and Count" and "Positional and Relational Context" (Table 1, p. 6). For OpenAI ViT-L@224, counting rises from 20.0 to 26.7 and positional/relational context falls from 13.3 to 6.7 (Table 1, p. 6). Every MMVP-VLM cell is a multiple of 6.7 points, consistent with 15 questions per pattern (inferred), so one question moves a cell by 6.7 points. Nothing tests two objects, binding or relations in generation.

## Numbers

| Metric (scale, direction) | DCR | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| MMVP-VLM average accuracy (percent, higher better) | 33.3 | 32.6 (un2CLIP); original 19.2 | MMVP-VLM | OpenAI CLIP ViT-L@224, SD2.1 | p. 6, Table 1 |
| MMVP-VLM average accuracy (percent, higher better) | 32.6 | 31.8 (GenHancer); original 23.7 | MMVP-VLM | MetaCLIP ViT-L@224 | p. 6, Table 1 |
| Zero-shot clustering, average NMI over 6 datasets (0 to 1, higher better) | 0.76 | 0.71 (original and DIVA) | 6 datasets | OpenAI CLIP ViT-L@224 | p. 7, Table 2 |
| Zero-shot clustering, average ACC (0 to 1, higher better) | 0.67 | 0.62 (un2CLIP) | 6 datasets | OpenAI CLIP ViT-L@224 | p. 7, Table 2 |
| Zero-shot clustering, average ARI (0 to 1, higher better) | 0.54 | 0.49 (original) | 6 datasets | OpenAI CLIP ViT-L@224 | p. 7, Table 2 |
| MNIST clustering NMI (0 to 1, higher better) | 0.73 | 0.69 (GenHancer) | MNIST | OpenAI CLIP ViT-L@224 | p. 7, Table 2 |
| MMVP-VLM accuracy, DCR against the weighted sum of two losses (percent) | 33.30 | 22.96 (weighted sum) | MMVP-VLM | OpenAI CLIP ViT-L@224 | p. 8, Table 4a |
| MMVP-VLM accuracy, two-stage against end-to-end (percent) | 33.30 | 25.93 (end-to-end) | MMVP-VLM | OpenAI CLIP ViT-L@224 | p. 8, Table 4b |
| MMVP-VLM accuracy with all patch tokens kept as condition (percent) | 20.74 at 100% local tokens | 33.30 at 0% | MMVP-VLM | OpenAI CLIP ViT-L@224 | p. 21, Table 7 |
| MMVP-MLLM accuracy in LLaVA-1.5 (percent, higher better) | 31.3 | 31.3 (DIVA, un2CLIP); original 24.7 | MMVP-MLLM | Vicuna-7B | p. 8, Table 3 |
| Share of training steps with negative gradient cosine, weighted-sum method (percent) | 86.3 | not applicable | CC3M training | OpenAI CLIP ViT-L@224 | p. 5, Fig. 2 |

Across SD backbones (SD-1.4, SD-1.5, SD-2.1, SD-XL), SD-2.1 is best and SD-XL drops because a single image embedding replaces its two text encoders (Fig. 6, p. 8). The figure values were not extracted.

## What it suggests for this loop

- Score-path encoder (inferred): DCR shows a working CLIP-style contrastive loss computed in noise-prediction space, with the batch's other conditions as negatives. The direction is the reverse of this loop's encoder: DCR trains the condition encoder and uses the frozen model's noise predictions as the comparison space, where this loop wants an encoder that reads the noise predictions. The transferable piece is the batch construction.
- Score-path encoder, seed (inferred): DCR feeds one shared $x_t$ to every condition in the contrast (p. 13). The SDXL cross-seed cosine of about 0.002 says the seed dominates the direction. Building each contrastive batch from score paths that share one seed and differ only in condition would keep the encoder from separating paths by seed. Mixing seeds inside a batch invites the encoder to learn the seed.
- Score-path encoder, what to compare (inferred): DCR compares full $\epsilon$ predictions by cosine. At a shared $x_t$, the shared part of $\epsilon$ is identical across conditions, so cosine on full $\epsilon$ is likely near 1 for every pair, and an unreported temperature does the separating. Comparing $d_t(c)$, as this loop defines it, removes that shared part.
- Two arithmetic tests (inferred): nothing in the paper bears on whether $d_t(a) + d_t(b)$ lands near the joint. Its bi-Lipschitz assumption (p. 14) is the property "sum lands on joint" needs in reverse: distinct conditions must give distinct noise predictions at each step. The paper assumes it and never measures it.
- Separate-experts paper (inferred): the 86.3% gradient-conflict result (p. 5) is a warning for any loss that sums a contrastive term with a reconstruction or denoising term. DCR's fix folds the true noise in as a second positive inside one contrastive loss (p. 5), which could replace a weighted sum when experts get a matching term.
- CO3 paper (inferred): no direct bearing. DCR is a training method for an image encoder and changes no sampler.

## Closer work it cites

- DIVA, "Diffusion feedback helps CLIP see better" (ICLR 2025, ref. 77, p. 3): trains the CLIP vision encoder through a frozen diffusion model's reconstruction (noise-prediction) loss, conditioning on CLIP embeddings. No arXiv id printed.
- GenHancer, "Imperfect generative models are secretly strong vision-centric enhancers" (ICCV 2025, ref. 57, p. 3): the same reconstruction-feedback route with retrained generators and discrete latents. No arXiv id printed.
- un2CLIP, "Improving CLIP's visual detail capturing ability via inverting unCLIP" (NeurIPS 2025, ref. 43, p. 3): aligns CLIP's image feature space with an unCLIP generator's latent space. No arXiv id printed.
- "Reconstructive visual instruction tuning" (ICLR 2025, ref. 76, p. 3): a reconstruction objective in a diffusion framework used to supervise a visual representation. No arXiv id printed.
