# Readout Guidance: Learning Control from Diffusion Features

arXiv 2312.02150v2, read from the full PDF (27 pages, appendix included).

## What it does

Readout Guidance trains small heads on the frozen U-Net's decoder features to predict a property of the image at any timestep (p. 1, p. 4). During sampling, the distance between a head's output and a target is differentiated with respect to the noisy image and added to the noise prediction (p. 4). The heads cover pose, depth, edges, correspondence, appearance similarity and face identity, on SD 1.5 and SDXL (p. 4, p. 5).

## What the classifier or loss reads

**Features.** The heads read the frozen decoder layers of the U-Net (p. 4). SDXL has 9 decoder layers and SD 1.5 has 12 (p. 16). The raw feature maps have 320, 640 or 1280 channels (p. 14, Fig. 12). Word cross-attention maps are not read.

**Text.** Features come from the unconditional branch, with the empty string as the prompt for all images (p. 4). The head therefore receives no word-specific signal from the user's prompt (inferred).

**Noise and time.** Training features come from images noised by the forward process, and the heads are timestep-conditional (p. 4). Layers are aggregated at one timestep, and features are not aggregated across timesteps (p. 4). The decoded image is not used, because the method works directly on latent U-Net features (p. 5).

**Architecture (p. 13, p. 14).**

1. Bottleneck layers bring each decoder feature map to a standard channel count and spatial size. A projected U-Net timestep embedding is added here.
2. A learned weighted sum combines the standardised maps across decoder layers.
3. Spatial heads add convolutions ending in a Tanh three-channel readout. Relative heads compute a distance directly on the aggregated feature map (p. 4).

The head has 5.9M parameters (35 MB) on SDXL and 8.5M (49 MB) on SD 1.5 (p. 15, Table 2). The paper does not print the final spatial size of the aggregated map.

## What it trained on

| Head | Data | Labels | Loss | Training on Nvidia A100 | Page |
|---|---|---|---|---|---|
| Pose | PascalVOC, up to 8.5k images | OpenPose pseudo labels | Per-pixel MSE, re-weighted for sparse targets | Up to 5k steps at batch 8: 37 min SD 1.5, 2 hr 25 min SDXL | p. 4, p. 5, p. 7, p. 13 |
| Depth | PascalVOC | MiDaS pseudo labels | Per-pixel MSE | 41 min SD 1.5, 2 hr 40 min SDXL | p. 4, p. 13 |
| Edge | PascalVOC | HED pseudo labels | Per-pixel MSE, re-weighted | 41 min SD 1.5, 2 hr 38 min SDXL | p. 4, p. 5, p. 13 |
| Correspondence | DAVIS video frame pairs | Point tracks from a tracker | Symmetric cross-entropy on cosine similarity | Up to 10k steps at batch 1: 48 min SD 1.5, 2 hr 15 min SDXL | p. 5, p. 13 |
| Appearance similarity | DAVIS | Same-video positive, SDEdit negative | Hinge with margin 0.5 | Up to 1k steps at batch 1: 8 min SD 1.5, 24 min SDXL | p. 5, p. 13 |
| Identity | CelebA-HQ face crops | SDEdit hard negatives | Same scheme as appearance | Not printed | p. 5 |

**How much data.** The paper says heads need "as few as 100" examples (p. 2, p. 7). Only the pose head has a data-size test, on SDXL at 100, 1k and 8.5k images (p. 7, p. 15). Accuracy rises with data. PCK at $\alpha = 0.05$, the percentage of keypoints within 5% of the pose's size (0 to 100, higher is better), is 15.24 at 100 images, 19.65 at 1k and 24.29 at 8.5k (p. 15, Table 2). Image-quality distance CMMD (lower is better) goes 1.305, 1.243, 1.134 across the same sizes (p. 15, Table 3). The 100-image head scores 63% of the 8.5k head's PCK (15.24 / 24.29, inferred).

## How its gradient enters

The gradient is added to the noise prediction (p. 4):

$$
\hat\epsilon_t \leftarrow \epsilon_\theta(x_t) + w \cdot \nabla_{x_t} d\big(r, f_\psi(x_t)\big)
$$

$\epsilon_\theta$ is the frozen model, $w$ is the guidance weight, $f_\psi$ is the head, $r$ is the target and $d$ is a distance. For relative heads the target is $f_\psi(z_t)$ from a separate reference diffusion process (p. 4).

**Normalisation.** All results use a normalised gradient (p. 13):

$$
w_t \leftarrow w_{t-1} - \alpha \cdot \frac{\nabla \mathcal{L}}{\sqrt{\nabla \mathcal{L}^2 + \epsilon}}, \qquad \alpha = 2 \times 10^{-2}
$$

Here $w_t$ is the "input/output noise" (p. 12), a different symbol from the guidance weight $w$. The update size is therefore independent of the gradient's magnitude (p. 13). The printed formula leaves open whether the division is per element or by the whole vector's norm, so check the code before copying it (inferred). The p. 12 and p. 13 updates subtract the gradient while the p. 4 form adds it to the noise, so the sign also needs checking in the code (inferred).

**Steps and scale (p. 12, p. 13, Table 1).** 100 sampling steps. Guidance weight 1.0 for drag, appearance and identity, and 0.5 for spatial control. Guidance runs over fractions [0.0, 0.5] of the steps for drag and spatial control, [0.05, 0.5] for appearance and [0.05, 0.8] for identity. Text guidance is 7.5, except 3.5 for drag on real images.

**Stability findings.** Removing timestep conditioning from the heads gives improbable outputs (p. 13). Spatial control needs the stochastic setting $\eta = 1.0$, and $\eta = 0$ gives artifacts (p. 12, p. 15). Weights well above 1.0 also give artifacts (p. 12, p. 14).

## Unseen concepts

This paper does not test held-out concepts. No head is scored on categories kept out of its training set.

**What it does test.** The pose evaluation uses 100 random images of humans from the MSCOCO validation set, which the paper calls unseen (p. 6, p. 15). That is a held-out image set of the same category the pose head was trained to read. Correspondence and appearance results on subjects such as a tiger, raccoon and ibex are qualitative figures (p. 6, p. 17), and no category split is defined for them.

## Word leakage and confounds

**Word leakage.** The paper does not discuss it. Reading the unconditional branch with an empty prompt removes the user's words from the features by construction (p. 4) (inferred). That does not stop a head from learning image content such as species, which is the concern with this project's current checker on SDXL UNet features (inferred).

**Confounds.**

- Pose labels and the evaluation both come from OpenPose (p. 4, p. 7), so the pose score measures agreement with the same tool that labelled the training data (inferred).
- The strong pose result combines Readout Guidance with T2I-Adapter (54.54 PCK at $\alpha = 0.05$). Readout Guidance alone reaches 24.29 (p. 15, Table 2).
- The quantitative tests cover pose only. Depth, edge, correspondence, appearance and identity are shown as images (p. 5 to p. 7, p. 15).

## Numbers

PCK is the percentage of correct keypoints against the input pose on 100 MSCOCO images, 0 to 100, higher is better. CMMD is the distance between generated and real image distributions, lower is better (p. 15, p. 16).

| Metric | This method | No control | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|---|
| PCK, $\alpha = 0.05$, 100 training images | 15.24 | 1.19 | T2I-Adapter 24.06 (3M images) | MSCOCO val | SDXL | p. 15, Table 2 |
| PCK, $\alpha = 0.05$, 1k training images | 19.65 | 1.19 | T2I-Adapter 24.06 | MSCOCO val | SDXL | p. 15, Table 2 |
| PCK, $\alpha = 0.05$, 8.5k training images | 24.29 | 1.19 | T2I-Adapter 24.06 | MSCOCO val | SDXL | p. 15, Table 2 |
| PCK, $\alpha = 0.05$, added to T2I-Adapter | 54.54 | 1.19 | T2I-Adapter alone 24.06 | MSCOCO val | SDXL | p. 15, Table 2 |
| PCK, $\alpha = 0.05$, 8.5k training images | 19.91 | 0.49 | ControlNet 42.34 (200k images) | MSCOCO val | SD 1.5 | p. 15, Table 2 |
| CMMD, 8.5k training images | 1.134 | 1.374 | T2I-Adapter 0.817 | MSCOCO val | SDXL | p. 15, Table 3 |

## What to copy, what to avoid

- Make every checker timestep-conditional. This paper finds that removing the conditioning sends samples off natural images (p. 13) (inferred for our checkers).
- Normalise the gradient so the step does not depend on its size (p. 13). This fits our cap on the push at 0.4 of PoE's prediction norm (inferred).
- Plan for more than 100 labelled examples. At 100 images the pose head reaches 15.24 PCK against 24.29 at 8.5k (p. 15), and no abstract property was tested at any size (inferred).
- Do not cite this paper as evidence that UNet-feature checkers transfer across concepts. It has no held-out category test, which leaves open the risk that our UNet checker learned species (inferred).
- Our result that a UNet-feature checker moved layout while a decoded-image checker changed only texture agrees with this paper's use of latent U-Net features for layout-level control (p. 5) (inferred).
- The empty-prompt trick cannot be copied directly to word attention maps, because those maps need the word in the prompt (inferred). It could serve as a comparison arm: a UNet checker on empty-prompt features against a word-map checker (inferred).
- Keep guidance in the early steps. The paper reports that composition is largely set in the first 10% of sampling (p. 15, p. 18), which is consistent with our push on steps 0 to 15 of 50 (inferred).

## Closer work it cites

**Mid-U Guidance: Fast Classifier Guidance for Latent Diffusion Models** (Whitaker, 2023, a Weights and Biases report with no arXiv id printed; p. 2, reference on p. 10). It learns a guidance function from diffusion features for aesthetics, which is an abstract property. Readout Guidance does not report whether it was tested on unseen concepts.
