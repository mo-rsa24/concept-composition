# InfSplign: Inference-Time Spatial Alignment of Text-to-Image Diffusion Models

arXiv 2512.17851v2, read from the full PDF (25 pages, appendix included).

## What it does

InfSplign adds a hand-written loss on word cross-attention maps to the noise prediction at every denoising step, so that two objects land in the prompted spatial relation and both stay present (p. 1, p. 5). The loss has three terms: centroid placement, attention variance for presence, and a balance between the two objects' variances (p. 5, p. 6). Its main comparisons are on SD 1.4 and SD 2.1, and SDXL appears only in an appendix ablation table (p. 7, p. 18).

## What the classifier or loss reads

**No classifier.** The loss reads the cross-attention maps of the two object tokens $A$ and $B$ from a single prompt of the form "A relation B" (p. 4, p. 5).

**Layers.** Coarse attention is cross-attention layers 1 to 3 of the first U-Net decoder block. Mid-level attention is layers 1 to 3 of the second decoder block (p. 7). Fine-grained layers are unused (p. 5). The spatial term uses coarse and mid-level maps, presence uses the first block, and balance uses the second (p. 5, p. 7).

**What is computed per token and layer (p. 5).**

$$
c_A^{(l)} = \left( \frac{\sum_{h,w} A_t^{(l)}[h,w]\, x_w}{\sum_{h,w} A_t^{(l)}[h,w]},\; \frac{\sum_{h,w} A_t^{(l)}[h,w]\, y_h}{\sum_{h,w} A_t^{(l)}[h,w]} \right)
$$

$$
\sigma_A^{2(l)} = \frac{\sum_{h,w} A_t^{(l)}[h,w]\, \lVert (x_w, y_h) - c_A \rVert^2}{\sum_{h,w} A_t^{(l)}[h,w]}
$$

The paper does not print the attention map resolutions. It also does not give the layer choice for SDXL, because the implementation details cover SD 1.4 and SD 2.1 only (p. 7).

## What it trained on

Nothing is trained. The hand-written loss is (p. 5, p. 6):

$$
\mathcal{L}_{\text{InfSplign}} = \lambda_s \mathcal{L}_{\text{spatial}} + \lambda_p \mathcal{L}_{\text{presence}} + \lambda_b \mathcal{L}_{\text{balance}}
$$

$$
\mathcal{L}_{\text{spatial}} = f_{\text{spatial}}(\alpha(m - \Delta)), \quad \mathcal{L}_{\text{presence}} = \sigma_A^{2(l)} + \sigma_B^{2(l)}, \quad \mathcal{L}_{\text{balance}} = \left| \sigma_A^{2(l)} - \sigma_B^{2(l)} \right|
$$

$\Delta$ is the signed centroid difference along the axis of the relation, for example $x_B - x_A$ for "left" (p. 5). $f_{\text{spatial}}$ is GeLU, $\alpha = 1.5$ sets the steepness, and $m$ is the required centroid margin: 0.25 on SD 1.4 and 0.5 on SD 2.1 (p. 6, p. 7).

**Tuned weights.** $\lambda_s = 0.5, \lambda_p = 1, \lambda_b = 0.5$ on SD 1.4 and $\lambda_s = 0.5, \lambda_p = 1, \lambda_b = 1.0$ on SD 2.1, chosen by grid search (p. 7, p. 16). The SDXL weights are not printed.

## How its gradient enters

The gradient is added to the classifier-free guidance noise prediction (p. 5, Eq. 3):

$$
\epsilon_t \leftarrow \epsilon_\theta(z_t; t) + \gamma\big(\epsilon_\theta(z_t; t, y) - \epsilon_\theta(z_t; t)\big) + \eta \nabla_{z_t} \mathcal{L}_{\text{InfSplign}}
$$

$$
z_{t-1} \leftarrow z_t - s_t \cdot \epsilon_t
$$

$\gamma = 7.5$ is the guidance scale, $\eta = 1000$ is the weight on the loss gradient, and $s_t$ is the step-size factor from the schedule (p. 4, p. 7). The right-hand form printed in Eq. 3 drops $\eta$, and Algorithm 1 keeps it (p. 5, p. 7).

- **Steps:** all reverse timesteps of 50 inference steps (p. 5, p. 7).
- **Iterations:** one gradient per step (p. 7, Algorithm 1).
- **Normalisation or cap:** none is stated.
- **Model:** SD 1.4 and SD 2.1 in the main results, with SDXL rows in Table 6 (p. 7, p. 18).

## Unseen concepts

This paper does not test held-out concepts. Nothing is trained, so there is no training vocabulary to hold out from. The hyperparameter grid search runs on VISOR3160, a 3160-prompt subset of VISOR that "still cover[s] all objects", and the chosen values are then scored on the complete VISOR benchmark (p. 8, p. 9).

## Word leakage and confounds

**Word leakage.** A hand-written loss cannot learn to read a word trace. The relation enters only by choosing the axis and sign of $\Delta$ (p. 5). The paper does not discuss whether the object tokens' maps carry traces of the relation word.

**Confounds.**

- Tuning on VISOR3160 and evaluating on full VISOR means the tuning prompts are part of the reported test set, so the VISOR margins are optimistic to an unknown degree (p. 8, p. 9) (inferred).
- The no-loss baseline differs between tables. Object accuracy for SD 1.4 is 29.86% in Table 1 and 32.74% in Table 3, and for SD 2.1 it is 47.83% in Table 1 and 44.37% in Table 3 (p. 8, p. 9). The paper does not explain the difference.
- SDXL has no comparison with any other method. The presence-only row is blank (p. 18, Table 6), and the caption names only SD 1.4 and 2.1.
- Its T2I-CompBench score should not be set beside Learn-to-Steer's. Vanilla SD 2.1 scores 0.1342 here (p. 8, Table 2) and 0.089 in Learn-to-Steer (Learn-to-Steer p. 5, Table 1), so the two papers' protocols differ (inferred).

## Numbers

Scales: OA is object accuracy, the percentage of images where both objects are detected. VISOR uncond is the percentage of images with a correct relation, whether or not both objects were generated correctly. VISOR-4 is read here as the percentage of prompts with all 4 images correct (inferred, since p. 15 words it as "the percentage of images out of N"). All are 0 to 100, higher is better (p. 15). T2I-CompBench spatial is a 0-to-1 score, higher is better (p. 8, p. 15).

| Metric | This method | No loss | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|---|
| OA (%) | 67.36 | 29.86 | CoMPaSS 65.56 | VISOR | SD 1.4 | p. 8, Table 1 |
| VISOR uncond (%) | 66.54 | 18.81 | STORM 57.58 | VISOR | SD 1.4 | p. 8, Table 1 |
| OA (%) | 77.28 | 47.83 | CoMPaSS 68.22 | VISOR | SD 2.1 | p. 8, Table 1 |
| VISOR uncond (%) | 76.26 | 30.25 | CoMPaSS 62.06 | VISOR | SD 2.1 | p. 8, Table 1 |
| VISOR-4 (%) | 50.23 | 4.70 | CoMPaSS 35.90 | VISOR | SD 2.1 | p. 8, Table 1 |
| Spatial score (0 to 1) | 0.4172 | 0.1342 | CoMPaSS 0.3200 | T2I-CompBench | SD 2.1 | p. 8, Table 2 |
| OA (%) | 81.01 | 68.06 | none compared | VISOR | SDXL | p. 18, Table 6 |
| VISOR uncond (%) | 78.44 | 45.10 | none compared | VISOR | SDXL | p. 18, Table 6 |
| VISOR uncond (%), spatial term only | 76.91 | 45.10 | none compared | VISOR | SDXL | p. 18, Table 6 |
| VISOR uncond (%), presence and balance only | 46.82 | 45.10 | none compared | VISOR | SDXL | p. 18, Table 6 |

## What to copy, what to avoid

- Use InfSplign's centroid-margin loss as the untrained baseline that a learned spatial checker must beat on SDXL. Neither this paper nor Learn-to-Steer compares against the other (inferred).
- Do not conclude from our 0-of-8 anti-correlation result that hand-written attention losses cannot move SDXL. Here a centroid loss lifts SDXL's VISOR uncond from 45.10% to 76.91% (p. 18, Table 6), though with a joint prompt (inferred).
- Test the placement term first. On SDXL the spatial term alone carries the gain, while presence and balance alone move VISOR uncond by 1.72 points, from 45.10% to 46.82% (p. 18, Table 6).
- Expect the PoE case to differ. InfSplign's two words share one prompt's attention, so they compete for each pixel. Under PoE each subject has its own prompt, so the maps may not compete in the same way (inferred).
- Compare dose before comparing methods. InfSplign applies an uncapped gradient at $\eta = 1000$ on all 50 steps, while our push is capped at 0.4 of PoE's prediction norm on steps 0 to 15 (p. 5, p. 7) (inferred).
- Avoid tuning on a subset of the test benchmark. Keep separate tuning prompts (inferred).

## Closer work it cites

None. Its cited methods are hand-written inference-time losses, layout-guided methods or fine-tuning, and none trains a classifier for an abstract property or tests one on unseen concepts (p. 2, p. 3).
