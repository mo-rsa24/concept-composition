# The Superposition of Diffusion Models Using the Itô Density Estimator

arXiv 2412.17762, read from the full PDF (54 pages, appendix included). Published as a conference paper at ICLR 2025 (printed on every page).

## What it does

SuperDiff combines several pre-trained diffusion models (or one model under several prompts) at sampling time by taking a weighted sum of their scores, with weights recomputed at every step from how likely the current sample is under each model (p. 5, p. 6). It tracks each model's log density along the sampling path with a new estimator that needs no divergence and no extra network calls (p. 6, Thm. 1). Two weightings are offered: OR samples the mixture of the models, and AND picks weights so the sample stays equally likely under every model (p. 2, p. 7).

## What it reads

At each step it reads each model's score $\nabla \log q^i_t(x_\tau)$ at the current noisy sample $x_\tau$ (in SD, the VAE latent), together with the step just taken $dx_\tau$, which includes the injected Wiener noise (p. 6, Algorithm 1 and Eq. 13). For Stable Diffusion each "model" is SD v1-4 under one concept prompt, run with classifier-free guidance (p. 7). The paper does not print the guidance scale or the number of SD sampling steps. The weights depend on the per-model log density accumulated over all steps so far, so they read the whole path up to the current step (p. 6, Algorithm 1). Internal UNet features and text embeddings are not read directly.

## What it trains

Nothing is trained for the Stable Diffusion, protein (Proteus, FrameDiff) or molecule (LDMol) experiments; all use pre-trained checkpoints (p. 8, p. 9, p. 30). In the CIFAR-10 experiment it trains two diffusion models on disjoint halves of CIFAR-10 (first 5 labels and last 5 labels) and one on the union, with the hyperparameters, architecture and preprocessing of Song et al. 2020 (p. 7). The appendix repeats this with class-conditional models on two random halves (p. 30, Table A1). The paper writes the training objective as denoising score matching (p. 19, Prop. 9):

$$
\int_0^1 dt\, \mathbb{E}_{q_t(x)} \lVert \nabla \log q_t(x) - \nabla \log q_t(x;\theta) \rVert^2 = \frac{1}{N}\sum_{i=1}^N \int_0^1 dt\, \mathbb{E}_{q^i_t(x)} \lVert \nabla \log q^i_t(x) - \nabla \log q_t(x;\theta)\rVert^2 + \text{const}
$$

Training length and noise levels are delegated to Song et al. 2020 and not printed.

## How noise predictions are combined or compared

The sampler is a reverse SDE whose drift uses a weighted sum of model scores (p. 6, Algorithm 1):

$$
u_t(x_\tau) = \sum_{i=1}^{M} \kappa^i_\tau \nabla \log q^i_t(x_\tau), \qquad dx_\tau = \big(-f_{1-\tau}(x_\tau) + g^2_{1-\tau} u_t(x_\tau)\big) d\tau + g_{1-\tau}\, dW_\tau
$$

$f$ is the forward drift, $g$ the diffusion coefficient, $W$ a Wiener process, $\tau = 1 - t$ the reverse time.

Each model's log density along the path is updated by the Itô density estimator (p. 6, Eq. 13):

$$
d \log q_{1-\tau}(x_\tau) = \langle dx_\tau, \nabla \log q_{1-\tau}(x_\tau)\rangle + \Big(\langle \nabla, f_{1-\tau}(x_\tau)\rangle + \big\langle f_{1-\tau}(x_\tau) - \tfrac{g^2_{1-\tau}}{2}\nabla \log q_{1-\tau}(x_\tau),\ \nabla \log q_{1-\tau}(x_\tau)\big\rangle\Big) d\tau
$$

The weights are:

- **OR:** $\kappa^i = \mathrm{softmax}(T \log q^i_t(x_\tau) + \ell)$, temperature $T$, bias $\ell$ (p. 6). This is the density-weighted average $q^i_t / \sum_j q^j_t$ that Prop. 4 gives for the mixture (p. 7, Eq. 15).
- **AND:** $\kappa$ solves $d\log q^i_{1-\tau}(x_\tau) = d\log q^j_{1-\tau}(x_\tau)$ for all $i, j$ with $\sum_j \kappa_j = 1$, a system of $M+1$ linear equations (p. 7, Prop. 6; p. 22, Eq. 78), with coefficients $a_{ij} = d\tau\,\langle u^j_\tau(x_\tau), \nabla \log q^i_{1-\tau}(x_\tau)\rangle$ (p. 22, Eq. 75).

The weights always sum to 1, so SuperDiff is a weighted average of scores. The baseline "average of scores (Liu et al., 2022a)" fixes $\kappa = 0.5$ (p. 8). The paper does not state that $\kappa$ must lie in $[0, 1]$ for AND; nothing in the linear system forces it (inferred from p. 22).

The weights are recomputed at every step; there is no separate weighting over timesteps. One noise sample per trajectory is used, and the same $x_\tau$ and the same $dW_\tau$ are shared by all models because there is one trajectory (p. 6, Algorithm 1). The SD results use the SDE sampler only (p. 8, footnote 1). Protein runs use 500 timesteps (p. 9).

## When adding scores or embeddings is valid

The paper proves that the vector field of a mixture of densities is the density-weighted average of the component vector fields, for ODEs (p. 4, Prop. 3, credited to Liu 2022) and for SDEs that share one diffusion coefficient (p. 4, Prop. 4). That is the only combination it proves exact. The density estimator is exact when the true scores of the marginals are available (p. 2). It says energy-based composition (summing scores to sample a product) rests on "a strong assumption that the marginal densities of the noising process are given", which fails for modern models (p. 10), and that guidance-based composition lacks a firm theoretical basis (p. 2). AND is defined as an equal-density condition on the trajectory; the paper never claims AND samples the product of the two densities. The paper does not discuss adding or subtracting text or image embeddings.

For this project's product of experts, averaging two CFG predictions with $\kappa = 0.5$ equals $\epsilon_\varnothing + \tfrac{w}{2}(d_t(a) + d_t(b))$, which is the product-of-experts prediction at half the guidance weight per concept (inferred; the paper does not write this out).

## What it says about the noise and the seed

The Itô estimator's variance matches that of Hutchinson's trace estimator (p. 5; p. 23, Thm. 3 and Eq. 86 to 88). The estimator's per-step update includes $\langle dx_\tau, \nabla\log q\rangle$, which contains the injected noise $g\,dW_\tau$, so each model's tracked density depends on the noise drawn along the path (inferred from p. 6, Eq. 13). Seed variance of the image results is not reported: Table 2 has no error bars (p. 8). The appendix figures show the first 6 seeds and the authors' favourite images out of 20 seeds per concept pair (p. 32). CIFAR-10 FLD values carry ± intervals of about 0.05 to 0.23 (p. 8, Table 1). Molecule metrics come from 5 runs of batch 512 (p. 10). Protein results average 50 proteins per length (p. 9). The ODE density route costs 1082 ± 7 seconds per SD image against 209 ± 2 seconds for the SDE route (p. 8, footnote 1).

## Which timesteps carry the signal

Not measured. The paper does not report how $\kappa$ or the density gap evolves over sampling steps.

## Unseen concepts and composition

This paper does not test held-out concepts. The composition test is 20 concept pairs on SD v1-4, 20 images each, such as "a sunflower" and "a lemon" (p. 7, p. 8), listed as Figs. A5 to A24 (p. 34 to p. 53). One pair is "a dog" and "a cat" (p. 40, Fig. A11). The paper calls AND "concept interpolation" (p. 7). In Fig. 1 and Fig. A11 the AND images show one animal mixing dog and cat features per tile, with no tile showing a separate dog and cat (read from the page images at low resolution, p. 2 and p. 40; inferred). AND and the fixed-average baseline give visually close images for the same seed in the first 6 seeds of Fig. A11 (read from image, p. 40; inferred). The joint-prompt baseline is "a sunflower that looks like a lemon" with both orders tried and the better kept (p. 8). The scoring takes the minimum of the per-concept metric over the two concept prompts (p. 8). Nothing on counting, relations or attribute binding.

## Numbers

| Metric (scale, direction) | SuperDiff | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Min. TIFA over the two concepts (higher better; values near 0 to 100, inferred as percent) | 39.92 (AND) | 32.48 (average of scores, κ = 0.5); joint prompt 27.58 | 20 concept pairs × 20 images | SD v1-4 | p. 8, Table 2 |
| Min. CLIP score over the two concepts (higher better; cosine similarity, values near 24 suggest ×100, inferred) | 24.79 | 24.23 (average); joint 23.87 | same | SD v1-4 | p. 8, Table 2 |
| Min. ImageReward (higher better) | −1.39 | −1.57 (average); joint −1.62 | same | SD v1-4 | p. 8, Table 2 |
| FID, ODE sampling (lower better) | 4.41 (OR); 4.11 at T = 100 | 4.28 random choice of model; 6.00 model on the union | CIFAR-10, disjoint halves | score SDE models (Song et al. 2020 setup) | p. 8, Table 1 |
| FID, SDE sampling (lower better) | 4.00 (OR) | 3.50 model on the union | same | same | p. 8, Table 1 |
| OR selection, max TIFA / abs. gap between concepts (higher better) | 95.83 / 84.10 | joint prompt 97.92 / 79.46; random prompt 97.58 / 88.21 | concept pairs | SD v1-4 | p. 33, Table A2 |
| Fraction of designable proteins, scRMSD under 2 Å (higher better) | 0.976 (OR, ℓ = 1) | 0.928 Proteus | 50 proteins per length | Proteus + FrameDiff | p. 9, Table 3 |
| Top-10 product of the two properties (higher better) | 0.177 (AND) | 0.171 joint prompt | GSK3β and QED | LDMol | p. 10, Table 4 |
| Seconds per image for density tracking (lower better) | 209 ± 2 (SDE) | 1082 ± 7 (ODE route) | n/a | SD v1-4 | p. 8, footnote 1 |

## What it suggests for this loop

- For a score-path encoder (inferred): Eq. 13 is a hand-built scalar summary of a score path. It sums $\langle dx_\tau, \nabla\log q^c\rangle$ plus a norm term over steps, with no extra network calls. On the digit canvases this gives a per-condition "log density along the path" for each caption, which is a cheap baseline the learned encoder should beat at retrieving the caption.
- For a score-path encoder (inferred): the tracked density includes the injected noise $dW_\tau$, which is one concrete route by which a path statistic carries the seed. This matches the project's seed-specific cosine of about 0.002 and argues for building negatives that share the seed.
- For "sum lands on joint" (inferred): SuperDiff forms a weighted average with weights summing to 1, where the product of experts forms the sum $\epsilon_\varnothing + d_t(a) + d_t(b)$. SuperDiff's paths are a third composition to embed beside the product's path and the joint path, which tests whether the encoder separates "average" from "sum" from "joint".
- For "gap predicts the term" (inferred): the AND weights solve a linear system in $\langle u^j, \nabla\log q^i\rangle$, inner products of one condition's prediction with another's. These cross terms of the two $d_t$ are the cheapest per-step quantities to correlate with the text gap $e(\text{joint}) - e(a) - e(b)$.
- For the separate-experts paper (inferred): the CIFAR-10 experiment is the closest published case of separately trained experts on disjoint data. There SuperDiff OR beats the model trained on the union in ODE FID (4.41 against 6.00) but not in SDE FID (4.00 against 3.50), and it does not beat picking one model at random (4.28) without the temperature T = 100 (p. 8, Table 1). The paper never tests AND on disjoint-data experts, where equal density may have no solution with positive weights.
- For the CO3 paper (inferred): SuperDiff AND targets one object that looks like both concepts (cat and dog fused, p. 40). That is the hybrid this project scores as failure, so SuperDiff AND is a baseline that should lose on two-separate-creatures scoring. Its per-step reweighting by tracked density could be applied inside CO3 as a weight rule, at no extra network cost when the sampler is stochastic.
- An SDXL implementation exists (p. 11, acknowledgments, huggingface.co/superdiff/superdiff-sdxl-v1-0), but the paper reports no SDXL results.

## Closer work it cites

- Training Data Protection with Compositional Diffusion Models, Golatkar et al. 2023, arXiv 2308.01937 (p. 10, p. 12). The paper says it learns "a separate model for the densities" to weight component vector fields, which is a learned network reading noisy samples to estimate per-model density (p. 10).
