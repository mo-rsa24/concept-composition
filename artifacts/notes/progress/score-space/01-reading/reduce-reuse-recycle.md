# Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC

arXiv 2302.11552 (v6), read from the full PDF (23 pages, appendix included). Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, as printed on p. 1.

## What it does
The paper shows that summing the scores of two diffusion models gives a score that is not the score of the noised product distribution at any $t > 0$, so plain reverse diffusion with that sum does not sample the product (p. 4, p. 5). It keeps the summed score and replaces the sampler with annealed MCMC: several Langevin or HMC steps at each noise level (Algorithm 1, p. 5). It also trains an energy-parameterized diffusion model, a scalar per $(x, t)$, so that Metropolis-corrected samplers and mixture composition become possible (p. 5, p. 6).

## What it reads
Each component model's noise prediction $\epsilon_\theta(x, t)$ at the current noisy sample, turned into a score with $\nabla_x \log q_{\sigma_t}(x) \approx -\epsilon_\theta(x,t)/\sigma_t$ (p. 3). For energy models it also reads the scalar $f_\theta(x,t) = -\|s_\theta(x,t)\|^2$, where $s_\theta$ is a U-Net of the usual $\epsilon_\theta$ shape (p. 6, p. 17), used in the Metropolis accept step (p. 16). Classifier guidance reads a noise-conditioned classifier's gradient $\nabla_x \log p_\theta(y\mid x,t)$ (Eq. 14, p. 7). All of this is on noisy pixels at every timestep; MCMC runs at every noise level (Algorithm 1, p. 5). No internal features and no text embeddings are read beyond conditioning the model.

## What it trains
Score models trained with the standard loss (Eq. 2, p. 2):

$$
L_t(\theta) = \mathbb{E}_{q(x_0)\,\mathcal N(\epsilon;0,I)}\,\big\|\epsilon - \epsilon_\theta(x_t(x_0,\epsilon), t)\big\|^2
$$

Energy models use the same loss with $\epsilon_\theta(x,t) = -\nabla_x f_\theta(x,t)$ and $f_\theta(x,t) = -\|s_\theta(x,t)\|^2$, chosen after an ImageNet pilot (p. 5, p. 6, Table A1 p. 17). Settings by domain:

- 2D toy data: residual MLP with 4 blocks and 128 hidden units, 15,000 iterations, 100 timesteps, linear β schedule (p. 18).
- CLEVR cubes: 200,000 images at 64 × 64 with 1 to 5 cubes; the model is trained on $p(x\mid y)$ where $y$ is the location of one cube; 100 timesteps, 20,000 iterations at batch 1024, about 8 hours on 8 TPUv2 cores (p. 7, p. 18).
- ImageNet 128 × 128: an unconditional model for 1 million iterations at batch 64 with 1,000 cosine-schedule timesteps, plus a noise-conditioned classifier whose training details are not given (p. 7, p. 19).
- Text-to-image: an energy-parameterized 64 × 64 model trained one week on an internal set of 400 million images, 1,000 timesteps, then two unconditional upsamplers to 1024 × 1024 (p. 19).

## How noise predictions are combined or compared
Product (Eq. 8, p. 4) and the summed score the paper samples with (Eq. 12, p. 4):

$$
q^{\text{prod}}(x) = \frac{1}{Z}\prod_{i=1}^{N} q^i(x), \qquad
\nabla_x \log q_t^{\text{prod}}(x_t) = \nabla_x \log \!\int q^1(x_0)\,q(x_t\mid x_0)\,dx_0 + \nabla_x \log \!\int q^2(x_0)\,q(x_t\mid x_0)\,dx_0
$$

The score reverse diffusion would need is that of the noised product (Eq. 11, p. 4):

$$
\nabla_x \log \tilde q_t^{\text{prod}}(x_t) = \nabla_x \log \int q^1(x_0)\,q^2(x_0)\,q(x_t\mid x_0)\,dx_0
$$

$q(x_t\mid x_0)$ is the forward noising kernel. The two agree only at $t = 0$. For CLEVR the product of one model with itself over $c$ cube locations is (Eq. 13, p. 7) $\log p_\theta(x\mid y_1,\dots,y_c) = \log p_\theta(x) + \sum_{i=1}^{c}\big(\log p_\theta(x\mid y_i) - \log p_\theta(x)\big)$, the same arithmetic as the brief's $\epsilon_{\text{PoE}}$ (inferred). Negation is $q^{\text{neg}}(x) \propto q^0(x)/q^1(x)^{\alpha}$ with $\alpha = 0.5$ (Eq. 10, p. 4). Mixtures $\frac1N\sum_i q^i(x)$ need energies and assume the ratio of normalizers is 1 (Eq. 9, p. 4).

Sampling: at each timestep run one reverse step, then $N$ MCMC steps at that noise level on the composed score (p. 15, p. 18). ULA can be run by repeating the reverse step at a fixed noise level, with the added noise scaled by $\sqrt 2$ (Appendix B.2, p. 13, p. 14). Steps per timestep: 2D uses 3 HMC steps with 3 leapfrog steps or 10 MALA steps; CLEVR 40 MALA or 13 HMC steps; ImageNet 6 MALA or 2 HMC steps; text-to-image 2 HMC steps (p. 18, p. 19). Step sizes are a constant times $\beta_t$, tuned so MALA accepts about 60% and HMC about 70% (p. 14). Comparisons hold the number of score evaluations fixed (p. 7). Intermediate samples must not be clipped to $[-1, 1]$ (p. 15). For image tapestries the overlapping models' noise vectors are summed with weights and the result is normalized to variance 1, with guidance weight 30 (p. 19).

## When adding scores or embeddings is valid
Adding scores gives the score of the product only at $t = 0$; for $t > 0$ the sum of the noised scores differs from the noised product's score (p. 4, Appendix C p. 15). The sequence of summed-score distributions still runs from the product at $t = 0$ to $\mathcal N(0, I/2)$ at $t = T$, so annealed MCMC on it reaches the right endpoint (p. 5, p. 16). The same holds for tempering ($\lambda\nabla\log q_t \ne \nabla\log q_t^{\lambda}$ for $t > 0$) and for classifier-free guidance with $\lambda > 1$, which does not sample a posterior with annealed likelihood (p. 15, p. 16). Mixtures cannot be built from scores at all (p. 4). The method fails for products of two distributions that do not overlap (Fig. A3, p. 17, p. 18). The paper's abstract concludes that "the sampler (not the model) is responsible" for composition failures (p. 1).

## What it says about the noise and the seed
Not addressed. No seed variance, repeated runs or error bars appear. The only noise-related controls are equal score-evaluation budgets across samplers (p. 7) and a reverse-diffusion baseline trained with more timesteps to match the MCMC cost, which does not close the gap (p. 7). ImageNet FID and Inception Score use 50,000 samples (p. 19).

## Which timesteps carry the signal
Not measured per timestep. The mismatch between the summed score and the noised product's score is zero at $t = 0$ and present for every $t > 0$ (p. 4); its size over $t$ is not reported. Fig. 5 plots 5-cube CLEVR accuracy (% of samples with every cube placed) against MCMC steps per timestep from 0 to 25: MALA rises from about 51% to about 66%, unadjusted Langevin from about 51% to about 55%, and reverse diffusion stays near 51% (p. 7, values read from the plot image). This measures how much correction per noise level helps, and says nothing about which noise levels need it. The tapestries skip the Langevin steps for the final 16 of 256 ancestral steps (p. 20).

## Unseen concepts and composition
This paper does not test held-out concepts. It tests held-out combinations: the CLEVR model is trained on one cube location per image and composed to 1 to 5 locations, with accuracy falling from 91.6% to 72.7% for the best sampler (Table 2, p. 6). Text-to-image composition is qualitative: detail ("Central Park 1900s" AND "Its snowy in New York"), color binding ("A lake with purple trees" AND "purple trees"), negation ("Forest" AND NOT "Green Foliage") and counting ("Two Oranges" AND "Two of the same fruit") (Figs. A4 to A7, p. 20 to p. 23).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| % of samples with every cube at its location by a classifier, higher better, 5 cubes | 72.7 (energy, HMC) | 57.4 (score, reverse diffusion = Liu et al. 2022) | CLEVR 64 × 64 | own U-Net | p. 6, Table 2 |
| Same, 1 cube | 91.6 (energy, HMC) | 70.8 (score, reverse) | CLEVR | own U-Net | p. 6, Table 2 |
| Same, 5 cubes, score model with MCMC only | 62.3 (score, U-HMC) | 57.4 | CLEVR | own U-Net | p. 6, Table 2 |
| Classifier accuracy % on guided samples, higher better | 94.61 (energy, HMC) | 18.64 (score, reverse) | ImageNet 128 × 128 | own U-Net + noisy classifier | p. 7, Table 3 |
| FID on 50,000 samples, lower better | 30.52 (energy, HMC); 26.89 (score, U-HMC) | 30.46 (score, reverse) | ImageNet 128 × 128 | same | p. 7, Table 3 |
| RAISE log-likelihood lower bound, higher better, 2D product | 2.71 (energy, HMC) | 1.55 (score, reverse) | 2D toy | residual MLP | p. 6, Table 1 |
| FID on 1,000 unconditional samples, lower better | 90.9 (L2 energy) | 86.7 (plain $\epsilon_\theta$) | ImageNet | own U-Net | p. 17, Table A1 |
| Sampling cost | up to 5 times standard sampling; energy models double memory and compute | | | | p. 8, p. 9 |

## What it suggests for this loop
- Gap predicts the term (inferred): even with perfect experts and a true product, the joint path minus the product path is nonzero at every $t > 0$ because the noised product's score is not the sum of noised scores. Part of the brief's "interaction term" at noisy steps is this noising effect and would exist even with no interaction in the clean data. The text gap $e(\text{joint}) - e(a) - e(b)$ has no such part, so alignment should be read separately at low noise, where the two agree.
- Score-path encoder (inferred): the energy parameterization gives one scalar per $(x, t)$ whose gradient is the score. A scalar energy path over steps 0 to 15 is a much lower-dimensional summary than the stack of $d_t$ and may be less dominated by the seed-specific direction the brief reports.
- Sum lands on joint (inferred): the summed path is the path of reverse diffusion with a sampler the paper calls wrong. Embedding the path produced after per-step MCMC correction is the comparison that separates sampler error from model error.
- CO3 paper (inferred): repeating the reverse step at the same noise level with $\sqrt 2$ noise is ULA at almost no code cost (p. 14). Score-model U-HMC lifts 5-cube accuracy from 57.4% to 62.3% at equal score evaluations (Table 2, p. 6). Two practical rules apply directly: do not clip intermediate samples (p. 15), and tune step size as a constant times $\beta_t$ by acceptance rate (p. 14).
- Separate-experts paper (inferred): the product construction assumes the experts' supports overlap; disjoint experts fail (p. 18). Energy parameterization costs a second backward pass (p. 9) but allows Metropolis correction, the best result in every table.

## Closer work it cites
- Salimans and Ho, "Should EBMs model the energy or the score?", ICLR 2021 EBM workshop, no arXiv id printed (p. 6, p. 16). Trains scalar energies on noisy inputs whose gradient is the score; the paper reuses its DAE, L2 and inner-product forms.
- Geffner, Papamakarios and Mnih, "Score modeling for simulation-based inference", arXiv 2209.14249 (p. 5). Uses MCMC to sample products of diffusion models; concurrent work, no encoder.

No cited paper trains an encoder or classifier on noise predictions or sampling trajectories.
