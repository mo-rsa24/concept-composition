# Diffusion Sampling Path Tells More: An Efficient Plug-and-Play Strategy for Sample Filtering

arXiv 2505.23343 (v1, 29 May 2025), read from the full PDF (25 pages, appendix included). The PDF is marked "Preprint. Under review." with no venue (p. 1).

## What it does

The paper sums, over the first sampling steps, the squared norm of the difference between the conditional and unconditional score at each step, and calls the total the Accumulated Score Difference (ASD) (p. 5). Seeds with low ASD are stopped early and the rest are finished, a method the authors call CFG-Rejection (p. 6). They report that high-ASD samples sit in high-density regions of the data, adhere better to the prompt, and score higher on preference and compositional benchmarks (p. 5, p. 7, p. 8).

## What it reads

At every step it reads the classifier-free guidance pair evaluated at the same noisy latent $x_t$: the conditional and the unconditional model output (p. 5, Eq. 5). The read differs by model:

- SD1.5: the conditional output minus the unconditional output, flattened to one vector, with its $\ell_2$ norm taken (p. 20).
- SDXL: the same, multiplied by $\sigma_t$ at each step (p. 20).
- EDM2-S on ImageNet: the "matrix two-norm" of the difference, averaged over the channel dimension, with no $\sigma_t$ scaling (p. 14).
- FLUX.1-dev: an extra model evaluation at $\omega = 1$ to expose the two branches of the guidance-distilled model, then the same as SDXL (p. 22).
- 2D toy: closed-form difference with outputs multiplied by $\sigma_t$ (p. 13).

Only the norm is kept. Direction is discarded, and the authors list cosine-similarity signals as future work (p. 9). The quantity read is the norm of this loop's $d_t(c)$, up to a $\sigma_t$ factor that the paper applies inconsistently across models (inferred from p. 14 and p. 20).

## What it trains

Nothing is trained (p. 1). The 2D toy model follows the training code of Karras et al. 2024 (p. 13); all other models are pretrained checkpoints (p. 14, p. 20, p. 22).

## How noise predictions are combined or compared

Guidance itself is the usual linear combination (p. 4, Eq. 3):

$$
S_w(x; \sigma, c) = w\,S_\theta(x; \sigma, c) + (1 - w)\,S_\theta(x; \sigma, \varnothing)
$$

The comparison is a per-step norm and a sum over steps (p. 5, Eqs. 5 and 6; p. 6, Eq. 7):

$$
G_t(c) = \big\lVert S_\theta(x_t; \sigma_t, c) - S_\theta(x_t; \sigma_t, \varnothing) \big\rVert_2, \qquad E_T(c) = \sum_{t=1}^{T} G_t(c)^2, \qquad E_{\tau:T}(c) = \sum_{t=T-\tau}^{T} G_t(c)^2
$$

- $S_\theta(x; \sigma, c)$ is the network's score estimate at noise level $\sigma$ under condition $c$; $\varnothing$ is the empty condition.
- $t = T$ is the noisiest step, so $E_{\tau:T}$ covers the first $\tau$ steps of sampling (inferred from the sampler running from $\sigma_T = \sigma_{\max}$ down, p. 4).
- A sample is discarded when $E_{\tau:T}(c) < \gamma$, where $\gamma$ is a percentile threshold (p. 6).

Each seed gets one trajectory, so the conditional and unconditional outputs at a step always share the same $x_t$ (p. 5). Steps carry equal weight: no per-timestep weighting beyond the $\sigma_t$ factor for some models (p. 5, p. 20).

## When adding scores or embeddings is valid

This paper does not address it. It uses only one condition against the empty condition, never a sum of two conditions' differences. It states that it has no theory for why ASD tracks density (p. 9).

## What it says about the noise and the seed

The method exists because quality varies across seeds: ASD ranks seeds of one prompt or class, and the ranking is the signal (p. 1, p. 6). ImageNet uses 10,000 samples over 50 classes, about 200 seeds per class (p. 6, inferred division). GenEval and DPG-Bench keep 4 of 20 or 4 of 50 seeds per prompt, and each experiment is repeated with three initial seeds and averaged (p. 8, p. 20). No variance or spread across the three repeats is reported. No experiment measures how much of $G_t$ comes from the seed and how much from the prompt.

## Which timesteps carry the signal

Measured indirectly by the cutoff $\tau$, the number of early steps summed.

- ImageNet, EDM2-S, 32 Heun steps: PickScore (a preference score, higher better) is 20.38 for all samples and 20.61 for the top 10% by full ASD. Summing the first 10 steps gives 20.52 and the first 20 gives 20.61, so the first 20 of 32 steps match the full sum (Table 1, p. 7).
- GenEval, SD1.5, 50 steps, guidance 5: the two-object score (fraction of prompts with both objects detected, 0 to 1) is 0.3540 at random, 0.3801 with $\tau = 10$, 0.4670 with $\tau = 20$ and 0.4988 with $\tau = 30$. Counting peaks earlier, 0.4110 at $\tau = 10$ against 0.3359 at random (Table 2, p. 8).
- DPG-Bench, SDXL: the Attribute category improves steadily as $\tau$ rises from 10 to 40, while Relation first improves and then falls (p. 21).
- The paper states that $G_t$ "typically decreases in later steps" (p. 6) and gives no curve or number for it.

## Unseen concepts and composition

This paper does not test held-out concepts. Nothing is trained, so nothing is held out.

It does test composition. GenEval (553 prompts, six categories: single object, two objects, counting, colors, position, color attribution) and DPG-Bench (1,065 prompts) are reported per category (p. 8, p. 9). With SD1.5 at guidance 5, keeping 4 of 50 seeds by full ASD raises two-object from 0.3540 to 0.5177 (the "+46%" in the text) and color attribution from 0.0579 to 0.09 (Table 2, p. 8). Counting falls when selecting by full ASD in several settings: SD1.5 at guidance 5 goes from 0.3359 to 0.3235 (4 of 20), and SDXL at guidance 5 goes from 0.4094 to 0.3250 (4 of 50) (Table 2, p. 8; Table 4, p. 21). Position stays near zero (0.0375 to 0.0575, SD1.5) (Table 2, p. 8). On FLUX text rendering, high-ASD samples render the full phrase more often in qualitative figures, with no success rate given (p. 22, Figs. 21 to 23).

## Numbers

| Metric (scale, direction) | CFG-Rejection or ASD selection | Baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| PickScore (preference score, higher better) | 20.61 (top 10% by full ASD); 20.52 ($\tau = 10$) | 20.38 (all samples) | ImageNet, 50 classes, 10k samples | EDM2-S, 32 Heun steps, $\omega = 1.4$ | p. 7, Table 1 |
| HPSv2 (preference score, higher better) | 26.57 (top 10%); 26.31 ($\tau = 10$) | 26.13 (all samples) | ImageNet | EDM2-S | p. 7, Table 1 |
| GenEval two-object (fraction, 0 to 1, higher better) | 0.5177 (4 of 50); 0.4670 ($\tau = 20$) | 0.3540 (random) | GenEval | SD1.5, guidance 5, 50 steps | p. 8, Table 2 |
| GenEval overall (fraction, 0 to 1, higher better) | 0.4594 ($\tau = 20$) | 0.4152 (random) | GenEval | SD1.5, guidance 5 | p. 8, Table 2 |
| GenEval overall | 0.5491 ($\tau = 20$) | 0.5393 (random) | GenEval | SDXL, guidance 5 | p. 21, Table 4 |
| GenEval overall | 0.5510 (4 of 50) | 0.5625 (random) | GenEval | SDXL, guidance 9 | p. 21, Table 4 |
| DPG-Bench overall (score out of 100, higher better) | 65.15 (4 of 50); 64.14 ($\tau = 20$) | 62.45 (random) | DPG-Bench | SD1.5, guidance 5 | p. 9, Table 3 |
| DPG-Bench overall | 74.66 ($\tau = 40$) | 73.54 (random) | DPG-Bench | SDXL, guidance 5 | p. 21, Table 5 |

Best-of-N with an external reward is compared only as curves under a matched time budget, with no table values (Fig. 6, p. 8; p. 20). No correlation coefficient between ASD and density is printed. The toy evidence is a binned log-linear plot over 50 ASD bins (p. 13).

## What it suggests for this loop

- Score-path encoder (inferred): the squared length of a score path as this brief defines it, $\sum_t \lVert d_t(c) \rVert^2$, is the paper's ASD. ASD varies enough across seeds of one prompt to rank them by quality (p. 6, p. 7), so the length alone carries seed-level information. An encoder fed raw paths can separate them by length and by seed before it learns the concept. Normalising each step's $d_t$ to unit length, or regressing length out, removes that route. Normalising also drops the quality signal.
- "Sum lands on joint" (inferred): compare the product's path length $\sum_t \lVert d_t(a) + d_t(b) \rVert^2$ with the joint's $\sum_t \lVert d_t(\text{joint}) \rVert^2$ per seed before any encoder exists. It is free on cached paths and tells whether product and joint differ in length as well as direction.
- "Gap predicts the term" (inferred): the paper's finding that low-ASD seeds are degenerate (p. 5) means the joint-minus-product gap may be largest on seeds where one path is near zero. Stratifying the gap test by path length would keep those seeds from dominating.
- Which steps (inferred): the first 20 of 32 steps carry all of the ASD ranking on ImageNet (p. 7), and counting peaks at the first 10 of 50 steps on GenEval (p. 8). This supports the loop's choice of steps 0 to 15 for the score path.
- Separate-experts paper (inferred): ASD-based seed selection is a no-training baseline that raises GenEval two-object from 0.354 to 0.518 on SD1.5 by picking 4 of 50 seeds (p. 8). A separate-experts method that guides one seed should report against picking seeds by ASD at matched compute. ASD on a product of experts can be computed from $\epsilon_{\text{PoE}} - \epsilon_\varnothing$ with no change to the code.
- CO3 paper (inferred): CFG-Rejection never touches the sampler, so it stacks on top of any sampler. Running it on CO3 trajectories would show whether CO3's failures are low-ASD seeds that early rejection removes. The counting drops (p. 8, p. 21) warn that ASD selection can hurt one composition category while helping another.

## Closer work it cites

None. The cited inference-time alignment work (Best-of-N, DNO, ReNO, loss-guided diffusion, p. 4) selects or steers with external rewards on finished images. None of it trains an encoder or classifier on noise predictions, score fields or trajectories.
