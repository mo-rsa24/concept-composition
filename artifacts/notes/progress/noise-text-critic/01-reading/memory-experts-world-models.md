# Composition of Memory Experts for Diffusion World Models

[ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/19eae75beed66321d62272e794a9c2ac-Paper-Conference.pdf), read from the full PDF (27 pages, appendix included). Printed on the PDF as a conference paper at ICLR 2026.

## What it does

The paper combines several video diffusion models at sampling time, each holding the past in a different way: recent frames, older frames stored in LoRA weights by fine-tuning at test time, and camera pose (pp. 5 to 6). Each conditional expert is raised to a power above 1 and multiplied by its own baseline raised to a negative power, which the paper calls a product of contrastive experts (Eq. 3, p. 4). The aim is to suppress modes that a plain product of experts would keep.

## What the model or score reads

Each expert is a video diffusion model that reads noisy frames $x_t$ plus its own context: 1 to 3 recent frames, a window of 10 to 100 frames, weights fine-tuned on 100 to 1000 frames, or spatial signals such as camera poses (pp. 5 to 6). On Memory Maze the input is 64 by 64 video through a VAE with 2× spatial downsampling (p. 18). RealEstate10K is resized to 256 by 256 (p. 16). No expert reads text, and no text and image score appears in the method.

## What it trained on

Memory Maze uses 30,000 trajectories of 1,000 frames each (p. 16). The base DiT trains for 150k steps at batch size 16 and predicts 17 frames from 3 context frames (p. 18). RealEstate10K reuses a pretrained model (p. 19). The long-term expert is fine-tuned at test time on the current episode with the diffusion loss, updating only LoRA adapters while the backbone stays frozen (p. 16). No labels or sentence pairs are involved.

## How its score or gradient enters

The contrastive product is:

$$
p(x \mid \mathcal{M}) \propto \prod_{i=1}^{K} \tilde p_i(x), \qquad \tilde p_i(x) \propto p_i(x)^{\alpha_i}\,\bar p_i(x)^{1-\alpha_i}
$$

This is Eq. 3 (p. 4). $p_i$ is expert $i$ conditioned on its context, $\bar p_i$ is that expert's baseline and $\alpha_i > 1$ is the contrast strength. The four-expert form is Eq. 4 (p. 6):

$$
p_{\text{CoME}} \propto \big[p_\theta(x|\varnothing)^{1-\alpha_0} p_\theta(x|c)^{\alpha_0}\big]\big[p_\phi(x|\varnothing)^{1-\alpha_1} p_\phi(x|c_{\text{ST}})^{\alpha_1}\big]\big[p_{\psi(\varnothing)}(x|c)^{1-\alpha_2} p_{\psi(c_{\text{LT}})}(x|c)^{\alpha_2}\big]\big[p_\lambda(x|\varnothing)^{1-\alpha_3} p_\lambda(x|\mathcal{S})^{\alpha_3}\big]
$$

At each noise level the sampler uses the summed score (Eq. 6, p. 15):

$$
\nabla_{x_t} \log p_{\text{CoM}}(x_t) = \sum_k \big[\alpha_k \nabla_{x_t} \log p_k(x_t) + (1-\alpha_k)\nabla_{x_t} \log \bar p_k(x_t)\big]
$$

**What plays the negative expert**

Each expert's own baseline is its negative expert, and its exponent $1-\alpha$ is negative (p. 4). The paper uses one $\alpha$ in the range 2 to 3 for all experts, so the negative weight is $-1$ to $-2$ (p. 18).

| Expert | Positive term | Negative term | Page |
|---|---|---|---|
| Pretrained prior | $p_\theta(x \mid c)$ | $p_\theta(x \mid \varnothing)$, same network with context dropped | p. 6 |
| Short-term memory | $p_\phi(x \mid c_{\text{ST}})$ | $p_\phi(x \mid \varnothing)$, same network with context dropped | p. 5 |
| Long-term memory | $p_{\psi(c_{\text{LT}})}(x \mid c)$, fine-tuned weights | Eq. 4 prints $p_{\psi(\varnothing)}(x \mid c)$, the weights before fine-tuning with the same frames. The text on p. 5 says $p_{\psi(\varnothing)}(x \mid \varnothing)$ | p. 5, p. 6 |
| Spatial memory | $p_\lambda(x \mid \mathcal{S})$ | $p_\lambda(x \mid \varnothing)$, same network with the spatial signal dropped | p. 6 |

Each bracket can be rewritten as $\bar s_k + \alpha_k(s_k - \bar s_k)$, which is classifier-free guidance applied to one expert (inferred). The negative is always "this model without its condition". It is never a specific wrong condition.

The paper prints no division by the number of experts in Eq. 6 and no cap (pp. 15 to 16). No classifier or text gradient enters the sampler.

## When adding scores is valid

Proposition 1 (pp. 4 to 5) assumes the conditional and baseline are mixtures over the same kernels $h_k$, with limited bandwidth and disjoint supports. The contrastive expert then keeps the kernels and changes only the weights:

$$
\tilde p_i(x) \propto \sum_{k=1}^{M} (\pi_k^i)^{\alpha_i} (\omega_k^i)^{1-\alpha_i} h_k(x)
$$

Modes where the conditional weight exceeds the baseline weight, $\rho_k = \pi_k^i / \omega_k^i > 1$, are amplified and modes with $\rho_k < 1$ are suppressed (p. 14). Removing spurious modes while keeping real secondary ones needs a margin: some $\tau > 1$ with $\rho_k \ge \tau$ for real modes and $\rho_k \le \tau^{-1}$ for spurious ones (p. 15). Valid secondary modes with small weights are also reduced (p. 5). The method assumes each expert is plausible and reasonably calibrated, and fusion degrades when one expert's output looks implausible to the others (p. 21).

Three limits follow from the printed mathematics (all inferred):

- Proposition 1 needs disjoint kernels, while Eq. 6 is applied at every noise level, where Gaussian noise makes the kernels overlap. The proposition therefore does not cover the sampling path.
- Proposition 1 covers one contrastive expert, whose exponents sum to 1. The product over $K$ experts has total exponent $K$ on a shared kernel, so a Gaussian kernel's variance shrinks by $1/K$. That is the same narrowing the paper uses to reject tempering (p. 4).
- A mode with zero weight in the base stays at zero under reweighting, so in this regime a contrastive term can choose among existing modes and cannot create one.

## Which of the five properties it bears on

- **Counting:** not tested.
- **Attribute binding:** not tested.
- **Spatial relation:** not tested. Spatial memory concerns camera pose and scene consistency in video (p. 6).
- **Position and size:** not tested.
- **Prompt match:** not tested. The method uses no prompts.

## Unseen concepts

This paper does not test held-out concepts. Its evaluations score video prediction and recall on generated test videos (p. 8, p. 18).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page and table |
|---|---|---|---|---|---|
| LPIPS perceptual distance (lower is better) | 0.097 | 0.113, full attention | Memory Maze | DiT | p. 7, Table 1 |
| SSIM structural similarity, 0 to 1 (higher is better) | 0.892 | 0.859, full attention | Memory Maze | DiT | p. 7, Table 1 |
| PSNR in dB (higher is better) | 23.07 | 22.78, full attention | Memory Maze | DiT | p. 7, Table 1 |
| LPIPS, all experts, with contrast against without (lower is better) | 0.097 | 0.192, same experts without contrast | Memory Maze | DiT | p. 9, Table 4 |
| Absolute trajectory error, units not printed (lower is better) | 0.96 | 1.13, NWM | RECON | CDiT | p. 8, Table 2 |
| Relative pose error, units not printed (lower is better) | 0.28 | 0.35, NWM | RECON | CDiT | p. 8, Table 2 |
| LPIPS on backward recall (lower is better) | 0.359 | 0.400, HG-t | RealEstate10K | UViT | p. 8, Table 3 |
| Tiles recalled, percent (higher is better) | 79.2% | 13%, standard diffusion | Memory Cards | U-Net | p. 17 |

Three printed details disagree. The long-term negative differs between Eq. 4 and p. 5, as shown in the table above. The base LPIPS is 0.209 in Table 1 (p. 7) and 0.200 or 0.203 in Table 4 (p. 9). The text on p. 9 says stacking experts without contrast fails to beat the base model, yet every row of the no-contrast column in Table 4 (0.170 to 0.192) is below its base of 0.203.

## What to copy, what to avoid

- Copy the margin test. Before pushing, measure $S(\text{right}) - S(\text{wrong})$ on pictures known to be right and wrong and require a gap. CLIP's bag-of-words tendency would put a swap pair's ratio near 1, where the paper's analysis says raising the strength cannot separate modes (p. 15) (inferred).
- Copy the ablation shape of Table 4: run the push with and without the negative term. For this loop that means $S(\text{right})$ alone against $S(\text{right}) - S(\text{wrong})$ (inferred).
- Keep the loop's net exponent at zero. The right-minus-wrong push adds $+\lambda$ and $-\lambda$, so it avoids the $K$-fold sharpening of this paper's sum (inferred).
- Expect selection among what the base already draws. CO3 already draws two bodies in all 96 test pictures, so a contrastive push may choose among its arrangements, and it has no mechanism here for creating a layout the base never produces (inferred).
- Avoid citing this paper as evidence for text-based contrast, because every negative here is an unconditional or unadapted video model (inferred).

## Closer work it cites

None.
