# Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis

arXiv 2411.16503, read from the full PDF (15 pages, appendix included).

## What it does

The method generates a whole image, asks a large vision-language model whether the image shows the prompt, and uses that probability to change the starting noise (p. 2 to 3). Each update mixes the starting noise with fresh Gaussian noise, and an approximate gradient picks which of 50 random noises to mix in (p. 4 to 5). After 50 rounds it returns the image with the highest score (Algorithm 1, p. 5).

**Plain answer on the two questions for this loop.** The score does not read a noisy input. It reads the finished, decoded image after all denoising steps (Eq. 5, p. 3). Layout does change in the figures, but the change comes from replacing part of the noise. In the paper's own comparison on one prompt, the two methods that follow the gradient directly (PGD and Mean-Variance) "show limited changes to the image structure" (p. 7, Fig. 6).

## What the model or score reads

- **Input to the score:** the decoded image $I = D(\Omega(z_T, C, \emptyset))$, where $\Omega$ is the full DDIM denoising chain from the starting noise $z_T$ and $D$ is the VAE decoder (Eq. 5, p. 3).
- **Question:** "Does this figure show '{prompt}'? Please answer yes or no." (p. 3).
- **Score:** $VQA(I, \text{prompt}) = P(\text{"Yes"} \mid I, \text{prompt})$, a probability from 0 to 1 (Eq. 4, p. 3).
- **Model:** CLIP-FlanT5-11B by default. InstructBLIP-FlanT5, LLaVA-1.5 and ShareGPT4V are also tried (p. 5).
- **Noisy input:** none. The vision-language model only ever sees a final image.

## What it trained on

Nothing is trained. The diffusion model and vision-language model are both used as supplied (p. 5, 9).

## How its score or gradient enters

The gradient is approximated by treating each noise prediction as a constant, so the Jacobian of the whole denoising chain collapses to one scalar (p. 4):

$$
\nabla_{z_T} s(z_T) = \sqrt{\frac{1}{\alpha_T}} \, \frac{\partial s(z_T)}{\partial I} \frac{\partial I}{\partial z_0}
$$

This is Eq. 9 (p. 4). The gradient at the final latent $z_0$ is therefore used as if it were the gradient at the starting noise (inferred). The paper does not compare this approximation with the true gradient.

The update mixes in new noise $\sigma \sim N(0, I)$ with a step size set by the current score (Eq. 10 and 11, p. 4):

$$
z'_T = \sqrt{1 - \gamma} \, z_T + \sqrt{\gamma} \, \sigma, \qquad \gamma = 1 - \sqrt{s(z_T)}
$$

The gradient chooses $\sigma$ among $N = 50$ random candidates. With $v_i = (\sqrt{1 - \gamma} - 1) z_T + \sqrt{\gamma} \sigma_i$, the chosen noise is (Algorithm 1, p. 5):

$$
\sigma = \arg\max_{i \in [1, \dots, N]} \nabla_{z_T} s(z_T) \, v_i / \|v_i\|_2^2
$$

The update happens only at the starting noise, with no change at intermediate sampling steps. The settings are $T = 50$ denoising steps, $M = 50$ optimisation rounds and $N = 50$ candidates (p. 5). There is no norm cap beyond the variance-preserving mix. Models are Stable Diffusion 1.4, 1.5, 2.0 and 2.1 (p. 5), and SDXL is not tested. One round costs 6.71 s on top of 6.08 s per image (p. 5), so 50 rounds add about 336 s per image (computed from those figures).

The paper's condition for a score increase is $\nabla_{z_T} s(z_T) v / \|v\|_2^2 \ge c/2 + \delta$, where $c$ bounds the Hessian of the score (Eq. 14 and 15, p. 5). The constant $c$ is never estimated, so the condition cannot be checked on a real run (inferred).

## When adding scores is valid

This paper does not address it.

## Which of the five properties it bears on

**Counting.** Not tested. Prompts name two things, and no count is measured.

**Attribute binding.** Shown only in pictures. The 276 simple prompts include colour pairs such as "a brown bowl and a green clock" (p. 4, 13), and the text says the method renders colours more accurately than InitNo (p. 6). No binding metric is reported.

**Spatial relation.** Shown only in pictures plus the overall VQA score. The 100 complex prompts use "on", "under", "above" and "below" (p. 5, 13 to 14). Fig. 4 shows, for example, an egg moved onto a rabbit (p. 6). No relation accuracy is reported, and the same VQA score is both the target and the main measure.

**Position and size.** Not tested.

**Prompt match.** Measured by VQA score and CLIP score (p. 6 to 8, Fig. 5 and Fig. 7).

## Unseen concepts

This paper does not test held-out concepts.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| VQA score after 10 rounds (0 to 1, higher better, the quantity being optimised) | 0.9352 | 0.8717 InitNo | 276 simple prompts, 25 seeds each | SD1.4 | p. 6 |
| VQA score after 50 rounds (same scale) | 0.9790 | InitNo lower by 0.1073 | 276 simple prompts, 25 seeds each | SD1.4 | p. 6 |
| VQA score after 10 rounds (same scale) | 0.8791 | InitNo lower by 0.1144 | 100 complex prompts, 25 seeds each | SD1.4 | p. 6 to 7 |
| VQA score after 50 rounds (same scale) | 0.9578 | InitNo lower by 0.1930 | 100 complex prompts, 25 seeds each | SD1.4 | p. 6 to 7 |
| Added time per optimisation round (seconds, lower better) | 6.71 | n/a | one RTX A6000 | SD1.4 | p. 5 |

## What to copy, what to avoid

- Copy: compare a direct gradient against a random-noise control on the same seeds. Fig. 6 is the only place the paper separates gradient following from noise replacement, and it covers one prompt (p. 7) (inferred).
- Take as a warning: PGD on the starting noise leaves the picture almost unchanged over 50 rounds, and Mean-Variance degrades quality by round 50 (Fig. 6, p. 7). This matches our CLIP checker, which raised its score by 1 to 13.5 logits over a random push without moving the animals (inferred).
- Avoid: reading the gain as gradient steering. The update replaces a share $\gamma$ of the noise, and the algorithm keeps the best of up to 51 images (p. 4 to 5). Random Sampling and Random Diffusion also change the layout (Fig. 6, p. 7). No table gives their average scores, so the gain from the gradient choice over best-of-N search is unmeasured (inferred).
- Avoid: judging by the score that was optimised. VQA is both target and main result, and CLIP score is the only other number (p. 6) (inferred).
- Be careful with the distribution claim. Eq. 10 is said to keep $z'_T \sim N(0, I)$ because $\sigma$ is Gaussian (p. 4), but the chosen $\sigma$ is an argmax that depends on $z_T$, so it is neither independent of $z_T$ nor exactly Gaussian (inferred).
- Does not transfer directly: a score on the finished image cannot act during early sampling, which is where this loop needs the push (inferred).

## Closer work it cites

- InitNo, "Initno: Boosting text-to-image diffusion models via initial noise optimization", CVPR 2024, no arXiv id printed (p. 9, reference 12). It optimises the starting noise using attention maps (p. 1).
- Attend-and-Excite, "Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models", ACM TOG 2023, no arXiv id printed (p. 9, reference 5). It updates the latent at each denoising step from cross-attention between words and image (p. 1, 3).
