# Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization

arXiv 2502.01051, read from the full PDF (26 pages, appendix included). 39th Conference on Neural Information Processing Systems (NeurIPS 2025), as printed on p. 1.

## What it does

The paper turns a copy of a diffusion U-Net and its text encoder into a reward model, the Latent Reward Model (LRM), which scores a noisy latent against a prompt at any timestep (p. 2, 4 to 5). At each denoising step the LRM ranks four candidate next latents, and the best and worst become a training pair (p. 6). Those pairs fine-tune a LoRA on the diffusion model, a method the paper calls Latent Preference Optimization (LPO) (p. 6, 13).

**Plain answer on the two questions for this loop.** The score does read a noisy input at the earliest sampling steps, up to t = 950 of 1000 (p. 6 to 7). No gradient of the score is ever taken with respect to the latent. The score only picks winners, so this paper has no evidence about what a score gradient does to layout. The fine-tuned model does change pose and framing when the highest-noise steps are included (Fig. 10, p. 21), and that change comes from selection plus retraining.

## What the model or score reads

- **Image side:** the noisy latent $x_t$ and timestep $t$ go straight through the U-Net, with no predicted clean image and no VAE decode (p. 4 to 5, Fig. 4).
- **Features:** down-block and middle-block features, each **average pooled over the spatial dimensions** (p. 4, Eq. 7 on p. 5). This removes where in the picture a feature was.
- **Text-conditioned boost:** the middle-block feature is pushed away from its null-prompt version, $V_{enh} = V_{mid} + (g_s - 1)(V_{mid} - V_{mid\_ucond})$, with $g_s = 7.5$ (Eq. 8, p. 5; p. 7).
- **Text side:** only the last token of the prompt, passed through a projection layer (p. 4). For SDXL only the OpenCLIP ViT-bigG encoder is used (p. 13).
- **Score:** $S(p, x_t) = \tau \, (l_2(V) \cdot l_2(T))$, a scaled cosine similarity (Eq. 9, p. 5).
- **Resolution:** 512 for SD1.5 and 1024 for SDXL (p. 13).

## What it trained on

- **Data:** Pick-a-Pic v1, triples of one prompt, a preferred image and a less preferred image (p. 5). It holds 511,840 win-lose pairs and 71,907 tie pairs (p. 6).
- **Filtering:** the chosen rule keeps pairs where the winner's CLIP and VQA scores are at least the loser's and its aesthetic score is at least the loser's minus 0.5, which leaves 168,539 win-lose pairs and 8,537 ties (Table 1, p. 6).
- **Loss:** a Bradley-Terry ranking loss between $S(p, x^w_t)$ and $S(p, x^l_t)$, with both images noised to the same uniformly sampled $t$ (Eq. 10 and 11, p. 5).
- **What moves in training:** the U-Net and text encoder train, and the VAE is frozen (p. 5).
- **Length:** LRM-1.5 trains for 4,000 steps at batch 128, and LRM-XL for 8,000 steps at batch 32, both at learning rate 1e-5 (p. 13).
- **The contrast it learns:** every training comparison is **two images under one sentence** (p. 5). It never compares two sentences on one image, which is the comparison this loop's checker needs (inferred).

## How its score or gradient enters

No gradient of the score enters a sampler. The score enters through selection (p. 6). From one $x_{t+1}$, LPO draws $K = 4$ candidates $x^i_t$ and scores each. If the softmax-normalised gap between the highest and lowest score exceeds a threshold $th_t$, the highest becomes $x^w_t$ and the lowest becomes $x^l_t$ (p. 6). The threshold depends on the sampler's noise level $\sigma_t$:

$$
th_t = \frac{\sigma_t - \sigma_{min}}{\sigma_{max} - \sigma_{min}} (th_{max} - th_{min}) + th_{min}
$$

The range $[th_{min}, th_{max}]$ is $[0.35, 0.5]$ for SD1.5 and $[0.45, 0.6]$ for SDXL (Eq. 12 and p. 7). The pairs train the diffusion model through the step-level preference loss (Eq. 6, p. 4):

$$
L_{SPO} = -\mathbb{E}_{x^w_t, x^l_t \sim p_\theta(x_t | x_{t+1}, c)} \left[ \log \sigma \left( \beta \log \frac{p_\theta(x^w_t | x_{t+1}, c)}{p_{ref}(x^w_t | x_{t+1}, c)} - \beta \log \frac{p_\theta(x^l_t | x_{t+1}, c)}{p_{ref}(x^l_t | x_{t+1}, c)} \right) \right]
$$

Here $p_\theta$ is the model being tuned, $p_{ref}$ is its frozen starting copy, and $\beta = 500$ (p. 4, 7). Sampling uses DDIM with 20 steps, and all steps $t \in \{0, 50, \dots, 950\}$ are used (p. 7). The LoRA rank is 4 for SD1.5 and 64 for SDXL (p. 13). Models tested are SD1.5, SD2.1, SDXL and SD3-Medium (p. 7, 9, 14 to 15).

Using the LRM for gradient-based reward fine-tuning is listed as future work (p. 10).

## When adding scores is valid

This paper does not address it. The nearest statement concerns ranking under noise: the loss assumes that if one clean image beats another, it still wins after equal noise is added, and the paper says this breaks down when the winner is better in only some respects (p. 5 to 6).

## Which of the five properties it bears on

All numbers below measure images from the **fine-tuned model**. None measures a score used as a checker on a single image.

**Counting.** GenEval Counting on SDXL at 20 steps rises from 30.94% to 43.75%, and Diffusion-DPO reaches 45.00% (Table 25, p. 18). T2I-CompBench++ Numeracy rises from 0.4874 to 0.5493 (Table 3, p. 8).

**Attribute binding.** GenEval Color Attribution on SDXL rises from 19.00% to 27.50%, below InterComp at 31.75% (Table 25, p. 18). T2I-CompBench++ Color rises from 0.5833 to 0.7351 (Table 3, p. 8).

**Spatial relation.** T2I-CompBench++ 2D-Spatial on SDXL rises from 0.1936 to 0.2414, and InterComp reaches 0.2406 (Table 3, p. 8). The absolute score stays low even though the features are spatially pooled.

**Position and size.** GenEval Position on SDXL rises from 9.25% to 14.00%, tied with InterComp (Table 25, p. 18). Object size is not tested.

**Prompt match.** GenEval Overall on SDXL rises from 49.40 to 59.27 (Table 2, p. 8). In the human study the paper says LPO's alignment advantage "is not particularly pronounced" (p. 13).

## Unseen concepts

This paper does not test held-out concepts. Its evaluation uses 500 unique validation prompts from Pick-a-Pic, and no concept is withheld from training (p. 7).

## Numbers

| Metric (scale, direction) | LPO | Best baseline | Original | Dataset | Model | Page, table |
|---|---|---|---|---|---|---|
| 2D-Spatial (0 to 1, higher better) | 0.2414 | 0.2406 InterComp | 0.1936 | T2I-CompBench++ | SDXL | p. 8, Table 3 |
| Numeracy (0 to 1, higher better) | 0.5493 | 0.5395 InterComp | 0.4874 | T2I-CompBench++ | SDXL | p. 8, Table 3 |
| Color (0 to 1, higher better) | 0.7351 | 0.7218 InterComp | 0.5833 | T2I-CompBench++ | SDXL | p. 8, Table 3 |
| Position (% correct, higher better) | 14.00 | 14.00 InterComp | 9.25 | GenEval, 20 steps | SDXL | p. 18, Table 25 |
| Color Attribution (% correct, higher better) | 27.50 | 31.75 InterComp | 19.00 | GenEval, 20 steps | SDXL | p. 18, Table 25 |
| Counting (% correct, higher better) | 43.75 | 45.00 Diffusion-DPO | 30.94 | GenEval, 20 steps | SDXL | p. 18, Table 25 |
| GenEval Overall when tuning only t in [750, 950] (%, higher better) | 48.28 | 48.39 for all of [0, 950] | 42.56 | GenEval | SD1.5 | p. 10, Table 9 |
| GenEval Overall when tuning only t in [0, 200] (%, higher better) | 40.11 | 48.39 for all of [0, 950] | 42.56 | GenEval | SD1.5 | p. 10, Table 9 |
| GenEval Overall with the LRM told t = 0 at every step (%, higher better) | 32.21 | 48.39 with the real t | 42.56 | GenEval | SD1.5 | p. 17, Table 22 |
| Preference accuracy on clean or lightly noised images (%, higher better) | 67.44 at t = 0, 69.31 at t = 200 | 71.93 PickScore | n/a | Pick-a-Pic v1 validation and test | LRM-XL | p. 16, Table 18 |

The paper reports the reward model's direct accuracy only at t = 0 and t = 200 (Table 18, p. 16). It gives no direct accuracy at high noise, so its reading quality at the earliest steps is supported only by downstream results.

## What to copy, what to avoid

- Copy: pass the true timestep to any U-Net-based checker. Telling the LRM t = 0 throughout drops GenEval Overall from 48.39 to 32.21 (p. 17) (inferred).
- Copy: put the effort on the first steps. Tuning only the first 5 of 20 steps recovers 48.28 of the full 48.39 and visibly changes pose and framing (p. 10, 21). This agrees with our checker reading the pair from step 0 at balanced accuracy 0.63 (inferred).
- Avoid: spatial average pooling for "a cat to the left of a dog" against its swap. A pooled feature cannot directly say which side an object is on (inferred).
- Avoid: expecting this training to rank two sentences on one image. It ranks two images under one sentence, so our sentence-pair contrast needs its own negatives (inferred).
- Avoid: a last-token text summary for binding pairs, because it adds to the bag-of-words risk already measured for CLIP (inferred).
- Avoid: reading this paper as support for a score gradient. Its layout change came from selecting among sampled candidates and retraining, while our CLIP gradient raised its score without moving the animals (inferred).
- Watch for reward hacking: the paper reports the metric rising while image quality falls under long training (p. 19).

## Closer work it cites

- Step-by-step Preference Optimization (SPO), "Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization", arXiv 2406.04314 (p. 10, reference 8). It trains a step-aware preference model from PickScore that scores intermediate steps through predicted clean images (p. 1, 4).
- ReNO, "Enhancing one-step text-to-image models through reward-based noise optimization", NeurIPS 2024, no arXiv id printed (p. 12, reference 33). The title describes optimising the noise against a reward model, and this paper lists it among training-free preference methods (p. 3).
- "Test-time alignment of diffusion models without reward over-optimization", ICLR 2025, no arXiv id printed (p. 12, reference 34). This paper lists it among training-free preference methods (p. 3).
