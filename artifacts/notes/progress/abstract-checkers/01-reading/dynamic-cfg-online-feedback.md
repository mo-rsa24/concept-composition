# Dynamic Classifier-Free Diffusion Guidance via Online Feedback

arXiv 2509.16131v2, read from the full PDF (19 pages, appendix included).

## What it does

At each denoising step the method makes one conditional and one unconditional prediction, recombines them at several classifier-free guidance (CFG) scales, and scores the candidates with small trained evaluators that read noisy latents (p. 2, p. 5). It keeps the scale with the highest score, so each prompt and seed gets its own guidance schedule (p. 2). The evaluators choose a number and send no gradient into the sample.

## What the classifier or loss reads

All five evaluators read the noisy diffusion latent $x_t$ directly (p. 2, p. 4). None reads UNet features, cross-attention maps or a decoded image in the proposed method; the paper states that a latent evaluator needs no VAE decoding at each step (p. 16).

The shared vision encoder is a CLIP ViT-B whose pixel embedding layer is replaced by a randomly initialised latent one, and it is conditioned on timestep $t$ (p. 4). On the Stable-Diffusion-like model (LDM) it is a ViT-B/4 that turns the latent of a 512 by 512 image into 256 tokens (p. 15). The text encoder is BERT-Base, and the two encoders total 194M parameters (p. 15).

| Evaluator | Reads the prompt? | Output | Page |
|---|---|---|---|
| Alignment | Yes, through the CLIP text encoder | Similarity of noisy-latent and text embeddings | p. 4 |
| Visual quality | No, scored "independently of $c$" | Log odds from a real-or-generated head | p. 4 |
| Reward (human preference) | Yes | CLIP similarity tuned on preference pairs | p. 4, p. 5 |
| Text rendering | Yes, through a multimodal head on both encoders | Predicted OCR-based score | p. 5 |
| Numerical reasoning | Yes | CLIP similarity tuned on count captions | p. 5 |

Equation 6 writes the score as $e_t(x_t,c)$ with no $s$ on the right (p. 5). Figure 1 shows each scale producing its own candidate next latent, and those candidates are what get scored (p. 2). The text does not say whether the score uses the next latent or another estimate.

## What it trained on

Every evaluator is trained on noisy latents, and a separate set is trained for each model family (p. 4, p. 6).

| Evaluator | Data and labels | Loss | Length | Page |
|---|---|---|---|---|
| Alignment | WebLI image-text pairs, encoded to latents and noised on a schedule similar to diffusion training | CLIP contrastive | 90k steps, batch 512, base learning rate $5\times10^{-5}$, 64 TPUv5e chips for 1.5 days | p. 4, p. 15 |
| Visual quality | A "small set" of real and generated MS-COCO images | Real-or-generated classification (loss not printed) | About 10k steps from the alignment model | p. 4, p. 15 |
| Reward | Pairs of generated images for one prompt with human preference labels | Bradley-Terry | About 10k steps | p. 4, p. 15 |
| Text rendering | Generated images labelled with OCR-model scores | Mean squared error | About 10k steps | p. 5, p. 15 |
| Numerical reasoning | 100K image-text pairs from WebLI-100B, filtered to countable entities and re-captioned by Gemini 2.5 Pro to describe object counts | CLIP contrastive | About 10k steps | p. 5, p. 15 |

The reward and text-rendering evaluators showed useful signal only for $t < t_{min}+\frac13(t_{max}-t_{min})$, the least noisy third of the timestep range, so their loss is weighted (p. 15):

$$
w_{loss}(t)=\begin{cases}0.05 & \text{if } t>t_{min}+\frac13(t_{max}-t_{min})\\[4pt] 0.05+0.95\cdot\dfrac{e^{k(t-\alpha)/\beta}-1}{e^{k}-1} & \text{otherwise}\end{cases}
$$

Here $\alpha=\frac{2(t_{max}-t_{min})}{3}$, $\beta=\frac{t_{max}+2t_{min}}{3}$ and $k=5$ (p. 15). The text says this weight reaches 1 at $t=t_{min}$ (p. 15). With $t_{min}=0$ the printed formula gives about 0.044 there, and it gives 1 only if $t$ in the formula is read as $t_{max}-t$ (inferred).

## How its gradient enters

The proposed method uses no evaluator gradient. The evaluator output changes guidance only by choosing the scale $s$ in ordinary CFG (p. 5):

$$
\epsilon_\theta(x_t\mid c)=\epsilon_\theta(x_t\mid\emptyset)+s\,\bigl(\epsilon_\theta(x_t\mid c)-\epsilon_\theta(x_t\mid\emptyset)\bigr)\tag{5}
$$

$$
\hat{s}_t=\arg\max_{s\in W} e_t(x_t,c)\tag{6}
$$

Equation 6 is printed with $s\in W$, although the set of scales is defined as $S=\{s_1,\dots,s_n\}$ (p. 5). Several evaluators are combined with a weight that depends on how much each score changed since the previous step (p. 6):

$$
\hat{e}_t=\sum_{e\in E}\alpha_{e,t}*e_t,\qquad \alpha_{e,t}=\frac{e_t-e_{t+1}}{e_{t+1}}\tag{7}
$$

The search runs at every denoising step (p. 5). LDM searches the scales $\{1, 3, 7.5, 11, 15\}$, with a best fixed scale of 7.5 (p. 15). Imagen 3 uses "a larger set", whose values are not printed (p. 16). The paper describes no normalisation or cap. The models are LDM (865M and 2B parameters) and Imagen 3 (p. 6).

The weight in Equation 7 has no absolute value and is undefined when $e_{t+1}=0$, and the first step has no $e_{T+1}$ (inferred). Equation 2 defines the visual-quality score as $-\log\frac{p(x_t\mid t)}{1-p(x_t\mid t)}$ with $p$ the probability of being real (p. 4). Maximising that as printed would prefer less real samples, so the sign is probably a misprint (inferred).

The gradient-guidance baselines in Table 2 are labelled as alignment guidance after Nichol et al. and visual-quality guidance after Kim et al. (p. 8). Their update equation and scales are not printed.

## Unseen concepts

This paper does not test held-out concepts. Evaluators are trained on web or COCO data and scored on prompt benchmarks (p. 6, p. 15). No category is withheld from training, and overlap between training and test concepts is not measured. Moving to Imagen 3 means training new evaluators on its latent space (p. 6).

## Word leakage and confounds

Four of the five evaluators receive the prompt through the text encoder, so their score can rise because the latent matches the prompt's words (p. 4, p. 5). The paper does not test or discuss this. The visual-quality evaluator is the only one that does not read the prompt (p. 4).

The method limits what a confounded evaluator can do, because it can only choose among CFG scales (inferred). CLIP-only selection still produced a "pasted-together" artifact in the LDM examples (p. 18).

## Numbers

Gecko score is an alignment score where higher is better. FID is a distance to real images where lower is better. Win rate is the percentage of side-by-side human comparisons won against default Imagen 3; ties are considered, but the formula is not printed (p. 6, p. 16).

| Metric (scale, direction) | This method | Best baseline or comparison | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Gecko score, higher is better | 47.2 (alignment + visual quality, adaptive) | 47.0 annealing; 43.8 default CFG | Gecko prompts | LDM 2B | p. 8, T2 |
| FID, lower is better | 24.8 (same method) | 25.5 gradient guidance; 25.6 default CFG | MS-COCO prompts | LDM 2B | p. 8, T2 |
| Gecko score with evaluator gradients added | Not this method | 46.1 CLIP alone; 44.6 discriminator alone; 45.3 both | Gecko prompts | LDM 2B | p. 8, T2 |
| Human win rate, %, higher is better | 53.6 (alignment + reward) | 46.4 annealing | Gecko | Imagen 3 | p. 9, T3 |
| Human win rate, % | 53.8 (alignment + reward) | 34.4 annealing | GenAI-Bench | Imagen 3 | p. 9, T3 |
| Human win rate on text, % | 55.5 (text rendering + reward) | 54.7 alignment + reward | MARIO-eval | Imagen 3 | p. 9, T3 |
| Human win rate on counts, % | 54.1 (numerical + reward) | 53.8 reward alone; 52.2 numerical alone | GeckoNum | Imagen 3 | p. 9, T3 |
| Gecko score after keeping the best of 4 seeds at 25% of denoising | 45.9 latent CLIP | 47.1 pixel CLIP; 42.9 no filtering | Gecko prompts | LDM 2B | p. 7, T1 |
| Sampling compute, billions of FLOPs, lower is better | 116,739 | 115,489 baseline; 493,239 pixel evaluator | LDM sampling | LDM 2B | p. 16, T5 |

The compute row is a 1.08% increase for the latent evaluator and a 327% increase for the pixel evaluator (inferred from Table 5).

## What to copy, what to avoid

- Copy the filtering test before trusting any checker: score seeds at a fixed step and measure whether the chosen seeds end up better (p. 7). For this project that means predicting a final two-subject label from steps 0 to 15. The paper's earliest check is 25% of denoising, so it does not contradict our chance-level CLIP result at step 0 (balanced accuracy 0.50) (inferred).
- Keep ranking and steering as separate tests. Pixel CLIP on a one-step denoised image ranked seeds better than latent CLIP (47.1 against 45.9, p. 7), while our checker read through the decoded estimate changed texture and left layout alone. A checker that ranks well may still steer badly (inferred).
- Avoid assuming that adding gradients helps. CLIP plus discriminator gradients scored 45.3 on Gecko against 46.1 for CLIP alone, with FID 25.5 against 25.6 (p. 8). This is the paper's closest test of our summed-gradient design, and its scales are not printed (inferred).
- Avoid assuming a count-specific evaluator beats a general one. On GeckoNum the numerical evaluator alone won 52.2% and general reward alone won 53.8%. Adding reward to numerical reached 54.1% (p. 9).
- Treat fine properties as late signals until shown otherwise. Reward and text evaluators had no useful signal above one third of the timestep range (p. 15). Our push runs on steps 0 to 15 of 50, which lies above that point if steps are evenly spaced, so a binding checker needs its own signal-by-step curve (inferred).
- Test a lower early push for counting. The paper reports GeckoNum benefiting from lower guidance early in generation (p. 9, p. 11), and our cap of 0.4 of PoE's norm acts on exactly those steps (inferred).
- Use scale search as a baseline. Searching PoE or CFG weights with the same checkers needs no backpropagation. It can only move along directions the predictions already contain, so it cannot add a separation PoE lacks (inferred).

## Closer work it cites

None.
