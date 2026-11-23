# Universal Guidance for Diffusion Models

arXiv 2302.07121, read from the full PDF (15 pages, appendix included). No venue is printed on the PDF.

## What it does

Universal Guidance steers a frozen diffusion model with off-the-shelf networks trained on clean images, such as CLIP, a segmenter, a face recognizer or an object detector (p. 1, p. 2). It evaluates those networks on the predicted clean image instead of the noisy latent (p. 2, p. 4). Two additions support this: backward guidance optimizes a clean-image change, and self-recurrence re-noises and repeats each step (p. 4, p. 5).

## What the classifier or loss reads

The guidance function $f$ reads the one-step clean estimate from Eqn. 3 (p. 2, p. 4):

$$
\hat z_0=\frac{z_t-\sqrt{1-\alpha_t}\,\epsilon_\theta(z_t,t)}{\sqrt{\alpha_t}}
$$

On Stable Diffusion, $\hat z_0$ is a latent and is passed through the image decoder to give a predicted clean image before $f$ reads it (p. 5).

| Guidance function $f$ | Loss $\ell$ | Page |
|---|---|---|
| CLIP image encoder | Negative cosine to a text embedding | p. 6 |
| MobileNetV3-Large segmentation | Summed per-pixel cross-entropy | p. 6 |
| MTCNN crop, then FaceNet | L1 between face embeddings, with the crop treated as fixed | p. 7 |
| Faster R-CNN, ResNet-50-FPN | Anchor classification, box regression and region label losses | p. 7 |
| CLIP image encoder on a style image | Negative cosine between image embeddings | p. 8 |

## What it trained on

Nothing is trained (p. 2). The guidance networks are pre-trained and the diffusion models stay frozen (p. 2, p. 3). The hand-written losses are listed above.

## How its gradient enters

**What it says about classifiers on noisy inputs.** The paper names "the domain shift between the noisy images used by the diffusion sampling process and the clean images on which the guidance models are trained" as the difficulty (p. 2). It states that a clean-trained $f$ "fails to provide meaningful guidance when the input is noisy" (p. 4). It cites GLIDE's CLIP, retrained on noisy inputs, as a way to close the gap at high cost (p. 2). The paper does not measure this failure. It reports no accuracy of any $f$ on noisy inputs or on $\hat z_0$ by step. The usefulness of $\hat z_0$ is described as "an empirical observation" (p. 3).

**Forward guidance (Eqn. 6, p. 4).** The gradient flows through $\hat z_0$ and the denoiser back to $z_t$:

$$
\hat\epsilon_\theta(z_t,t)=\epsilon_\theta(z_t,t)+s(t)\cdot\nabla_{z_t}\ell\big(c,f(\hat z_0)\big)
$$

Here $c$ is the target prompt and $s(t)$ is the strength. Most experiments use $s(t) = w\sqrt{1-\alpha_t}$ with a hand-picked $w$ (pp. 6 to 9).

**Backward guidance (Eqns. 7 to 9, p. 4).** Find a clean-space change by $m$ steps of gradient descent from $\Delta = 0$:

$$
\Delta z_0=\arg\min_{\Delta}\ \ell\big(c,f(\hat z_0+\Delta)\big)
$$

Then convert that change to a noise prediction that satisfies $z_t=\sqrt{\alpha_t}(\hat z_0+\Delta z_0)+\sqrt{1-\alpha_t}\,\tilde\epsilon$:

$$
\tilde\epsilon=\epsilon_\theta(z_t,t)-\sqrt{\frac{\alpha_t}{1-\alpha_t}}\,\Delta z_0
$$

Backward guidance exists because forward guidance can favour realism over matching the prompt, while a larger $s(t)$ causes instability (p. 4). In Algorithm 1 it is applied on top of the forward-guided prediction (p. 4).

**Per-step self-recurrence (Eqn. 10, p. 5).** After sampling $z_{t-1}$, fresh noise returns it to level $t$:

$$
z_t'=\sqrt{\alpha_t/\alpha_{t-1}}\;z_{t-1}+\sqrt{1-\alpha_t/\alpha_{t-1}}\;\epsilon',\qquad \epsilon'\sim\mathcal N(0,I)
$$

The step repeats $k$ times before moving to $t-1$ (p. 5, Algorithm 1 on p. 4). The authors say this explores different regions at the same noise level and keeps images realistic (p. 5). Run time grows linearly with $k$ (p. 9).

**Which steps and models.** Algorithm 1 guides every step from $T$ to 1 (p. 4). The models are Stable Diffusion and an unconditional ImageNet diffusion model (p. 5). The Stable Diffusion version and the number of sampling steps are not printed. No gradient normalization or cap is printed.

**Backward guidance was not used on Stable Diffusion.** All Stable Diffusion experiments set $m = 0$ (p. 5). Backward guidance appears only with the ImageNet model, for object location and for the inpainting term (p. 8, p. 9). The value of $m$ is not printed for those runs.

## Unseen concepts

This paper does not test held-out concepts.

The guidance networks are used as supplied, and no category split is defined for them. The ImageNet CLIP test uses prompts outside ImageNet's labels (p. 8). That tests the generator's range under CLIP guidance, with no learned checker held out from any class.

## Word leakage and confounds

**The checker's input.** Segmentation, face and detection networks read only the decoded estimate, so they do not receive the prompt words directly (p. 5 to p. 7). CLIP guidance compares against a text embedding by design (p. 6).

**Indirect path.** In most Stable Diffusion runs the text is a fixed extra input to the denoiser, so $\hat z_0$ is text-conditioned before $f$ reads it (p. 6 to p. 8). For CLIP guidance, Stable Diffusion is used as an unconditional generator (p. 6). The paper does not discuss word leakage.

**A noisy classifier is still used.** In the combined inpainting experiment the authors use the Dhariwal and Nichol classifier that accepts noisy input, with original classifier guidance (Eqn. 4), "instead of our forward guidance" (p. 9). So their one three-way combination does not rely on clean-estimate guidance for its class signal.

## Numbers

The paper reports no quantitative metric and no baseline score. All evidence is figures (p. 5 to p. 9, p. 12 to p. 15).

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| None reported | Qualitative figures only | Conditional SD shown side by side, unscored | Hand-chosen prompts | Stable Diffusion, ImageNet diffusion | pp. 5 to 9, no table |

Guidance settings, with $s(t) = w\sqrt{1-\alpha_t}$:

| Guidance | Model | $w$ | Recurrence $k$ | Backward $m$ | Page |
|---|---|---|---|---|---|
| CLIP text | Stable Diffusion | 10 | 8 | 0 | p. 5, p. 6 |
| Segmentation | Stable Diffusion | 400 | 10 | 0 | p. 5, p. 6 |
| Face | Stable Diffusion | 20000 | 2 | 0 | p. 5, p. 7 |
| Object location | Stable Diffusion | 100 | 3 | 0 | p. 5, p. 7 |
| Style | Stable Diffusion | 6 | 6 | 0 | p. 5, p. 8 |
| CLIP text, three images | ImageNet | 2, 5, 2 | 10, 5, 10 | 0 | p. 8 |
| Object location | ImageNet | 100 | 3 | 0 in the forward-only run; used in the other run, value not printed | p. 8 |
| Segmentation in inpainting | ImageNet | 200 | Not printed | Used for inpainting, value not printed | p. 9 |

## What to copy, what to avoid

- Copy self-recurrence as the cheapest next test on our decoded-image checker. We saw it change texture and leave layout alone, while this paper moves layout through a decoder with segmentation at $k = 10$ and detection at $k = 3$ (p. 6, p. 7). Fig. 2 is a single example of self-recurrence helping segmentation guidance, at $k$ = 1, 4 and 10 (p. 5). If our run used $k = 1$, a $k = 4$ test separates a weak signal from too few attempts (inferred).
- Keep the push in $s(t) = w\sqrt{1-\alpha_t}$ form, since it matches our $\sigma_t \nabla \log C$ form with $\ell = -\log C$ (inferred).
- Do not reuse a single $w$ across checkers. Values here range from 6 to 20000, and the paper says hyper-parameters must be chosen per guidance network (p. 9). Our norm cap at 0.4 of PoE's prediction is a safer common scale (inferred).
- Do not treat $\hat z_0$ as readable at the earliest steps. The paper never measures this, and our CLIP checker reads the pair at balanced accuracy 0.50 on step 0 decoded images. Measure accuracy per step before guiding through $\hat z_0$ (inferred).
- Do not assume backward guidance works on SDXL. This paper offers no latent-diffusion evidence for it because $m = 0$ on Stable Diffusion (p. 5) (inferred).
- Copy the idea of different objectives for forward and backward terms when combining checkers (p. 5), with the limit that the evidence is one qualitative figure on an ImageNet model (p. 9) (inferred).
- Budget for cost. With $k = 10$, each sampling step needs 10 denoiser evaluations, and PoE needs several predictions per evaluation (p. 9, inferred).

## Closer work it cites

**GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**, arXiv 2112.10741 (p. 2, p. 10). It is cited as using a CLIP model retrained from scratch on noisy inputs for guidance. That is a learned prompt-match checker on noisy inputs, the same property as our overall prompt match.
