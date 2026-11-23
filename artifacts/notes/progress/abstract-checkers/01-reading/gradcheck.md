# GradCheck: Analyzing classifier guidance gradients for conditional diffusion sampling

arXiv 2406.17399, read from the full PDF (9 pages, appendix included).

## What it does

The paper compares classifier guidance from a classifier trained on noisy images ("robust") with one trained only on clean images ("non-robust") on a small synthetic dataset (p. 2 to 3). It tracks how much the guidance direction changes from one sampling step to the next, then tests two fixes: reading the classifier through the one-step denoised estimate $\hat{x}_0$, and smoothing the gradient with ADAM (p. 2 to 4).

## What the classifier or loss reads

Both classifiers are MobileNetV3 small models with 1.5M parameters (p. 2, p. 6). The input is either the noisy image $x_t$ or the one-step estimate $\hat{x}_0^{(x_t)}$, both at 3 × 64 × 64 (p. 2, p. 6). The model is an unconditional pixel-space DDPM, so there is no text and no cross-attention (p. 2, p. 6). The paper calls $x_t$ "noisy latents", but it describes no autoencoder (p. 2, inferred).

## What it trained on

The data is SportBalls, a synthetic set of four sport-ball classes placed on a white background with random position, rotation and scale (p. 2, p. 6). The non-robust classifier trains on clean images and reaches 99.2% validation accuracy on them and 26.65% on noisy images (p. 2). With four classes, 26.65% is close to the 25% chance rate (inferred).

The robust classifier trains on noisy images at uniformly sampled timesteps of the DDPM noise schedule and reaches 64.1% on noisy images and over 99% on clean ones (p. 2 to 3). Both train for up to 200 epochs with early stopping at 99% validation accuracy (p. 6). The paper reports no accuracy per timestep. The dataset size is not printed.

## How its gradient enters

The gradient shifts the DDPM mean after being rescaled to the mean's norm (p. 2):

$$
\mu_t' = \mu_t + \frac{s\,\Sigma_t(x_t)\,\lVert \mu_t \rVert_2\, g_t}{\lVert g_t \rVert_2}, \qquad g_t = \nabla_{x_t} \log p_{cl}(y \mid x_t)
$$

$\mu_t$ is the denoising mean, $\Sigma_t$ its covariance, $s$ the scale and $y$ the target class (p. 1 to 2). The scale is fixed at $s = 0.04$ for all runs, chosen from a sweep on the robust classifier (p. 3, p. 7). Guidance is applied through the 1000-step sampling process (p. 3, p. 6).

The two variants change $g_t$ (p. 2):

$$
g_t = \nabla_{x_t} \log p_{cl}\!\left(y \mid \hat{x}_0^{(x_t)}\right), \qquad \hat{x}_0^{(x_t)} = \frac{x_t}{\sqrt{\bar{\alpha}_t}} - \frac{\sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

$$
g_t = \nu\!\left(\nabla_{x_t} \log p_{cl}(y \mid x_t)\right)
$$

$\nu$ is the ADAM adjustment with $\beta_1 = 0.9$, $\beta_2 = 0.999$ and step size $\eta = 1$ (p. 2).

## Unseen concepts

This paper does not test held-out concepts. All four classes appear in classifier training, and every guided run targets the one class "baseball" (p. 6).

## Word leakage and confounds

There are no words in this setup, so word leakage does not apply (p. 2).

Three confounds limit the numbers. Accuracy is judged by the same classifier that does the guiding, so a sample can score well by exploiting that classifier (p. 3, inferred). The robust and non-robust accuracy columns therefore use different judges. Each cell also uses only 64 generated images, and FID compares those 64 with 64 training images (p. 3). One image is 1.56 accuracy points, and FID on 64 images is unreliable (inferred).

## Numbers

Table 1 (p. 3), 64 generated images per cell, target class baseball, unconditional pixel DDPM on SportBalls. Accuracy is the guiding classifier's accuracy on the target class (0 to 100%, higher is better). FID compares the generated images with target-class training images (lower is better).

| Classifier | Fix applied | Accuracy | FID | Step-to-step cosine of guidance, read from the image (Figures 1 to 3, p. 3 to 4) |
|---|---|---|---|---|
| Robust | none | 96.88% | 26.641 | about 0.8 after the first step, then 0.85 to 0.88 |
| Robust | $\hat{x}_0$ | 81.25% | 38.053 | 0.6 to 0.75 early, 0.3 to 0.5 in mid-sampling, about 0.85 at $t$ = 0 (Figure 6, p. 7) |
| Robust | ADAM | 92.19% | 26.575 | about 0.98 (Figure 6, p. 7) |
| Robust | $\hat{x}_0$ + ADAM | 78.13% | 34.484 | about 0.95 to 0.98 (Figures 3 and 6) |
| Non-robust | none | 15.63% | 126.526 | about 0.03 near $t$ = 1000, about 0.15 at $t$ = 0 |
| Non-robust | $\hat{x}_0$ | 39.06% | 103.301 | about 0.05 near $t$ = 1000, 0.4 to 0.5 from about $t$ = 750 to 300, about 0.85 at $t$ = 0 |
| Non-robust | ADAM | 15.63% | 127.526 | about 0.9 from the first steps |
| Non-robust | $\hat{x}_0$ + ADAM | 85.94% | 24.348 | about 0.95 from the first steps |

The caption of Figure 6 says "non-robust", while the text above it says the figure shows the robust classifier and its red curve matches the robust curve in Figure 1 (p. 3, p. 7). This note treats it as the robust classifier (inferred).

The scale sweep in Figure 5 has FID (lower is better) near its minimum of about 27 at $s$ = 0.03 to 0.04, about 65 at $s$ = 0.10, and above 250 from $s$ = 0.20 upward (p. 7, read from the image).

## What it concludes about classifier gradients at noisy inputs

**What the paper states**

A classifier trained only on clean images gives guidance vectors that are almost orthogonal between consecutive steps. From that, the authors infer it "has almost random gradients for most of the denoising process" (p. 3). This classifier reached 15.63% target accuracy and FID 126.526 (p. 3).

A classifier trained on noisy images keeps a step-to-step cosine of about 0.8 to 0.88 from the first steps to the last, and reaches 96.88% and FID 26.641 (p. 3, Figure 1 read from the image).

Reading a clean-trained classifier through $\hat{x}_0$ helps partly. Its cosine starts near 0.05 and only reaches about 0.85 at the end, with 39.06% accuracy (p. 3 to 4, Figure 2 read from the image). The authors tie this to the one-step estimate improving as denoising proceeds (p. 3).

ADAM alone raises the cosine to about 0.9 and leaves the result unchanged at 15.63% and FID 127.526. The authors conclude "The gradients have to be meaningful and stable over time" (p. 4).

**What the paper does not establish**

No experiment turns guidance on or off by timestep, so nothing here measures what an early-step gradient contributes to the final image (p. 3 to 4, inferred from the listed experiments).

The evidence for "random" and "meaningful" is the cosine between consecutive steps. The paper's own ADAM-only row shows a high cosine with no gain, so a steady direction does not prove useful content. With momentum 0.9, a high cosine under ADAM is expected by construction (p. 2, p. 4, inferred).

There is no accuracy broken down by timestep. The robust classifier's 64.1% is an average over all noise levels, so its accuracy at the earliest steps is unknown (p. 3).

The setting is a 64 × 64 pixel DDPM with four simple classes and one target class. It has no text, latents, classifier-free guidance or composition (p. 2, p. 6).

**What this means for our design (inferred)**

GradCheck supports one rule directly: a classifier that has never seen noisy inputs should not be trusted to give a useful gradient at those inputs. Our CLIP checker reading decoded early images at chance (balanced accuracy 0.50 at step 0) fits that rule.

It gives only indirect support for the claim our design needs. A noise-trained classifier's gradient is steady from the first steps and its full guidance run works, which makes early-step meaning plausible. Whether the early steps carry the effect remains untested here, so we have to test it with a step-window ablation.

## What to copy, what to avoid

- Copy noise-matched training: train each checker on word maps from the same timesteps where it will push (inferred).
- Copy the per-step cosine plot as a cheap diagnostic, and add a checker trained on shuffled labels as its control, because high cosine alone does not show meaning (p. 4, inferred).
- Avoid using cosine stability as the success criterion. ADAM reached about 0.9 with zero improvement (p. 4).
- Avoid adding ADAM or $\hat{x}_0$ on top of a checker trained on noisy inputs without a test. Both together cut the robust classifier from 96.88% to 78.13% (p. 3 to 4).
- Copy the norm-relative scale, which sets the push as a fraction of the model's own output as our cap at 0.4 of PoE's prediction norm does (p. 2, inferred). Sweep it finely, because FID here goes from about 27 to above 250 between $s$ = 0.04 and $s$ = 0.20 (p. 7).
- Avoid judging success with the guiding checker. Use a separate judge and more than 64 images per setting (p. 3, inferred).
- Test the early steps directly by running guidance on steps 0 to 15 only, on later steps only, and on all steps, then comparing layout outcomes (inferred).

## Closer work it cites

None.
