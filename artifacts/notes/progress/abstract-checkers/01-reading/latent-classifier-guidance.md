# Exploring Compositional Visual Generation with Latent Classifier Guidance

arXiv 2304.12536v2, read from the full PDF (10 pages, appendix included; the PDF has no separate appendix).

## What it does

The paper trains an unconditional diffusion model over the semantic latent vector of an already-trained generator (StyleGAN2 or Diffusion Autoencoder) and adds the gradients of attribute classifiers on that vector to compose attributes with AND and NOT (p. 2, p. 3, p. 5). An extra term keeps an edited image near its source latent (p. 3). With a flat prior and linear classifiers, the update reduces to adding classifier weight vectors (p. 4).

## What the classifier or loss reads

Each classifier reads one global semantic latent vector and no text (p. 4, p. 5). For StyleGAN2 this is a $w_s$ vector from the intermediate $\mathcal{W}_s$ space, and $\mathcal{W}_s+$ is a concatenation of 18 such 512-dimensional vectors (p. 5). For Diffusion Autoencoder it is that model's semantic latent (p. 5). This latent is different from a Stable Diffusion VAE latent, and the classifier receives no spatial map, attention map or image (inferred from the input described on p. 4 and p. 5).

The classifiers are linear, $z\mapsto w^{T}z+b$ (p. 4). On StyleGAN2 the same classifiers trained on clean $w_s$ vectors are applied during sampling, although the paper says Equation 5 requires time-dependent classifiers (p. 5).

## What it trained on

A latent DDIM is trained on 100,000 $w_s$ vectors sampled from StyleGAN2's style network (p. 5). Linear classifiers are trained on $\mathcal{W}_s$ using latent-label pairs supplied by StyleFlow (p. 5). Seven attributes appear in Table 1: smile, gender, glasses, beard, hair colour, yaw and age, with test accuracy from 75.50% for hair colour to 98.13% for yaw (p. 5). For Diffusion Autoencoder the paper reuses that model's pre-trained latent diffusion model and linear classifiers (p. 5).

The classifier loss, number of labelled pairs, training length and source image dataset are not printed.

## How its gradient enters

The conjunction of attributes is written as follows (p. 3), under the stated assumption of "assuming the conditions to be independent of each other":

$$
\nabla_{z_t}\log p(z_t\mid y^1,y^2,\dots,y^n)=\nabla_{z_t}\log p(z_t)+\sum_{i=1}^{n}\alpha_t^i\,\nabla_{z_t}\log p(y^i\mid z_t)\tag{1}
$$

Negated attributes are subtracted with weights $\beta_t^i$ (p. 3):

$$
\nabla_{z_t}\log p(z_t\mid y^1,\dots,y^n)=\nabla_{z_t}\log p(z_t)+\sum_{i=1}^{m-1}\alpha_t^i\nabla_{z_t}\log p(y^i\mid z_t)-\sum_{i=m}^{n}\beta_t^i\nabla_{z_t}\log p(y^i\mid z_t)\tag{2}
$$

Editing adds a term that holds the sample near the source latent $\hat z$ (p. 3):

$$
\nabla_{z_t}\log p(z_t\mid y^1,\dots,y^n,\hat z)=\nabla_{z_t}\log p(z_t)+\sum_{i=1}^{n}\alpha_t^i\nabla_{z_t}\log p(y^i\mid z_t)+\gamma_t\nabla_{z_t}\log p(\hat z\mid z_t)\tag{3}
$$

The paper calls $\alpha$ a temperature on the classifier posterior (p. 3). The values of $\alpha_t^i$, $\beta_t^i$ and $\gamma_t$, the sampled steps, and any normalisation or cap are not printed.

### What it proves about adding log probabilities

**The claim.** Lemma 2 gives an evidence lower bound (ELBO) for one condition, and states the multi-condition form "with independent conditions" (p. 4):

$$
\mathbb{E}_{q(z_{1:T}\mid z_0)}\left[\sum_{t=1}^{T}\left[\sum_{i=1}^{n}\log p(y^i\mid z_{t-1})+\log p(\hat z\mid z_{t-1})\right]\right]+\mathcal{L}_{uncond}+C\tag{6}
$$

**What is actually derived.** The printed proof covers a single condition. The paper says the derivation "can be easily extended" to several (p. 4). The sum over classifiers in Equation 6 is therefore stated without a printed proof.

**The independence needed.** Bayes' rule gives $p(z_t\mid y^{1:n})\propto p(z_t)\,p(y^{1:n}\mid z_t)$. Equation 1 with all $\alpha_t^i=1$ needs $p(y^{1:n}\mid z_t)=\prod_i p(y^i\mid z_t)$, which is independence of the labels **given the noisy latent** (inferred). The paper says only that the conditions are independent of each other (p. 3).

**The weights leave the derivation.** With $\alpha_t^i\neq1$ the right side of Equation 1 is the score of $p(z_t)\prod_i p(y^i\mid z_t)^{\alpha_t^i}$ up to a constant, and the equality with the posterior no longer holds (inferred). Equation 6 has no weights (p. 4).

**The noisy-level sum depends on the treatment of $Z$.** The proof uses $p(z_{t-1}\mid z_t,y)=Z\,p(z_{t-1}\mid z_t)\,p(y\mid z_{t-1})$ and calls $Z$ a normalising constant (p. 4). If $y$ is independent of $z_t$ given $z_{t-1}$, then $Z=1/p(y\mid z_t)$, which depends on both $z_t$ and the classifier (inferred). Keeping it makes the classifier terms telescope:

$$
\sum_{t=1}^{T}\bigl[\log p(y\mid z_{t-1})-\log p(y\mid z_t)\bigr]=\log p(y\mid z_0)-\log p(y\mid z_T)
$$

Under that reading the bound depends on the classifier only at the two ends of the chain, so Lemma 2 gives weak support for summing noisy-level classifier logs (inferred).

**A sign misprint.** Equation 8 is printed as $\sum_i\alpha_0^iw^i+\gamma_0(z_0-\hat z)=0$, which solves to $z_0=\hat z-\frac{1}{\gamma_0}\sum_i\alpha_0^iw^i$, while Equation 7 has a plus sign (p. 4). A Gaussian $p(\hat z\mid z)$ contributes $-\gamma_0(z_0-\hat z)$, so Equation 7 is the consistent form (inferred).

**Evidence offered for independence.** Figure 4 shows pairwise correlations between linear classifiers on the Diffusion Autoencoder latent, and the paper reads them as frequently orthogonal (p. 7). Visible correlated blocks remain in that heatmap (p. 7). Table 4 reports that editing one attribute changes the measured accuracy of the others by at most 0.013 on a 0-to-1 scale (p. 8). Neither measures independence of labels given a noisy latent (inferred).

## Unseen concepts

This paper does not test held-out concepts. Adding a new attribute means training a classifier on it (p. 4), and generation of unseen classes is listed as future work (p. 8).

## Word leakage and confounds

No text enters the classifier, so word leakage does not arise (p. 4, p. 5). The confound the paper discusses is shortcut learning: an image classifier for "old" may rely on "white hair" (p. 3). It argues a compact, disentangled latent reduces this when the two semantic directions are orthogonal (p. 3), and it does not run a test isolating that shortcut.

## Numbers

FID is a distance to real images where lower is better. ACC is the attribute accuracy of an off-the-shelf image classifier on a 0-to-1 scale where higher is better. ID is face identity distance where lower is better (p. 5).

| Metric (scale, direction) | This method | Best baseline | Dataset or task | Model | Page, table |
|---|---|---|---|---|---|
| FID, gender + smile + age | 22.46 linear; 26.49 diffusion | 22.03 LACE-ODE | Compositional generation | StyleGAN2 | p. 7, T2 |
| ACC gender / smile / age | Linear 0.980 / 0.982 / 0.863; diffusion 0.981 / 0.968 / 0.863 | LACE-ODE 0.964 / 0.967 / 0.925 | Same | StyleGAN2 | p. 7, T2 |
| FID, NOT gender + smile + NOT hair colour | 23.94 linear; 29.62 diffusion | 22.86 LACE-LD | Compositional generation with negation | StyleGAN2 | p. 7, T2 |
| ACC gender / smile / hair colour, same task | Linear 0.948 / 0.995 / 0.936; diffusion 0.987 / 0.954 / 0.906 | LACE-ODE 0.649 / 0.970 / 0.935 | Same | StyleGAN2 | p. 7, T2 |
| Sequential edit FID | 29.48 linear; 24.06 diffusion | 27.49 LACE-ODE | Four sequential edits | StyleGAN2 | p. 7, T3 |
| Sequential edit ID | 0.290 linear; 0.445 diffusion | 0.501 LACE-ODE | Same | StyleGAN2 | p. 7, T3 |
| Linear classifier test accuracy, % | 75.50 to 98.13 | None | Seven attributes | StyleGAN2 $\mathcal{W}_s$ | p. 5, T1 |

## What to copy, what to avoid

- Avoid citing this paper as proof that the sum $\lambda_b\log C_{bind}+\lambda_c\log C_{count}+\lambda_s\log C_{spatial}$ is the correct score. The multi-condition case is stated without a printed proof, the weights are outside the bound, and our base is PoE where the paper's is an unconditional score (p. 3, p. 4) (inferred).
- State the assumption precisely: the design treats binding, count and spatial labels as independent given $x_t$. This is least plausible on steps 0 to 15, where one $x_t$ is compatible with many images and "left of" can only hold when two subjects exist (inferred).
- Measure that dependence before relying on the sum. On held-out images at each step, compare the product of the checkers' probabilities with the joint rate of all labels being true (inferred).
- Copy the correlation check from Figure 4 (p. 7), applied to gradients. Compute pairwise cosine between the three checker gradients at each step, because opposing gradients cancel before our 0.4 norm cap applies (inferred).
- Copy the linear baseline. LCG-Linear matched or beat the diffusion version on FID in compositional generation, 22.46 against 26.49 (p. 7). A linear checker on the attention maps is a cheap control (inferred).
- Avoid clean-only checker training. The paper's clean classifiers worked on a low-dimensional disentangled vector (p. 5). Our decoded-image CLIP checker read the pair at chance at step 0, so ours should be trained across noise levels (inferred).
- Avoid carrying over the disentanglement argument. All tested attributes describe one face in a global latent (p. 5, p. 7). Nothing here tests two subjects, spatial relations or attention-map inputs (inferred).

## Closer work it cites

None. The nearest miss is LACE, "Controllable and compositional generation with latent-space energy-based models" (reference [27], p. 9), which this paper describes as using latent energy-based models with non-linear classifiers (p. 2) and compares on the same face attributes (p. 7). LACE itself was not read for this note, so whether it tests unseen concepts is unknown.
