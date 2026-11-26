# Rebuild the sentence-score push from fundamentals

This page takes each equation the push depends on down to a rung the learning tree already teaches, in the order you would rebuild it. CLIP's own insides have [their own rebuild](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/scorers/rebuild-clip-from-fundamentals.md) and classifier guidance has [another](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md), so this page links their rungs and writes only what NoisyCLIP adds.

## Table of contents

1. [The noisy latent](#1-the-noisy-latent)
2. [The fixed linear map](#2-the-fixed-linear-map)
3. [The clamp and how many pixels it holds](#3-the-clamp-and-how-many-pixels-it-holds)
4. [The image encoder and its unit vector](#4-the-image-encoder-and-its-unit-vector)
5. [The contrastive loss with one side frozen](#5-the-contrastive-loss-with-one-side-frozen)
6. [The score difference is a two-sentence classifier](#6-the-score-difference-is-a-two-sentence-classifier)
7. [The gradient of a cosine](#7-the-gradient-of-a-cosine)
8. [From a gradient to a noise prediction](#8-from-a-gradient-to-a-noise-prediction)
9. [The DDIM step](#9-the-ddim-step)
- [The order to rebuild it](#the-order-to-rebuild-it)
- [Gaps](#gaps)

## 1. The noisy latent

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon
$$

**What the code does.** The released hook saves the U-Net's scaled input, which is this quantity ([block 1 of the code walk](3-code/walk-noisyclip.md#1-catching-the-half-drawn-latents)).

**Where it is taught.** [The schedule and forward noising](../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/textbook/02-one-denoising-step-on-one-small-network.md#21-the-schedule-and-forward-noising) works it on a small network, and [the forward process by hand](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/09-the-forward-process-by-hand.md#93-abar-from-a-beta-schedule-including-the-log-sum-shortcut) builds $\bar\alpha$ from SDXL's scaled-linear betas.

## 2. The fixed linear map

$$
c = W z + b
$$

### Rung 1: a matrix times a vector is one weighted sum per row

Red is row one of $W$ dotted with the four channels, plus its bias. On pixel A of [the miniature](2-miniature/README.md#2-work-the-linear-map), $60(1) - 60(0.5) + 25(0) - 70(0.5) + 150 = 145$.

**Where it is taught.** The [linear map card](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Linear%20Map.md#where-to-learn-this) lists the textbook pages and videos.

### Rung 2: the gradient of a linear map is its transpose

If $c = Wz + b$, a gradient $u$ on $c$ becomes $W^\top u$ on $z$. So each latent channel collects its colours' gradients, each weighted by the number that connected them.

**Where it is taught.** The [backpropagation card](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Backpropagation.md#where-to-learn-this), and [the chain rule run back into the input](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-16-the-chain-rule-run-backwards-reaches-the-input-as-well-as-the-weights).

## 3. The clamp and how many pixels it holds

$$
\text{clamp}(c) = \min(\max(c, 0), 255), \qquad \frac{d\,\text{clamp}}{dc} = \begin{cases} 1 & 0 < c < 255 \\ 0 & \text{otherwise} \end{cases}
$$

### Rung 3: a flat function has no slope to pass back

The clamp is flat outside its range, the same way a ReLU is flat below zero. [Understanding deep learning, page 52](http://127.0.0.1:8765/machine-learning/understanding-deep-learning.pdf#page=52) names that zero derivative, and [Probabilistic machine learning, page 495](http://127.0.0.1:8765/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=495) names the "dead ReLU" it causes.

### Rung 4: a weighted sum of independent unit Gaussians is Gaussian

Red's standard deviation is $\sqrt{60^2 + 60^2 + 25^2 + 70^2} = 112.81$, around a mean of 150. The share clamped is the area of that bell below 0 and above 255:

$$
\Phi\!\left(\frac{-150}{112.81}\right) + \Phi\!\left(\frac{150-255}{112.81}\right) = 9.2\% + 17.6\% = 26.8\%.
$$

**Where it is taught.** [Probabilistic machine learning, page 67](http://127.0.0.1:8765/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=67) covers the variance of a scaled variable, and [Pattern classification, page 720](http://127.0.0.1:8765/machine-learning/pattern-classification.pdf#page=720) covers the area under a Gaussian.

**The limit of this number.** It assumes independent unit-variance channels, which holds for pure noise. A half-drawn SDXL latent has structure, so its clamped share has to be measured.

## 4. The image encoder and its unit vector

$$
I = \frac{\text{projection}(\text{ViT}(\Phi(x_t)))}{\lVert \text{projection}(\text{ViT}(\Phi(x_t))) \rVert}
$$

**Where it is taught.** [The CLIP rebuild](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/scorers/rebuild-clip-from-fundamentals.md#5-image-path-the-vision-towers-parameters) walks the vision tower from its parameters to its output. NoisyCLIP changes the weights of this tower and keeps its shape.

## 5. The contrastive loss with one side frozen

$$
\mathcal L = \tfrac12\Big[\mathrm{CE}\big(e^{t_s} s,\ \text{rows}\big) + \mathrm{CE}\big(e^{t_s} s,\ \text{columns}\big)\Big]
$$

**What is new.** Only the image side receives the update. In [the miniature's training step](2-miniature/README.md#8-one-training-step), the loss falls and the sentence vectors stay where they were, so the image encoder learns to place noisy latents near a text space that never moves.

**Where it is taught.** [CLIP's dot product as a cosine](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/scorers/rebuild-clip-from-fundamentals.md#rung-20-the-dot-product-of-two-unit-vectors-is-the-cosine-of-the-angle-between-them) and the [CLIP chapter on one small network](../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/textbook/01-clip-on-one-small-network.md).

## 6. The score difference is a two-sentence classifier

### Rung 5: a two-way softmax has a one-line gradient

Let $a = e^{t_s} S_{\text{right}}$ and $b = e^{t_s} S_{\text{wrong}}$. Then $\log p(\text{right}) = a - \log(e^a + e^b)$, and differentiating gives

$$
\nabla \log p(\text{right}) = \big(1 - p(\text{right})\big)\, \nabla(a - b) = \big(1 - p(\text{right})\big)\, e^{t_s}\, \nabla\big[S_{\text{right}} - S_{\text{wrong}}\big].
$$

So the push's direction is a classifier's gradient. Its size differs, because the factor $(1-p)\,e^{t_s}$ shrinks as the classifier grows confident and a fixed $\lambda$ does not.

**Where it is taught.** [The classifier's slope is one minus its probability](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-26-the-classifiers-slope-is-one-minus-its-probability-times-the-log-odds-slope) is the same identity for a cat and dog classifier. [The miniature](2-miniature/README.md#5-see-why-this-is-a-classifiers-gradient) checks it to rounding error.

## 7. The gradient of a cosine

### Rung 6: differentiating a unit vector removes the part along itself

For $I = h / \lVert h \rVert$ and a fixed vector $u$,

$$
\frac{\partial (I \cdot u)}{\partial h} = \frac{u - (u \cdot I)\,I}{\lVert h \rVert}.
$$

The gradient is perpendicular to $I$, so it turns the embedding and cannot lengthen it. With $u = \Delta T$, this is the first card of [the miniature's backward pass](2-miniature/README.md#4-work-the-gradient-back).

**Where it is taught.** The [cosine similarity card](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Cosine%20Similarity.md#where-to-learn-this) teaches the cosine. This derivative is listed under [Gaps](#gaps).

## 8. From a gradient to a noise prediction

$$
\hat\epsilon_t = \epsilon_{\text{base}} - \lambda\,\sigma_t\,\nabla_{x_t}\big[S_{\text{right}} - S_{\text{wrong}}\big]
$$

**Why $\sigma_t$ appears.** A noise prediction is minus the noise level times the score, so a change to the score becomes $-\sigma_t$ times that change in the noise prediction.

**Where it is taught.** [Noise prediction is minus the noise level times the score](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-13-noise-prediction-is-minus-the-noise-level-times-the-score), and [Algorithm 2 takes the push off the noise prediction](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#3-steering-algorithm-2-takes-the-push-off-the-noise-prediction).

## 9. The DDIM step

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\frac{x_t - \sigma_t\,\hat\epsilon_t}{\sqrt{\bar\alpha_t}} + \sigma_{t-1}\,\hat\epsilon_t
$$

**Where it is taught.** [The DDIM update](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#11-ordinary-sampler-the-ddim-update-with-no-noise-at-eta-0) and the [DDIM card](../../../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/DDIM.md#where-to-learn-this).

## The order to rebuild it

1. Noise one toy latent with the schedule (section 1).
2. Write the linear map and its transpose (section 2).
3. Add the clamp and count the pixels it holds (section 3).
4. Write the unit vector and the cosine gradient (sections 4 and 7).
5. Build the score difference and check the two-sentence identity (section 6).
6. Join the push to a noise prediction and run one DDIM step (sections 8 and 9).
7. Train one contrastive step with the text side frozen (section 5).

[The miniature's script](2-miniature/miniature.py) is this order in about 150 lines.

## Gaps

**The gradient of a cosine with respect to its input.** No page in the learning tree derives $\partial(I \cdot u)/\partial h$. Section 7 holds the derivation, and a short textbook section would fill the gap.

**A soft clamp's trade-off.** Nothing in the tree compares a hard clamp with a smooth one for guidance. [The miniature's fourth break](2-miniature/README.md#7-break-it-four-ways) shows the mechanics on two pixels only.

**Bicubic resize as a linear operator.** The push passes back through a resize from 128 to 224, and no page teaches the transpose of that resampling.

## Cross-references

- The mention of **the noisy latent** in [the context page on the readout checker, the push and its cap](../../../../context/methods/02-the-readout-checker/03-the-push-and-its-cap.md).
- The mention of **the noisy latent** in [the context page on the readout checker, inside the kept checker](../../../../context/methods/02-the-readout-checker/02-inside-the-kept-checker.md).
- The mention of **the noisy latent** in [the finding "Does adding the kept push to CO3 beat CO3 alone?"](../../../report/the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md#2-what-the-push-follows).
