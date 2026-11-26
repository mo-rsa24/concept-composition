# Rebuild the digit-map checkers from fundamentals

Each equation the checkers use, taken down to rungs the learning tree already teaches. A rung with no page to point at is marked **Gap** and worked here. The symbols are those of the [ladder's notation](NOTATION.md).

## Table of contents

- [1. The noisy canvas](#1-the-noisy-canvas)
- [2. The two maps](#2-the-two-maps)
- [3. The network](#3-the-network)
- [4. The loss and its slope](#4-the-loss-and-its-slope)
- [5. Backprop into the weights, and on into the canvas](#5-backprop-into-the-weights-and-on-into-the-canvas)
- [6. The push](#6-the-push)
- [7. The score](#7-the-score)

## 1. The noisy canvas

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

### Rung 1: one jump from the clean canvas to any noise level

**Where it is taught.** The Stable Diffusion rebuild's [forward noising in one jump](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/rebuild-stable-diffusion-from-fundamentals.md#1-forward-noising-x_t-from-a-clean-latent-in-one-jump). The same line runs on pixels here instead of SDXL's latents; [the canvas and its noise](0.5-network-view/dc1-the-canvas-and-its-noise.png) shows it at four of the sixteen timesteps.

## 2. The two maps

$$m_a = 2048 \cdot \frac{|\varepsilon_a - \varepsilon_\varnothing|}{\sum_{\text{pixels}} |\varepsilon_a - \varepsilon_\varnothing|}$$

### Rung 2: dividing by the total makes the parts add to 1

**Where it is taught.** The Readout Guidance rebuild's [dividing by the total](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-21-dividing-by-the-total-makes-the-parts-sum-to-1-so-a-sum-of-such-parts-counts-them). Multiplying by the 2,048 pixels then makes the map's mean 1, so a map keeps its shape and loses its overall size.

### Rung 3: the difference from the no-digit guess is what asking for a digit adds

**Gap**, worked here. $\varepsilon_\varnothing$ is the U-Net's guess with no digit asked (label 10). $\varepsilon_a - \varepsilon_\varnothing$ is therefore what the label $a$ changes, pixel by pixel. In the [miniature](2-miniature/README.md) the expert subtracts $0.5\,T_a\,\sigma(3x)$, so on the apart canvas $m_a$ reads 0.57, 2.37, 0.75, 0.31: largest on pixel 2, where a "1" has ink.

## 3. The network

### Rung 4: a convolution slides one small kernel over the grid

**Where it is taught.** Understanding Deep Learning on [convolutional networks for 2D inputs, §10.3](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=184), after [the 1D case, §10.2](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=177), which covers stride and padding.

### Rung 5: stride 2 halves the grid

$$n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k}{s} \right\rfloor + 1, \qquad \left\lfloor \frac{32 + 2 - 3}{2} \right\rfloor + 1 = 16$$

**Where it is taught.** Understanding Deep Learning's [§10.2](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=177). The four sizes this network passes through are counted in [the checker network](0.5-network-view/dc3-the-checker-network.png).

### Rung 6: the timestep as sines and cosines

**Where it is taught.** The SDXL module build's [sinusoidal time embedding](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/sources/sdxl-module-build/plans/module_5_plan.md#sinusoidaltimeembedding). The checker feeds it log-SNR/10 instead of the timestep number, and uses 8 frequencies; the 16 real rows are in [the timestep as sixteen numbers](0.5-network-view/dc4-the-timestep-as-sixteen-numbers.png).

### Rung 7: a scale and a shift made from the timestep

$$h \leftarrow h\,(1+\gamma) + \beta, \qquad (\gamma, \beta) = W\,\phi(t)$$

**Where it is taught.** The SDXL module build's [FiLM-conditioned ResBlock](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/sources/sdxl-module-build/plans/module_5_plan.md#resblock-film-conditioned), which applies the same line inside a U-Net block.

### Rung 8: mean and max over positions make one vector whatever the grid

**Where it is taught.** Understanding Deep Learning on [downsampling, §10.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=187). Here the pooling runs over all 128 positions at once, so the 128 channels give 128 means and 128 maxes.

## 4. The loss and its slope

$$L = -\big[\,y\log\sigma(\ell) + (1-y)\log(1-\sigma(\ell))\,\big]$$

### Rung 9: one logit through the sigmoid is a chance between 0 and 1

**Where it is taught.** Understanding Deep Learning on [binary classification, §5.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=78), and the vault's [classifier card](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Classifier.md#where-to-learn-this).

### Rung 10: binary cross-entropy is minus the log of the chance given to the right label

**Where it is taught.** Understanding Deep Learning on [cross-entropy loss, §5.7](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=85), and the vault's [cross-entropy card](../../../../../../Obsidian/Mathematics/Statistics/Information%20Theory/Cross-Entropy.md#where-to-learn-this).

### Rung 11: the loss's slope in the logit is the chance minus the label

$$\frac{\partial L}{\partial \ell} = \sigma(\ell) - y$$

**Where it is taught.** The classifier guidance rebuild's [slope as one minus the probability](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-26-the-classifiers-slope-is-one-minus-its-probability-times-the-log-odds-slope), which is the $y = 1$ case. In the [miniature's training step](2-miniature/mm2-one-training-step-per-branch.png) the wrong canvas at $\sigma(\ell) = 0.587$ gets slope $+0.587$ under identity ($y = 0$) and $-0.413$ under count ($y = 1$).

## 5. Backprop into the weights, and on into the canvas

### Rung 12: the chain rule run backwards reaches the input as well as the weights

**Where it is taught.** The Readout Guidance rebuild's [chain rule reaching the input](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-16-the-chain-rule-run-backwards-reaches-the-input-as-well-as-the-weights), Understanding Deep Learning on [backpropagation, §7.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=117), and the vault's [backpropagation card](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Backpropagation.md#where-to-learn-this). Training takes the weights' branch ([dc6](0.5-network-view/dc6-training-backprop-into-the-weights.png)); the push takes the input's branch ([dc7](0.5-network-view/dc7-the-push-gradient-into-the-canvas.png)).

### Rung 13: Adam scales each weight's step by its own running gradient size

**Where it is taught.** Understanding Deep Learning on [Adam, §6.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=102).

### Rung 14: the slope through $|d|/\sum|d|$ has a quotient-rule term

$$\frac{\partial m_i}{\partial |d_j|} = 2048\left(\frac{[i=j]}{S} - \frac{|d_i|}{S^2}\right), \qquad S = \sum_k |d_k|$$

**Gap**, worked here and checked in code: `maps_backward` in [the miniature's script](2-miniature/miniature.py) applies this line and matches finite differences to 1.4e-09. Raising one pixel's $|d|$ raises that pixel's map value and lowers every other pixel's a little, because the total grew.

## 6. The push

$$\hat\varepsilon = \varepsilon_{\text{product}} - \lVert\varepsilon_{\text{product}}\rVert\; b\; \frac{\nabla_{x_t}\ell}{\lVert\nabla_{x_t}\ell\rVert}$$

### Rung 15: subtracting a classifier's gradient from the noise guess steers the sample toward that class

**Where it is taught.** Journey 13's [chapter 1, the guided noise prediction](../../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/01-the-guided-noise-prediction.md#the-slope-is-not-the-probability).

### Rung 16: dividing by the gradient's length keeps its direction and throws away its size

**Gap**, worked here. $g/\lVert g\rVert$ has length 1 whatever $g$ was, so the push's length is exactly $b\,\lVert\varepsilon_{\text{product}}\rVert$. In the [miniature's push](2-miniature/mm3-the-push-on-the-four-pixels.png) $\lVert g\rVert$ is 1.441 on the apart canvas and 0.850 on the wrong one, and both pushes come out at 0.40 of the product's length.

## 7. The score

### Rung 17: balanced accuracy averages the two classes' hit rates

$$\text{BA} = \tfrac12\big(P(\text{called }1 \mid y=1) + P(\text{called }0 \mid y=0)\big)$$

**Gap**, worked here. The identity test on pairs holding digit 4 has 400 `apart` and 100 `wrong` canvases, so a checker that calls everything 1 scores 0.8 plain accuracy and 0.5 balanced accuracy. That is why every bar in this ladder is balanced (`balanced` in `v2_learn.py`, lines 136 to 139).

## Cross-references

- The mention of **the noisy canvas** in [the context page on the digit-map checkers, the code that runs them](../../../../context/methods/03-the-digit-map-checkers/04-the-code.md).
- The mention of **the noisy canvas** in [the finding "Does a checker learn plurality on unseen digits?"](../../../report/abstract-checkers-on-digits/does-a-checker-learn-plurality-on-unseen-digits.md#words-this-page-uses).
- The mention of **the noisy canvas** in [the finding "Does the count checker's push compose unseen digits?"](../../../report/abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md#words-this-page-uses).
- The mention of **the noisy canvas** in [the finding "Does removing one checker remove its property, and only its property?"](../../../report/abstract-checkers-on-digits/does-removing-one-checker-remove-its-property.md#words-this-page-uses).
- The mention of **the noisy canvas** in [the finding "Does the digit model's score path carry which digits are present?"](../../../report/does-the-score-path-carry-which-digits-are-present.md).
- The mention of **the noisy canvas** in [the context page on the sentence-score push, training our own encoder on digits](../../../../context/methods/04-the-sentence-score-push/05-training-our-own-encoder.md#the-loss-by-what-moves).
- The mention of **the noisy canvas** in [the findings index](../../../../report/00-INDEX.md#project-2-classifier-guidance).
