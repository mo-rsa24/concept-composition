# Rebuild the word-map checker from fundamentals

<a href="../../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/artifacts/figures/network-view/nv4b-each-patch-asks-the-words.png"><img src="../../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/artifacts/figures/network-view/nv4b-each-patch-asks-the-words.png" width="400" alt="One patch of the picture asks every word of the caption how much to read from it, and gets a row of weights that adds to 1"></a>

The picture is where the word-map checker starts: each patch of the picture asks every word of the caption how much to read from it. Learn-to-Steer keeps those weights as one map per word, trains a small classifier to name the spatial relation between two words' maps, and moves the latent down the gradient of that classifier's error. This page takes each equation of that method down to first-course maths, in the order the [walk through the authors' code](3-code/walk-learn-to-steer.md) and the [walk through our sampler](3-code/walk-our-sampler.md) meet them. Under each equation sits a ladder of single moves, each with one number from [the miniature's answer key](2-miniature/miniature_answer_key.json) and a link to where the learning tree teaches it. [The order to rebuild it](#the-order-to-rebuild-it) lists all 25 moves.

Learn-to-Steer ran on FLUX and SD 1.4 and 2.1, and never on SDXL. Anything said here about SDXL is our proposal.

## Words this uses

**The miniature**: a 2 × 2 latent with a cat channel and a dog channel, two cross-attention layers and a head with hand-set weights. It shows what each part computes and says nothing about how well a trained head performs.

**$A^{(\ell)}_s$, $\hat A^{(\ell)}_s$, $\bar A_s$**: one word's map from layer-and-head number $\ell$, the same map resized and divided by its sum, and the word's maps pooled into one. [The ladder's notation](NOTATION.md) defines every symbol.

**A rung**: one move of algebra, probability or code, checked on one line with one number.

**Gap**: a rung no textbook, card or plan in the learning tree teaches, worked in full on this page.

## Table of contents

1. [The cross-attention softmax](#1-the-cross-attention-softmax)
2. [Resize and renormalise](#2-resize-and-renormalise)
3. [Pooling with a learned query](#3-pooling-with-a-learned-query)
4. [Softmax cross-entropy](#4-softmax-cross-entropy)
5. [The chain rule through the map into the latent](#5-the-chain-rule-through-the-map-into-the-latent)
6. [The latent update](#6-the-latent-update)
7. [The product of experts](#7-the-product-of-experts)
8. [The capped push](#8-the-capped-push)
- [The order to rebuild it](#the-order-to-rebuild-it)
- [Gaps](#gaps)
- [Cross-references](#cross-references)

## 1. The cross-attention softmax

**What the code does.** The U-Net's attention layer computes these weights for every patch and all 77 text tokens, and the authors' processor keeps a copy ([block 1 of the code walk](3-code/walk-learn-to-steer.md#1-catching-each-attention-map)).

**As the paper prints it.** [Page 3](https://arxiv.org/pdf/2509.02295v2#page=3): "Each map $A_t[:, :, i]$ indicates how token $i$ influences each spatial location."

$$
A^{(\ell)}[p, i] = \frac{\exp\big(q_p^\top k_i / \sqrt{d_{\text{head}}}\big)}{\sum_{j} \exp\big(q_p^\top k_j / \sqrt{d_{\text{head}}}\big)}
$$

### Rung 1: a dot product scores how well a patch's query matches a word's key

Multiply two vectors entry by entry and add. A large result means they point the same way. The miniature's query is the patch's latent value and its key is 2, so a patch with cat-ness 2 scores $2 \times 2 = 4$. The real scores are then divided by the square root of the head's width.

**Where it is taught.** The Stable Diffusion rebuild's [score as a query dotted with a key](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/rebuild-stable-diffusion-from-fundamentals.md#rung-15-a-score-is-a-query-dotted-with-a-key-divided-by-d), the transformers journey's [why scores are divided by the root of the width](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/25-show-why-attention-scores-are-divided.md#maths), and the trigonometry handbook's [dot product](../../../../../../goal-setting/learning/mathematics/textbooks/trigonometry/trigonometry-handbook.pdf#page=87).

### Rung 2: softmax turns a row of scores into shares that add to 1

Raise $e$ to each score, then divide by the total. With a score of 4 for "cat" and 0 for the background token, "cat" takes $e^4 / (1 + e^4) = 0.982$ of the patch.

**Where it is taught.** The transformers journey's [softmax weights and the blend](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/20-turn-scores-into-softmax-weights-and-blend.md#maths) and the vault's [softmax card](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Softmax.md#where-to-learn-this).

### Rung 3: one word's column of the patch-by-word table is that word's map

The softmax runs along each patch's row, so a word's weights down all the patches form a picture of where that word lands. The miniature's layer 1 map for "cat", over top left, top right, bottom left and bottom right, is (0.982, 0.982, 0.1192, 0.1192).

**Where it is taught.** The Stable Diffusion rebuild's [keys and values from another sequence](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/rebuild-stable-diffusion-from-fundamentals.md#rung-17-keys-and-values-from-another-sequence-make-it-cross-attention) and the small-network reading of [each cell asking the words](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/stable-diffusion-read-through-a-small-network.md#4-each-cell-asks-the-words-through-cross-attention).

### Rung 4: two words in one softmax compete, and two words in two prompts do not

Learn-to-Steer reads both words from one joint prompt, so they share each patch's total. Our product of experts runs "a cat" and "a dog" as separate prompts, so each word only competes with its own prompt's other tokens. At a patch where both channels are 2, each word takes 0.982 in its own prompt. In one joint prompt, "cat" takes $e^4 / (1 + e^4 + e^4) = 0.4955$.

**Where it is taught.** **Gap**, worked here. The arithmetic is rung 2 with one more term in the denominator.

## 2. Resize and renormalise

**What the code does.** `resize_and_concat_attn_maps_from_dict` brings every map to 16 × 16 with bilinear interpolation and divides it by its sum over the grid ([block 4 of the code walk](3-code/walk-learn-to-steer.md#4-bringing-every-map-to-16-by-16)).

**As the paper prints it.** It does not. [Page 15](https://arxiv.org/pdf/2509.02295v2#page=15) gives the maps' size as $h \times w$ with no values, and the code settles it.

$$
\hat A^{(\ell)}_s = \frac{\operatorname{bilinear}_{16}\big(A^{(\ell)}_s\big)}{\sum_{p} \operatorname{bilinear}_{16}\big(A^{(\ell)}_s\big)[p] + 10^{-8}}
$$

### Rung 5: bilinear is a straight line across, then a straight line down

Take the straight-line value along each of the two rows a point falls between, then one more straight line between those two results. Shrinking 32 to 16 places each output cell at the middle of a 2 × 2 block, so it takes 0.25 of each of the 4 cells.

**Where it is taught.** The Readout Guidance rebuild's [bilinear rung](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-4-bilinear-is-that-straight-line-done-across-then-down), and Understanding Deep Learning on [downsampling and upsampling, §10.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=187).

### Rung 6: shrinking by four without antialiasing reads only the middle of each block

An output cell's centre lands between input cells 1 and 2 of its block of 4 along each axis, so it reads those two and ignores cells 0 and 3. In two dimensions a 64 to 16 cell reads 4 of its block's 16 cells, each at weight 0.25. The other 12 have no influence. Moving one input cell at a time through `F.interpolate(mode='bilinear', align_corners=False)` on CPU gave exactly these weights. This touches the maps from SD 2.1's 64 grid, and SDXL's 100 maps from its 64 grid would be read the same way.

**Where it is taught.** **Gap**, worked here. Understanding Deep Learning's [§10.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=187) names sub-sampling, which is the same effect.

### Rung 7: dividing by the total makes a map add to 1

A word that covers little of the picture and a word that covers much now enter the head with the same total, so the head reads where a word is and loses how much attention it got. The miniature's layer 1 cat map sums to 2.2024, and $0.982 / 2.2024 = 0.4459$, $0.1192 / 2.2024 = 0.0541$.

**Where it is taught.** The Readout Guidance rebuild's [dividing by the total](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-21-dividing-by-the-total-makes-the-parts-sum-to-1-so-a-sum-of-such-parts-counts-them).

## 3. Pooling with a learned query

<a href="../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv5b-four-ways-to-pool.png"><img src="../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv5b-four-ways-to-pool.png" width="200" alt="One token row pooled four ways: the end token, the first token, the mean, and a learned query"></a>

The fourth column of this picture is the move here: one trained vector asks every token how much to take from it.

**What the code does.** Each of the $L \cdot H$ maps becomes one token, a small transformer lets them exchange information, and one `MultiheadAttention` layer with a learned query pools them into a single vector, reshaped to a 16 × 16 map ([block 5 of the code walk](3-code/walk-learn-to-steer.md#5-the-aggregator-many-maps-into-one)).

**As the paper prints it.** [Page 15](https://arxiv.org/pdf/2509.02295v2#page=15) calls the pooling step "another four-layer, single-head transformer". The released code uses one attention layer with 2 heads on SD 2.1 and 4 on FLUX.1-dev.

$$
z = \sum_{\ell} \operatorname{softmax}_\ell\big(q^\top k(T_\ell)\big)\, v(T_\ell), \qquad \bar A_s = \operatorname{reshape}_{16 \times 16}\big(W_{\text{out}}\, z\big)
$$

### Rung 8: a learned query is one vector, trained, that asks every token the same question

The query belongs to the head and has no input of its own. Its scores against the tokens' keys go through a softmax, so it spends a total weight of 1 across the maps. On SDXL that would be 1300 maps per word (our proposal), and the 2-head layer runs this twice in parallel.

**Where it is taught.** Understanding Deep Learning on [dot-product attention, §12.2](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=222), and the vault's [attention card](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Attention.md#where-to-learn-this).

### Rung 9: the pooled result is a weighted sum of the tokens

Multiply each token's value by its weight and add. The miniature fixes the weights at 3/4 and 1/4 in place of a trained query, so the cat's top-left cell becomes $0.75 \times 0.4459 + 0.25 \times 0.383 = 0.4302$. In the real head the weights depend on the maps, and the output projection is free, so $\bar A_s$ need not stay positive or add to 1.

**Where it is taught.** The transformers journey's [softmax weights and the blend](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/20-turn-scores-into-softmax-weights-and-blend.md#mechanism).

### Rung 10: attention over a single token has weight 1

The classifier's transformer receives the whole flattened pair of maps as one token. A softmax over one score is $e^s / e^s = 1$ whatever $s$ is, so its attention step cannot compare two positions. Any comparison between the subject's and object's maps happens in the linear layer before it and the feed-forward layers around it.

**Where it is taught.** **Gap**, worked here. It is rung 2 with one entry.

## 4. Softmax cross-entropy

**What the code does.** `nn.CrossEntropyLoss` on the head's five logits against the relation in the prompt ([block 7 of the code walk](3-code/walk-learn-to-steer.md#7-the-loss)).

**As the paper prints it.** [Page 4](https://arxiv.org/pdf/2509.02295v2#page=4):

$$
\mathcal L_t = -\log \hat p[r]
$$

### Rung 11: the log of a softmax is a score minus the log-sum-exp

$\log \hat p[r] = \ell_r - \log \sum_j e^{\ell_j}$. On the miniature's fused body all five logits are 0, so $\log \hat p[r] = 0 - \ln 5 = -1.6094$.

**Where it is taught.** The classifier guidance rebuild's [softmax and its log](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-25-softmax-turns-scores-into-probabilities-and-its-log-is-a-score-minus-the-log-sum-exp).

### Rung 12: cross-entropy with one right answer is minus the log of that answer's probability

With a one-hot target, every term of the cross-entropy vanishes except the right class. The head gives "left of" 0.2 on the fused body, so $\mathcal L_t = -\ln 0.2 = 1.6094$. A head certain of the right answer would score 0.

**Where it is taught.** Understanding Deep Learning on [cross-entropy loss, §5.7](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=85), and the vault's [cross-entropy card](../../../../../../Obsidian/Mathematics/Statistics/Information%20Theory/Cross-Entropy.md#where-to-learn-this).

### Rung 13: the loss's slope in each logit is that class's probability, minus 1 for the right class

$\partial \mathcal L / \partial \ell_j = \hat p_j - \mathbb 1[j = r]$. At the tie that is $0.2 - 1 = -0.8$ for "left of" and $0.2$ for each other class. The miniature's logits are $-4\Delta_u$ for "left of" and $+4\Delta_u$ for "right of", so the slope in $\Delta_u$ is $(-4)(-0.8) + (4)(0.2) = 4$, the key's value.

**Where it is taught.** Pattern Recognition and Machine Learning on [multiclass logistic regression, §4.3.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=227), and the Deep Learning textbook on [softmax units, §6.2.2.3](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/deep-learning-textbook.pdf#page=200).

## 5. The chain rule through the map into the latent

**What the code does.** One call to `torch.autograd.grad(loss, latents)` runs the chain rule from the loss back through the classifier, the aggregator, the resize and the U-Net's attention to the latent ([block 8 of the code walk](3-code/walk-learn-to-steer.md#8-the-refine-loop)).

**On the miniature.**

$$
\frac{\partial \mathcal L}{\partial x_p} = \sum_{q} \frac{\partial \mathcal L}{\partial \bar A_s[q]}\ \frac{\partial \bar A_s[q]}{\partial x_p}
$$

### Rung 14: the chain rule multiplies each link's slope, from the output back to the input

The same backward pass that trains a network's weights continues one link further and reaches its input.

**Where it is taught.** The Readout Guidance rebuild's [chain rule reaching the input](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-16-the-chain-rule-run-backwards-reaches-the-input-as-well-as-the-weights), Understanding Deep Learning on [backpropagation, §7.4](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=117), and the vault's [backpropagation card](../../../../../../Obsidian/Artificial%20Intelligence/Deep%20Learning/Backpropagation.md#where-to-learn-this).

### Rung 15: a softmax share moves fastest where it is undecided

For a two-way softmax $A = e^{kx}/(1 + e^{kx})$, the slope is $\partial A/\partial x = k A (1 - A)$. At the fused body's top row $A = 0.982$ and the slope is 0.0353. At the bottom row $A = 0.1192$ and the slope is 0.21, about six times larger. The gradient therefore acts mostly on patches the word has not yet claimed or abandoned.

**Where it is taught.** Journey 13's [chapter 1, the slope is not the probability](../../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/01-the-guided-noise-prediction.md#the-slope-is-not-the-probability), and the classifier guidance rebuild's [slope as one minus the probability](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-26-the-classifiers-slope-is-one-minus-its-probability-times-the-log-odds-slope).

### Rung 16: renormalising adds a quotient-rule term

With $S = \sum_q A[q]$, the slope of $\hat A[q] = A[q]/S$ in $A[p]$ is $\mathbb 1[q=p]/S - A[q]/S^2$. Raising one cell raises its own share and lowers everyone else's. On the fused body the second term drops out of the gradient, because $\sum_q u_q A[q] = 0$ when the left and right columns match.

**Where it is taught.** The calculus textbook on [the quotient rule, §2.4](../../../../../../goal-setting/learning/mathematics/textbooks/calculus/calculus-textbook.pdf#page=189).

### Rung 17: the whole chain, on one latent number

The slope of the loss in the pooled map is $4u_q$, so for the cat channel at the bottom-left patch, where $u = -1$:

$$
\begin{aligned}
\frac{\partial \mathcal L}{\partial x_{\text{bottom left}}} &= 4 u \left(w_1 \frac{k_1 A_1 (1-A_1)}{S_1} + w_2 \frac{k_2 A_2 (1-A_2)}{S_2}\right)\\
&= -4\left(0.75 \times \frac{0.209987}{2.202433} + 0.25 \times \frac{0.196612}{2.299477}\right)\\
&= -4\,(0.071507 + 0.021376) = -0.3715
\end{aligned}
$$

A finite-difference check on all 8 latent numbers agrees to within $1.2 \times 10^{-10}$.

**Where it is taught.** **Gap**, worked here. Rungs 13, 15 and 16 joined by rung 14.

## 6. The latent update

**What the code does.** `update_latents` subtracts step size times gradient from the latent, and `refine_latents` repeats it up to 16 times per sampler step ([block 8 of the code walk](3-code/walk-learn-to-steer.md#8-the-refine-loop)).

**As the paper prints it.** [Page 4](https://arxiv.org/pdf/2509.02295v2#page=4), with $z_t$ and $\alpha$ where this ladder writes $x_t$ and $\eta$:

$$
x_t \leftarrow x_t - \eta\, \nabla_{x_t} \mathcal L_t
$$

### Rung 18: gradient descent steps against the slope

Move each number by minus the step size times its slope. With $\eta = 5$ the cat's bottom-left value goes from $-1$ to $-1 - 5(-0.3715) = 0.8577$, and the head's probability for "left of" goes from 0.2 to 0.7961 in one update.

**Where it is taught.** The Readout Guidance rebuild's [gradient descent rung](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#rung-17-gradient-descent-steps-against-the-slope-by-a-learning-rate-times-the-slope), and the vault's [gradient descent card](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Gradient%20Descent.md#where-to-learn-this).

### Rung 19: repeating the step gets slower as the head grows sure

The loop recomputes the maps and the gradient each pass. On the miniature the loss reads 1.6094, 0.2280, 0.1884, 0.1573, 0.1223 over the first passes, all 16 updates fire, and it ends at 0.0064 with $\hat p[r] = 0.9936$. By the last pass the gradient's norm has fallen from 0.7664 to 0.0098, because a sure classifier has little left to ask for.

**Where it is taught.** Understanding Deep Learning on [gradient descent, §6.1](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/understanding-deep-learning.pdf#page=91).

## 7. The product of experts

**What the code does.** Our sampler runs the U-Net on "a cat", "a dog" and the empty prompt, applies guidance to each, and combines them ([block 1 of the sampler walk](3-code/walk-our-sampler.md#1-the-products-noise-prediction)). Learn-to-Steer has no such step.

$$
\varepsilon_{\text{PoE}} = \varepsilon_{\text{cat}} + \varepsilon_{\text{dog}} - \varepsilon_\varnothing
$$

### Rung 20: the log of a product is a sum

Multiplying two densities adds their logs, so it adds their slopes.

**Where it is taught.** The classifier guidance rebuild's [log of a product](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-14-the-log-of-a-product-is-a-sum) and PoE foundations' [chapter 3.1](../../../../../../goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/textbook/03-moving-into-logs.md#31-the-log-of-a-product-and-the-quotient-rule).

### Rung 21: a noise prediction is minus the noise level times the score

$\varepsilon = -\sigma_t \nabla_{x_t} \log p(x_t)$, so adding scores adds noise predictions.

**Where it is taught.** The classifier guidance rebuild's [noise prediction as a score](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-13-noise-prediction-is-minus-the-noise-level-times-the-score).

### Rung 22: the empty prompt is subtracted once, because each expert already contains it

Each prompt's prediction is the shared unconditional part plus that prompt's change, and the product should count the shared part once. The miniature's predictions are $\pm 0.25$ for each prompt and 0 for the empty one, so $\varepsilon_{\text{PoE}}$ has entries of $\pm 0.5$ and norm 1.4142.

**Where it is taught.** Journey 13's [chapter 5, the product as addition in score space](../../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/05-the-whole-pipeline-on-one-small-network.md#product-multiplication-becomes-addition-in-score-space), and the small-network reading of [what the product leaves out](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/classifier-guidance-read-through-a-small-network.md#4-what-the-product-of-two-experts-leaves-out).

## 8. The capped push

**What the code does.** Our sampler takes the head's gradient, scales it by the strength and the noise level, shortens it to at most a fraction of the prediction's norm and subtracts it from the prediction ([blocks 5 to 7 of the sampler walk](3-code/walk-our-sampler.md#5-from-gradient-to-push)). This is our route, and Learn-to-Steer states no cap.

$$
\text{push} = \lambda\sigma_t\nabla_{x_t}\log\hat p[r]\ \cdot\ \min\!\left(1,\ \frac{c\,\lVert\varepsilon_{\text{PoE}}\rVert}{\lVert\lambda\sigma_t\nabla_{x_t}\log\hat p[r]\rVert}\right), \qquad \varepsilon' = \varepsilon_{\text{PoE}} - \text{push}
$$

### Rung 23: classifier guidance subtracts the noise level times the gradient

Adding a classifier's log-probability to the density adds its slope to the score, and rung 21 turns that into $-\sigma_t \nabla \log p$ on the prediction. With $\lambda = 2$ and $\sigma_t = 0.8$, the miniature's raw push has norm 1.2262.

**Where it is taught.** The classifier guidance rebuild's [Algorithm 2 section](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#3-steering-algorithm-2-takes-the-push-off-the-noise-prediction).

### Rung 24: dividing a vector by a number above 1 shortens it and keeps its direction

The cap is $0.4 \times 1.4142 = 0.5657$. The raw norm is above it, so every entry is multiplied by $0.5657 / 1.2262 = 0.4613$.

**Where it is taught.** The Deep Learning textbook on [clipping a gradient by its norm, §10.11](../../../../../../goal-setting/learning/mathematics/textbooks/machine-learning/deep-learning-textbook.pdf#page=431).

### Rung 25: the DDIM step carries a change in the prediction into the latent at a known rate

Write $\hat x_0 = (x_t - \sigma_t \varepsilon')/\sqrt{\bar\alpha_t}$ and $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sigma_{t-1}\varepsilon'$. The slope of $x_{t-1}$ in $\varepsilon'$ is $\sigma_{t-1} - \sigma_t\sqrt{\bar\alpha_{t-1}/\bar\alpha_t}$. With $\bar\alpha_t = 0.36$ and $\bar\alpha_{t-1} = 0.64$ that is $0.6 - 0.8 \times 0.8/0.6 = -0.4667$. A push of norm 0.5657 therefore moves the latent by 0.264, and the head's probability for "left of" goes from 0.2 to 0.2367. Learn-to-Steer's 16 updates move the same latent by 6.0072, to 0.9936. The weights are hand-set, so the size of that gap applies to the miniature alone.

**Where it is taught.** Journey 13's [chapter 5, turning the prediction into the next image](../../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/05-the-whole-pipeline-on-one-small-network.md#step-turn-the-prediction-into-the-next-image), and the classifier guidance rebuild's [DDIM rung](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md#rung-23-re-noise-the-clean-guess-to-the-next-level-with-the-same-noise-guess).

## The order to rebuild it

1. A dot product scores a query against a key.
2. Softmax turns scores into shares.
3. A word's column is its map.
4. Two words in one softmax compete.
5. Bilinear is a line across, then a line down.
6. Shrinking by four reads the middle of each block.
7. Dividing by the total makes a map add to 1.
8. A learned query asks every token.
9. The pooled result is a weighted sum.
10. Attention over one token has weight 1.
11. The log of a softmax is a score minus the log-sum-exp.
12. Cross-entropy is minus the log of the right answer's probability.
13. The loss's slope in a logit is its probability, minus 1 for the right class.
14. The chain rule multiplies slopes backward.
15. A softmax share moves fastest where it is undecided.
16. Renormalising adds a quotient-rule term.
17. The whole chain lands on one latent number.
18. Gradient descent steps against the slope.
19. Repeated steps slow as the head grows sure.
20. The log of a product is a sum.
21. A noise prediction is minus the noise level times the score.
22. The empty prompt is subtracted once.
23. Classifier guidance subtracts the noise level times the gradient.
24. Scaling a vector keeps its direction.
25. The DDIM step carries a change in the prediction into the latent.

## Gaps

Five rungs have no teaching page in the learning tree, so each is worked in full above:

| Rung | The move | Why no existing page covers it |
|---|---|---|
| 4 | words compete in one softmax and do not across two prompts | the tree teaches softmax and the product separately |
| 6 | bilinear shrinking by four reads 4 of 16 cells | textbooks name sub-sampling without working the weights |
| 10 | attention over one token has weight 1 | a property of this code's input shape |
| 17 | the chain through a renormalised map | specific to this head |
| 25 (its rate) | the factor by which a push on the prediction reaches the latent | the tree teaches the DDIM step and leaves this slope unwritten |

## Cross-references

- [The walk through the authors' code](3-code/walk-learn-to-steer.md) shows the lines each section computes, at commit `a2ae427`.
- [The walk through our sampler](3-code/walk-our-sampler.md) shows the push and what a port to SDXL would change.
- [The miniature's script](2-miniature/miniature.py) computes every worked value on this page.
- [The ladder's numbers file](numbers/numbers.json) holds every full-scale size.
- [The classifier guidance rebuild](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md) and [the Readout Guidance rebuild](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md) teach the rungs this page links to.
