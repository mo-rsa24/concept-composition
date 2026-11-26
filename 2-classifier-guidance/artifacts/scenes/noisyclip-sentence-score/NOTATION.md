# Notation for the NoisyCLIP sentence-score ladder

Every rung of this ladder writes these symbols. The right-hand column gives what the paper or the code prints, so its equations can be read directly.

## The sampler

| Symbol | What it is | Size on SDXL | Paper or code writes |
|---|---|---|---|
| $x_t$ | the noisy latent at sampler step $t$, on the scale $\sqrt{\bar\alpha_t}\,x_0 + \sigma_t\,\varepsilon$ | $4 \times 128 \times 128$ | $z_t$; `input[0]` in `generate_latents.py` |
| $\bar\alpha_t$ | the share of signal variance left at step $t$ | one number | not printed |
| $\sigma_t$ | the noise scale, $\sqrt{1-\bar\alpha_t}$ | one number | `sigma` in `co3_push.py` |
| $\epsilon_{\text{base}}$ | the noise prediction before any push, from the product of experts or CO3 | $4 \times 128 \times 128$ | `eps` in `co3_push.py` |
| $\hat\epsilon_t$ | the noise prediction after the push | $4 \times 128 \times 128$ | not printed |
| $\lambda$ | the strength of the push | one number | `strength` in `co3_push.py` |
| $g$ | the gradient $\nabla_{x_t}[S_{\text{right}} - S_{\text{wrong}}]$ | $4 \times 128 \times 128$ | `grad` in `co3_push.py` |
| $p$ | the push, $\lambda\,\sigma_t\,g$ | $4 \times 128 \times 128$ | `push` in `co3_push.py` |

## The input path

| Symbol | What it is | Size | Paper or code writes |
|---|---|---|---|
| $z_0, z_1, z_2, z_3$ | the four latent channels at one position | 4 numbers | `latents` |
| $W$ | the fixed weights from 4 channels to 3 colours | $3 \times 4$ | `weights` |
| $b$ | the fixed bias of each colour | 3 numbers | `biases` |
| $c$ | a colour value after the map, 0 to 255 after the clamp | $3 \times 128 \times 128$ | `rgb_tensor` |
| $\Phi$ | the whole path from the latent to the image encoder's input | | $\Phi$ (p. 5) |

## The two towers and the score

| Symbol | What it is | Size | Paper or code writes |
|---|---|---|---|
| $I$ | the image embedding, unit length | 768 | $\nu_\theta(\Phi(z_t))$ |
| $T_{\text{right}},\ T_{\text{wrong}}$ | the embeddings of the right and wrong sentences, unit length | 768 | $\tau_\theta(y)$ |
| $S$ | the sentence score, the cosine $I \cdot T$, from $-1$ to $1$ | one number | $S_{\text{latent}}$ (p. 5) |
| $D$ | the score difference, $S_{\text{right}} - S_{\text{wrong}}$, from $-2$ to $2$ | one number | not printed |
| $e^{t_s}$ | the score multiplier stored as `logit_scale`, 98.86 in NoisyCLIP | one number | `logit_scale` |
| $\mathcal L$ | the contrastive loss | one number | InfoNCE (p. 6) |

The multiplier is written $e^{t_s}$ because $t$ already means the sampler step.

## The miniature only

| Symbol | What it is |
|---|---|
| A, B | the two pixels |
| $p_{A,R}$ | pixel A's red after scaling, $c/255 - 0.5$ |
| $M$ | the hand-set $2 \times 6$ image layer |
| $h$ | the layer's output before it is made unit length |
| $\Delta T$ | $T_{\text{right}} - T_{\text{wrong}}$ |

## Cross-references

- The mention of **the input path** in [the finding "Can a sentence score read the five properties on a half-drawn picture?"](../../../report/the-sentence-score/can-a-sentence-score-read-the-five-properties.md).
