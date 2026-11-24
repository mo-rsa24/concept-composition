# Notation for the digit-map checker ladder

Every rung of this ladder writes these symbols. The right-hand column gives the name the code uses, so a line of `v2_learn.py` can be read against an equation directly.

## The canvas and its noise

| Symbol | What it is | Size | The code writes |
|---|---|---|---|
| $x_0$ | the clean canvas, pixels from −1 (background) to +1 (ink) | 1 × 32 × 64 | `x0`, from `to_x` |
| $t$ | the timestep, one of the 16 from 712 down to 480 | one number | `t`, from `SDXL_MATCHED` |
| $\bar\alpha_t$ | the share of the clean canvas's variance left at $t$: 0.0058 at 712, 0.0949 at 480 | one number | `ABAR[t]` |
| $\varepsilon$ | the noise mixed in, one standard normal number per pixel | 1 × 32 × 64 | `noise` |
| $x_t$ | the noisy canvas, $\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ | 1 × 32 × 64 | `xt` |

## The frozen model and the maps

| Symbol | What it is | Size | The code writes |
|---|---|---|---|
| $a,\ b$ | the two asked digits, $a < b$ | | `pair[:, 0]`, `pair[:, 1]` |
| $\varepsilon_a,\ \varepsilon_b,\ \varepsilon_\varnothing$ | the frozen U-Net's noise guess asked for digit $a$, digit $b$, and no digit (label 10) | 1 × 32 × 64 each | `ea`, `eb`, `e0` |
| $\varepsilon_{\text{product}}$ | the product of the two digit experts, $\varepsilon_a + \varepsilon_b - \varepsilon_\varnothing$ | 1 × 32 × 64 | not in `v2_learn.py` |
| $m_a,\ m_b$ | the two maps: $2048 \cdot |\varepsilon_a - \varepsilon_\varnothing| / \sum |\varepsilon_a - \varepsilon_\varnothing|$, and the same for $b$ | 1 × 32 × 64 each, stacked to 2 × 32 × 64 | `out` from `maps` |

## The checker and its training

| Symbol | What it is | The code writes |
|---|---|---|
| $\phi(t)$ | the 16 time features: sin and cos of $e^{3k/7}\log\frac{\bar\alpha_t}{1-\bar\alpha_t}/10$, $k = 0..7$ | `time_features(t)` |
| $\gamma,\ \beta$ | the scale and shift made from $\phi(t)$, 128 each, applied after `c1`: $h \leftarrow h(1+\gamma)+\beta$ | `g`, `b` in `PixelHead.forward` |
| $h$ | the hidden values after a layer | `h` |
| $\ell$ | the checker's one output, a logit; $\sigma(\ell)$ is its chance that the label is 1 | `head(...)[:, 0]` |
| $y$ | the label, 0 or 1; count and identity set it differently | `target` |
| $L$ | binary cross-entropy, $-[y\log\sigma(\ell) + (1-y)\log(1-\sigma(\ell))]$ | `F.binary_cross_entropy_with_logits` |
| $W$ | any trained weight of the checker | `head.parameters()` |

## The push

| Symbol | What it is | Where |
|---|---|---|
| $g$ | $\nabla_{x_t}\ell$: how the logit changes per pixel of $x_t$, through the checker, the maps and the three U-Net calls | `03-design.md`, step V3 |
| $b$ | the push's length as a share of $\lVert\varepsilon_{\text{product}}\rVert$; 0.4 for one checker | `03-design.md` |
| $\hat\varepsilon$ | what the sampler uses: $\varepsilon_{\text{product}} - \lVert\varepsilon_{\text{product}}\rVert\, b\, g/\lVert g\rVert$ | `03-design.md` |

## The miniature only

| Symbol | What it is |
|---|---|
| $p_1 .. p_4$ | the four pixels: $p_1, p_2$ the left half, $p_3, p_4$ the right half |
| $T_y$ | digit $y$'s stroke on both halves, the frozen expert's idea of where that digit's ink goes |
| $k = 0.5,\ g_e = 3$ | the frozen expert's strength and sharpness: $\varepsilon_y - \varepsilon_\varnothing = -k\,T_y\,\sigma(g_e x)$ |
| $K$ | the one convolution's kernel, 2 output channels × 2 maps × 2 taps, one window per half |
