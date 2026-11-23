# Notation for the word-map checker ladder

Every rung of this ladder writes these symbols. The right-hand column gives the symbol Learn-to-Steer prints, so its equations can be read directly.

## The sampler

| Symbol | What it is | Size on SDXL | The paper writes |
|---|---|---|---|
| $x_t$ | the noisy latent at timestep $t$ | $4 \times 128 \times 128$ | $z_t$ |
| $\bar\alpha_t$ | the share of signal variance left at timestep $t$ | one number | not used |
| $\sigma_t$ | the noise scale, $\sqrt{1-\bar\alpha_t}$ | one number | not used |
| $\varepsilon_{\text{cat}},\ \varepsilon_{\text{dog}},\ \varepsilon_\varnothing$ | the U-Net's noise prediction for the prompt "a cat", the prompt "a dog" and the empty prompt | $4 \times 128 \times 128$ each | not used |
| $\varepsilon_{\text{PoE}}$ | the product of experts' prediction, $\varepsilon_{\text{cat}}+\varepsilon_{\text{dog}}-\varepsilon_\varnothing$ after guidance on each prompt | $4 \times 128 \times 128$ | not used |

## The word maps

| Symbol | What it is | Size | The paper writes |
|---|---|---|---|
| $s,\ o$ | the subject word and the object word ("cat", "dog") | | $s,\ o$ |
| $A^{(\ell)}_{s}$ | the subject word's map from layer-and-head number $\ell$: the softmax weight that word gets at each patch | one grid | one entry of $A_s$ |
| $L,\ H$ | the number of cross-attention layers and of heads per layer | SDXL: $L = 70$, heads 10 or 20 | $L,\ H$ |
| $\hat A^{(\ell)}_{s}$ | the same map resized to $16 \times 16$ and divided by its sum, so it adds to 1 over the grid | $16 \times 16$ | not named |
| $\bar A_{s},\ \bar A_{o}$ | one word's maps pooled into a single map | $16 \times 16$ | $A^{\text{agg}}$ |

## The head and its loss

| Symbol | What it is | The paper writes |
|---|---|---|
| $C$ | the trained classifier that reads the two words' maps | $C$ |
| $r$ | the relation asked for: above, below, left of, right of | $r$ |
| $\hat p$ | the head's probabilities over the relation classes | $\hat p$ |
| $\mathcal L_t$ | the loss at timestep $t$, $-\log \hat p[r]$ | $\mathcal L_t$ |
| $\eta$ | Learn-to-Steer's step size on the latent, 5 on SD 2.1 and Flux | $\alpha$ |
| $\lambda$ | our push's strength on the product's prediction (the code's `strength`) | not used |
| $c$ | our cap: the push's norm may be at most $c\,\lVert\varepsilon_{\text{PoE}}\rVert$ | no cap stated |

The paper's $\alpha$ is written $\eta$ here because $\bar\alpha_t$ already means the noise schedule.

## The miniature only

| Symbol | What it is |
|---|---|
| $p$ | one of the four patches: top left, top right, bottom left, bottom right |
| $u_p,\ v_p$ | the patch's horizontal position ($-1$ left, $+1$ right) and vertical position ($+1$ top, $-1$ bottom) |
| $k_1,\ k_2$ | a word's key in cross-attention layer 1 and layer 2, set to 2 and 1 |
| $w_1,\ w_2$ | the pooling weights, set to $3/4$ and $1/4$; they stand for the learned query |
| $\Delta_u,\ \Delta_v$ | the head's two hidden units: how far the subject's centre of mass sits right of, and above, the object's |
| $\kappa$ | the head's scale from hidden units to logits, set to 4 |
| $\tau$ | the trace of the prompt's relation word in a map, in the leakage example ($+1$ "right of", $-1$ "left of") |
