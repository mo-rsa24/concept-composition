# Notation shared by this ladder's levels

| Symbol | Meaning | Where it is defined |
|---|---|---|
| $x_0$ | a clean 32 by 64 canvas, pixels from -1 to +1 | `build_env.make_canvas`; `canvas_unet.training_batch` scales it |
| $x_t$ | the canvas at timestep $t$: $\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ | `canvas_unet.py` lines 37 to 38 |
| $t$, step $k$ | $t$ is the timestep from 0 to 999; step $k$ is the $k$-th of 50 DDIM steps, $t = $ `ddim_timesteps(50)[k]` | `canvas_unet.py` line 168 |
| $c$ | one label: a digit 0 to 9, or 10 for no digit ($\varnothing$) | `canvas_unet.py` line 35 |
| $\epsilon_\theta(x_t, t, c)$, $\epsilon_c$ | the frozen digit U-Net's noise prediction | `CanvasUNet.forward` |
| $d_t(c)$, $d(c)$ | $\epsilon_c - \epsilon_\varnothing$: what label $c$ adds | `product_eps`, `v2_learn.maps` |
| $\epsilon_{\mathrm{PoE}}$ | $\epsilon_\varnothing + d_t(a) + d_t(b)$, the product's prediction | `product_eps` at $g = 1$ |
| $S(c)$ | the score path: $d_t(c)$ at steps 0 to 15, 16 by 32 by 64 | the session note; no code yet |
| $T$, $I$, $S$ | text, canvas and score-path embeddings | figures sp3 and sp4 |
| $z$, $e$ | the miniature's path embedding and caption vector | `2-miniature/miniature.py` |
| $r$ | the miniature's hand-set interaction term in its toy joint label | `2-miniature/miniature.py` |
| $\tau$ | the contrastive temperature, 0.07 in W4 | `w4_train.py` line 48 |
