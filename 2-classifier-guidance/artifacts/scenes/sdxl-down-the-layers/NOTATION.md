# SDXL ladder notation

One line per symbol the SDXL ladder uses, with the name the code gives it. Values come from `sdxl_numbers.json`. Where
a symbol is shared with the CO3 ladder it is written the same way, and the third column says so; the CO3 ladder's
symbols are in [its notation file](../co3-sampler/NOTATION.md).

| Symbol | What it is | In the code, or in the CO3 ladder |
|---|---|---|
| $t$ | the timestep, one of 981, 961, ..., 1 (50 values, 20 apart) | `t`; same as the CO3 ladder |
| $i$, step | the sampling step, 0 to 49, with $t = 981 - 20i$ | `i`; same as the CO3 ladder |
| $x_t$ | the SDXL latent at timestep $t$, 4 × 128 × 128 | `sample` in the U-Net, `x` in CO3; same as the CO3 ladder |
| $x_T$ | the starting noise, a standard Gaussian draw of 4 × 128 × 128 | same as the CO3 ladder |
| $x_t^{(2)}$ | the latent after CO3's passes at one of steps 0 to 5 | `x_2`; same as the CO3 ladder |
| $x_0$ | the clean latent the noise was added to | |
| $\epsilon$ | Gaussian noise the size of $x_t$ | |
| $\bar\alpha_t$ | the share of $x_t$'s variance that is picture: 0.0058 at step 0, 0.9171 at step 45 (the scheduler's `alphas_cumprod[t]`; CO3's code reads the value at $t-1$) | `alphas_cumprod`; same as the CO3 ladder |
| $\hat\epsilon$ | the U-Net's noise guess after guidance, 4 × 128 × 128 | `noise_pred` |
| $\epsilon^{\phi}$ | the U-Net's noise guess for the empty prompt | `noise_pred_uncond`; same as the CO3 ladder |
| $C$ | the joint prompt, "an eagle and a fox" | `prompt`; same as the CO3 ladder |
| $\epsilon^{C}$ | the noise guess for the joint prompt | `noise_pred_cond`; same as the CO3 ladder |
| $c_1$, $c_2$ | each animal's own prompt, "eagle" and "fox" | `prompts_single`; same as the CO3 ladder |
| $\epsilon^{c_k}$ | the noise guess for animal $k$'s prompt alone | `eps[2]`, `eps[3]` in Z2's capture; same as the CO3 ladder |
| $\lambda$ | guidance strength, 0.8 in the released CO3 settings | `guidance_scale`; same as the CO3 ladder |
| $\hat x_0$ | the clean guess, $(x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$, the Tweedie mean divided by $\sqrt{\bar\alpha_t}$ | `denoised_tweedie`; same as the CO3 ladder |
| $d_t(c_k)$ | Z2's map for animal $k$: $\epsilon^{c_k} - \epsilon^{\phi}$ on one latent, 4 × 128 × 128 | `dA`, `dB` in `z2_predictors.py` |
| $Q$ | the queries, one row per patch: 1,024 × 1,280 at 32 × 32, 4,096 × 640 at 64 × 64 | `to_q(hidden_states)` |
| $K$, $V$ | the keys and values, one row per word slot: 77 × 1,280 at 32 × 32, 77 × 640 at 64 × 64 | `to_k`, `to_v` of the word vectors |
| $d$ | the numbers per head: 64 in SDXL, 2 in the hand-worked head | `head_dim`; `scale` = $1/\sqrt{d}$ |
| $A$ | the attention weights, softmax of each row of $QK^\top/\sqrt{d}$; 1,024 × 77 per head at 32 × 32, each row summing to 1 | `attention_probs` |
| $A_{p,w}$ | the weight patch $p$ gives word slot $w$ | one entry of `attention_probs` |
| $q_p$, $k_w$, $v_w$ | in the hand-worked head: patch $p$'s query, word $w$'s key and value, each 2 numbers | |
