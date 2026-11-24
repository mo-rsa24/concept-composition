# CO3 notation

One line per symbol the CO3 ladder uses, in the paper's notation, with the name the code gives it. Values come from
`co3_numbers.json`. Every figure and the code walk use these symbols and no others.

| Symbol | What it is | In the code |
|---|---|---|
| $t$ | the timestep, one of 981, 961, ..., 1 (50 values, 20 apart) | `t`, from `self.scheduler.timesteps` |
| $i$ | the sampling step, 0 to 49, with $t = 981 - 20i$ | `i` in `sample_loop` |
| $T$ | the first timestep, 981 | `self.start_t` |
| $T_c$ | the timestep after which CO3 stops; the release stops after 6 steps (t = 881) | `num_ts_to_correct` = 6 |
| $T_R$ | the timestep after which the resampler hands over to the corrector; the release resamples 3 steps (t = 981 to 941) | `num_resampling_steps` = 3 |
| $x_t$ | the SDXL latent at timestep t, 4 × 128 × 128 | `x` |
| $x_T$ | the starting noise, a standard Gaussian draw | `normal` in `run_sampling`, `start_noise` in `co3_common.py` |
| $x_t^{(2)}$ | the latent after one CO3 pass | `x_2` |
| $\bar\alpha_t$ | the share of the clean latent's variance still in $x_t$ (0 is pure noise, 1 is clean); the code reads the scheduler's value at $t-1$ | `at`, from `self.alpha(t)` |
| $\sqrt{1-\bar\alpha_t}$ | the noise scale at t | `(1 - at).sqrt()` |
| $\psi$ | the SDXL U-Net, the denoiser, frozen | `unet` |
| $C$ | the joint prompt, "an eagle and a fox" | `prompt_orig`, `noise_pred_multi` is its prediction |
| $c_k$ | concept k's prompt; $c_1$ = "eagle", $c_2$ = "fox" | `prompts_single` |
| $c_0$ | another name for $C$ in Algorithms 2 and 3 | |
| $K$ | the number of concepts, 2 here | `num_concepts` |
| $\epsilon^{c_0}$, $\epsilon^{C}$ | the U-Net's noise prediction for the joint prompt | `noise_pred_multi` |
| $\epsilon^{c_k}$ | the noise prediction for concept k | `noise_pred_concepts[k-1]` |
| $\epsilon^{\phi}$ | the noise prediction for the empty prompt | `noise_pred_uncond` |
| $\lambda$ | guidance strength; 0.8 inside CO3 (`lmda`) and 0.8 in the ordinary step (`guidance_scale`), two separate settings | `self.lmda`, `self.config.guidance_scale` |
| $\epsilon^{\lambda,c}$ | the guided prediction $\epsilon^{\phi} + \lambda(\epsilon^{c} - \epsilon^{\phi})$ (equation 4, p. 3) | `noise_pred` in lines 107, 112 and 354 |
| $\hat x_{\mathrm{tweedie}}[\epsilon]$ | the Tweedie mean of a prediction, $x_t - \sqrt{1-\bar\alpha_t}\,\epsilon$ (p. 3) | `tweedie_multi`, `tweedie_single` |
| $\hat x_0$ | the clean-latent estimate, the Tweedie mean divided by $\sqrt{\bar\alpha_t}$ (equation 5) | `denoised_tweedie` |
| $\tilde x_{\mathrm{tweedie}}$ | the composed Tweedie mean, CO3's weighted sum (equation 13) | `denoised_tweedie_at` (resampler), `composed_tweedie` (corrector) |
| $w_k$ | composition weight k; $w_0$ = 1 in the resampler (sum 0) and 2 in the corrector (sum 1) | `composition_weights` |
| $P$ | the number of CO3 passes at one timestep: 10, 10, 5, 5, 5, 5 | `num_corrector_steps` |
| $d_k$ | the length of $\epsilon^{C} - \epsilon^{c_k}$ over all 65,536 numbers of the latent | `dists` |
| $a_k$ | concept k's affinity, $\exp(-\beta d_k)$ (equation 14) | `a` |
| $\beta$ | how sharply affinity falls with distance; 1.0 at t = 981, 0.9 after | `beta` |
| $r$ | the corrector's rescale factor; a ratio of lengths in the paper, a ratio of largest entries in the code | `r` |
