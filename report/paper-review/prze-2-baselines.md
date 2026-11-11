# PRZE-2: the Langevin corrector and the qualitative comparison

Answers PrzE's second weakness (only CO3, no MCMC corrector).

## Outcomes

- [x] ✅ check passed: our update rule on two Gaussians with a known product lands at mean -0.2308 and variance 0.3077 within a stated tolerance, and the corrector-first order and the paper's step-first order agree or their difference is measured ✓ verified 2026-11-11: `python3 scripts/check_langevin_two_gaussians.py` prints `PASS: all four step sizes`, judged against the chain's own stationary variance V / (1 − δ/(2V)) with the gap to 0.3077 printed beside it; `python3 scripts/check_langevin_order.py` measures the two orders on SDXL's schedule
- [x] 💻 code ran: a joint-prompt Langevin driver and an unadjusted HMC driver, each byte-identical to the plain sampler with zero corrector steps ✓ verified 2026-11-11: Slurm 84583, `PASS: both drivers at k = 0`, `max |diff| = 0.000e+00` on both
- [x] 🖼️ screenshot: one figure with five columns (joint, joint + Langevin, PoE, PoE + Langevin, CO3), every column from the same starting noise, shown seeds fixed in advance ✓ 2026-11-11: <a href="../../artifacts/results/does-a-langevin-corrector-fix-the-product/five_columns.png"><img src="../../artifacts/results/does-a-langevin-corrector-fix-the-product/five_columns.png" width="200" alt="R6, four pairs by five samplers at seed 10"></a> seed 10 (rule 1's shared seed), every column from `latents/seed10.pt`
- [ ] ✏️ markdown updated: Figure 5's caption and Section 4.1 agree on whether CO3 starts from the same noise
- [ ] ✏️ markdown updated: the related-work paragraph says which samplers ran, which cannot run on SDXL (adjusted MALA and HMC need an energy), and why
