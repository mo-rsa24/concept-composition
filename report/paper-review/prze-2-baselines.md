# PRZE-2: the Langevin corrector and the qualitative comparison

Answers PrzE's second weakness (only CO3, no MCMC corrector).

## Outcomes

- [ ] ✅ check passed: our update rule on two Gaussians with a known product lands at mean -0.2308 and variance 0.3077 within a stated tolerance, and the corrector-first order and the paper's step-first order agree or their difference is measured
- [ ] 💻 code ran: a joint-prompt Langevin driver and an unadjusted HMC driver, each byte-identical to the plain sampler with zero corrector steps
- [ ] 🖼️ screenshot: one figure with five columns (joint, joint + Langevin, PoE, PoE + Langevin, CO3), every column from the same starting noise, shown seeds fixed in advance
- [ ] ✏️ markdown updated: Figure 5's caption and Section 4.1 agree on whether CO3 starts from the same noise
- [ ] ✏️ markdown updated: the related-work paragraph says which samplers ran, which cannot run on SDXL (adjusted MALA and HMC need an energy), and why
