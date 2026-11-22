# Du: first-read notes

PDF: arXiv 2302.11552v6, 23 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. The dashes in the Score rows of the Table 1 Mixture columns are printed as dashes and have no row.

## Plot-only curves

Figure 5 (p7) plots accuracy against MCMC sampling steps for MALA, LA and Reverse on combinations of 5 cubes. It prints no values, so it has no rows.

## How rows are attributed

Page 6 says the score model with standard reverse diffusion sampling "is exactly the approach of Liu et al. (2022)". Rows for `Score, Reverse` in Tables 1, 2 and 3 therefore carry `paper` = `Liu`. They were measured by Du, on Du's models.

## Protocol facts a comparison depends on

- 2D densities: residual MLP with 4 residual blocks, 100 discrete timesteps, linear beta schedule (p18). HMC uses 3 steps per timestep with 3 leapfrog steps, and MALA uses 10 steps per timestep (p18).
- All MCMC experiments in Table 1 use the same number of score function evaluations (p7).
- CLEVR: 200,000 images at 64x64 with 1 to 5 cubes, each conditioned on one cube position (p7, p18). A cube counts when binary classifier confidence exceeds 0.5 (p18).
- CLEVR samplers: 40 MALA steps or 13 HMC steps with 3 leapfrog steps, per timestep (p18). The sample count behind Table 2 is not stated.
- ImageNet: 128x128, 1000 discrete timesteps, cosine beta schedule (p19). Inception Score and FID use 50,000 generated samples (p19). Accuracy counts a sample when the classifier gives the specified class more than 50% (p19), and its sample count is not stated.
- ImageNet samplers: 6 MALA steps or 2 HMC steps with 3 leapfrog steps, per timestep (p19).
- Table A1 FID uses 1k samples (p17), so it is not comparable with the 50,000-sample FID in Table 3.
- Step sizes are tuned so average acceptance is approximately 60% for MALA and 70% for HMC (p14). This is a tuning target and has no row.

## Hardware and training time

- 2D models: 15,000 iterations, 10 minutes on 8 TPUv2 cores (p18).
- CLEVR models: 20,000 iterations at batch size 1024, roughly 8 hours on 8 TPUv2 cores (p18).
- ImageNet models: 1 million iterations at batch size 64, 3 days on 16 TPUv2 cores (p19).
- Text-to-image models: one week on 400 million images with 32 TPUv3 cores at batch size 256 (p19).

## Contradictions inside the paper

- Table 1 (p6) bolds HMC at 2.72 as the best product LL, where higher is better. MALA in the same column prints 2.73.
- Page 6 defines the energy as f = -||s||^2. Page 17 defines the L2 parameterization as f = -(1/2)||s||^2.
- Figure 5 (p7) shows the MALA curve above its 65.0 tick by 25 sampling steps on 5 cubes. Table 2 (p6) prints 63.9 for EBM MALA on 5 cubes, and p18 states 40 MALA steps per timestep for CLEVR. The paper does not say whether the two runs differ in setting.
