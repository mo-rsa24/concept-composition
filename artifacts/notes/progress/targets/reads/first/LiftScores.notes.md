# LiftScores first read

Improving Compositional Generation with Diffusion Models Using Lift Scores (CompLift), arXiv 2505.13740v2, 28 PDF pages.
Page numbers are PDF pages counted from 1, and they match the printed footer.

## What was read

Thirteen tables and 556 rows. Table 1 (page 3) holds no results because it only lists the Compose rule for each algebra.
The inventory misses Table 12 (minCLIP, page 28), so the coverage file adds it. Rows outside the tables come from Figure 5
(page 5), Figure 9 (page 8), Figure 10 (page 9), Figure 20 (page 23) and the running text on page 27.

Method names: Composable Diffusion is `Liu`, the EBM samplers are `Du`, Attend-and-Excite is `AttendExcite`, and every
CompLift variant is `LiftScores`. The printed row name stays in `model`.

## Numbers that could not be read

Figure 18 (page 21) prints classifier confidence percentages on individual boxes, and several labels overlap on the
crowded rows. They describe single boxes in example images, so no row was written.

Figure 9 (page 8) prints 100.0, 100.0 and 99.7 where the curves overlap, so those three labels cannot be assigned to one
curve from the figure alone and have no row. Table 5 carries the same accuracies by method.

## Plot-only, no rows written

Figure 2 (page 4), accuracy against number of trials. Figure 3 (page 4), accuracy against timestep. Figure 6 (page 5),
lift-score heat maps. The FID panel of Figure 9 (page 8), whose values are in Table 6. Figure 17 (page 18), histograms of
CompLift scores. The scatter points of Figures 10 and 20 (pages 9 and 23).

## Protocol facts a comparison depends on

**2D synthetic (page 7).** 8000 data points per component distribution, one 3-layer MLP energy model per distribution, 50
diffusion timesteps for training and inference, 8000 generated samples per method. Cached CompLift uses 50 trials and
vanilla CompLift uses 50 or 1000.

**CLEVR Position (pages 8 and 22).** Pretrained model of Liu et al., CFG scale 7.5, 10 initial samples per composition, one
trial at t = 928 with importance sampling. All methods are tested on 5000 combinations of positions with various numbers
of constraints, and the paper does not say whether 5000 is per constraint count. Tables 5 and 6 use a SAM2 background-mask
verifier and Table 7 uses the classifier of Liu et al. The two verifiers give very different accuracies: Composable
Diffusion at 3 constraints scores 99.0 in Table 5 and 3.1 in Table 7.

**Text to image (page 8).** SD 1.4, SD 2.1 and SDXL on the Attend-and-Excite prompts: 66 animal, 66 object and 144
animal-object combinations. 5 initial samples per prompt, 50 inference timesteps, CFG scale 7.5, tau 250. When every
sample is rejected, one image is drawn uniformly. Resolution and seeds are not stated.

**CompLift is a rejection method.** Its scores are computed on the accepted subset of the initial samples, so they are
not comparable with a method scored on all its samples without saying so.

**Empty sets count as accurate (pages 8 and 19).** The paper says rejection can generate empty sets by construction. In
Table 4, Scenario 3 of Product and of Negation shows CompLift (T=1000) at accuracy 100.0 with an acceptance ratio of 0.0,
which means no sample was kept. Table 2 averages the three scenarios, so these 100.0 entries raise its Product and
Negation accuracies.

**MCMC caveat (page 22).** The authors could not reproduce the MCMC results on CLEVR and say those rows should be read
with caution.

**Rebuttal tables (pages 27 and 28).** Tables 10 to 13 come from the rebuttal appendix and state fewer settings. Table 13
uses 100 hand-labelled samples, 62 positive and 38 negative.

## Hardware and training time

No hardware is named. Figure 5 gives average running times on the 2D task from 0.24 s to 23.67 s. The paper states no
training time.

## Contradictions inside the paper

**The average gain on page 27 does not match Table 10.** The text says 40% for vanilla CompLift and 30% for cached. The
mean of the 18 cells in Table 10 is 47.84% for vanilla and 40.79% for cached. CLIP cells alone average 37.00% and 31.20%,
and IR cells alone average 58.67% and 50.38%, so no reading of the table gives 40% and 30%.

**The setup and Table 3 differ on trial count.** Page 8 says vanilla CompLift uses 200 trials on text to image, and
Table 3 also prints a T=50 row.

Table 2 agrees with the mean of the three Table 4 scenarios on all 48 cells, within the rounding of Table 4's one-decimal
accuracies and three-decimal distances. The shared rows of Tables 3 and 9 are identical.
