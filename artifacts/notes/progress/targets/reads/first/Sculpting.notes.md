# Sculpting: notes on the first read

PDF: arXiv 2309.16115v1, 36 pages. Page numbers are PDF pages.

## Numbers with no row

Every printed result digit was readable. These are the choices that shaped the rows.

- **Table F.4 is missing from the inventory**, because its caption sits on the same line as Table F.3. It is on p36 and is in the coverage file with 36 rows.
- **Table F.4 cells print as A / B.** Each cell has two rows, one for the mode count A and one for the molecule count B.
- **Elapsed times in Tables F.1 and F.2** print as hours, minutes and seconds. Each printed part has its own row, so no time was converted.
- "More than 70%, or even 90%" (p33) is a loose restatement of the 73% and 97% in Tables F.1 and F.2, so it has no row.

## Plot-only curves

Figure 4 (reward densities, p12), Figure 5 (t-SNE, p13) and Figures F.6, F.7 and F.8 (classifier loss and L1 curves, p35) print no values at their points. The other figures are image samples.

**The diffusion experiments have no quantitative image result.** Colored MNIST and MNIST composition appear only as sample grids (Figure 6 p14, Figures F.1 to F.5 p31 to p34). The only diffusion numbers are the training costs in Table F.2. This paper can set a target on molecules or the 2D grid, and none on image composition.

## Protocol facts a comparison depends on

- **2D grid.** 32 x 32 cells. L1 distance is against the ground-truth composition computed from the known base probabilities (p29).
- **Molecules.** Rewards SEH, SA and QED normalized to [0, 1]. 5000 molecules are sampled from each distribution (p13). Table 1 and Table F.5 use beta = 32 (p14).
- **Table 1 bins.** Low and high split at SEH 0.5, SA 0.6, QED 0.25 (p13).
- **Table F.3.** Similarity is over the top 1000 of the 5000 molecules (p36).
- **Table F.4.** Reward thresholds are SEH 0.875, SA 0.75, QED 0.75. A new mode has Tanimoto similarity below 0.7 to every earlier mode (p36).
- **Table F.5.** Earth mover's distance with ground distance (max{s, 10^-3})^-1 - 1 on Tanimoto similarity over Morgan fingerprints (p13).
- **Diffusion models.** VE SDE with a U-Net score network, trained for 200 steps at batch size 32. Sampling uses the Predictor-Corrector sampler over 500 time steps, with classifier guidance scaled by 10 for the first classifier and 75 for the second (p30). The two-model experiment scales guidance by 20 (p31).
- The paper states no seed count and prints no uncertainty on any result.

## Hardware and training time

- **Molecules, one GeForce RTX 2080 Ti (Table F.1, p35).** Base GFlowNet 6h 47m 11s. Classifier 9h 2m 19s, of which data generation is 6h 35m 58s (73%).
- **MNIST diffusion, one Tesla V100 (Table F.2, p35).** Base diffusion 10m 6s. Classifier 30m 12s, of which data generation is 29m 22s (97%).

## Where the paper contradicts itself

1. **Base GFlowNet training steps.** Table F.1 prints 20 000 steps for the molecule domain (p35). Appendix E.2 says the molecule GFlowNets trained for 15 000 steps (p29).
2. **QED similarity.** The text says average pairwise similarity for the SA and QED models does not grow as beta increases (p33). Table F.3 has QED rising from 0.398 at beta 32 to 0.404 at beta 96 (p36).
3. **SEH modes.** The text says the total number of separated SEH modes grows as beta increases (p33). Table F.4 has it falling from 15 at beta 1 to 12 at beta 4 (p36).
4. **Possible label mismatch.** Figure 3 and the text label the parameterized contrast with subscript 0.95 and alpha = 0.05 (p12). The Figure F.6 legend prints subscript 0.05 (p35). The paper does not say whether the two labels name one operation, so the 0.167 L1 row (p29) keeps the Figure 3 label.
5. **Printed learning-rate decay.** Appendix E.2 prints the policy decay exponent as step/20 0000, with a digit grouping that reads as neither 20 000 nor 200 000 (p29). It is a training setting, so no row depends on it.
