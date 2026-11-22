# CoInD: notes on the first read

PDF: arXiv 2503.01145v1, 33 pages. Page numbers are PDF pages.

## Numbers with no row

Every printed result digit was readable. These printed items have no CSV row, each for the reason given.

- The dashes in Table 2 (p9), Table 3 (p19), Table 4 (p20), Figure 4a (p8), Figure 6a (p9) and Figure 15b (p31) are cells the paper leaves empty. LACE has no JSD because it trains one model per attribute (p22). LACE also has no CS or FID for the joint prompt on CelebA (p9).
- ">4x" (p9), "nearly 4x" (p25) and "2 to 10x" (p8) are a bound or a range, so none is a single value.
- H(uniform) = 3.32 (p9) is the theoretical maximum of the entropy for 10 colors and measures no model.

## Result tables printed as figures

Six figures print exact result values. Their rows use `kind` `plot` and `source_table` `CoInD Figure <label>`, as the reading rule says for a number printed on a figure. The coverage file lists only numbered tables, because the checker counts rows by `CoInD Table <label>`.

| Figure | Page | Rows | What it holds |
|---|---|---|---|
| 4a | 8 | 37 | Colored MNIST JSD and CS, the paper's main result |
| 5 | 8 | 3 | Color entropy H per method |
| 6a | 9 | 28 | Shapes3D JSD, R2 and CS |
| 10 | 24 | 10 | Accuracy of the attribute classifier that computes CS |
| 15b | 31 | 11 | Colored MNIST with Gaussian support |
| 17 | 32 | 100 | CS per digit and color, partial support |

A target that filters on `kind` = `published` drops Figures 4a and 6a. Table 3 repeats every Figure 4a value, and Table 4 repeats every Figure 6a value, with those two tables carrying `published` and `baseline` rows.

## Plot-only curves

Figures 4b (p8), 7a and 7b (p10), 8 (p20), 12a and 12b (p26) and 15c (p31) print no values at their points, so they have no rows.

## Protocol facts a comparison depends on

- **Method names.** Composed GLIDE is Liu et al. 2022, the Liu rival, so its rows use `paper` `Liu`. CoInD retrained it on these datasets, so these are CoInD's numbers for Liu's method (p7).
- **Colored MNIST.** 28x3x3 images, 10 digits and 10 colors. Diagonal partial support gives digit d the colors d and d+1, and digit 9 one color (p23).
- **Shapes3D.** 64x3x3 images, six attributes, orthogonal partial support following Schott et al. (p7, p22, p23).
- **CelebA.** 128x128, encoded with the SD3 latent encoder, smiling and male attributes, every combination seen except smiling male (p22, p23).
- **Evaluation set.** On Colored MNIST and Shapes3D the composition tasks score unseen compositions under partial support. Non-uniform support scores the compositions seen less often (p22).
- **CS.** A single ResNet-18 with one head per attribute, trained on full support (p23). Its accuracy is in Figure 10 (p24).
- **FID.** Computed on the smiling male subset from 10000 generated samples (p22). The paper does not state a sample count for CS or for the SDv1.5 FIDs in Table 6.
- **Training.** DDPM linear schedule, 1000 noise steps, 50000 steps for Colored MNIST CoInD and Composed GLIDE and 100000 for the others (Table 5, p22). The paper states no seed count and prints no uncertainty on any result.
- **Model size.** 8.2M parameters on Colored MNIST, 8.2M x 2 for LACE. On Shapes3D, 17.2M, and 8.2M x 6 for LACE (Table 5, p22).

## Hardware and training time

CelebA trained for 500,000 steps on one A6000 GPU (p22). The paper states no training time.

## Where the paper contradicts itself

1. **Sampling steps.** Table 5 prints 150 DDIM steps for Colored MNIST (p22). The text on the same page says DDIM with 100 steps.
2. **CelebA composition.** The Table 2 caption (p9) and App. E.1 (p25) say CoInD beats the baselines on CS across tasks. On "smiling" ∧ "male", CoInD has CS 8.79, below LACE at 24.20 and Composed GLIDE at 10.55.
3. **CelebA joint ratio.** The text says CoInD beats the baseline by more than 4x in joint sampling (p9), and "nearly 4x" on p25. Table 2 gives 8.63 against 2.51, a ratio of 3.44.
4. **SDv1.5 negation ratio.** The text says a 2x improvement in ¬ CS (p27). Table 6 gives 18.80 against 11.02, a ratio of 1.71. The stated 3.46x for ∧ does match Table 6 (49.15 against 14.19).
5. **Ratio against LACE.** App. F.4 (p30) says 2.4x better CS on Colored MNIST diagonal partial support. The partial-support rows give 4.83 for ∧, 5.90 for ¬ Color and 1.86 for ¬ Digit, so no task gives 2.4. The stated 1.4x on Shapes3D sits near the ∧ ratio of 1.47, and the ¬ ratio is 1.79.
6. **Figure 17 against the headline.** The text says CoInD reaches 52.38% CS on unseen combinations in this task (p32). The mean of the 81 printed cells outside the seen diagonal is 60.78. The cells are rounded to integers, which cannot explain a gap of 8.40.
7. **Figure 17 against its description.** The text says colors 2 and 3 have zero CS with more digits than others, and that colors 4, 5 and 6 have high CS with all digits (p32). The printed zeros are both in color 8. Color 5 has 49 and 39 at digits 8 and 9, and color 6 has 53 and 52 at digits 0 and 1.
8. **Missing table number.** The text on p24 cites "Table 10". The classifier accuracies are in Figure 10, and the paper has no Table 10.
9. **Support label.** Table 4 calls a row "Partial" (p20) and Figure 6a calls the same numbers "Orthogonal" (p9).
