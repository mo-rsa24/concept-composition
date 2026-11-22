# Bradley-Projective: first-read notes

PDF: arXiv 2502.04549v3, 40 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. Blank cells in Table 1 (p17) are not printed numbers and have no row. They are not recorded as zero.

The text gives two approximate results that have no row: explicit composition reaches "about 8 objects" (p37), and color-conditioned explicit composition does not generalize "beyond 2-3 objects" (p39).

## Plot-only curves

Figure 5 histograms (p5), Figure 8 cosine heatmaps (p11) and Figure 17 score-difference curves (p40) print no values, so they have no rows.

## How rows are attributed

Page 37 says the learned composition results in Table 2 are from Bradley (2025), which is arXiv 2509.16447. The `Learned` column and the color result of 7 (p37, p39) therefore carry `paper` = `Bradley-Local`.

## Protocol facts a comparison depends on

- CLEVR images are rendered at 320x240 and downsampled to 128x128 (p15).
- Datasets: 50,000 background images, 1,550,000 single-object images, and 500,000 images per object count for 1 to 5 objects (p15).
- The model is an EDM2 edm2-img64-xs re-implementation with 124M trainable weights (p15).
- Sampling uses 65 model calls with a 32-step Heun sampler (p15). CLEVR experiments use no guidance (p15).
- Weights are averaged over the last 4096 training updates, with no post-hoc EMA (p15).
- Table 1 counts 100 samples per method by hand, for N = 1 to 6 (p17).
- Table 2 uses the success rule of Bradley (2025) (p39). This paper does not state the sample count behind Table 2.
- The switch background uses the learned background for the first 12 denoising steps (p39).

## Hardware and training time

Training uses a batch size of 2048 over 128 x 2^20 samples and takes around 16 hours on 32 A100 GPUs (p15).

## Contradictions inside the paper

- The Table 1 caption (p17) says 100 samples per method. The Single-object Bayes row at N = 2 sums to 109 (10 + 67 + 32), and the row at N = 6 sums to 43 (34 + 6 + 3). The digits were confirmed in a 300 dpi crop and in the extracted text, and are copied as printed. Every other row sums to 100.
- Page 17 prints P(k > 5) = 0.026 for Binomial(9, 0.3). The exact value is 0.0253. This is an illustration and has no row.

## Disagreement with Bradley-Local

Table 2 (p39) and p37 give the learned composition trained on 1-3 objects a Kmax of 9, citing Bradley (2025). Bradley-Local v3 prints 7 for its grid-conditioned model with all objects labelled, trained on 1-3 objects (Table 1 and text, p3). It prints 9 only for its local patch models, Exp.2L and Exp.3L. The row is copied as printed here, so any target using it must choose between the two.
