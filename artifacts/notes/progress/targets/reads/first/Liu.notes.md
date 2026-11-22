# Liu: first-read notes

PDF: arXiv 2206.01714v6, 30 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. The dashes in the StyleGAN2-ADA rows of Tables 1, 2 and 3 (2 and 3 Components) are printed as dashes and have no row.

## Plot-only curves

None. Figures 1 to 15 are image grids and diagrams, with no plotted curves.

## Protocol facts a comparison depends on

- Each method generates 5,000 images per test setting (p9).
- FID is Clean-FID on 5,000 generated images, where the paper notes 50,000 is the usual count (p9). These FIDs are not comparable with a 50,000-image FID.
- Accuracy is the percentage of generated images capturing all concepts, judged by a binary classifier trained on real images (p9).
- CLEVR has 30,000 training images at 128x128, each with 1 to 5 objects and one object position label (p8).
- Relational CLEVR has 50,000 training images at 128x128, each with one relation label (p9).
- FFHQ has 51,067 images labelled by classifiers (p9). Its evaluation classifiers are trained on 30,000 CelebA-HQ pairs, split 24,000 and 6,000 (p20).
- CLEVR classifier: 24,000 training and 6,000 validation pairs, 99.05% validation accuracy (p19, p20).
- Relational CLEVR classifier: 40,000 training and 10,000 validation images, 99.80% validation accuracy (p20).
- FFHQ classifiers reach 95.01% (smiling), 99.20% (glasses) and 97.49% (gender) validation accuracy (p20).
- FFHQ images are generated at 64x64 and upsampled to 256x256. CLEVR images are generated directly at 128x128 (p21).
- Diffusion models use 1,000 diffusion steps and the cosine noise schedule (p23).
- Seeds and guidance weights for the table numbers are not stated.

## Hardware and training time

- Each model is trained on a single Tesla V100 32GB GPU (p22).
- The diffusion model trains for seven days on CLEVR (750,000 iterations) and two days on FFHQ (250,000 iterations) (p23).
- StyleGAN2-ADA, StyleGAN2 and EBM each train for two days (p22, p23).
- A binary classifier takes 2 hours to train. The LACE classifier takes less than 10 minutes (p22).

## Contradictions inside the paper

- The Table 2 caption (p12) says the EBM FID "is much lower than other methods". The table shows EBM with the highest FID in all three settings (44.41, 55.89, 58.66), and p13 says its FID scores are "much worse".
- EBM is cited as [7] in Tables 1 and 3 and as [28] in Table 2. The `paper` column keeps each printed citation so the rows are not merged.
