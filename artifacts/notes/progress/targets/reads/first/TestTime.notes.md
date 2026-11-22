# TestTime: notes on the first read

PDF: arXiv 2605.07078v1, 18 pages. Page numbers are PDF pages.

## Numbers with no row

Every printed result digit was readable. The running text on p8 and p9 repeats values from Tables 1 and 2, so those repeats have no rows of their own. "Near 5% to 7%" (p8) is an approximate range.

## Plot-only curves

The paper has no plotted result curves. Figures 1 to 6 are diagrams and image samples.

## Method names

- `PoE` and `+LoRA` in Tables 1 and 2 are the two configurations of this paper's own method (p7), so their rows use `paper` `TestTime`, and the `model` column names the configuration. `PoE` here is the analytic PoE teacher built from discovered prototypes, and it is a different method from plain PoE of two prompts.
- `Top-1`, `Top-3` and `Query-only` are baselines defined on p7. `Top-3` composes the three nearest trained classes with the composition of Liu et al. 2022 (reference 11), and keeps its printed name because it is a retrieval baseline.

## Protocol facts a comparison depends on

- **FID sample size.** FID is computed at N=100. The paper says it is biased upward and must not be compared with published N=50,000 FIDs (p15). Absolute FID values here cannot set a target against another paper.
- **Generated sample count.** Table 8 states 16 saved generations per query per teacher (p16). The paper does not state how many generated images enter each metric, so the `samples` column is empty.
- **Reference sets.** Faithfulness scores against the 100 query images. Generalization scores against 100 other held-out images of the same class (p8).
- **Uncertainty.** Each cell is a mean with a standard error over 40 classes on ColorMNIST and 23 on CelebA (p8, p9).
- **ColorMNIST.** 32x32 RGB, 10 digits, 4 digit colors, 4 backgrounds, 160 slots: 120 seen, 40 held out, 1,000 images per slot (p7, Table 3 p13).
- **CelebA.** 128x128, 14 binary attributes, 74 realized classes: 51 seen with 26,775 training and 3,825 validation images, 23 held out with 9,914 images (Table 4, p14).
- **Metrics.** CLIP is image-image cosine. Precision and recall use Inception-V3 features with a k-NN estimator at k=3 (p8).
- **Sampling.** DDIM with eta 0 and 50 steps. Canonical LoRA sampling weight is 0.8 on ColorMNIST and 1.2 on CelebA. Seed 42 is fixed across methods (Table 8, p16).
- **LoRA.** Rank 8 on ColorMNIST and 16 on CelebA, distilled per query from a pool of 256 PoE samples (Table 7, p16).
- **Backbones.** UNet2DModel from google/ddpm-cifar10-32 for ColorMNIST, and from google/ddpm-celebahq-256 at sample size 128 for CelebA (p14). ColorMNIST trains for 100 epochs and CelebA for 500,000 steps (Table 5, p15).

## Hardware and training time

Backbone training uses one NVIDIA A40 for ColorMNIST, roughly 6 hours, and four NVIDIA RTX 6000 GPUs for CelebA, roughly 50 hours. Concept discovery and PoE sampling run on one A40, and the full pipeline takes about 10 to 17 minutes per query (p15).

## Where the paper contradicts itself

1. **ColorMNIST baselines.** The text says PoE and LoRA outperform Query-only and the top-k baselines on both Faithfulness and Generalization (p8). On Generalization CLIP, PoE scores 90.9 and Top-1 scores 91.0 (Table 1, p8).
2. **CelebA Faithfulness.** The text says Query-only attains the strongest Faithfulness scores (p9). On Faithfulness CLIP, PoE has 68.5 and +LoRA 67.5, both above Query-only at 66.6 (Table 2, p9).
3. **Conclusion against Table 2.** The conclusion says PoE and its LoRA variant improve over query-only sampling on ColorMNIST and CelebA (p9). On CelebA Faithfulness, Query-only beats both on FID, precision, recall and F1. On CelebA Generalization FID, +LoRA at 160.1 is worse than Query-only at 147.5.
4. **Reported sweep.** The Table 8 caption says the paper reports a sweep of the LoRA sampling weight over 0.8, 1.0 and 1.2 (p16). No table or figure prints results for that sweep.
5. **Empty section.** Section D.3, "Sampling hyperparameters", has a heading and no body (p15). Its settings are in Table 8 (p16).
