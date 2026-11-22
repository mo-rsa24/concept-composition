# LogicalGuidance first read

Logical Guidance for Diffusion Models (LoGDiff), arXiv 2602.05549v2, 32 PDF pages. Page numbers are PDF pages counted from
1, and they match the printed footer.

## What was read

Eleven tables and 222 rows. Table 1 (page 3) holds composition rules and Table 9 (page 21) holds hyperparameters, so both
are marked `no results`. Rows outside the tables are the conformity scores printed in the captions of Figures 10 to 18
(pages 28 to 32), one query per figure.

Method names: LoGDiff is `LogicalGuidance`, Liu et al. 2022 is `Liu`, Du et al. 2023 is `Du`, and Skreta et al. 2025b (the
superposition paper in this PDF's references) is `SuperDiff`. Skreta et al. 2025a is FKC. In Tables 7 and 8 FKC is applied
on top of LoGDiff or SDE, so those rows carry the base method in `paper` and the printed name in `model`.

Table 11 prints one AND formula credited to both LoGDiff and Liu. That row is under `LogicalGuidance`, with the shared
credit kept in `model`. Its OR rows with fixed one-half weights carry no method name, so `paper` holds the printed formula.

## Numbers that could not be read

None. Every digit in the ten result tables and nine figure captions was legible.

## Plot-only, no rows written

Figure 3 (page 6), conformity score against joint entropy on CMNIST for guidance weights from 1.0 to 2.5. Figure 4 left
(page 7), ImageNet FID against NOT guidance weight. The ImageNet FID result exists only as this curve, with axis ticks
from 1.31 to 1.34 and no printed value, so it has no row.

## Protocol facts a comparison depends on

**Conformity score (page 23).** The percentage of samples satisfying the query, judged by a ResNet-18 attribute classifier
whose accuracies are in Table 10. It is averaged over 10,000 samples from 100 unique queries per task type with 100
samples each. Tables 2, 4 and 11 print it as a percentage, and Table 3 and the figure captions on a 0-to-1 scale.

**Models (page 21, Table 9).** CMNIST: U-Net at 28 x 28, DDPM sampler with 50 steps. Shapes3D: U-Net at 64 x 64, DDPM
sampler with 50 steps. CelebA: SiT on 16 x 16 latents, DDIM sampler with 1000 steps. ImageNet-512: pretrained EDM2-XXL
(page 19).

**One conditional model.** Every image method is run on a single conditional model. Table 11 applies methods designed for
separately trained models to this one model (page 26), so its `Liu`, `Du` and `SuperDiff` rows are not those papers' own
setups.

**CelebA (pages 6 and 24).** Only the attributes Blond and Male are used. FID uses clean-fid on 5,000 samples per task, 100
images for each of 50 queries.

**Molecules (pages 7 and 25).** GRM5-RRM1 pair only, 1000 steps, inverse temperature 2.0, 32 ligands of size 23 per
experiment over 8 experiments, docking by AutoDock Vina. Quality requires QED at least 0.6 and SA at most 4.0.

**Figures 10 to 18.** Each caption score is for one query, so it is not comparable with a table average.

Image seeds are not stated. Page 19 says the ImageNet FID differs slightly from the EDM2 publication because the exact
sampling seeds were unavailable.

## Hardware and training time

No hardware is named and no training time is stated. Table 9 gives training steps: 50,000 for CMNIST, 100,000 for Shapes3D
and 500,000 for CelebA.

## Contradictions inside the paper

**Tables 2 and 11 disagree on CMNIST NOT.** LoGDiff is 96.5 in Table 2 and 97.38 in Table 11. The plain negation rule is
99.5 against 99.41, and the unconditional row is 90.0 against 90.19. The AND, OR-ME and OR-CI cells agree within rounding.

**The conformity gap on page 6 is not met everywhere.** The text says the constant baseline has a gap above 20%. In Table 2
on CMNIST, LoGDiff leads by 17.7 points at N = 2 and 19.2 points at N = 5.

**The direction of the docking score is stated both ways.** Page 7 says a lower docking score indicates better binding, and
Table 5 marks A and B with a down arrow. The Table 5 caption says a higher average docking score should correlate with
higher binding affinity.

**The Table 6 caption names a method the table lacks.** It lists TargetDiff and DualDiff as baselines, and the rows are
DualDiff, SDE and LoGDiff.

**The RRM1 identifier differs.** Figures 5 and 6 print UniProt ID PP23921, and the text on pages 7 and 25 prints P23921.
