# FKC first read

Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts, arXiv 2503.02819v2,
44 PDF pages. Page numbers below are PDF pages counted from 1, which match the printed footer.

## What was read

Twelve tables (1, 2, 3, 4, A1 to A8), two figures that print numbers beside each rendered molecule
(Figure 5 on page 8 and Figure A8 on page 43), and six numbers stated in running text or in a table
caption that no table cell carries. The inventory listed all twelve tables with the right page
numbers, and no thirteenth exists: the only `Table` labels anywhere in the extracted text are 1 to 4
and A1 to A8.

Table 1 (page 5) holds no results. It is the conversion table between terms of the Feynman-Kac PDEs
and the corresponding weighted SDE, one proof reference per line.

Rows outside the twelve tables carry `FKC Figure 5`, `FKC Figure A8`, `FKC text p9`,
`FKC text p40` or `FKC text p43` in `source_table`, so they sit outside the coverage file.

## Numbers that could not be read

None. Every printed digit in all eleven result tables was legible at 400 dpi, and the layout text
from `pdftotext` agrees with the page image on every row of every table.

## Plot-only, no rows written

- Figure 2 (page 7), samples from the mixture of 40 Gaussians: scatter panels with axis ticks only.
- Figure 3 (page 7), EDM2+CFG against EDM2+FKC samples: images, no scores.
- Figure 4 (page 8), 2-Wasserstein against target temperature: curves with no printed values.
- Figures A1 to A4 (pages 35 and 37) and Figure A6 (page 41): curves, energy histograms and
  molecule renderings with no printed values.
- Figure A5 (page 39), SDXL against SDXL+FKC examples: images with prompts, no scores.
- Figure A7 (page 42), Spearman correlation against timestep: curves with no printed values. The
  two averages those curves summarise are stated in the text on page 40 and are recorded as rows.

## Numbers recorded from text and captions

- Page 9, Table 4 caption: the reference molecules' mean docking score, -7.915 ± 2.841, which is
  the level "Better than ref." counts against over the 100 pairs.
- Page 40, Table A5 caption: the reference level for the single validation pair, -8.255 ± 1.849.
  **The two reference levels differ because the protein sets differ**, so a "Better than ref."
  fraction from Table 4 and one from Table A5 are not measured against the same bar.
- Page 40: the average Spearman rank correlation between FKC weight and docking score, 0.275 for a
  pair that improves with resampling and 0.087 for one that does not.
- Page 43: the docking scores of known binders, -8.168 for ATP1A1 and -9.174 for CPT2, quoted from
  Zhou et al. (2024).

Page 38 says roughly 2 to 3% of samples with unacceptably bad energy were excluded from the Figure
A3 comparison. The range makes it a bound rather than a measurement, so it has no row.

## Protocol facts a comparison would depend on

**Tables 2 and A3, images.** EDM2-XS trained on ImageNet-512, sampled in RGB pixel space with the
default classifier-free guidance weight β = 1.4. CLIP and ImageReward are averaged over 10,000
images. γ is the churn parameter and N the number of sampling steps. **The γ sweep uses FKC batch
size 32 and the steps sweep uses FKC batch size 8** (page 38), which Table 2 itself does not print.
K = 1 is the no-resampling baseline.

**Tables 3, A1 and A2, samplers.** LJ-13 is a Lennard-Jones system of 13 particles. DEM models are
trained for 166 epochs on 4 NVIDIA A100 80GB GPUs, with the lowest-energy-W2 checkpoint used for
inference. Sampling uses 1000 integration steps with dt = 0.001, σ_min = 0.01, σ_max = 4.0 and the
score clipped to norm 1000 per particle; FKC inference assumes a Gaussian at t_start = 0.99.
Table 3 is 3 seeds, Table A2 is 3 runs with DEM run for one seed only, Table A1 is 5 runs with 10k
samples. Reference distributions come from a No-U-Turn sampler with 20k warmup steps and 20k
samples from 10 chains per temperature.

**Tables 4 and A5, docking.** 32 ligands for each of 5 molecule sizes {15, 19, 23, 27, 35} per
protein pair, coordinates sampled with an SE(3)-equivariant graph neural network over 1000
integration steps, docked with AutoDock Vina. **Table 4 is 100 protein pairs; Table A5 is a single
validation pair (UniProt P23786/P05023)**, so the two tables are different experiments. The PoE
weight computation needs equal sample dimensionality, so molecules within a batch have the same
atom count (page 38).

**Table A4, SDXL.** GenEval, two prompt sets, the original 553 prompts and 1000 prompts. Starred
rows are taken from the GenEval leaderboard rather than run by the authors.

**Tables A6, A7 and A8, LDMol.** Molecules generated from text prompts of the form "This molecule
inhibits {protein_name}", scored by Therapeutic Data Commons oracles for JNK3, GSK3β and DRD2.
FKC weights are divided by a temperature term T = 100, the top 20% clipped, and resampling stops
after 70% of the timesteps (page 43). Table A6's top-10 set comes from a batch of 512 averaged over
5 runs.

**Hardware and training time.** The only statement is on page 36: all DEM models are trained for
166 epochs on 4 NVIDIA A100 80GB GPUs. No hardware or wall-clock is given for the EDM2, SDXL,
docking or LDMol experiments.

## Where the paper contradicts itself

**Table 2 prints two rows with the same visible settings and different numbers.** In the left half,
FKC on with γ = 40 and N = 32 gives CLIP 29.00 and ImageReward 0.04; in the right half, FKC on with
γ = 40 and N = 32 gives CLIP 29.14 and 0.05. The table shows no column that separates them. Page 38
resolves it: the two halves use FKC batch sizes 32 and 8, and Table A3 prints exactly those two
values against batch size 32 and 8. Anyone reading Table 2 alone cannot tell which batch size a row
used.

**Table A8's caption defines quality with a different threshold from the rest of the paper.** It
says molecules must have "QED ≥ 0.6 and SA < 0.4", while Table A5, Table A6 and page 10 all say
SA ≤ 4.0. The A8 numbers are in the same range as A6's, so the caption looks like a typo, but the
printed definition disagrees.

**Table A8 names an SDE the paper never defines.** Six of its rows are labelled "Tempered Score"
where every other table, and Section 3.3, says "Tempered Noise". The paper defines only target
score and tempered noise simulation.

**One Table A2 interval is zero.** Target temperature 1.0, tempered noise without FKC, reports
Energy-W2 of 6.560 ± 0.000 over 3 runs, while its own distance-W2 varies (0.094 ± 0.002).

**Table A8 prints one interval at four decimals among three-decimal neighbours**: 0.351 ± 0.0340
and 0.678 ± 0.0187. Copied as printed.

**The DEM rows of Table A2 print "± —"**, so those nine values carry no interval.
