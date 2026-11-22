# SuperDiff first read

Superposition of Diffusion Models Using the Itô Density Estimator, arXiv 2412.17762v2, 54 PDF pages.
Page numbers below are PDF pages counted from 1, which match the printed footer in this PDF.

## What was read

Six tables (1, 2, 3, 4, A1, A2), two figures that print numbers beside each drawn molecule
(A3 and A4, both on page 32), and one footnote on page 8 that states a runtime no table carries.
The table inventory listed all six tables with the right page numbers, and no seventh table exists:
the only `Table` labels anywhere in the extracted text are 1, 2, 3, 4, A1 and A2.

Rows that are not from a table carry `SuperDiff Figure A3`, `SuperDiff Figure A4` or
`SuperDiff text p8` in `source_table`, so they sit outside the coverage file's six table lines.

## Numbers that could not be read

None. Every printed digit in all six tables was legible, and the layout text from `pdftotext` agrees
with the page image for every row of every table.

## Plot-only, no rows written

- Figure 2 (page 5), an illustration of model superposition: no numbers.
- Figure 3 (page 9), a UMAP of protein structures: points only, no axis values worth reading.
- Figures A1 and A2 (page 31), UMAP and protein renderings: no numbers.
- Figures A5 to A24 (pages 34 to 54), the generated image grids: prompts in the captions, no scores.
  Searching those pages for any token of the form `d.dd` returns nothing.

## A stated number deliberately left out

Page 8, footnote 1, says SuperDiff requires "almost 30% less memory". The qualifier makes the
number a bound rather than a measurement, so it has no row. The two timings in the same footnote
(1082 ± 7 seconds per image against 209 ± 2) are recorded as rows.

## Protocol facts a comparison would depend on

**Tables 1 and A1, CIFAR-10.** Table 1 trains two models on two disjoint halves (first five labels
and last five labels); Table A1 trains two conditional models on two random partitions, so the two
tables are different experiments and their numbers are not comparable. Architecture,
hyperparameters and data preprocessing follow Song et al. (2020). ODE inference integrates the
Prop. 1 ODE with densities from Prop. 5; SDE inference is the OR branch of Algorithm 1 with
densities from Thm. 1. **The number of generated images behind FID, IS and FLD is never stated.**

**Table 2 and Table A2, Stable Diffusion.** 20 images for each of 20 concept pairs, SD v1-4 with
classifier-free guidance, SDE inference only (footnote 1, page 8). Table 2 reports the average
minimum of the two per-concept scores; Table A2 reports the maximum and the absolute difference.
The joint-prompt baseline joins the concepts with "that looks like", generates both orderings and
keeps the better one for every metric (page 8 and page 32). Figures use the first 6 seeds, plus
favourites from 20 seeds (page 32).

**Table 3, proteins.** Averages over 50 generated proteins at each of the lengths
{100, 150, 200, 250, 300}, for 500 timesteps (page 9). Designability predicts a sequence with
ProteinMPNN and refolds it with ESMFold, repeated 8 times keeping the lowest scRMSD (page 30).
Temperature T = 1 for every SuperDiff variant, and ℓ biases the superposition towards Proteus
(ℓ > 0) or FrameDiff (ℓ < 0).

**Table 4, molecules.** LDMol conditioned on a text prompt per property; metrics from 5 runs of
batch-size 512 (page 10). Properties scored with Therapeutic Data Commons oracles, diversity is
Tanimoto distance on Morgan fingerprints, quality is the fraction valid, unique, with QED ≥ 0.6 and
SA ≤ 4.0. The joint-prompt baseline samples 50% of its molecules from each property ordering.

**Hardware and training time.** The paper states neither. No GPU type, no device count and no
training or sampling wall-clock appears anywhere in the text, so the only cost number in this paper
is the per-image density-evaluation time in the page 8 footnote.

## Where the paper contradicts itself

**The union-model claim fails on one of the four columns it covers.** Page 7 says SuperDiff
"performs even better than the model trained on the union of both parts of the dataset". In Table 1
that holds for ODE FID (4.41 against 6.00), ODE FLD (6.10 against 8.06) and SDE FLD (5.33 against
7.51), but not for SDE FID, where the union model is better: 3.50 against SuperDiff's 4.00, and FID
is a lower-is-better metric. The same holds in Table A1, where the union model's SDE FID of 2.83
beats every SuperDiff variant.

**A metric is named wrongly in the running text.** Page 9 says FrameDiff "is substantially less
designable than Proteus (0.392 vs. 0.928 scRMSD)". Those two numbers are the `< 2Å scRMSD` column
of Table 3, which is the fraction of designable proteins, not the scRMSD. The scRMSD column holds
4.315 and 1.014 for the same two methods.

## A judgement call in the `better` column

Table A2's three column headers each carry one ↑ arrow over a cell that prints two numbers,
`|Δ|/Max.`. Both halves are recorded as `higher`, which is what the single printed arrow and the
bolding in the table say, but the paper never states separately which direction |Δ| should move.
