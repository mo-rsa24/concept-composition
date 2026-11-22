# Catastrophic: notes on the first read

arXiv 2606.23920v1, 41 PDF pages. The CSV holds 685 rows: 680 from the 27 tables and 5 printed on Figure 1.

## Numbers that could not be read

None. An OCR pass over the rendered pages disagreed on four cells: p28 Table 3(c) K=256 N=1000, p38 Table 16 K=256
N=1000, p38 Table 18 K=4 N=1000, and the std on p39 Table 22 K=256 N=100. A 600 dpi view of each shows 4.2434, 4.2434,
2.2133 and 1.0936, which are the CSV values.

## How the rows are labelled

K=1 rows are `Naive denoising (K=1)` and `baseline`, because p22 states that FKC with one particle equals the naive
denoising SDE. Rows with K above 1 are `FKC` and `published`, because FKC is one of the 19 rivals.

Tables 1, 2 and 3 repeat appendix means, so they are the same measurements printed twice. Table 1 equals Tables 4, 6, 8
and 10. Table 2 equals Tables 20 and 22. Table 3 equals Tables 12, 14, 16 and 18. All 200 cells match.

`-0.0000` is copied as printed. The MMD2 estimate is an unbiased U-statistic (p30), so it can fall below zero.

## Plot-only

Figure 1 (p9) is a set of scatter plots. Its five printed SW2 labels are rows, and nothing is read off its axes. Figures
2 (p12), 3 (p28) and 4 (p33) are image grids of rooms with no printed numbers, so the objects-in-a-room experiment has no
numeric result anywhere in the paper.

## Protocol facts a comparison depends on

**Grid.** K is 1, 4, 16, 64 or 256 particles. N is 100, 1000 or 10000 training samples, or analytic scores (p8, p10).

**Network.** A 4-layer MLP of width 512 with SiLU activations, either one conditional network with a one-hot condition or
one network per distribution (p29).

**Training.** Adam at learning rate 2e-4, batch size 512, 20,000 iterations (p29).

**Noising and sampling.** VP SDE with beta from 0.1 to 20. The FKC sampler resamples at every step over 500 integration
steps (p29). `g_clip` is 15.0 for the 2D Gaussian and mixture experiments and unset for images (p30).

**Metrics.** SW2 uses 2000 random projections. MMD2 uses a Gaussian RBF kernel with a median-heuristic bandwidth. Both
compare 5000 generated samples with 5000 ground-truth samples (p29, p30).

**Seeds.** Run r uses seed 1 + r, over 30 runs, reported as mean and standard deviation (p30).

**Gaussian mixtures.** Network, noising schedule and optimizer match the 2D Gaussian setup (p30). Ground truth comes from
rejection sampling in-distribution and importance sampling out-of-distribution (p30, p31). Table 2 inherits the 5000-sample
convention of Table 1, and Tables 20 and 22 print the same means. Tables 21, 23, 24, 25, 26 and 27 do not restate a sample
count, so their `samples` cell is empty.

**Room images.** 256 x 256 images from FLUX.1-schnell, 2000 accepted per class, a class-conditional U-Net with
v-prediction, T=1000, 100 inference steps, AdamW at 1e-4, batch size 16, 50 epochs (p31, p32).

## Hardware and training time

NVidia L40S or A6000 GPUs. The image models take about 9 hours on one GPU. Each table takes roughly 7 GPU hours across all
configurations and 30 runs. About 48GB of GPU memory is recommended (p29).

## Where the paper disagrees with itself

**Separate models are not worse everywhere outside the stated exception.** Appendix E (p27) calls the out-of-distribution
Gaussian mixture "the only exception" where separate models beat one conditional model. Separate models also have the lower
mean in 11 of the 96 learned 2D Gaussian cells with K above 1 (Tables 12 to 15 and 19 against Tables 4 to 7 and 11). Ten
are at N=10000 and one is at N=100, and every gap is smaller than the printed standard deviations.

**The stated exception holds on one metric only.** In the out-of-distribution mixture, separate models have the lower mean
in 11 of 12 learned SW2 cells with K above 1 (Table 26 against Table 22), and in 6 of 12 MMD2 cells (Table 27 against Table
23). At N=100 their MMD2 is worse at every K above 1.

**"No improvements" on factorized in-distribution data.** p9 says FKC yields no improvements there. Table 1(a) at N=1000
falls from 0.1089 at K=1 to 0.0937 at K=64. The drop is smaller than the 0.0266 standard deviation in Table 4.

**"Only for poorly learned models."** p11 says more particles harm in-distribution mixtures only at N=100 and large K.
Table 2(a) also shows the mean rising at N=10000, from 0.1454 at K=16 to 0.1478 at K=64, and with analytic scores, from
0.0736 at K=64 to 0.0771 at K=256. Both rises are smaller than the standard deviations in Table 20.

**Figure 1 against Table 8.** Figure 1 prints SW2 0.1746 for FKC with analytic scores at K=16 in the factorized
out-of-distribution setting. Table 8 gives 0.1133 with standard deviation 0.0228 for that setting, so the figure sits 2.7
standard deviations above the mean. The paper does not say which run the figure shows.

**The symbol N has three meanings.** It is the training-set size in the tables (p8), the number of particle swarms, each
giving one output sample, so "N = 5000" output samples (p29), and the number of runs (p30).
