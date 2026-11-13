# What PoE loses when it composes: figures and toy runs

The figures, scripts and checked references behind the
[write-up on structural and causal composition](../../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md).

## figures/

**Computed by [`make_figures.py`](make_figures.py).**

| File | What it shows | Made from |
|---|---|---|
| `fig-causal-renders.png` | camel and dog at seeds 1 and 4 across six prompts, with the reader's calls | SDXL tiles in `3-understanding-compositionality/artifacts/results/does-a-camel-in-the-rain-come-out-wet/tiles/`, crops of the dog sheet, the two calls files |
| `fig-exact-product-is-a-blend.png` | the exact product of two centred one-picture Gaussians is the pixel average | exact arithmetic on 16 × 16 images |
| `fig-why-rain-and-dog-do-not-fuse.png` | contention (cat and dog) against separate pixels (dog and rain) | exact Gaussian products, spreads chosen by hand |
| `fig-mixture-reweighting.png` | the product's wet share against the subject's own wet share | the formula of the write-up's §5.4; values illustrative |
| `fig-caption-dial.png` | exact product mass on two objects against the share of single-name pair captions | exact mixture-product weights |
| `fig-caption-evidence.png` | caption counts, 25 wet-word captions read, the pilot funnel | Re-LAION-2B-en counts and pilot jobs 72598, 72599 |

**Computed by [`fig_noise_on_off.py`](fig_noise_on_off.py).**

| File | What it shows | Made from |
|---|---|---|
| `fig-noise-on-off.png` | the blue value of pixel `[4, 8]` under the centre circle's and centre square's opinions, multiplied, then scaled to total 1, at σ = 0.1 against σ = 0: a purple bell at 0.5 (raw peak 2.2 × 10⁻¹⁰) against a product that is zero everywhere | exact one-dimensional Gaussian arithmetic, drawn while walking the six-checkpoint toy |

**Computed by [`fig_sum_vs_product.py`](fig_sum_vs_product.py).**

| File | What it shows | Made from |
|---|---|---|
| `fig-sum-vs-product.png` | the blue value of pixel `[4, 8]`: the product of the centre circle's and centre square's opinions (one bell at 0.5) beside their half-and-half sum (bumps at 0 and 1, height 1.5 × 10⁻⁵ at 0.5 against 1.99 at 0) | exact one-dimensional Gaussian arithmetic, σ = 0.1, drawn while walking the six-checkpoint toy |

**Computed by [`fig_caption_dial_pairings.py`](fig_caption_dial_pairings.py).**

| File | What it shows | Made from |
|---|---|---|
| `fig-caption-dial-pairings.png` | the caption dial's 16 pairings at caption share 0.05: each circle-word picture averaged with each square-word picture, with its d² (0 for the pair with itself, 32, 48, 80 and up) and its weight against the best | exact mixture-product weights on the six-checkpoint toy's 8 × 16 pictures, σ = 0.1, drawn while walking checkpoint 5 |

**Computed by [`fig_error_a.py`](fig_error_a.py).**

| File | What it shows | Made from |
|---|---|---|
| `fig-error-a.png` | error A in closed form: one pixel at ᾱ = 0.5 (exact noisy product variance 0.5025 against the summed scores' 0.2525, same centre), and the share on the two shapes apart against ᾱ under the summed scores (σ_eff² = σ² + (1 − ᾱ)/ᾱ) against the exact noisy product | the six-checkpoint toy's mixtures, σ = 0.1, caption share 0.05 and 0.5, drawn while walking checkpoint 6 |

**Computed by [`fig_ddim_guided.py`](fig_ddim_guided.py).**

| File | What it shows | Made from |
|---|---|---|
| `fig-ddim-guided.png` | checkpoint 6's first column: guided-sum DDIM samples (unconditional over all seven pictures, w = 1 and 7.5, pair not included and at 1 in 20) beside each one's nearest library picture | 200 samples per row, seed 0, 200 DDIM steps on the cosine schedule, σ = 0.1; counts at 50, 200 and 1000 steps in `fig-ddim-guided.json` |

**Computed by [`toy_sampler_hybrids.py`](toy_sampler_hybrids.py).**

| File | What it shows |
|---|---|
| `fig-toy-du-layout.png` | Du et al.'s product and sum rows redone on red circles and blue squares, with the pair-captioned product between them |
| `fig-toy-sampler-hybrids.png` | the same runs, eight samples per row with counts |
| `fig-toy-sampler-hybrids-B-strong-mala.png` | setup B with a stronger corrector |
| `fig-toy-sampler-hybrids-D-w0.png`, `-D-w0.1.png` | condition D at pair weights 0 and 0.1, the two checks against conditions A and B |

**Computed by [`toy_cooccurrence_sweep.py`](toy_cooccurrence_sweep.py), [`toy_cooccurrence_figure.py`](toy_cooccurrence_figure.py), [`toy_cooccurrence_sheet.py`](toy_cooccurrence_sheet.py) and [`toy_caption_competition.py`](toy_caption_competition.py),** filed with their counts and caveats in
[the co-occurrence card](../../results/how-much-co-occurrence-the-product-needs/README.md).

| File | What it shows |
|---|---|
| `../../results/how-much-co-occurrence-the-product-needs/weight-sweep.png` | the crisp-pair fraction of 300 draws against the pair image's weight inside each expert, log x, one curve per sampler |
| `../../results/how-much-co-occurrence-the-product-needs/draws-across-weights.png` | eight draws per sampler at five pair weights, with counts |
| `../../results/how-much-co-occurrence-the-product-needs/competing-captions.png` | both competing-caption forms against the hand-weighted condition at the same implied weight |

**How it was made.** Each condition is an exact expert: a Gaussian mixture of spread 0.1 around its
training images, 16 × 32 RGB. Under the variance-preserving forward process every diffused density and
score stays exact, so each sampler can be set against the exact composed distribution. Every sample is
labelled by its nearest picture among the single shapes, the pair, and every circle-square pixel
average. Counts and corrector acceptance rates are in `toy-sampler-hybrids*.json`.

**Fetched.** `du-2023-product-sampling.png` (Du et al. 2023, Figure 2) and
`bradley-2025-projective-composition.png` (Bradley et al. 2025, Figure 5), with their captions and URLs
in [the reference check](references-verified.md).

## Rebuild

```bash
cd /home/molef/PhD/poe_repair_min-dog-and-rain
/home/molef/miniconda3/bin/python3 -I 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/make_figures.py
cd 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes
/home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py                     # about 3 minutes on CPU
/home/molef/miniconda3/bin/python3 -I toy_sampler_hybrids.py --only B --mala-k 25 --mala-step 0.3 --tag=-B-strong-mala
```
