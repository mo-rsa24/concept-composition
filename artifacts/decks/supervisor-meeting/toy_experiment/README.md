# Toy experiment: figures for the supervisor meeting

The figures from the co-occurrence write-up
(`3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/plain-figures/README.md`), plus the
result from Du et al. (2023) that the toy builds on.

## Du et al. (2023): adding scores does not sample the product

Du, Durkan, Strudel, Tenenbaum, Dieleman, Fergus, Sohl-Dickstein, Doucet, Grathwohl, "Reduce,
Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC", ICML 2023,
arXiv 2302.11552. Cropped at 200 dpi from pages 5 and 6 of the copy on the paper shelf
(`goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-generation-with-energy-based-diffusion-models-and-mcmc.pdf`).

| File | What it shows |
|---|---|
| `du-2023-figure-2-product-reverse-diffusion-vs-hmc.png` | Two trained 2D diffusion models: a ring of eight blobs times a vertical bar. The true product is the two blobs inside the bar. Reverse diffusion with the two scores added draws a smeared ring; HMC, a corrector, draws the two blobs. The bottom row is a mixture (sum), not a product. |
| `du-2023-table-1-2d-composition-numbers.png` | The same 2D experiment in numbers: for the product, reverse diffusion scores a log-likelihood of −6.47 (score model) where U-HMC scores 2.40 and HMC on the energy model 2.72; higher is better. |
| `du-2023-figure-3-and-table-2-shape-positions.png` | One CLEVR model trained to place one shape at one location, composed with itself to place 1 to 5 shapes. Reverse diffusion places 57.4% of shapes correctly at 5 combinations; HMC on the energy model 72.7%. |

**How this maps onto the toy.** Du trains the experts (neural networks) and shows the sampler is
the problem. Our model 1 writes the experts as formulas, so the correct score of the product can
be computed exactly: with it the same reverse diffusion draws the pair 300 of 300 times, where
adding the scores draws it 15 times (figure 04 in the write-up). Our model 3 then tests Du's fix,
a Langevin corrector, on a trained caption model: it completes the pair only where both-shapes
training pictures also carry the single captions (figures 07b and 08).
