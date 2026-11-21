# The network views as drawn from the adding design

The versions of the network-view figures drawn before replacing became the main design. Here each
single caption grows to 2,000 + k pictures, and the reference model is the one retrained on CPU
with 2,000 relabelled (seed 20261022). Adding is now the check, and [the current figures](../README.md)
are drawn from replacing.

| File | What it shows |
|---|---|
| `02-what-relabelling-adds-to-each-caption` | each single caption's pool growing from 2,000 to 4,000, and one picture filed under three captions |
| `03-how-long-each-network-trains` | passes over the data falling from 341 to 205 (denoiser) and 171 to 102 (VAE) as the set grows |
| `04-the-vae` | the VAE on the adding 2,000 model's training picture |
| `05-the-denoiser`, `08-training-the-denoiser` | the training example from the adding 2,000 model (loss 0.00625) |
| `07-the-caption`, `07b` to `07e` | the caption table, first-layer split and caption shifts of the adding 2,000 model (shift norms 11.8, 15.9, 15.6, 10.5) |
| `09-sampling` | the adding 2,000 model's joint-caption chain and reverse step 100 |
| `10a-the-product-and-the-corrector` | the caption arrows at step 100 in the adding 2,000 model (cosine −0.76) |
| `10b-the-product-step-by-step`, `11-the-scorer` | 0 and 2,000 relabelled, plain and with 20 corrector steps (207 and 300 separate at 2,000) |
| `13-replacing-against-adding` | the earlier version of figure 13, made before the replacing sweep existed |

**How they were made.** Copied here on 2026-11-18 before the redraw. Each is PNG and SVG. The
numbers behind them stay in `../../toy_relabelling_numbers.json` under the adding keys, and
`fig02_adding` and `fig03_adding` in `../../draw_network_views.py` still draw figures 2 and 3 into
this folder. The others would need the drawing script pointed back at the adding model.
