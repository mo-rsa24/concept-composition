# How CO3 corrects a step

One round of the CO3 corrector on SDXL, taken apart symbol by symbol on a picture you can drive. It runs from the noisy latent x
through four noise predictions, three Tweedie estimates, the concept weights, their weighted sum, the rescale and the re-noise to
the next x. Wherever the released code does something Algorithm 3 does not print, the page shows the two side by side.

| File | What it is |
|---|---|
| `index.html` | the page source, 21 presses, published as the private artifact below |

**Published at**

[CO3 Corrector Round](https://claude.ai/artifact/9hHS8hf9vtTWHikPVcm52x). It is private to its owner until shared from the page's
Share menu. Each press opens at `#p1` to `#p21`, and each part at `#predictions`, `#estimates`, `#weights`, `#sum`, `#rescale`,
`#disagree` or `#drive`.

**How it was made**

The latent and the four noise predictions are invented two-entry vectors, labelled on the picture. The numbers come from
`co3_corrector` in `composers/utils/attn_utils.py` at commit 35ea972, called on CPU with a stub U-Net that returns those fixed
predictions and a small NumPy stand-in for the tensor calls. The page's JavaScript reached the same final x on the 2 settings it
shows and 400 random inputs, with a largest difference of 0.0000017 per entry. Its 14 quoted code lines were compared with the
released file. Codex reviewed the claims against the paper and code, then polished the wording over two rounds.

**What it does not show**

No SDXL model ran, so the gap between the paper's and the code's rescale on this toy says nothing about its size on a real
65,536-entry latent. The real distances between noise predictions were not measured. The page takes no side on which guidance
setting a reproduction should use: section 4.1 prints 5.0, appendix A.8 uses 0.8 and 0.9, and the released script uses CFG++ at
0.8. That choice is board question 16.
