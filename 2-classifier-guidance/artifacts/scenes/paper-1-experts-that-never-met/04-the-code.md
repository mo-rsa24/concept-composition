# Level 3: the code, where each part of the formula lives

**Said plainly.** Paper 1's digit run is this project's own code, and the code walks already exist. This page points each part of the
formula at the function that computes it and links the walks, without copying the code. Line numbers were read at commit `33cea79b`.

## Each part of the formula, in the digit run

| Part | Where it lives | What it computes |
|---|---|---|
| the run itself, four rows on the same seeds | `render` in [paper1_rows.py](../../../scripts/method_search/paper1_rows.py), line 55 | paste, the method without a push, with the checker push, with a random push |
| the settings | the same file, lines 33 to 36 | $\lambda$ = 10, the cut at 0.1, the side held from t = 800, $\eta$ = 1 |
| sample and paste, the control | `paste`, line 44 | the inked half of each expert's lone sample, placed on opposite sides |
| the merge by region | `inferred_eps` in [two_experts.py](../../../scripts/mnist/two_experts.py), line 105, called with `occlude` on | the side vote, the hold, the blanked half, each expert kept on its half |
| the pair checker's push | `pair_push` in [pixel_checker.py](../../../scripts/mnist/pixel_checker.py), line 70 | $-\lambda\,\sigma_t\,\nabla_{x_t} z_{\mathrm{pair}}$, only while the logit is inside its clip window |
| the separate checker's push | `separate_push`, line 82 | $-\lambda\,\sigma_t\,\nabla_{x_t}[\log\sigma(z_a) + \log\sigma(z_b)]$ |
| the two pushes added, and the random control | `make_push` in two_experts.py, line 178 | the sum, or a random direction rescaled to the same length each step |
| the checkers' network | `PixelHead` in pixel_checker.py, lines 33 to 58 | the pixels and the step in, logits out |
| the cut | `per_sample_cap` in [canvas_unet.py](../../../scripts/mnist/canvas_unet.py), line 180 | each canvas's push shortened to at most 0.1 of its prediction's length |
| the sampler step | `sample`, canvas_unet.py, line 197 | DDIM with $\eta$ = 1; the push is added after the cut, at line 218 |

## The code walks, linked

| Walk | What it reads |
|---|---|
| [the PoE ladder's code page](../product-of-experts-interaction-term/03-the-code.md) | the product, the side vote and hold, the blanked half and the sampler, block by block, beside the toy's twin of each |
| [the digit-map checkers' code walk](../digit-map-checkers/3-code/walk-v2-learn.md) | how a checker is trained on the experts' own samples and pushes the canvas, for Loop 8's checkers |
| [the product of experts method, its code page](../../../../context/methods/01-the-product-of-experts/04-the-code.md) | the product as the project runs it |
| [the digit-map checkers method, its code page](../../../../context/methods/03-the-digit-map-checkers/04-the-code.md) | the blocks, the schedule and the noising |

## The toy's twin

[make_paper1_numbers.py](scripts/make_paper1_numbers.py) computes the same parts on the toy: `grad_log_checker` is the exact
checker's gradient, `with_push` adds $-\lambda\sigma_t\nabla\log C_t$ to any base, and the base is `poe_score` or `hide_eps` from
[the PoE toy](../product-of-experts-interaction-term/scripts/poe_toy.py).

**How it fails and how you know it worked.** The run's own check is the zero push: the method with a push of exactly 0 must match
the method with no push. On the digit run the largest difference was 0.0 against a bar of 1e-5
([the paste finding](../../../report/the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md#what-would-have-counted)).
The toy's check is the Bayes row on [the toy page](03-the-toy.md): 4,000 of 4,000.

**Not covered here.** No two-colour CLEVR expert has trained, so the table has no CLEVR row. The fixed-place CLEVR step's scripts
(`clevr_render.py`, `clevr_scorer.py`, `clevr_paste.py` under `2-classifier-guidance/scripts/method_search/`) render, score and paste;
this ladder did not read them.

Back to [the closing page](README.md).
