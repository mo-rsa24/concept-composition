# Level 3: the real code, block by block

**Said plainly.** The toy's `poe_toy.py` and the real runs' code compute the same three things: the product of the experts, the committed layout with the other half hidden, and the sampler step. The real code is this project's, not an author's reference implementation, so this page reads it where it lives and does not copy it.

| Block | Where it lives | The equation it computes |
|---|---|---|
| the product of two experts | `two_eps` in [two_experts.py](../../../scripts/mnist/two_experts.py), line 49 | ε = ε_A(x, a) + ε_B(x, b) − ε_ref, with ε_ref the base's prediction |
| the layout vote and the hold | `inferred_eps`, lines 105 to 145 | D = Σ over steps with t ≥ 800 of (s₁ − s₂), the layout is 1 if D ≥ 0, then frozen |
| hiding the other half | the `occlude` branch of `inferred_eps`, lines 135 to 141 | x̃_A = m ⊙ x + (1 − m) ⊙ blank_t, blank_t = −√ᾱ_t + √(1 − ᾱ_t) z |
| each expert keeps its half | the last lines of `inferred_eps` | ε = m ⊙ ε_A(x̃_A) + (1 − m) ⊙ ε_B(x̃_B) |
| the pushes | `per_sample_cap`, [canvas_unet.py](../../../scripts/mnist/canvas_unet.py) line 180, and the pushes in `pixel_checker.py` | push capped to 10% of the prediction's norm, added to ε |
| the sampler step | `sample`, canvas_unet.py line 197, the update near lines 233 to 240 | x_0 = (x − √(1−ᾱ) ε)/√ᾱ, x' = √ᾱ' x_0 + √(1−ᾱ'−σ²) ε + σ z |
| the toy's twin of each block | [poe_toy.py](scripts/poe_toy.py): `poe_score`, `hide_eps`, `sample` | the same equations on exact scores |

**How it fails and how you know it worked.** The layout flips if `lock_t` is left unset: the printed figure "canvases whose layout never changes after the first fifth of the run" is then 0.0, and the clean-pair count falls from 310 of 320 to 56 on the five tuning pairs. It worked when that figure prints 1.0. The toy shows the same break: re-voting at every step draws two apart in 1810 of 4000 against 3840 when the layout is held.

**What is not claimed.** This page does not re-run the real code. The counts quoted for it come from the review file of plan 33 and are marked there as read on five tuning pairs (310 of 320) and on 20 unseen pairs (1,200 of 1,280 by the judge).

Next: [where each equation is taught](04-where-each-equation-is-taught.md).
