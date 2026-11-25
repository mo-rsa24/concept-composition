# Notation shared by every page of Paper 1's ladder

One meaning keeps one symbol and one colour on every rung, here and on the DCFM ladder. Words first, symbols second. Append new
symbols at the bottom and never change one a page already uses.

## Shared with the DCFM ladder

| Say it in words | Symbol | Colour on the figures |
|---|---|---|
| the picture partway along its path. Diffusion runs t from T (noise) down to 0; DCFM's flow runs the other way, so every maths step states its direction | x_t | blue (an input) |
| the concepts asked for | y = (y_1, y_2) | blue |
| no concept asked for | ∅ | grey (fixed) |
| an expert's predicted noise | ε_θ(x_t, y) | green (an expert) |
| a flow model's velocity (DCFM) | v_θ(x_t, t, y) | green |
| the composed prediction (the sum rule, or the merge by region) | ε̂ | green, with orange where a push is added |
| the interaction term: the joint prediction minus the sum rule | I_t | orange |
| the checker | C_t | green (a trained network) |
| the push strength | λ_t | orange |
| the noise level at step t | σ_t | |
| DCFM's penalty weight | λ_CI | orange |

Colour key: inputs `#1f5fd0`, experts and hidden units `#1d6b3a`, outputs `#c0262d`, the interaction term and any push
`#c8680a`, fixed, frozen or off `#6b7a88`.

---

## Used only on Paper 1's pages

| Say it in words | Symbol |
|---|---|
| the two regions, one per expert | R_A, R_B |
| expert A's and expert B's noise prediction | ε^A_t, ε^B_t |
| the mask: 1 on R_A, 0 on R_B | m |
| an empty canvas noised to step t, put in the half an expert must not see | b_t |
| what expert A and expert B are shown: the canvas with the other half blanked | x̃^A_t, x̃^B_t |
| the score of one density, the direction that raises its log density | s = ∇ log p (s_1, s_2, s_∅) |
| the same interaction term in noise units | r_t = −σ_t I_t |
| signal kept after noising | ᾱ_t, with σ_t = √(1 − ᾱ_t) |
| the DDIM fresh-noise size at step t (η = 1 in the runs) | η_t |
| the prediction the push is added to: the merge, or the plain product | ε_base |

## Where this differs from the PoE ladder's notation

The [PoE ladder's notation](../product-of-experts-interaction-term/NOTATION.md) is kept wherever the two agree: x_t, ε_A and ε_B
(written ε^A_t and ε^B_t here), s = ∇ log p, ε = −σ_t s, I_t in score units, r_t = −σ_t I_t, ᾱ_t and σ_t = √(1 − ᾱ_t), and the colour key.

**The base model.** The PoE ladder writes ε_0 for the prediction with no label. This ladder writes ε_∅, the symbol shared with the
DCFM ladder, because a subscript 0 reads as "clean picture at t = 0" once x_0 is on the same page.

**Units of I_t.** The shared line defines I_t as "the joint prediction minus the sum rule". This ladder reads "prediction" as a score
prediction, as the PoE ladder does: I_t = ∇ log p_t(x_t | y_1, y_2) − [s_1 + s_2 − s_∅]. The same term as a noise prediction is
r_t = −σ_t I_t = ε_joint − ε_PoE. The DCFM ladder should state which unit it uses for I_t.

**The push strength.** The digit guidance scene writes s for the push strength
([its derivation, step 3](../mnist-guidance-explained/4-the-derivation.md)); here it is λ_t, and s is kept for a score.
