# Notation shared by every page of this scene

One meaning keeps one symbol and one colour on every rung. Words first, symbols second.

| Say it in words | Symbol | Colour on the figures |
|---|---|---|
| the noisy canvas at sampler step t | x_t | blue (an input) |
| what an expert predicts when told its label, as noise | ε_A, ε_B | green (an expert) |
| what the base predicts when told nothing | ε_0 | grey (fixed) |
| the score of a density, the direction that raises its log density | s = ∇ log p | same colour as its owner |
| the noise prediction and the score of one density are linked | ε = −σ_t s | |
| the product of the two experts, as a noise prediction | ε_PoE = ε_A + ε_B − ε_0 | green |
| the term the product drops, in score units | I_t = ∇ log p(x_t \| c₁, c₂) − [s_1 + s_2 − s_0] | orange |
| the same term in noise units | r_t = −σ_t I_t | orange |
| a push (classifier guidance) or a layout commit | the other orange things | orange |
| what the sampler draws | x_0 | red (an output) |
| signal level kept after noising | ᾱ_t (a cosine schedule in the toy, the project's schedule in the real runs) | |
| noise level | σ_t = √(1 − ᾱ_t) | |

Append new symbols here and never change one that a page already uses.
