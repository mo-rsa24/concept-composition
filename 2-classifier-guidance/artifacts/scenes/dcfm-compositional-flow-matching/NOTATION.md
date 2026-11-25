# Notation shared by every page of the DCFM ladder

One meaning keeps one symbol and one colour on every rung, here and on Paper 1's ladder. Words first, symbols second. Append new
symbols at the bottom and never change one a page already uses.

## Shared with Paper 1's ladder

| Say it in words | Symbol | Colour on the figures |
|---|---|---|
| the picture partway along its path. Diffusion runs t from T (noise) down to 0; DCFM's flow runs the other way, from t = 0 (noise) to t = 1 (data), so every maths step states its direction | x_t | blue (an input) |
| the attributes asked for, such as digit 4 and colour red | y = (y_1, y_2) | blue |
| an attribute left out | ∅ | grey (fixed) |
| an expert's predicted noise (Paper 1's experts) | ε_θ(x_t, y) | green (an expert) |
| a flow model's predicted velocity (DCFM) | v_θ(x_t, t, y) | green |
| the composed prediction, the sum rule | v̂ (or ε̂) | green, with orange where a push is added |
| the interaction term: the joint prediction minus the sum rule; DCFM's penalty drives it to zero | I_t | orange |
| DCFM's penalty weight (the paper writes λ) | λ_CI | orange |
| Paper 1's push strength | λ_t | orange |
| Paper 1's checker | C_t | green (a trained network) |

Colour key: inputs `#1f5fd0`, experts and hidden units `#1d6b3a`, outputs `#c0262d`, the interaction term, the penalty's gradient and
any push `#c8680a`, fixed, frozen or off `#6b7a88`.

**The unit of I_t on this ladder is velocity.** I_t = v_θ(x_t, y) − v̂_θ(x_t, y), in data units per unit of t. Appendix A.2 of the paper
writes every velocity as v = a_t x + b_t ∇ log p, and the a_t x parts cancel inside the sum rule, so the velocity I_t is b_t times
the score I_t that Paper 1's ladder uses. On DCFM's straight path b_t = (1 − t)/t. In noise units, ε = x_t − t v on that path, so the
noise I_t is −t times the velocity I_t, which equals −(1 − t) times the score I_t: Paper 1's r_t = −σ_t I_t with σ_t = 1 − t.

---

## Used only on the DCFM ladder's pages

| Say it in words | Symbol |
|---|---|
| the starting noise (t = 0) and the data image (t = 1) | x_0, x_1 |
| the straight path between them, and its target velocity | x_t = (1 − t) x_0 + t x_1, u_t = x_1 − x_0 |
| the number of attributes, and attribute i's set of values | k, 𝒴_i |
| the set of all attribute combinations, and the share of them some silo holds | 𝒴, 𝒞 (total coverage) |
| DCFM's minimum share for composition to be possible (Eq. 2) | 𝒞_min |
| the share of attribute i's values held somewhere | ℳ_i (marginal coverage) |
| silo (client) a, its data, and its expert | c_a, D_a, v_θ^(a) |
| the mask over attributes, and the masked label | m ∈ {0, 1}^k, y ⊙ m |
| the masking weights: full label, one attribute, none | π_full, π_marg, π_uncond |
| the guidance weight on attribute i in the sum rule | w_i |
| the routing vector: which silo's expert answers the unconditional call (r_0) and each attribute (r_i) | r = (r_0, r_1, …, r_k) |
| a peer's expert held frozen (StopGrad) | v̄_θ^(r) |
| the sum built from routed experts (Eq. 12), and from the student's own calls (Eq. 16) | z_θ^(r_0), z_θ |
| a point on a frozen teacher's path, reached by integrating it from noise (Eq. 15) | x̂_t |
| the factors relating velocity to score on a Gaussian path (A.2) | a_t = α̇_t / α_t, b_t = (α̇_t σ_t − α_t σ̇_t) σ_t / α_t |
| the signal and noise scales of that path; on the straight path α_t = t, σ_t = 1 − t | α_t, σ_t |
| the number of DCFM-A exchange rounds, and of silos | R, N |
