# One CO3 corrector pass, worked with the run's real numbers

This page follows one pass of CO3's corrector (Algorithm 3, p. 6) on "a cheetah and a cougar", seed 6001, at
step 3 (t = 921), the first of that step's 5 passes. It is pass 26 of
the run's 40. Each line is one equation of the chain $d_k \to a_k \to w_k \to \tilde x \to r \to x_t^{(2)}$ with this
pass's values in it. Every number is read from `exact_plane_numbers.json` (key `trace`), which `exact_plane_numbers.py`
computes from the run's records; symbols are in [the ladder's notation file](../NOTATION.md), and [the corrector
figure](../co3-corrector.png) draws the same chain as a small network.

All lengths are taken over the latent's 65,536 numbers (4 × 128 × 128). The records are stored in float16, so a value
recomputed here can differ from the one the run logged in float32 in the seventh digit.

## What goes in

The latent entering the pass has length $\|x_t\|$ = 246.59. CO3 reads
$\bar\alpha_t$ = 0.011460 at this step (the scheduler's value one timestep earlier, as the
[code walk](../code-walk.md) explains), so the noise scale is $\sqrt{1-\bar\alpha_t}$ = 0.99425.
Guidance inside CO3 is $\lambda$ = 0.8, and $\beta$ = 0.9 after the first step.

The U-Net returns four noise predictions from this one latent: $\epsilon^\phi$ (empty prompt), $\epsilon^C$ (the joint
prompt), $\epsilon^{c_1}$ ("cheetah") and $\epsilon^{c_2}$ ("cougar").

## 1. The two distances, $d_k = \|\epsilon^{c_k} - \epsilon^{C}\|$

$$d_1 = \|\epsilon^{c_1} - \epsilon^{C}\| = 0.8167 \qquad d_2 = \|\epsilon^{c_2} - \epsilon^{C}\| = 0.9895$$

Recomputed from the stored predictions: 0.8167 and 0.9895. The joint prompt's prediction sits closer to the
cheetah's at this pass. By the step's last pass the order has flipped (0.96 for cheetah,
0.80 for cougar), as [the per-step figure](co3-steps-cheetah-cougar.png) shows on its step 3 row.

## 2. The affinities, $a_k = e^{-\beta d_k}$ (equation 14)

$$a_1 = e^{-0.9 \times 0.8167} = 0.4795 \qquad a_2 = e^{-0.9 \times 0.9895} = 0.4104$$

The closer animal gets the larger affinity.

## 3. The weights, $w_k = -a_k / (a_1 + a_2)$ (equation 15)

$$w_1 = -\frac{0.4795}{0.4795 + 0.4104} = -0.5388 \qquad w_2 = -\frac{0.4104}{0.4795 + 0.4104} = -0.4612$$

The run logged -0.5388 and -0.4612. With the joint prompt's weight $w_0$ = 2 the three sum to
2 + (-0.5388) + (-0.4612) = 1.0000.

## 4. The composed mean, $\tilde x = w_0\,\hat x^{C} + w_1\,\hat x^{c_1} + w_2\,\hat x^{c_2}$ (Algorithm 3 line 4)

Each guess is the guided Tweedie mean $\hat x^{c} = x_t - \sqrt{1-\bar\alpha_t}\,\big(\epsilon^\phi + \lambda(\epsilon^{c} - \epsilon^\phi)\big)$.
Their lengths:

| Guess | Length |
|---|---|
| $\hat x^{C}$, joint prompt | 4.0888 |
| $\hat x^{c_1}$, "cheetah" | 3.7899 |
| $\hat x^{c_2}$, "cougar" | 3.5831 |

$$\tilde x = 2\,\hat x^{C} -0.5388\,\hat x^{c_1} -0.4612\,\hat x^{c_2}, \qquad \|\tilde x\| = 4.5369, \qquad \max \tilde x = 0.0455$$

$\tilde x$ lies on the flat sheet through the three guesses, on the far side of the joint's guess from the animals'
weighted average. [The exact-plane figure](co3-exact-plane.png) draws that sheet at the step's last pass.

## 5. The rescale, $r$

Algorithm 3 sets $r = \|\hat x^{C}\| / \|\tilde x\|$, which here would be
4.0888 / 4.5369 = 0.9012.

The code (`attn_utils.py` lines 252 and 255) instead sets $r = \max \hat x^{C}$ = 0.0409, the largest single
entry, and divides $\tilde x$ by its own largest entry:

$$\tilde x \leftarrow \tilde x \cdot \frac{r}{\max \tilde x} = \tilde x \times \frac{0.0409}{0.0455} = \tilde x \times 0.8976, \qquad \|\tilde x\| = 4.0723$$

At this pass the two factors are close (0.8976 against 0.9012). On eagle × fox
the code's factor is 0.85
at step 4, which moves the composed mean 1.54
off the sheet.

## 6. The new latent, $x_t^{(2)} = \tilde x + \sqrt{1-\bar\alpha_t}\,\epsilon^\phi$ (Algorithm 3 line 7)

$$\|x_t^{(2)}\| = 246.42$$

The code then adds a step Algorithm 3 does not have (line 257), matching the old latent's length:

$$x_t^{(2)} \leftarrow x_t^{(2)} \cdot \frac{\|x_t\|}{\|x_t^{(2)}\|} = x_t^{(2)} \times 1.00069, \qquad \|x_t^{(2)}\| = 246.59$$

This pass moved the latent by 0.78% of its length.

## The check

The next pass recorded its incoming latent, of length 246.66. The latent computed above differs
from it by $\|x_t^{(2)} - x_{\mathrm{next}}\| / \|x_{\mathrm{next}}\|$ = 0.0006. The run's own control,
computed on the float32 records before they were stored, gave 0.0006 for this pass, against its limit of
0.01.
