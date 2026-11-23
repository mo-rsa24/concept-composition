# CFGComposer: Compositional Audio-Text-to-Motion Generation via Copula Score Correction

arXiv 2611.15101, read from the full PDF (9 pages, no separate appendix).

## What it does

The paper generates whole-body motion from speech audio and a gesture caption by composing a frozen audio-to-motion model with a frozen text-to-motion model (p. 1 to 3). It writes the joint score as the product-of-experts score plus an interaction term, which it calls the copula score (Eq. 9, p. 3). It estimates that term as the difference between a small jointly trained model and the product of experts, and adds a fraction $\lambda$ of it at each step (Eq. 11 and 12, p. 4).

**Plain answer on the two questions for this loop.** Yes, all three networks read the noisy motion $x_t$ at every sampling step, from the first step onward (Algorithm 2, p. 5). There is no classifier and no gradient of a score with respect to $x_t$, because the correction is a difference between two denoiser outputs. It does change the large-scale result, here body pose inside the gesture window (SRGR 0.9916, p. 6). The domain is motion, so the paper says nothing about the layout of a picture.

## What the model or score reads

- **Motion:** a noisy sequence $x_t$ of $T = 300$ frames and $J = 60$ joints (p. 3).
- **Audio:** Whisper features of size $250 \times 512$, at 50 frames per second for 5 seconds, condensed by a 2-layer Transformer to one conditioning token (p. 3 to 5).
- **Text:** a gesture caption such as "thumbs up", through a frozen text encoder that projects to a fixed-length embedding (p. 3 to 5).
- **Window:** a per-frame embedding of the gesture window $(g_s, g_e)$ for the joint model (p. 5).
- **Networks:** Transformer backbones with $d = 512$ and 8 layers, cosine schedule, $x_0$ prediction, 1000 steps (p. 3 to 5).

## What it trained on

- **Experts:** the audio expert trains for 300k steps on the full BEAT audio-motion corpus. The text expert trains for 200k steps on gesture captions plus the authors' motion-caption dataset (p. 6).
- **Joint model:** 775 training clips and 195 validation clips, built from BEAT's annotated gestures, with each annotation placed at 5 frame offsets (p. 5 to 6). It trains for 50k steps with an $x_0$ mean squared error loss, $t$ uniform on 0 to 999 and joint condition dropout of 0.1 (Eq. 16 and Algorithm 1, p. 5).
- **Split:** 775 and 195 are both multiples of 5, which fits a split by annotation (155 and 39). The paper does not state how the split was made (inferred).

## How its score or gradient enters

The exact decomposition, stated for all $x$, $t$, $a$ and $c$ (Eq. 9, p. 3):

$$
s_{ac} = s_a + s_c - s_u + s_I, \qquad s_I = \nabla_x I_t(x; a, c), \qquad I_t = \log \frac{p_t(x|a,c)\, p_t(x)}{p_t(x|a)\, p_t(x|c)}
$$

The estimate and the update (Eq. 11 and 12, p. 4):

$$
\hat\epsilon_{AT2M} - \hat\epsilon_{PoE} \approx -\sqrt{1 - \bar\alpha_t}\,(s_{ac} - s_{PoE}) = \hat\epsilon_I
$$

$$
\hat\epsilon_{total} = \hat\epsilon_{PoE} + \lambda \cdot (\hat\epsilon_{AT2M} - \hat\epsilon_{PoE})
$$

Here $\hat\epsilon_{AT2M}$ is the joint model's prediction converted from $x_0$ to $\epsilon$, and $\lambda = 0.7$ is the chosen strength (p. 5 to 6). Rearranged, the update is $(1-\lambda)\hat\epsilon_{PoE} + \lambda\hat\epsilon_{AT2M}$, a straight interpolation between the two predictions (inferred). The correction runs at every step $t = T, \dots, 1$ with no cap or normalisation (Algorithm 2, p. 5).

The baseline as implemented differs from the unit-weight product (Eq. 14 and 15, p. 4):

$$
\hat\epsilon^g_y = g(\hat\epsilon_y - \hat\epsilon_u), \qquad \hat\epsilon_{PoE} = \hat\epsilon_u + \frac{w_a \hat\epsilon^g_a + w_c \hat\epsilon^g_c}{w_a + w_c} \text{ inside the window}, \qquad \hat\epsilon_u + \hat\epsilon^g_a \text{ outside}
$$

The settings are $g = 5$, $w_a = 1$ and $w_c = 1.6$, with a 15-frame smoothstep at the window edges (p. 4). Inside the window this puts coefficients of about 1.92 on the audio direction and 3.08 on the text direction, where the unit-weight product of Eq. 7 has 1 and 1 (computed). Algorithm 2 shows no guidance applied to the joint model (p. 5). So the measured difference mixes the interaction term with a guidance-scale difference, and it is not the clean $\hat\epsilon_I$ of Eq. 10 (inferred).

## When adding scores is valid

The paper states that adding scores (the product of experts) is exact only when the copula score is zero, and it says this holds "if and only if" $a \perp c \mid x_t$ (p. 4). It also argues that independence given clean data $x_0$ does not survive noising, because marginalising through $q(x_t|x_0)$ creates dependence between $a$ and $c$ given $x_t$ (p. 4). This is the same reason this project's interaction term exists.

The "only if" is stronger than the maths supports. $s_I$ is a gradient in $x$, so it vanishes whenever $I_t$ is constant in $x$ for the given $(a, c)$, and that constant need not be zero (inferred).

## Which of the five properties it bears on

**Counting.** Not tested as a property. One robot figure shows a "counting two" gesture (p. 8).

**Attribute binding.** Not tested.

**Spatial relation.** Not tested.

**Position and size.** Not tested. The gesture window fixes when a gesture occurs in time, and body-part-level spatial control is listed as future work (p. 8).

**Prompt match.** Tested in the motion domain only: SRGR and G-MSE inside the gesture window, and a blind human rating of semantic accuracy (p. 6).

## Unseen concepts

This paper does not test held-out concepts.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| SRGR, keypoint recall weighted to the gesture window (0 to 1, higher better) | 0.9916 | 0.8018 RoboGesture fine-tuned | BEAT | CFGComposer, $\lambda = 0.7$ | p. 7, Table II |
| FID on a motion encoder (lower better) | 27.18 | 62.31 DiffSHEG on BEAT plus AT2M | BEAT | same | p. 7, Table II |
| G-MSE in the gesture window (lower better) | 0.0079 | 0.0139 DiffSHEG fine-tuned | BEAT | same | p. 7, Table II |
| Beat alignment (closer to the ground-truth 0.1743 is better) | 0.1826 | 0.1808 SemTalk | BEAT | same | p. 6, Table I |
| Human net preference on semantic accuracy (-1 to +1, higher better) | +0.570 | -0.132 DiffSHEG | 17 raters, 50 test clips | same | p. 6, Table I |
| SRGR at $\lambda = 0$ (the product of experts alone) | about 0.915 | n/a | BEAT | same | read from the plot, p. 8, Fig. 4 |
| SRGR at $\lambda = 1$ (the joint model alone) | about 0.987 | n/a | BEAT | same | read from the plot, p. 8, Fig. 4 |

The product of experts has no row in Table I or Table II, so its numbers appear only as the left edge of Fig. 4 (p. 6 to 8). The $\lambda$ was chosen on the same four metrics that are reported (p. 7).

## What to copy, what to avoid

- Copy: the split of $\epsilon_{joint}$ into product plus interaction is the same object this project calls the correction, so Eq. 9 and the noising argument on p. 4 can be cited for it (inferred).
- Copy: plot metrics across the whole range of $\lambda$, with $\lambda = 0$ and $\lambda = 1$ marked as the two pure models (Fig. 4, p. 8).
- Avoid: reading this as evidence for a score-difference gradient. The method has no sentence score and takes no gradient (Algorithm 2, p. 5).
- Avoid: its reliance on a joint model. It needs one trained on paired data, and most of the gain over the product of experts is already there at $\lambda = 1$: about 0.987 against 0.9916 on SRGR (Fig. 4, p. 8) (inferred).
- Avoid: the mismatch between guided and unguided predictions if a similar correction is built on SDXL. Compute both sides of the difference at the same guidance scale (inferred).
- Does not bear on our failure modes: it never shows that a correction moves objects in a picture, and CO3's borrowed parts and wrong species have no analogue here (inferred).

## Closer work it cites

None.
