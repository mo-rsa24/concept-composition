"""Write worked-pass.md: one real corrector pass with every equation filled in, read from exact_plane_numbers.json.

  python3 write_worked_pass.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = json.loads((HERE / "exact_plane_numbers.json").read_text())
t = J["trace"]
P = J["pairs"][t["pair"]]
A, B = P["single_prompts"]
last = P["steps"][t["step"]]["last_pass"]
d1, d2 = t["dists_logged"]
r1, r2 = t["dists_recomputed"]
a1, a2 = t["affinities"]
w1, w2 = t["weights_logged"]
v1, v2 = t["weights_recomputed"]
f = lambda x, n=4: f"{x:.{n}f}"

md = f"""# One CO3 corrector pass, worked with the run's real numbers

This page follows one pass of CO3's corrector (Algorithm 3, p. 6) on "{P['prompt']}", seed {J['source']['seed']}, at
step {t['step']} (t = {t['t']}), the first of that step's {t['passes_in_step']} passes. It is pass {t['pass_index'] + 1} of
the run's 40. Each line is one equation of the chain $d_k \\to a_k \\to w_k \\to \\tilde x \\to r \\to x_t^{{(2)}}$ with this
pass's values in it. Every number is read from `exact_plane_numbers.json` (key `trace`), which `exact_plane_numbers.py`
computes from the run's records; symbols are in [the ladder's notation file](../NOTATION.md), and [the corrector
figure](../co3-corrector.png) draws the same chain as a small network.

All lengths are taken over the latent's 65,536 numbers (4 × 128 × 128). The records are stored in float16, so a value
recomputed here can differ from the one the run logged in float32 in the seventh digit.

## What goes in

The latent entering the pass has length $\\|x_t\\|$ = {f(t['length_x'], 2)}. CO3 reads
$\\bar\\alpha_t$ = {t['abar']:.6f} at this step (the scheduler's value one timestep earlier, as the
[code walk](../code-walk.md) explains), so the noise scale is $\\sqrt{{1-\\bar\\alpha_t}}$ = {f(t['sqrt_one_minus_abar'], 5)}.
Guidance inside CO3 is $\\lambda$ = {t['lambda_']}, and $\\beta$ = {t['beta']} after the first step.

The U-Net returns four noise predictions from this one latent: $\\epsilon^\\phi$ (empty prompt), $\\epsilon^C$ (the joint
prompt), $\\epsilon^{{c_1}}$ ("{A}") and $\\epsilon^{{c_2}}$ ("{B}").

## 1. The two distances, $d_k = \\|\\epsilon^{{c_k}} - \\epsilon^{{C}}\\|$

$$d_1 = \\|\\epsilon^{{c_1}} - \\epsilon^{{C}}\\| = {f(d1)} \\qquad d_2 = \\|\\epsilon^{{c_2}} - \\epsilon^{{C}}\\| = {f(d2)}$$

Recomputed from the stored predictions: {f(r1)} and {f(r2)}. The joint prompt's prediction sits closer to the
{A}'s at this pass. By the step's last pass the order has flipped ({f(last['dists'][0], 2)} for {A},
{f(last['dists'][1], 2)} for {B}), as [the per-step figure](co3-steps-cheetah-cougar.png) shows on its step 3 row.

## 2. The affinities, $a_k = e^{{-\\beta d_k}}$ (equation 14)

$$a_1 = e^{{-{t['beta']} \\times {f(d1)}}} = {f(a1)} \\qquad a_2 = e^{{-{t['beta']} \\times {f(d2)}}} = {f(a2)}$$

The closer animal gets the larger affinity.

## 3. The weights, $w_k = -a_k / (a_1 + a_2)$ (equation 15)

$$w_1 = -\\frac{{{f(a1)}}}{{{f(a1)} + {f(a2)}}} = {f(v1)} \\qquad w_2 = -\\frac{{{f(a2)}}}{{{f(a1)} + {f(a2)}}} = {f(v2)}$$

The run logged {f(w1)} and {f(w2)}. With the joint prompt's weight $w_0$ = 2 the three sum to
2 + ({f(w1)}) + ({f(w2)}) = {f(2 + w1 + w2)}.

## 4. The composed mean, $\\tilde x = w_0\\,\\hat x^{{C}} + w_1\\,\\hat x^{{c_1}} + w_2\\,\\hat x^{{c_2}}$ (Algorithm 3 line 4)

Each guess is the guided Tweedie mean $\\hat x^{{c}} = x_t - \\sqrt{{1-\\bar\\alpha_t}}\\,\\big(\\epsilon^\\phi + \\lambda(\\epsilon^{{c}} - \\epsilon^\\phi)\\big)$.
Their lengths:

| Guess | Length |
|---|---|
| $\\hat x^{{C}}$, joint prompt | {f(t['length_joint_guess'])} |
| $\\hat x^{{c_1}}$, "{A}" | {f(t['length_a_guess'])} |
| $\\hat x^{{c_2}}$, "{B}" | {f(t['length_b_guess'])} |

$$\\tilde x = 2\\,\\hat x^{{C}} {w1:+.4f}\\,\\hat x^{{c_1}} {w2:+.4f}\\,\\hat x^{{c_2}}, \\qquad \\|\\tilde x\\| = {f(t['length_combination'])}, \\qquad \\max \\tilde x = {f(t['max_combination'])}$$

$\\tilde x$ lies on the flat sheet through the three guesses, on the far side of the joint's guess from the animals'
weighted average. [The exact-plane figure](co3-exact-plane.png) draws that sheet at the step's last pass.

## 5. The rescale, $r$

Algorithm 3 sets $r = \\|\\hat x^{{C}}\\| / \\|\\tilde x\\|$, which here would be
{f(t['length_joint_guess'])} / {f(t['length_combination'])} = {f(t['r_paper'])}.

The code (`attn_utils.py` lines 252 and 255) instead sets $r = \\max \\hat x^{{C}}$ = {f(t['r_code'])}, the largest single
entry, and divides $\\tilde x$ by its own largest entry:

$$\\tilde x \\leftarrow \\tilde x \\cdot \\frac{{r}}{{\\max \\tilde x}} = \\tilde x \\times \\frac{{{f(t['r_code'])}}}{{{f(t['max_combination'])}}} = \\tilde x \\times {f(t['code_scale_factor'])}, \\qquad \\|\\tilde x\\| = {f(t['length_after_rescale'])}$$

At this pass the two factors are close ({f(t['code_scale_factor'])} against {f(t['r_paper'])}). On eagle × fox
the code's factor is {f(J['pairs']['an_eagle__x__a_fox']['steps'][4]['sheet']['after_code_rescale']['scale'], 2)}
at step 4, which moves the composed mean {f(J['pairs']['an_eagle__x__a_fox']['steps'][4]['sheet']['after_code_rescale']['off_sheet'], 2)}
off the sheet.

## 6. The new latent, $x_t^{{(2)}} = \\tilde x + \\sqrt{{1-\\bar\\alpha_t}}\\,\\epsilon^\\phi$ (Algorithm 3 line 7)

$$\\|x_t^{{(2)}}\\| = {f(t['length_x2_before_norm_match'], 2)}$$

The code then adds a step Algorithm 3 does not have (line 257), matching the old latent's length:

$$x_t^{{(2)}} \\leftarrow x_t^{{(2)}} \\cdot \\frac{{\\|x_t\\|}}{{\\|x_t^{{(2)}}\\|}} = x_t^{{(2)}} \\times {f(t['norm_match_factor'], 5)}, \\qquad \\|x_t^{{(2)}}\\| = {f(t['length_x2'], 2)}$$

This pass moved the latent by {100 * t['change_of_latent_relative']:.2f}% of its length.

## The check

The next pass recorded its incoming latent, of length {f(t['length_next_recorded'], 2)}. The latent computed above differs
from it by $\\|x_t^{{(2)}} - x_{{\\mathrm{{next}}}}\\| / \\|x_{{\\mathrm{{next}}}}\\|$ = {f(t['rel_diff_here'])}. The run's own control,
computed on the float32 records before they were stored, gave {f(t['rel_diff_in_run'])} for this pass, against its limit of
0.01.
"""
(HERE / "worked-pass.md").write_text(md)
print("wrote worked-pass.md")
