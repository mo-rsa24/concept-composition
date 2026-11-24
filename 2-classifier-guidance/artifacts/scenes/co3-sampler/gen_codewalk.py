"""Level 3: writes code-walk.md, the authors' CO3 code line by line, each block under the equation it computes.

Every code block is read byte for byte from the checkout, every block's anchor lines are asserted to hold the text
they are expected to hold (a moved line stops the build), and every number comes from co3_numbers.json.

Run:  python3 gen_codewalk.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "co3_numbers.json").read_text())
REPO = Path(N["code"]["checkout"])
FILES = {"Co3.py": REPO / "composers" / "Co3.py", "attn_utils.py": REPO / "composers" / "utils" / "attn_utils.py"}
LINES = {k: v.read_text().splitlines() for k, v in FILES.items()}


def block(name, a, b, anchors):
    """Lines a..b of a file, numbered, after asserting each anchor {line: text}."""
    for n, must in anchors.items():
        assert a <= n <= b, f"anchor {n} outside {a}-{b}"
        assert must in LINES[name][n - 1], f"{name}:{n} no longer holds {must!r}: {LINES[name][n - 1]!r}"
    body = "\n".join(f"{n:>3} | {LINES[name][n - 1]}" for n in range(a, b + 1))
    return f"`composers/{'utils/' if name == 'attn_utils.py' else ''}{name}`, lines {a} to {b}:\n\n```python\n{body}\n```\n"


def thumb(png, alt):
    return f'<a href="{png}"><img src="{png}" width="200" alt="{alt}"></a>'


s, st, W = N["settings"], N["steps"], N["weights"]
corr = next(r for r in st if r["algorithm"] == "corrector")
fs = N["first_step"]

sections = []

sections.append(f"""## 1. Ordinary sampler: the schedule and the list of steps

**In words.** Fifty timesteps, 20 apart, from {N['timesteps'][0]} down to {N['timesteps'][-1]}, and the noise level
$\\bar\\alpha_t$ at each one. CO3 is switched on for the first {s['num_ts_to_correct']} of them.

**The source.** Algorithm 1 line 1 (p. 4): $\\text{{for }} t = T \\text{{ down to }} 1$, with $\\bar\\alpha_t$ the
cumulative product of the noising schedule (p. 3).

{block("Co3.py", 85, 91, {85: "DDIMScheduler.from_pretrained", 87: "set_timesteps(self.config.n_timesteps", 89: "self.skip = N_ts // self.config.n_timesteps", 91: "torch.tensor([1.0])"})}
{block("Co3.py", 99, 103, {103: "self.scheduler.timesteps[:t_cond]"})}
{block("Co3.py", 200, 202, {201: "self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod"})}
**Symbols.** `self.scheduler.timesteps` is the list $t_0 = {N['timesteps'][0]}, \\dots, t_{{49}} = {N['timesteps'][-1]}$
(the scheduler's "leading" spacing plus an offset of 1). `self.skip` is the gap between steps, {N['skip']}.
`self.alpha(t)` is $\\bar\\alpha_t$. `latent_corrector_ts` is the first {s['num_ts_to_correct']} timesteps,
{N['co3_timesteps']}.

**Exactness.** Line 91 puts 1.0 in front of the scheduler's list, so `self.alpha(t)` returns the scheduler's
$\\bar\\alpha$ at $t-1$. At t = {fs['t']} the code uses {fs['alpha_bar']:.6f} where the scheduler holds
{fs['alpha_bar_without_prepend']:.6f} at t itself, {100 * fs['relative_shift']:.1f}% apart. The paper does not say which
index it means. Every $\\bar\\alpha_t$ in the figures is the code's value.

{thumb("co3-schedule.png", "the 50-step schedule with the six CO3 steps zoomed")}
""")

sections.append(f"""## 2. Ordinary sampler: the loop over the 50 steps

**In words.** Start from Gaussian noise the size of one SDXL latent and call one denoising step per timestep.

**The source.** Algorithm 1 (p. 4): require $x_T$; for $t = T$ down to 1, run lines 2 to 6.

{block("Co3.py", 366, 381, {367: "self.start_t = self.scheduler.timesteps[0]", 373: "torch.randn(1,4,self.config.resolution_h//8", 380: "for i, t in enumerate(tqdm(self.scheduler.timesteps", 381: "x = self.denoise_step(x, t, i)"})}
**Symbols.** `normal` is $x_T$, shape 1 × 4 × {s['resolution_h'] // 8} × {s['resolution_w'] // 8}
({W['latent_numbers']:,} numbers), drawn from a standard Gaussian and multiplied by `init_noise_sigma`, which is 1.0
for DDIM. The project draws this noise itself (`start_noise` in `co3_common.py`) so every method shares one starting
latent per seed. `self.start_t` is $t_0 = {N['timesteps'][0]}$, used below to pick $\\beta$.

**Exactness.** The loop is Algorithm 1's loop exactly. After it, lines 383 to 415 decode the latent with the VAE.

{thumb("co3-overview.png", "the whole run annotated stage by stage")}
""")

sections.append(f"""## 3. The method: whether CO3 runs at this step, and which one

**In words.** On the first {s['num_ts_to_correct']} steps, run CO3 on $x_t$ before the ordinary step. The first
{s['num_resampling_steps']} of those use the resampler, the next {s['num_ts_to_correct'] - s['num_resampling_steps']} the
corrector.

**The source.** Algorithm 1 lines 2 to 4 (p. 4):

$$\\text{{if }} t > T_c:\\quad x_t^{{(2)}} \\leftarrow \\mathrm{{CO3}}(x_t, C, c_1, \\dots, c_K),\\qquad x_t \\leftarrow x_t^{{(2)}}$$

and the hybrid rule on p. 5: resample while $t > T_R$, correct while $T_C < t < T_R$.

{block("Co3.py", 279, 283, {279: "if t in self.config.latent_corrector_ts:", 280: "text_embed_cond[:-1]"})}
{block("Co3.py", 302, 333, {302: "elif self.config.corrector_algo == 'co3-hybrid':", 306: "beta = 1.0 if t == self.start_t else self.config.beta", 307: "num_corrector_steps = 10 if t >= self.scheduler.timesteps[1]", 309: "if t >= self.scheduler.timesteps[self.config.num_resampling_steps-1]:", 312: "self.latentCorrector.co3_resampling(", 324: "self.latentCorrector.co3_corrector("})}
**Symbols.** `text_embed_corrector` holds four prompt encodings in this order: {', '.join(f'"{p}"' for p in N['corrector_conditions'])}.
The release appends the full prompt once more at the end of its prompt string, and line 280 drops that last copy.
`num_corrector_steps` is $P$, the number of passes. `beta` is $\\beta$ in equation 14.

**What the branch decides**, computed in `co3_numbers.json` and matching the run log
`source/run-log-excerpt-l6-m1-p2-93468.txt` line for line:

| step | t | CO3 | passes P | $\\beta$ |
|---|---|---|---|---|
""" + "\n".join(f"| {r['step']} | {r['t']} | {r['algorithm']} | {r['iterations']} | {r['beta']:g} |" for r in st[:6]) + f"""

**Exactness.** This is the paper's hybrid with $T_R$ and $T_C$ set by step count. The comment on line 303 says only
$T$ gets 10 passes; line 307 gives 10 to both t = {N['timesteps'][0]} and t = {N['timesteps'][1]}.

{thumb("co3-schedule.png", "which steps resample, which correct, and how many passes")}
""")

sections.append(f"""## 4. The method: the resampler, weights summing to 0

**In words.** Ask the U-Net for four noise predictions at the same $x_t$, add the joint prompt's prediction and
subtract a weighted share of each concept's, scale that onto the noise level, and add the empty prompt's prediction
back. The old $x_t$ cancels, so the new latent is noise built from the predictions alone.

**The source.** Algorithm 2 (p. 6):

$$\\epsilon^{{c_0}} = \\psi(x_t, C),\\quad \\epsilon^{{c_k}} = \\psi(x_t, c_k)$$
$$\\tilde{{x}}_{{\\mathrm{{tweedie}}}} = -\\lambda\\sqrt{{1-\\bar\\alpha_t}}\\sum_{{k=0}}^{{K}} w_k\\,\\epsilon^{{c_k}},\\qquad
x_t^{{(2)}} = \\tilde{{x}}_{{\\mathrm{{tweedie}}}} + \\sqrt{{1-\\bar\\alpha_t}}\\,\\epsilon^{{\\phi}}_t$$

with $\\sum_k w_k = 0$ and $w_0 = {W['w0_resampler']:g}$ (p. 5).

{block("attn_utils.py", 170, 177, {176: "concept_weights = [-1.0/num_concepts]*num_concepts", 177: "composition_weights = [1.0] + concept_weights"})}
Lines 176 and 177 start from equal weights: $w_0 = {W['w0_resampler']:g}$ and $w_k = {W['uniform_concept_weight']:g}$ for
the K = {W['K']} concepts, sum {W['sum_resampler']:g}.

{block("attn_utils.py", 179, 185, {182: "torch.cat([x] * (num_concepts + 2))", 183: "noise_pred = unet(latent_model_input, t", 184: "noise_pred_uncond, noise_pred_multi = noise_pred[:1], noise_pred[1:2]"})}
Algorithm 2 lines 1 to 3: one U-Net call on a batch of {W['K'] + 2} copies of $x_t$, giving $\\epsilon^{{\\phi}}$
(`noise_pred_uncond`), $\\epsilon^{{c_0}}$ (`noise_pred_multi`) and $\\epsilon^{{c_1}}, \\epsilon^{{c_2}}$
(`noise_pred_concepts`), each 1 × 4 × 128 × 128.

{block("attn_utils.py", 186, 192, {187: ".norm().item())", 191: "self._get_concept_weights_from_dists(concept_dists, method='exp', beta=beta)"})}
Equations 14 and 15, recomputed on every pass from that pass's predictions (section 5).

{block("attn_utils.py", 194, 200, {194: "self._get_sumzero_tweedie_composed_noise(", 196: "denoised_tweedie_at = x - (1 - at).sqrt() * composed_noise_lmda", 198: "x_2 = denoised_tweedie_at + (1 - at).sqrt() * noise_pred_uncond", 200: "x = x_2"})}
{block("attn_utils.py", 67, 84, {77: "composed_noise = composed_noise + weights[0] * noise_pred_multi", 81: "composed_noise = lmda * composed_noise", 83: "composed_noise = (x + (1-alpha_t).sqrt() * composed_noise) / (1-alpha_t).sqrt()"})}
**Exactness.** The helper returns $(x_t + \\sqrt{{1-\\bar\\alpha_t}}\\,\\lambda\\sum_k w_k\\epsilon^{{c_k}})/\\sqrt{{1-\\bar\\alpha_t}}$,
and line 196 computes $x_t - \\sqrt{{1-\\bar\\alpha_t}}$ times that, which is
$-\\lambda\\sqrt{{1-\\bar\\alpha_t}}\\sum_k w_k\\epsilon^{{c_k}}$: Algorithm 2 line 5 exactly, up to rounding in half
precision. Line 198 is line 6 and line 200 is line 7. At t = {st[0]['t']}, $\\lambda$ = {s['lmda']:g} and
$\\sqrt{{1-\\bar\\alpha_t}}$ = {st[0]['sqrt_one_minus_alpha_bar']:.4f}. $\\lambda$ here is `self.lmda`, a different
number from the ordinary step's guidance scale.

{thumb("co3-resampler.png", "Algorithm 2 drawn as a small network with its equations")}
""")

sections.append(f"""## 5. The method: the concept weights of equation 15

**In words.** Measure how far each concept's prediction sits from the joint prompt's, turn each distance into an
affinity that shrinks as the distance grows, and share a total weight of $-1$ among the concepts by affinity. The
closest concept gets the most negative weight, so CO3 pushes hardest away from it.

**The source.** p. 5 and equations 14 and 15 (pp. 5 and 6):

$$d_k = \\lVert \\epsilon^{{C}} - \\epsilon^{{c_k}} \\rVert,\\qquad a_k = \\exp(-\\beta\\, d_k),\\qquad
w_k = -\\frac{{a_k}}{{\\sum_{{j=1}}^{{K}} a_j}}$$

{block("attn_utils.py", 118, 156, {119: "def _get_concept_weights_from_dists(", 137: "w_min_mag, w_max_mag: clipping range on |weights|", 145: "a = torch.exp(-beta * dists)", 151: "if torch.all(a == 0):", 155: "w = -a / a.sum()"})}
**Symbols.** `dists` is $(d_1, \\dots, d_K)$, each the length of the difference between two predictions over all
{W['latent_numbers']:,} numbers of the latent (lines 187 and 243). `beta` is $\\beta$: {N['beta_first_step']:g} at
t = {N['timesteps'][0]} and {s['beta']:g} afterwards (section 3).

**Exactness.** Line 145 is equation 14 and line 155 is equation 15. With two concepts the weight depends only on the
gap: $w_1 = -1/(1 + e^{{\\beta (d_1 - d_2)}})$. At $\\beta$ = {s['beta']:g} a gap of 1 gives $w_1$ =
{W['w1_at_gap'][str(s['beta'])]['1.0']:.3f}, a gap of 2 gives {W['w1_at_gap'][str(s['beta'])]['2.0']:.3f} and a gap of 5
gives {W['w1_at_gap'][str(s['beta'])]['5.0']:.3f}. Lines 151 and 152, not in the paper, fall back to equal weights if
every affinity underflows to 0. The docstring (line 137) names a clip range on $|w_k|$ of 0.01 to 0.9 that the
function never applies.

{thumb("co3-weights.png", "equations 14 and 15 as a small network beside the weight curve")}
""")

sections.append(f"""## 6. The method: the corrector, weights summing to 1

**In words.** Guide each prediction with the empty prompt's, form each one's Tweedie mean, add the joint prompt's
mean twice and subtract a weighted share of each concept's, rescale, and add the empty prompt's prediction back at the
noise level. The weights sum to 1, so $x_t$ survives and the corrector edits it.

**The source.** Algorithm 3 (p. 6), with $\\epsilon^{{\\lambda,c}} = \\epsilon^{{\\phi}} + \\lambda(\\epsilon^{{c}} - \\epsilon^{{\\phi}})$
(equation 4, p. 3) and $\\hat{{x}}_{{\\mathrm{{tweedie}}}}[\\epsilon] = x_t - \\sqrt{{1-\\bar\\alpha_t}}\\,\\epsilon$ (p. 3):

$$\\tilde{{x}}_{{\\mathrm{{tweedie}}}} = \\sum_{{k=0}}^{{K}} w_k\\,\\hat{{x}}_{{\\mathrm{{tweedie}}}}[\\epsilon^{{\\lambda,c_k}}],\\qquad
r = \\frac{{\\lVert \\hat{{x}}^{{c_0}}_{{\\mathrm{{tweedie}}}} \\rVert_2}}{{\\lVert \\tilde{{x}}_{{\\mathrm{{tweedie}}}} \\rVert_2}},\\qquad
\\tilde{{x}}_{{\\mathrm{{tweedie}}}} \\leftarrow r\\,\\tilde{{x}}_{{\\mathrm{{tweedie}}}},\\qquad
x_t^{{(2)}} = \\tilde{{x}}_{{\\mathrm{{tweedie}}}} + \\sqrt{{1-\\bar\\alpha_t}}\\,\\epsilon^{{\\phi}}_t$$

with $\\sum_k w_k = 1$ and $w_0 = {W['w0_corrector']:g}$ (p. 5).

{block("attn_utils.py", 226, 233, {232: "concept_weights = [-1.0/num_concepts]*num_concepts", 233: "composition_weights = [2.0] + concept_weights"})}
$w_0 = {W['w0_corrector']:g}$ and $w_k = {W['uniform_concept_weight']:g}$, sum {W['sum_corrector']:g}. The comment on line
233 says the weights sum to 0; they sum to 1.

{block("attn_utils.py", 235, 249, {238: "torch.cat([x] * (num_concepts + 2))", 243: ".norm().item())", 248: "self._get_concept_weights_from_dists(concept_dists, method='exp', beta=beta)"})}
Algorithm 3 lines 1 to 3, then equations 14 and 15, the same as the resampler.

{block("attn_utils.py", 251, 258, {251: "self._get_contrastive_tweedie(", 252: ".max() #at.sqrt() * x.max()", 255: "composed_tweedie = composed_tweedie / composed_tweedie.max() * r", 256: "x_2 = composed_tweedie + (1-at).sqrt() * noise_pred_uncond", 257: "x_2 = x_2 * (x.norm()/ x_2.norm())"})}
{block("attn_utils.py", 99, 115, {107: "noise_pred = noise_pred_uncond + lmda * (noise_pred_multi - noise_pred_uncond)", 108: "tweedie_multi = x - (1 - alpha_t).sqrt() * noise_pred", 112: "noise_pred_single = noise_pred_uncond + lmda * (noise_pred_concepts[i:i+1] - noise_pred_uncond)", 114: "tweedie_multi = tweedie_multi + weights[i+1] * tweedie_single"})}
**Exactness.** The helper (lines 99 to 115) is Algorithm 3 line 4 exactly, and line 256 is line 7. Lines 252 and 255
differ from lines 5 and 6: the code sets $r$ to the largest single entry of the joint prompt's Tweedie mean and divides
the composed mean by its own largest entry before multiplying by $r$, so it matches the largest entries where the paper
matches the lengths. Line 257 has no line in Algorithm 3: it rescales $x_t^{{(2)}}$ to the length of the incoming
$x_t$. At t = {corr['t']}, $\\sqrt{{1-\\bar\\alpha_t}}$ = {corr['sqrt_one_minus_alpha_bar']:.4f}.

{thumb("co3-corrector.png", "Algorithm 3 as a small network with the printed and coded rules boxed beneath")}
""")

sections.append(f"""## 7. Ordinary sampler: one DDIM step with CFG++

**In words.** Ask the U-Net for the empty prompt's and the joint prompt's predictions, mix them with guidance
{s['guidance_scale']:g}, estimate the clean latent, and step to the next timestep. The renoise uses the empty prompt's
prediction, which is what CFG++ changes about ordinary guidance.

**The source.** Equations 4 to 6 (p. 3):

$$\\epsilon^{{\\lambda,c}}_t = \\lambda\\,\\epsilon_\\theta(x_t, c, t) + (1-\\lambda)\\,\\epsilon_\\theta(x_t, \\varnothing, t),\\qquad
\\hat{{x}}_0 = \\frac{{x_t - \\sqrt{{1-\\bar\\alpha_t}}\\,\\epsilon^{{\\lambda,c}}_t}}{{\\sqrt{{\\bar\\alpha_t}}}},\\qquad
x_{{t-1}} = \\sqrt{{\\bar\\alpha_{{t-1}}}}\\,\\hat{{x}}_0 + \\sqrt{{1-\\bar\\alpha_{{t-1}}}}\\,\\epsilon^{{\\lambda,c}}_t$$

{block("Co3.py", 263, 277, {265: "next_t = t - self.skip", 266: "at = self.alpha(t)", 267: "at_next = self.alpha(next_t)", 274: "text_embed_multi = text_embed_cond[1].unsqueeze(0)"})}
{block("Co3.py", 338, 364, {338: "torch.cat([x]+[x])", 348: "noise_pred = self.unet(latent_model_input, t", 354: "noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)", 355: "denoised_tweedie = (x - (1-at).sqrt() * noise_pred) / at.sqrt()", 357: "denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond", 361: "if t == 1:"})}
**Symbols.** `noise_pred_uncond` is $\\epsilon^{{\\phi}}$, `noise_pred_cond` the joint prompt's prediction,
`self.config.guidance_scale` is $\\lambda$ = {s['guidance_scale']:g}, `at` and `at_next` are $\\bar\\alpha_t$ and
$\\bar\\alpha_{{t-20}}$ (the paper writes $t-1$ for the next step), `denoised_tweedie` is $\\hat{{x}}_0$.

**Exactness.** Line 354 is equation 4 written as $\\epsilon^{{\\phi}} + \\lambda(\\epsilon^{{c}} - \\epsilon^{{\\phi}})$ and line
355 is equation 5. Line 357 differs from equation 6 by design: with `use_cfgpp` on, the renoise uses $\\epsilon^{{\\phi}}$
in place of the guided prediction. On the last step, t = 1, line 362 returns $\\hat{{x}}_0$ directly. The paper's
experiments state guidance $\\lambda$ = 5.0 with plain DDIM (p. 6); the release runs {s['guidance_scale']:g} with CFG++.

{thumb("co3-overview.png", "the whole run, with this step in the small forward diagram")}
""")

disagree = f"""## Where the paper and the code disagree

Pages are in the PDF on the shelf, arXiv 2509.25940v2. Lines are in the checkout at commit `{N['code']['commit'][:7]}`.

| What | Paper | Code | Effect |
|---|---|---|---|
| Corrector's mean normalisation | Algorithm 3 lines 5 and 6 (p. 6): $r$ is a ratio of lengths, $\\lVert\\hat{{x}}^{{c_0}}\\rVert_2 / \\lVert\\tilde{{x}}\\rVert_2$ | `attn_utils.py` 252 and 255: $r$ is the largest entry of $\\hat{{x}}^{{c_0}}$, and $\\tilde{{x}}$ is divided by its own largest entry | matches peaks, where the paper matches lengths |
| Extra rescale after the corrector | not in Algorithm 3 | `attn_utils.py` 257: $x^{{(2)}} \\leftarrow x^{{(2)}}\\,\\lVert x_t\\rVert_2 / \\lVert x^{{(2)}}\\rVert_2$ | each pass keeps the latent's length |
| Guidance in the ordinary step | $\\lambda$ = 5.0, plain DDIM (p. 6); equation 6 renoises with the guided prediction (p. 3) | guidance {s['guidance_scale']:g} with CFG++, renoise with $\\epsilon^{{\\phi}}$ (`Co3.py` 354 to 357; `co3_common.py`) | a different sampler from the one the paper's equations describe |
| How many steps get CO3, and $\\beta$ | 10 steps, $\\beta$ = 0.8 (pp. 6 and 17) | {s['num_ts_to_correct']} steps, $\\beta$ = {s['beta']:g}, and {N['beta_first_step']:g} at t = {N['timesteps'][0]} (`Co3.py` 306; `co3_common.py`) | the release's settings sit inside the paper's own ablation range (A.8, pp. 19 and 20) |
| Algorithm 1 line 6's label | "Reverse DDIM" in the v2 PDF (p. 4); the screenshot in `source/` reads "Inverse DDIM", and which version it came from is not recorded | an ordinary denoising DDIM step (`Co3.py` 354 to 362) | a label only |
| Which $\\bar\\alpha$ belongs to t | not stated | `Co3.py` 91 prepends 1.0, so t uses the scheduler's value at $t-1$ | {100 * fs['relative_shift']:.1f}% at t = {fs['t']} |
| Code comments | none | line 303 says only T gets 10 passes (line 307 gives two steps 10); line 233 says the corrector's weights sum to 0 (they sum to 1); the docstring at line 137 names a clip on $\\lvert w_k\\rvert$ that is never applied | comments only |

Agreements worth stating: Algorithm 2 is computed exactly (section 4), Algorithm 3 line 4 and line 7 exactly
(section 6), and equations 14 and 15 exactly (section 5). The anchors $w_0$ = {W['w0_resampler']:g} and
{W['w0_corrector']:g} match p. 5.
"""

header = f"""# CO3's code, line by line

The authors' code for "Steer Away From Mode Collisions" (CO3), read in the order it runs, each block of lines under the
equation it computes. It is level 3 of the CO3 ladder; the [reading page](index.md) puts it after the
[overview](co3-overview.png) and the four figures it links below.

Code: `{N['code']['checkout']}` at commit `{N['code']['commit']}`. Settings: the released ones the project runs,
from `co3_common.py` ({', '.join(f'{k} {v}' for k, v in s.items() if k in ('guidance_scale', 'use_cfgpp', 'n_timesteps', 'num_ts_to_correct', 'num_latent_corrector_steps', 'num_resampling_steps', 'corrector_algo', 'modulate_comp_weights', 'beta', 'lmda'))}).
Written by `gen_codewalk.py`, which reads every block from the files and stops if a line has moved; every number is
from `co3_numbers.json`. Symbols are in [the notation file](NOTATION.md).

**The roles.** Sections 1, 2 and 7 are the ordinary sampler, the same for any SDXL run. Sections 3 to 6 are the
paper's own method, CO3.

"""

(HERE / "code-walk.md").write_text(header + "\n".join(sections) + "\n" + disagree)
print("wrote code-walk.md:", len(sections), "sections")
