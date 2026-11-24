"""Level 0.5: Algorithm 3, the CO3 corrector (weights sum to 1), drawn as a small network with the numbers on the
edges, then three boxed rules beneath: the algorithm as printed, what the code computes, and why the weights
sum to 1. Values at the first corrector step, t = 921, from co3_numbers.json. Writes co3-corrector.png and
co3-corrector.svg.
"""
import numpy as np

from co3_figkit import INK, KEY, N, Canvas, fmt

st = next(r for r in N["steps"] if r["algorithm"] == "corrector")
lam, root = N["settings"]["lmda"], st["sqrt_one_minus_alpha_bar"]
w0, wu = N["weights"]["w0_corrector"], N["weights"]["uniform_concept_weight"]
c1, c2 = N["concepts"]
c = Canvas(18.4, 11.6)

c.text(0.3, 11.25, "CO3 corrector, Algorithm 3 (p. 6): the weights sum to 1, so the result is still a Tweedie mean",
       size=15, weight="bold", ha="left")
c.text(0.3, 10.88, f"One pass at t = {st['t']}, the first corrector step. Edge numbers are this step's values from "
       f"co3_numbers.json. Magenta marks where the code departs from the algorithm.", size=9.4, ha="left")

# input, bus, four U-Net runs (same layout as the resampler figure)
c.box(0.35, 6.55, 1.1, 0.9, "$x_t$", kind="input", size=17, name="x_t")
c.text(0.9, 7.72, "4 × 128 × 128", size=8.4)
c.line([(1.45, 7.0), (2.2, 7.0)], KEY["input"], arrow=False)
c.line([(2.2, 5.0), (2.2, 9.0)], KEY["input"], arrow=False)
UX, R = 4.6, 0.45
rows = [(9.0, "$\\epsilon^{\\phi}$", "fixed", "empty prompt"),
        (7.75, "$\\epsilon^{c_0}$", "input", "C: an eagle and a fox"),
        (6.5, "$\\epsilon^{c_1}$", "concept1", f"$c_1$: {c1}"),
        (5.0, "$\\epsilon^{c_2}$", "concept2", f"$c_2$: {c2}")]
for y, sym, kind, cond in rows:
    c.line([(2.2, y), (UX - R, y)], KEY["input"])
    c.text(3.08, y + 0.25, cond, size=8.4, color=KEY[kind] if kind != "input" else INK)
    c.node(UX, y, R, sym, kind=kind, size=13, name=sym)
c.text(UX, 9.68, "U-Net $\\psi$, one batch", size=9, color=KEY["denoiser"], weight="bold")

# classifier-free guidance per condition: lambda on the condition, 1 - lambda on the empty prompt (eq. 4, p. 3)
GX, GR = 7.0, 0.45
lab_lam = {7.75: 6.2, 6.5: 5.5, 5.0: 5.8}
lab_phi = {7.75: 6.3, 6.5: 6.45, 5.0: 6.62}
for (y, sym, kind, _), sup in zip(rows[1:], ("c_0", "c_1", "c_2")):
    c.node(GX, y, GR, f"$\\epsilon^{{\\lambda,{sup}}}$", kind="white", ec=KEY[kind], size=11.5, name=f"cfg {sup}")
    e = c.line([(UX + R, y), (GX - GR, y)], KEY[kind], lw=1.5)
    c.text(lab_lam[y], y, f"{lam:g}", size=9, on_edge=e, weight="bold", color=KEY[kind])
    d = np.array([GX, y]) - np.array([UX, 9.0])
    u = d / np.linalg.norm(d)
    a, b = np.array([UX, 9.0]) + R * u, np.array([GX, y]) - GR * u
    e2 = c.line([tuple(a), tuple(b)], KEY["fixed"], lw=1.2)
    t = (lab_phi[y] - a[0]) / (b[0] - a[0])
    c.text(lab_phi[y], a[1] + t * (b[1] - a[1]), f"{1 - lam:.1f}", size=9, on_edge=e2, color=KEY["fixed"])

# the Tweedie mean of each guided prediction
TX, TW = 8.2, 2.0
c.text(9.3, 8.62, f"each box: $x_t - \\sqrt{{1-\\bar\\alpha_t}}\\;\\epsilon^{{\\lambda,c_k}}$, "
       f"$\\sqrt{{1-\\bar\\alpha_t}}$ = {fmt(root)}", size=9.2)
for (y, sym, kind, _), sup in zip(rows[1:], ("c_0", "c_1", "c_2")):
    c.box(TX, y - 0.33, TW, 0.66, f"$\\hat{{x}}_{{\\mathrm{{tweedie}}}}[\\epsilon^{{\\lambda,{sup}}}]$", kind="white",
          ec=KEY[kind], size=10.5)
    c.line([(GX + GR, y), (TX, y)], KEY[kind], lw=1.5)

# the weighted sum, weights sum to 1
SX, SY = 11.55, 6.5
c.node(SX, SY, 0.42, "$\\Sigma$", kind="method", size=15, name="sum")
c.line([(TX + TW, 7.75), (SX - 0.3, SY + 0.3)], KEY["input"], lw=1.6)
c.line([(TX + TW, 6.5), (SX - 0.42, SY)], KEY["concept1"], lw=1.6)
c.line([(TX + TW, 5.0), (SX - 0.3, SY - 0.3)], KEY["concept2"], lw=1.6)
c.text(10.75, 7.6, f"$w_0$ = {w0:g}", size=10, color=KEY["method"], weight="bold")
c.text(10.6, 6.68, "$w_1$", size=10.5, color=KEY["method"], weight="bold")
c.text(10.65, 5.2, "$w_2$", size=10.5, color=KEY["method"], weight="bold")
c.text(13.0, 5.35, f"$w_0 + w_1 + w_2 = 1$\n({wu:g} each when $d_1 = d_2$)", size=8.8, color=KEY["method"])

# mean normalisation (paper and code differ), add the empty prompt's prediction, the code-only rescale
c.box(12.5, 6.12, 1.2, 0.76, "$\\times\\, r$", kind="broken", size=13)
c.line([(SX + 0.42, SY), (12.5, SY)], KEY["method"], lw=1.6)
c.text(13.1, 7.2, "mean normalisation:\npaper and code differ", size=8.4, color=KEY["broken"])
PX = 14.35
c.node(PX, SY, 0.32, "+", kind="white", ec=INK, size=16, name="plus")
c.line([(13.7, SY), (PX - 0.32, SY)], KEY["method"], lw=1.6)
ephi = c.line([(UX + R, 9.0), (PX, 9.0), (PX, SY + 0.32)], KEY["fixed"], lw=1.6)
c.text(12.2, 9.0, f"$\\times\\,\\sqrt{{1-\\bar\\alpha_t}}$ = {fmt(root)}", size=9.6, on_edge=ephi, color=KEY["fixed"])
c.box(14.95, 6.02, 1.55, 0.96, "code only:\n$\\times\\, \\|x_t\\|_2 / \\|x^{(2)}_t\\|_2$", kind="broken", ls="--", size=8.6)
c.line([(PX + 0.32, SY), (14.95, SY)], INK, lw=1.4)
c.box(16.8, 6.05, 1.35, 0.9, "$x_t^{(2)}$", kind="output", size=16)
c.line([(16.5, SY), (16.8, SY)], INK, lw=1.4)

loop = c.line([(17.47, 6.05), (17.47, 4.02), (0.9, 4.02), (0.9, 6.55)], KEY["fixed"], lw=1.5)
c.text(8.6, 4.24, "$x_t \\leftarrow x_t^{(2)}$, repeat P = 5 times at t = 921, 901 and 881  (Algorithm 3 lines 1, 8)",
       size=9.2, color=KEY["fixed"])

# three boxed rules beneath
def rule_box(x, w, title, colour, lines, note=""):
    c.box(x, 0.25, w, 3.6, "", kind="white", ec=colour, lw=1.3, container=True)
    c.ax.text(x + 0.18, 3.5, title, ha="left", va="center", fontsize=9.8, weight="bold", color=colour, zorder=4)
    for k, (where, tex) in enumerate(lines):
        y = 2.88 - k * 0.6
        c.ax.text(x + 0.18, y, where, ha="left", va="center", fontsize=8.4, color=KEY["fixed"], zorder=4)
        c.ax.text(x + 1.15, y, tex, ha="left", va="center", fontsize=11.5, color=colour if where.startswith("257") else INK, zorder=4)
    if note:
        c.ax.text(x + 0.18, 0.42, note, ha="left", va="bottom", fontsize=8.6, style="italic", color=INK, zorder=4, linespacing=1.25)

TWD = "\\mathrm{tweedie}"
rule_box(0.3, 5.75, "Algorithm 3 as printed (p. 6)", INK, [
    ("line 4", f"$\\tilde{{x}}_{{{TWD}}} = \\sum_{{k=0}}^{{K}} w_k\\,\\hat{{x}}_{{{TWD}}}[\\epsilon^{{\\lambda,c_k}}]$"),
    ("line 5", f"$r = \\|\\hat{{x}}^{{c_0}}_{{{TWD}}}\\|_2 \\,/\\, \\|\\tilde{{x}}_{{{TWD}}}\\|_2$"),
    ("line 6", f"$\\tilde{{x}}_{{{TWD}}} \\leftarrow \\tilde{{x}}_{{{TWD}}}\\cdot r$"),
    ("line 7", f"$x_t^{{(2)}} = \\tilde{{x}}_{{{TWD}}} + \\sqrt{{1-\\bar\\alpha_t}}\\,\\epsilon^{{\\phi}}_t$")],
    "r matches the length of the composed mean\nto the joint prompt's Tweedie mean.")
rule_box(6.25, 6.15, "What attn_utils.py computes (lines 251 to 257)", KEY["broken"], [
    ("251, 256", "same as lines 4 and 7"),
    ("252", f"$r = \\max\\,\\hat{{x}}_{{{TWD}}}[\\epsilon^{{\\lambda,c_0}}]$"),
    ("255", f"$\\tilde{{x}}_{{{TWD}}} \\leftarrow \\tilde{{x}}_{{{TWD}}} \\,/\\, \\max(\\tilde{{x}}_{{{TWD}}}) \\cdot r$"),
    ("257", "$x_t^{(2)} \\leftarrow x_t^{(2)}\\cdot \\|x_t\\|_2 \\,/\\, \\|x_t^{(2)}\\|_2$")],
    "max is the largest single entry of the latent, not a\nlength; line 257 has no line in Algorithm 3.")
rule_box(12.6, 5.5, "Why the weights sum to 1 (Prop. 1a, p. 5)", KEY["method"], [
    ("p. 5", f"$w_0 = {w0:g},\\quad w_1 + w_2 = -1$"),
    ("p. 5", f"$\\tilde{{x}}_{{{TWD}}} = x_t - \\sqrt{{1-\\bar\\alpha_t}}\\;\\tilde\\epsilon^{{\\lambda,C}}$"),
    ("p. 5", "$\\tilde\\epsilon^{\\lambda,C} = \\epsilon^{\\phi} + \\lambda\\,(\\sum_k w_k\\epsilon^{c_k} - \\epsilon^{\\phi})$")],
    "With a sum of 1, $x_t$ survives with weight 1 and the\ncomposition is one guided prediction, so the\ncorrector edits $x_t$ instead of replacing it.")

c.save("co3-corrector")
