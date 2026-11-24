"""Level 0.5: Algorithm 2, the CO3 resampler (weights sum to 0), drawn as a small network with every symbol on
the part it belongs to, the numbers on the edges and the equations stacked beneath. Values at the first CO3
step, t = 981, from co3_numbers.json. Writes co3-resampler.png and co3-resampler.svg.
"""
from co3_figkit import FILL, INK, KEY, N, Canvas, fmt

st = N["steps"][0]
lam, root, K = N["settings"]["lmda"], st["sqrt_one_minus_alpha_bar"], N["weights"]["K"]
w0, wu = N["weights"]["w0_resampler"], N["weights"]["uniform_concept_weight"]
c1, c2 = N["concepts"]
c = Canvas(16.6, 10.6)

c.text(0.3, 10.32, "CO3 resampler, Algorithm 2 (p. 6): the weights sum to 0, so $x_t$ drops out", size=15, weight="bold", ha="left")
c.text(0.3, 9.95, f"One pass at t = {st['t']}, the first CO3 step. Edge numbers are this step's values from co3_numbers.json; "
       f"$w_1$ and $w_2$ come from equation 15 (see the weights figure).", size=9.4, ha="left")

# input and the bus to the four U-Net runs
c.box(0.35, 6.55, 1.1, 0.9, "$x_t$", kind="input", size=17, name="x_t")
c.text(0.9, 7.72, "4 × 128 × 128", size=8.4)
c.line([(1.45, 7.0), (2.2, 7.0)], KEY["input"], arrow=False)
c.line([(2.2, 5.0), (2.2, 9.0)], KEY["input"], arrow=False)
rows = [(9.0, "$\\epsilon^{\\phi}$", "fixed", "empty prompt"),
        (7.75, "$\\epsilon^{c_0}$", "input", "C: an eagle and a fox"),
        (6.5, "$\\epsilon^{c_1}$", "concept1", f"$c_1$: {c1}"),
        (5.0, "$\\epsilon^{c_2}$", "concept2", f"$c_2$: {c2}")]
UX, R = 4.6, 0.45
for y, sym, kind, cond in rows:
    c.line([(2.2, y), (UX - R, y)], KEY["input"])
    c.text(3.08, y + 0.25, cond, size=8.4, color=KEY[kind] if kind != "input" else INK)
    c.node(UX, y, R, sym, kind=kind, size=13, name=sym)
c.text(UX, 9.6, "U-Net $\\psi$, one batch", size=9, color=KEY["denoiser"], weight="bold")

# weighted sum of the three conditioned predictions
SX, SY = 7.55, 6.5
c.node(SX, SY, 0.42, "$\\Sigma$", kind="method", size=15, name="sum")
e0 = c.line([(UX + R, 7.75), (SX - 0.3, SY + 0.3)], KEY["input"], lw=1.6)
e1 = c.line([(UX + R, 6.5), (SX - 0.42, SY)], KEY["concept1"], lw=1.6)
e2 = c.line([(UX + R, 5.0), (SX - 0.3, SY - 0.3)], KEY["concept2"], lw=1.6)
c.text(5.75, 7.72, f"$w_0$ = {w0:g}", size=10, color=KEY["method"], weight="bold")
c.text(5.78, 6.68, f"$w_1$", size=10.5, color=KEY["method"], weight="bold")
c.text(5.75, 5.12, f"$w_2$", size=10.5, color=KEY["method"], weight="bold")
c.text(SX, 5.5, f"$w_1 + w_2 = -1$\n({wu:g} each when $d_1 = d_2$)", size=8.8, color=KEY["method"])

# scale onto the noise scale
c.box(8.75, 6.1, 1.9, 0.8, "$\\times\\,(-\\lambda\\sqrt{1-\\bar\\alpha_t})$", kind="white", ec=KEY["method"], size=11)
c.line([(SX + 0.42, SY), (8.75, SY)], KEY["method"], lw=1.6)
c.text(9.7, 7.12, f"$\\lambda$ = {lam:g},  $\\sqrt{{1-\\bar\\alpha_t}}$ = {fmt(root)}", size=8.8)
c.text(9.7, 5.78, f"= {fmt(-lam * root)}", size=9.5, weight="bold", color=KEY["method"])
c.box(11.35, 6.1, 1.25, 0.8, "$\\tilde{x}_{\\mathrm{tweedie}}$", kind="method", size=13)
c.line([(10.65, SY), (11.35, SY)], KEY["method"], lw=1.6)

# add the unconditional prediction back at the noise scale
PX = 13.55
c.node(PX, SY, 0.33, "+", kind="white", ec=INK, size=16, name="plus")
c.line([(12.6, SY), (PX - 0.33, SY)], KEY["method"], lw=1.6)
ephi = c.line([(UX + R, 9.0), (PX, 9.0), (PX, SY + 0.33)], KEY["fixed"], lw=1.6)
c.text(9.1, 9.0, f"$\\times\\,\\sqrt{{1-\\bar\\alpha_t}}$ = {fmt(root)}", size=9.6, on_edge=ephi, color=KEY["fixed"])
c.box(14.55, 6.05, 1.55, 0.9, "$x_t^{(2)}$", kind="output", size=16)
c.line([(PX + 0.33, SY), (14.55, SY)], INK, lw=1.6)

# the loop lane, below everything
loop = c.line([(15.32, 6.05), (15.32, 4.15), (0.9, 4.15), (0.9, 6.55)], KEY["fixed"], lw=1.5)
c.text(8.1, 4.38, "$x_t \\leftarrow x_t^{(2)}$, repeat P times: 10 at t = 981 and 961, 5 at t = 941  (Algorithm 2 lines 1, 7)",
       size=9.2, color=KEY["fixed"])
c.text(11.0, 8.15, "$x_t$ enters only through the four predictions:\nthe weights sum to 0, so it cancels\n(Proposition 1b, p. 5)",
       size=8.8, style="italic")

# the equations, stacked beneath
c.box(0.3, 0.25, 9.9, 3.45, "", kind="white", ec=INK, lw=1.0, container=True)
eqs = [("lines 2 and 3", "$\\epsilon^{c_0}=\\psi(x_t, C),\\quad \\epsilon^{c_k}=\\psi(x_t, c_k),\\quad k = 1,\\dots,K$"),
       ("line 5", "$\\tilde{x}_{\\mathrm{tweedie}} = -\\lambda\\,\\sqrt{1-\\bar\\alpha_t}\\;\\sum_{k=0}^{K} w_k\\,\\epsilon^{c_k}$"),
       ("line 6", "$x_t^{(2)} = \\tilde{x}_{\\mathrm{tweedie}} + \\sqrt{1-\\bar\\alpha_t}\\;\\epsilon^{\\phi}_t$"),
       ("p. 5", "$\\sum_{k=0}^{K} w_k = 0:\\quad w_0 = 1,\\quad w_1 + \\dots + w_K = -1$")]
for k, (where, tex) in enumerate(eqs):
    y = 3.15 - k * 0.8
    c.ax.text(0.5, y, where, ha="left", va="center", fontsize=9, color=KEY["fixed"], zorder=4)
    c.ax.text(2.05, y, tex, ha="left", va="center", fontsize=14, color=INK, zorder=4)
c.box(10.45, 0.25, 5.85, 3.45, "", kind="white", ec=KEY["method"], lw=1.0, container=True)
c.ax.text(10.65, 3.35, "Same result in the code (attn_utils.py 194 to 198)", ha="left", va="center", fontsize=9.6,
          weight="bold", color=KEY["method"], zorder=4)
c.ax.text(10.65, 2.62, "$x_t^{(2)} = \\sqrt{1-\\bar\\alpha_t}\\,\\left(\\epsilon^{\\phi} - \\lambda\\,[\\,\\epsilon^{c_0} + w_1\\epsilon^{c_1} + w_2\\epsilon^{c_2}\\,]\\right)$",
          ha="left", va="center", fontsize=12.5, color=INK, zorder=4)
c.ax.text(10.65, 1.45, "Line 194 adds $x_t$ to the scaled sum and line 196\nsubtracts it again, so the result is Algorithm 2\n"
          "line 5 exactly. It is noise built from the four\npredictions, which is why the paper calls it valid\nonly at t = T (p. 5).",
          ha="left", va="center", fontsize=9.2, color=INK, zorder=4, linespacing=1.3)

c.save("co3-resampler")
