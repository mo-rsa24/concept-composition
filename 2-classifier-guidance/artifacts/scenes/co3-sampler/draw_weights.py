"""Level 0.5: equations 14 and 15, the concept weights, drawn as a small network from the three predictions to the
two weights, beside equation 15's curve for two concepts (computed in co3_numbers.json), with the equations
stacked beneath and the code that computes them. Writes co3-weights.png and co3-weights.svg.
"""
from co3_figkit import INK, KEY, MONO, N, Canvas, code_line, fmt, wrap_code

W = N["weights"]
ATTN = "/home/molef/PhD/replications/co3/composers/utils/attn_utils.py"
b_main, b_first = N["settings"]["beta"], N["beta_first_step"]
c = Canvas(17.0, 10.9)

c.text(0.3, 10.55, "Equations 14 and 15 (pp. 5 and 6): the closer a concept's prediction, the harder CO3 pushes away from it",
       size=14.5, weight="bold", ha="left")
c.text(0.3, 10.15, f"$d_k$ is measured between two noise predictions over all {W['latent_numbers']:,} numbers of the latent "
       f"(4 × 128 × 128). The curve is equation 15 for K = 2, computed in co3_numbers.json; real $d_k$ values wait for the Z2 smoke run.",
       size=9.2, ha="left")

# ---- the small network: predictions -> distances -> affinities -> weights --------------------------------
c.box(0.35, 8.25, 1.25, 0.7, "$\\epsilon^{c_1}$", kind="concept1", size=14)
c.box(0.35, 6.65, 1.25, 0.7, "$\\epsilon^{c_0}$", kind="input", size=14)
c.box(0.35, 5.05, 1.25, 0.7, "$\\epsilon^{c_2}$", kind="concept2", size=14)
c.text(0.97, 9.22, f"$c_1$: {N['concepts'][0]}", size=8.6, color=KEY["concept1"])
c.text(0.97, 6.33, "C, the joint prompt", size=8.6)
c.text(0.97, 4.73, f"$c_2$: {N['concepts'][1]}", size=8.6, color=KEY["concept2"])
D, A, Wx, S = (3.55, 3.55), (5.75, 5.75), (8.35, 8.35), (7.05, 4.75)
YU, YL = 8.15, 5.85
c.node(D[0], YU, 0.42, "$d_1$", kind="white", ec=KEY["method"], size=13)
c.node(D[1], YL, 0.42, "$d_2$", kind="white", ec=KEY["method"], size=13)
e = c.line([(1.6, 8.6), (D[0] - 0.42, YU + 0.1)], KEY["concept1"], lw=1.4); c.text(2.45, 8.4, "+", size=12, on_edge=e, color=KEY["concept1"], weight="bold")
e = c.line([(1.6, 7.15), (D[0] - 0.38, YU - 0.18)], KEY["input"], lw=1.4); c.text(2.45, 7.47, "−", size=12, on_edge=e, color=KEY["input"], weight="bold")
e = c.line([(1.6, 6.85), (D[1] - 0.38, YL + 0.18)], KEY["input"], lw=1.4); c.text(2.45, 6.53, "−", size=12, on_edge=e, color=KEY["input"], weight="bold")
e = c.line([(1.6, 5.4), (D[1] - 0.42, YL - 0.1)], KEY["concept2"], lw=1.4); c.text(2.45, 5.6, "+", size=12, on_edge=e, color=KEY["concept2"], weight="bold")
c.text(D[0], YU + 0.62, "length $\\|\\cdot\\|_2$", size=8.4)
c.text(D[1], YL - 0.62, "length $\\|\\cdot\\|_2$", size=8.4)
c.node(A[0], YU, 0.42, "$a_1$", kind="white", ec=KEY["method"], size=13)
c.node(A[1], YL, 0.42, "$a_2$", kind="white", ec=KEY["method"], size=13)
for y in (YU, YL):
    e = c.line([(D[0] + 0.42, y), (A[0] - 0.42, y)], KEY["method"], lw=1.5)
    c.text(4.65, y + 0.27, "$e^{-\\beta\\,d}$", size=10.5, color=KEY["method"])
c.text(4.65, 7.0, f"$\\beta$ = {b_main:g}\n({b_first:g} at t = {N['steps'][0]['t']})", size=8.8, color=KEY["method"])
c.node(S[0], S[1], 0.4, "$\\Sigma_j a_j$", kind="white", ec=KEY["method"], size=9.5)
c.line([(A[0] + 0.3, YU - 0.3), (S[0] - 0.12, S[1] + 0.38)], KEY["fixed"], lw=1.1)
c.line([(A[0] + 0.3, YL - 0.3), (S[0] - 0.3, S[1] + 0.27)], KEY["fixed"], lw=1.1)
c.node(Wx[0], YU, 0.42, "$w_1$", kind="method", size=13)
c.node(Wx[1], YL, 0.42, "$w_2$", kind="method", size=13)
e = c.line([(A[0] + 0.42, YU), (Wx[0] - 0.42, YU)], KEY["method"], lw=1.6); c.text(6.85, YU + 0.27, "$-a_1$", size=10.5, color=KEY["method"])
e = c.line([(A[0] + 0.42, YL), (Wx[1] - 0.42, YL)], KEY["method"], lw=1.6); c.text(6.85, YL + 0.27, "$-a_2$", size=10.5, color=KEY["method"])
e = c.line([(S[0] + 0.25, S[1] + 0.31), (Wx[0] - 0.24, YU - 0.35)], KEY["fixed"], lw=1.1)
c.text(7.7, 6.43, "÷", size=12, on_edge=e, color=KEY["fixed"], weight="bold")
e = c.line([(S[0] + 0.38, S[1] + 0.13), (Wx[1] - 0.3, YL - 0.3)], KEY["fixed"], lw=1.1)
c.text(7.76, 5.2, "÷", size=12, on_edge=e, color=KEY["fixed"], weight="bold")
c.text(Wx[0], YU + 0.68, f"$w_1 + w_2 = -1$", size=9.2, color=KEY["method"])
c.text(Wx[0], 9.4, f"anchor $w_0$ = {W['w0_resampler']:g} (resampler), {W['w0_corrector']:g} (corrector)", size=8.6, color=KEY["method"])

# ---- equation 15's curve for K = 2 --------------------------------------------------------------------------
def X(g):
    return 10.15 + (g + 6) / 12 * 6.2
def Y(w):
    return 4.98 + (w + 1) * 3.7
c.line([(X(-6), Y(-1)), (X(6) + 0.1, Y(-1))], INK, lw=1.0, arrow=False)
c.line([(X(-6), Y(-1)), (X(-6), Y(0) + 0.1)], INK, lw=1.0, arrow=False)
for g in (-6, -3, 0, 3, 6):
    c.ax.plot([X(g)] * 2, [Y(-1), Y(-1) - 0.07], color=INK, lw=1.0)
    c.text(X(g), Y(-1) - 0.25, f"{g:g}", size=8.6)
for w in (-1.0, -0.5, 0.0):
    c.ax.plot([X(-6) - 0.07, X(-6)], [Y(w)] * 2, color=INK, lw=1.0)
    c.text(X(-6) - 0.12, Y(w), f"{w:g}", size=8.6, ha="right")
c.ax.plot([X(-6), X(6)], [Y(-0.5)] * 2, color=KEY["fixed"], lw=0.8, ls=":", zorder=1)
c.text(X(0), Y(-1) - 0.62, "$d_1 - d_2$: how much farther $\\epsilon^{c_0}$ sits from $c_1$ than from $c_2$", size=9.2)
c.text(X(-6) - 0.62, Y(-0.5), "$w_1$", size=12, rotation=90)
gaps, curve = W["curve"]["gap"], W["curve"]
c.curve([X(g) for g in gaps], [Y(v) for v in curve[str(b_main)]], KEY["method"], lw=2.2)
c.curve([X(g) for g in gaps], [Y(v) for v in curve[str(b_first)]], KEY["fixed"], lw=1.4, ls="--")
c.text(X(-3.0), Y(-0.86), f"$\\beta$ = {b_main:g}", size=9.2, color=KEY["method"], weight="bold", ha="right")
c.text(X(-1.0), Y(-0.9), f"$\\beta$ = {b_first:g} (dashed)", size=8.8, color=KEY["fixed"], ha="left")
pts = W["w1_at_gap"][str(b_main)]
for g, dy in (("0.0", 0.33), ("1.0", -0.3), ("2.0", -0.3), ("5.0", -0.3)):
    v = pts[g]
    c.ax.plot([X(float(g))], [Y(v)], "o", ms=6, color=KEY["method"], zorder=5)
    c.text(X(float(g)) + 0.06, Y(v) + dy, fmt(v, 3), size=8.8, ha="left" if g != "0.0" else "right", color=KEY["method"])
c.text(X(-4.4), Y(-0.36), "$c_1$ closer: $w_1 \\to -1$,\nCO3 pushes away\nfrom $c_1$ alone", size=8.6, ha="left")
c.text(X(5.9), Y(-0.6), "$c_1$ farther:\n$w_1 \\to 0$", size=8.6, ha="right")

# ---- equations beneath, and the code -----------------------------------------------------------------------
c.box(0.3, 0.25, 9.0, 3.75, "", kind="white", ec=INK, lw=1.0, container=True)
eqs = [("p. 5", "$d_k = \\|\\,\\epsilon^{C} - \\epsilon^{c_k}\\,\\|$"),
       ("eq. 14", "$a_k = \\exp(-\\beta\\, d_k),\\quad \\beta > 0$"),
       ("eq. 15", "$w_k = -\\dfrac{a_k}{\\sum_{j=1}^{K} a_j}\\quad\\Rightarrow\\quad \\sum_{k=1}^{K} w_k = -1$"),
       ("K = 2", "$w_1 = -\\dfrac{1}{1 + e^{\\,\\beta\\,(d_1 - d_2)}},\\qquad w_2 = -1 - w_1$")]
for k, (where, tex) in enumerate(eqs):
    y = 3.45 - k * 0.86
    c.ax.text(0.5, y, where, ha="left", va="center", fontsize=9, color=KEY["fixed"], zorder=4)
    c.ax.text(1.55, y, tex, ha="left", va="center", fontsize=14, color=INK, zorder=4)
c.ax.text(0.5, 0.5, "Only the gap between the distances matters: adding the same amount to every $d_k$ leaves every $w_k$ unchanged.",
          ha="left", va="center", fontsize=8.8, style="italic", color=INK, zorder=4)

c.box(9.55, 0.25, 7.15, 3.75, "", kind="white", ec=KEY["method"], lw=1.0, container=True)
c.ax.text(9.75, 3.7, "The code (attn_utils.py)", ha="left", va="center", fontsize=9.8, weight="bold", color=KEY["method"], zorder=4)
items = [(191, "concept_weights = self._get_concept_weights_from_dists(concept_dists, method='exp', beta=beta)"),
         (187, "dists.append((noise_pred_concepts[i].flatten() - noise_pred_multi.flatten()).norm().item())"),
         (145, "a = torch.exp(-beta * dists)"), (151, "if torch.all(a == 0):"), (152, "a = torch.ones_like(a)"),
         (155, "w = -a / a.sum()")]
body = "\n".join(wrap_code(n, code_line(ATTN, n, must), 60) for n, must in items)
c.ax.text(9.75, 3.42, body, ha="left", va="top", fontsize=7.8, family=MONO, color=INK, zorder=4, linespacing=1.3)
c.ax.text(9.75, 0.42, "Lines 151 and 152 fall back to equal weights if every $a_k$\nunderflows to 0. The docstring (line 137) names a clip range on\n"
          "$|w_k|$ of 0.01 to 0.9 that the function never applies.", ha="left", va="bottom", fontsize=8.6,
          style="italic", color=KEY["broken"], zorder=4, linespacing=1.3)

c.save("co3-weights")
