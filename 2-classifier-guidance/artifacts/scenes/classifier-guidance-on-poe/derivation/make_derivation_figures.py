"""Classifier guidance on the product of experts, derived from its assumptions, one step per figure.

d1  why sqrt(1 - abar_t): a noise prediction is a rescaled score
d2  Bayes on the noisy canvas: why the guidance term is grad log p(y | x_t), and why p(y) drops out
d3  from score to noise prediction, and the dial s
d4  the product of experts is the same algebra with two implicit classifiers, and what it assumes
d5  the two checkers as two more terms of the same shape
d6  the whole sampler equation, every term coloured, with step 7b's weights and measured sizes

Left of each figure, a small one-dimensional picture (a toy, labelled as one) or real numbers from step 7b;
right, the derivation lines with a plain gloss under each. Schedule numbers are computed from the U-Net's
own schedule (1,000 levels, beta 0.0001 to 0.02, 50 DDIM steps); measured numbers are read from the
step 7b results (arch_numbers.json, grad_numbers.json). Usage: python make_derivation_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "results/how-good-must-the-checker-be-on-mnist"
ARCH = json.load(open(RES / "architecture/arch_numbers.json"))
GRAD = json.load(open(RES / "gradient-flow/grad_numbers.json"))
BLUE, GREEN, RED, ORANGE, GREY, INK = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88", "#1f2a33"
BETA = np.linspace(1e-4, 0.02, 1000)
ABAR = np.cumprod(1 - BETA)
STEPS = np.round(np.linspace(999, 0, 50)).astype(int)
r2 = lambda k: float(np.sqrt(1 - ABAR[STEPS[k]]))
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})


def sheet(title, sub):
    fig = plt.figure(figsize=(20, 10.5))
    fig.text(0.015, 0.965, title, size=21, weight="bold", color=INK)
    fig.text(0.015, 0.932, sub, size=12.5, color=GREY)
    return fig


def lines(fig, x, y, items, dy=0.098):
    """items: (kind, text, gloss, colour). kind 'eq' equation, 'as' assumption, 'box' boxed result.
    Glosses hang from their top edge, so a two-line gloss grows downward and pushes the next line down."""
    for kind, tex, gloss, col in items:
        nl = gloss.count("\n") if gloss else 0
        if kind == "as":
            fig.text(x, y, tex, size=14, color=BLUE, weight="bold")
            if gloss:
                fig.text(x + 0.012, y - 0.018, gloss, size=12, color=GREY, va="top")
            y -= dy * 0.75 + 0.024 * (nl + (1 if gloss else 0))
            continue
        t = fig.text(x + 0.012, y, tex, size=19 if kind == "box" else 17, color=col or INK)
        drop = 0.022
        if kind == "box":
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(fig.transFigure.inverted())
            fig.patches.append(FancyBboxPatch((bb.x0 - 0.008, bb.y0 - 0.012), bb.width + 0.016, bb.height + 0.024,
                                              boxstyle="round,pad=0,rounding_size=0.006", fc="none", ec=col or INK,
                                              lw=1.6, transform=fig.transFigure, figure=fig))
            drop = bb.y0 - 0.02 - y if bb.y0 - 0.02 < y - 0.022 else -0.022
            drop = -drop if drop < 0 else drop
        if gloss:
            fig.text(x + 0.012, y - abs(drop), gloss, size=12, color=GREY, va="top")
        y -= dy + 0.024 * nl + (0.015 if kind == "box" else 0)
    return y


def clean_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=10, colors=GREY)


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


# -------------------------------------------------------------------------------- d1
def d1():
    fig = sheet("Step 1: why every guidance term carries $\\sqrt{1-\\bar\\alpha_t}$",
                "A noise prediction and a score (the uphill direction of log-probability) are the same arrow, "
                "measured in different units. $\\sqrt{1-\\bar\\alpha_t}$ converts one into the other.")
    k = 25; t = STEPS[k]; ab = ABAR[t]; m, sd = np.sqrt(ab) * 1.0, np.sqrt(1 - ab); eps = 1.2; xt = m + sd * eps
    ax = fig.add_axes([0.05, 0.52, 0.38, 0.33]); clean_ax(ax)
    xs = np.linspace(-3, 4, 400)
    ax.plot(xs, gauss(xs, m, sd), color=GREEN, lw=2.2)
    ax.axvline(m, color=GREY, ls=":", lw=1)
    ax.plot([xt], [gauss(xt, m, sd)], "o", color=BLUE, ms=9)
    ax.annotate("", xy=(m + 0.05, gauss(xt, m, sd)), xytext=(xt - 0.05, gauss(xt, m, sd)),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2, mutation_scale=16))
    ax.text(m, 0.445, r"mean $\sqrt{\bar\alpha_t}\,x_0=%.3f$" % m, color=GREY, size=11, ha="center")
    ax.text(xt + 0.1, gauss(xt, m, sd) + 0.02, r"$x_t$ = %.2f" % xt, color=BLUE, size=12)
    ax.text((m + xt) / 2, gauss(xt, m, sd) - 0.06, "score = %.2f" % (-eps / sd), color=GREEN, size=12, ha="center")
    ax.set_ylim(0, 0.48); ax.set_xlabel("one pixel's value", size=11, color=GREY)
    ax.set_title(f"toy, one pixel: clean value $x_0$ = 1, noised to t = {t} with $\\varepsilon$ = {eps}", size=12,
                 color=INK, loc="left")
    ax2 = fig.add_axes([0.05, 0.08, 0.38, 0.33]); clean_ax(ax2)
    ax2.plot(range(50), np.sqrt(1 - ABAR[STEPS]), color=ORANGE, lw=2.2)
    for kk in (10, 25, 40):
        ax2.plot([kk], [r2(kk)], "o", color=ORANGE, ms=7)
        ax2.text(kk, r2(kk) - 0.09, f"step {kk}\n{r2(kk):.3f}", color=INK, size=11, ha="center", va="top")
    ax2.set_ylim(0, 1.1); ax2.set_xlabel("sampler step (0 = pure noise, 49 = clean)", size=11, color=GREY)
    ax2.set_title(r"real schedule: $\sqrt{1-\bar\alpha_t}$ at each of the 50 steps", size=12, color=INK, loc="left")
    lines(fig, 0.47, 0.86, [
        ("eq", r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,I)$",
         "how a canvas is noised: shrink the clean canvas, add noise of size sqrt(1 - abar_t)", None),
        ("eq", r"$q(x_t\mid x_0)=\mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\,I\right)$",
         "so given the clean canvas, the noisy one is a Gaussian (the bump, top left)", None),
        ("eq", r"$\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}=-\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}}$",
         "its uphill direction points back to the mean; put line 1 into the top to get the right side", None),
        ("eq", r"$\nabla_{x_t}\log p_t(x_t)=\mathbb{E}\,[\,\nabla_{x_t}\log q(x_t\mid x_0)\mid x_t\,]=-\frac{\mathbb{E}[\varepsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}$",
         "the sampler only sees x_t, so average over every clean canvas that could have made it", None),
        ("eq", r"$\hat\varepsilon_\theta(x_t)\approx\mathbb{E}[\varepsilon\mid x_t]$",
         "training on squared error of the noise makes the U-Net predict exactly that average", None),
        ("box", r"$\hat\varepsilon_\theta(x_t)=-\sqrt{1-\bar\alpha_t}\ \nabla_{x_t}\log p_t(x_t)$",
         "a noise prediction is a score times -sqrt(1 - abar_t): the exchange rate between the two units.\n"
         "Every guidance term below is a score, so it is multiplied by this rate to be added to a noise prediction.", GREEN),
    ])
    fig.text(0.48, 0.07, f"In numbers: t = {STEPS[10]} (step 10): rate {r2(10):.3f};   t = {t} (step 25): rate {r2(25):.3f};   "
             f"t = {STEPS[40]} (step 40): rate {r2(40):.3f}.\nLate steps shrink every push by this factor on their own.",
             size=12, color=INK)
    fig.savefig(HERE / "d1-why-the-square-root.png", dpi=80); plt.close(fig)


# -------------------------------------------------------------------------------- toy for d2, d3
XS = np.linspace(-4, 4, 800)
PX = 0.75 * gauss(XS, -1.2, 0.7) + 0.25 * gauss(XS, 1.6, 0.6)
PY = 1 / (1 + np.exp(-2.5 * XS))
norm = lambda p: p / np.trapezoid(p, XS)
grad = lambda p: np.gradient(np.log(np.maximum(p, 1e-300)), XS)


def toy_axes(ax):
    clean_ax(ax)
    ax.set_xticks([-1.2, 1.6]); ax.set_xticklabels(["fused-looking", "two apart"], size=11, color=INK)


def d2():
    fig = sheet("Step 2: what guidance aims at, and why the term is $\\nabla_{x_t}\\log p(y\\mid x_t)$",
                "Bayes' rule on the noisy canvas. The label's overall frequency p(y) never has to be known, "
                "because it does not change when the canvas changes.")
    ax = fig.add_axes([0.05, 0.52, 0.38, 0.34]); toy_axes(ax)
    post = norm(PX * PY)
    ax.plot(XS, norm(PX), color=GREEN, lw=2.2); ax.text(-2.75, 0.4, "the model\n$p(x_t)$", color=GREEN, size=12)
    ax.plot(XS, PY * 0.55, color=ORANGE, lw=2.2, ls="--"); ax.text(3.1, 0.62, "the checker\n$p(y\\mid x_t)$, scaled",
                                                                  color=ORANGE, size=12, ha="center")
    ax.plot(XS, post, color=RED, lw=2.4); ax.text(2.45, 0.36, "$p(x_t\\mid y)$", color=RED, size=13, ha="left")
    ax.set_title("toy: canvases on one line; y = 'two digits apart'", size=12, color=INK, loc="left")
    ax.set_ylim(0, 0.75)
    ax2 = fig.add_axes([0.05, 0.08, 0.38, 0.32]); toy_axes(ax2)
    sel = slice(None, None, 1)
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.plot(XS, grad(PX), color=GREEN, lw=2); ax2.plot(XS, grad(PY), color=ORANGE, lw=2, ls="--")
    ax2.plot(XS, grad(PX * PY), color=RED, lw=2.2)
    ax2.set_ylim(-6, 6); ax2.set_ylabel("uphill direction (score)", size=11, color=GREY)
    ax2.text(-3.9, -5.0, "red = green + orange at every point: line 3", color=INK, size=12)
    ax2.text(1.2, 4.6, "positive: move right", color=GREY, size=10)
    lines(fig, 0.47, 0.86, [
        ("as", "Assumption 1: we want the model's pictures, kept only where the label y holds", None, None),
        ("eq", r"$p_t(x_t\mid y)=\frac{p_t(x_t)\;p_t(y\mid x_t)}{p(y)}$", "Bayes' rule, on the noisy canvas at step t", None),
        ("eq", r"$\log p_t(x_t\mid y)=\log p_t(x_t)+\log p_t(y\mid x_t)-\log p(y)$",
         "take logs: the product becomes a sum", None),
        ("box", r"$\nabla_{x_t}\log p_t(x_t\mid y)=\nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(y\mid x_t)$",
         "take the gradient with respect to the canvas: log p(y) is one fixed number, so its gradient is 0", RED),
        ("as", "Assumption 2: a checker trained on noisy canvases stands in for the true noisy-label probability",
         None, None),
        ("eq", r"$p_\phi(y\mid x_t,t)\ \approx\ p_t(y\mid x_t)=\int p(y\mid x_0)\;p(x_0\mid x_t)\;dx_0$",
         "the chance y will hold for the clean canvas this noisy one becomes. This is why the checkers\n"
         "read x_t and the step t: a checker trained on clean canvases only would answer the wrong question.", None),
    ])
    fig.savefig(HERE / "d2-bayes-on-the-noisy-canvas.png", dpi=80); plt.close(fig)


def d3():
    fig = sheet("Step 3: from scores to noise predictions, and the dial s",
                "Multiply step 2 by the exchange rate from step 1. The checker's term becomes something the "
                "sampler can add to a noise prediction: the push.")
    ax = fig.add_axes([0.05, 0.12, 0.38, 0.72]); toy_axes(ax)
    for s, a in ((0, 0.35), (1, 0.6), (3, 0.8), (10, 1.0)):
        p = norm(PX * PY ** s)
        ax.plot(XS, p, color=RED if s else GREEN, lw=2.2, alpha=a)
        i = int(np.argmax(p))
        ax.text(XS[i] + (0.25 if s else -0.1), p[i] + 0.02, f"s = {s}" + (" (no checker)" if s == 0 else ""),
                color=RED if s else GREEN, size=12, ha="left" if s else "center")
    ax.set_title(r"toy: $p(x_t)\,p(y\mid x_t)^{s}$, normalised, for s = 0, 1, 3, 10", size=12, color=INK, loc="left")
    lines(fig, 0.47, 0.86, [
        ("eq", r"$-\sqrt{1-\bar\alpha_t}\ \times\ [\ \nabla\log p_t(x_t\mid y)=\nabla\log p_t(x_t)+\nabla\log p_\phi(y\mid x_t)\ ]$",
         "step 2's last line, times the exchange rate from step 1", None),
        ("eq", r"$\tilde\varepsilon(x_t)=\hat\varepsilon_\theta(x_t)-\sqrt{1-\bar\alpha_t}\ \nabla_{x_t}\log p_\phi(y\mid x_t)$",
         "the conditional noise prediction = the plain one + a term from the checker (the push)", None),
        ("as", "Assumption 3, a choice: sharpen the checker by raising it to a power s", None, None),
        ("eq", r"$p(x_t)\,p_\phi(y\mid x_t)^{\,s}\quad\Rightarrow\quad\nabla\log p(x_t)+s\,\nabla\log p_\phi(y\mid x_t)$",
         "a power inside the log becomes a factor outside it", None),
        ("box", r"$\tilde\varepsilon(x_t)=\hat\varepsilon_\theta(x_t)-s\ \sqrt{1-\bar\alpha_t}\ \nabla_{x_t}\log p_\phi(y\mid x_t)$",
         "s = 1 is Bayes exactly; s > 1 keeps fewer, surer pictures (left). Past s = 1 the target is no longer an exact\n"
         "distribution, because the noised version of p(y|x)^s is not p_t(y|x_t)^s; Dhariwal and Nichol (2021) use s as a dial.", ORANGE),
    ])
    fig.text(0.48, 0.17, "Reading the three factors of the push:", size=14, color=INK, weight="bold")
    fig.text(0.48, 0.125, r"$s$: how hard to push (10 in step 7b)      $\sqrt{1-\bar\alpha_t}$: score-to-noise exchange rate (step 1)",
             size=14, color=INK)
    fig.text(0.48, 0.08, r"$\nabla_{x_t}\log p_\phi(y\mid x_t)$: which way to change each pixel so the checker grows surer that y holds",
             size=14, color=INK)
    fig.text(0.48, 0.035, "The minus sign: the sampler subtracts predicted noise, so lowering the noise prediction moves the "
             "canvas uphill.", size=12, color=GREY)
    fig.savefig(HERE / "d3-scores-to-noise-and-the-dial.png", dpi=80); plt.close(fig)


# -------------------------------------------------------------------------------- d4
def d4():
    fig = sheet("Step 4: the product of experts is the same algebra, with two classifiers hidden inside the U-Net",
                "Asking for digit a and digit b at once, with one more assumption, gives the product. Each of its "
                "terms is a classifier-guidance term with s = 1.")
    ax = fig.add_axes([0.05, 0.40, 0.38, 0.45]); clean_ax(ax)
    xs = np.linspace(-5, 5, 800)
    pa, pb, p0 = gauss(xs, -1.3, 0.8), gauss(xs, 1.3, 0.8), gauss(xs, 0, 2.2)
    prod = pa * pb / p0; prod /= np.trapezoid(prod, xs)
    ax.plot(xs, p0, color=GREY, lw=2, ls=":"); ax.text(-4.2, 0.22, "$p(x)$\nany digit", color=GREY, size=12)
    ax.plot(xs, pa, color=GREEN, lw=2.2); ax.text(-1.3, 0.53, "$p(x\\mid 4)$", color=GREEN, size=13, ha="center")
    ax.plot(xs, pb, color=GREEN, lw=2.2); ax.text(1.3, 0.53, "$p(x\\mid 7)$", color=GREEN, size=13, ha="center")
    ax.plot(xs, prod, color=RED, lw=2.6); ax.text(0, prod.max() + 0.03, "the product", color=RED, size=13, ha="center")
    ax.set_xticks([-1.3, 0, 1.3]); ax.set_xticklabels(["4-shaped", "a bit of both", "7-shaped"], size=11, color=INK)
    ax.set_ylim(0, 0.85)
    ax.set_title("toy: shapes on one line; the product keeps only what both experts like", size=12, color=INK, loc="left")
    fig.text(0.05, 0.30, "On this line the only shape both experts like is the overlap: a fused glyph.\n"
             "On a real canvas, a 4 beside a 7 is also liked by both, but nothing in the\n"
             "product prefers it to the fused glyph. That is the gap the checkers fill.", size=12.5, color=INK)
    lines(fig, 0.47, 0.86, [
        ("as", "Assumption 4: the two labels are independent once the canvas is known", None, None),
        ("eq", r"$p(a,b\mid x)=p(a\mid x)\,p(b\mid x)$", None, None),
        ("eq", r"$p(x\mid a,b)\ \propto\ p(x)\,p(a\mid x)\,p(b\mid x)\ \propto\ \frac{p(x\mid a)\;p(x\mid b)}{p(x)}$",
         "Bayes on both labels, then Bayes again on each: two experts over the unconditional model", None),
        ("eq", r"$\nabla\log p(x\mid a,b)=\nabla\log p(x\mid a)+\nabla\log p(x\mid b)-\nabla\log p(x)$",
         "the log turns the fraction into a sum and a difference", None),
        ("box", r"$\varepsilon_{\mathrm{PoE}}=\varepsilon_\varnothing+1\cdot(\varepsilon_a-\varepsilon_\varnothing)+1\cdot(\varepsilon_b-\varepsilon_\varnothing)$",
         "times -sqrt(1 - abar_t) (step 1): the U-Net asked three times, no digit, digit a, digit b", GREEN),
        ("eq", r"$\varepsilon_a-\varepsilon_\varnothing=-\sqrt{1-\bar\alpha_t}\ \nabla_{x_t}\log p_t(a\mid x_t)$",
         "each bracket is step 3's push with s = 1, and its classifier is implicit: the U-Net's own\n"
         "difference between asked-for-a and asked-for-nothing (Bayes on p(x|a) and p(x))", None),
        ("as", "What it gets wrong: assumption 4 is used at every noise level",
         "p_t(x_t | a, b) is not p_t(x_t | a) p_t(x_t | b) / p_t(x_t): the product of the noised experts is not the noised\n"
         "product. The missing part is the interaction term r_t this project measures.", None),
    ])
    fig.savefig(HERE / "d4-the-product-is-the-same-algebra.png", dpi=80); plt.close(fig)


# -------------------------------------------------------------------------------- d5
def d5():
    fig = sheet("Step 5: the two checkers are two more terms of the same shape",
                "Each picks its own label y. The separate-label checker names the digits; the pair-aware checker "
                "scores 'two clear digits, the asked pair' against everything else.")
    ans = ARCH["answers"]
    names = [("clean", "4 and 7\napart"), ("twin", "two 4s"), ("single", "a lone 4"), ("fused", "4 and 7\nfused")]
    ax = fig.add_axes([0.05, 0.53, 0.38, 0.30]); clean_ax(ax)
    xi = np.arange(4)
    ax.bar(xi - 0.18, [ans[n]["sep_p4"] for n, _ in names], 0.34, color=ORANGE, alpha=0.55, label=r"$\sigma(z_4)$")
    ax.bar(xi + 0.18, [ans[n]["sep_p7"] for n, _ in names], 0.34, color=ORANGE, label=r"$\sigma(z_7)$")
    for i, (n, _) in enumerate(names):
        ax.text(i - 0.18, ans[n]["sep_p4"] + 0.02, f"{ans[n]['sep_p4']:.2f}", ha="center", size=10)
        ax.text(i + 0.18, ans[n]["sep_p7"] + 0.02, f"{ans[n]['sep_p7']:.2f}", ha="center", size=10)
    ax.set_xticks(xi); ax.set_xticklabels([l for _, l in names], size=11); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=11)
    ax.set_title(f"measured: separate-label answers, asked 4 and 7, t = {ans['t']}", size=12, color=INK, loc="left")
    ax2 = fig.add_axes([0.05, 0.10, 0.38, 0.30]); clean_ax(ax2)
    ax2.bar(xi, [ans[n]["pair_prob"] for n, _ in names], 0.5, color=ORANGE)
    for i, (n, _) in enumerate(names):
        ax2.text(i, ans[n]["pair_prob"] + 0.02, f"d = {ans[n]['pair_prob']:.2f}\nlog-odds {ans[n]['pair_logodds']:.2f}",
                 ha="center", size=10)
    ax2.set_xticks(xi); ax2.set_xticklabels([l for _, l in names], size=11); ax2.set_ylim(0, 1.15)
    ax2.set_title(f"measured: pair-aware answer d, same canvases", size=12, color=INK, loc="left")
    lines(fig, 0.47, 0.86, [
        ("as", "Separate-label: two labels, 'a is present' and 'b is present', independent (assumption 4 again)", None, None),
        ("eq", r"$p(x_t)\ \sigma(z_a)^{\,s}\ \sigma(z_b)^{\,s}$", "sigma(z_d) = the checker's chance that digit d is on the canvas", None),
        ("box", r"$-\,s\,\sqrt{1-\bar\alpha_t}\ \nabla_{x_t}\left[\log\sigma(z_a)+\log\sigma(z_b)\right]$",
         "the product's two terms again, with a trained checker in place of the implicit one and s = 10.\n"
         "A fused 4 and 7 was labelled 'both present', so this term names the digits but cannot punish fusing.", ORANGE),
        ("as", "Pair-aware: one label y = 'two clear digits, the asked pair';  d = p(y = 1 | x_t)", None, None),
        ("eq", r"$\log\frac{d}{1-d}=\log p_\phi(y{=}1\mid x_t)-\log p_\phi(y{=}0\mid x_t)$",
         "the log-odds: chance of 'yes' over chance of 'no'", None),
        ("box", r"$p(x_t)\left(\frac{d}{1-d}\right)^{s}\ \Rightarrow\ -\,s\,\sqrt{1-\bar\alpha_t}\ \nabla_{x_t}\log\frac{d}{1-d}$",
         "toward 'two clear, right digits' and away from all it was shown as 'no': fused, lone, wrong pair.", ORANGE),
        ("eq", r"$\nabla\log d=(1-d)\ \nabla\log\frac{d}{1-d}$",
         "why the log-odds and not log d: the plain log fades as d nears 1, the log-odds keeps pushing;\n"
         "the code switches the push off once the log-odds passes 11.5 (d above 0.99999).", None),
    ], dy=0.093)
    fig.savefig(HERE / "d5-the-two-checkers-as-terms.png", dpi=80); plt.close(fig)


# -------------------------------------------------------------------------------- d6
def chain(fig, x, y, parts, size=22):
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for tex, col in parts:
        t = fig.text(x, y, tex, size=size, color=col, va="baseline")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(inv)
        x = bb.x1 + 0.004
    return x


def d6():
    g = GRAD
    fig = sheet("Step 6: the whole sampler step, every term from the steps before",
                "One noisy canvas in; the U-Net asked three times, both checkers asked once; one noise prediction "
                "out, which the DDIM step uses as usual.")
    chain(fig, 0.03, 0.80, [(r"$\varepsilon_{\mathrm{used}}=$", INK), (r"$\varepsilon_\varnothing$", GREEN),
                             (r"$+\,1\cdot(\varepsilon_a-\varepsilon_\varnothing)$", GREEN),
                             (r"$+\,1\cdot(\varepsilon_b-\varepsilon_\varnothing)$", GREEN),
                             (r"$-\ \mathrm{cap}_{0.1}\,[$", INK),
                             (r"$10\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log\frac{d}{1-d}$", ORANGE),
                             (r"$+\ 10\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\left[\log\sigma(z_a)+\log\sigma(z_b)\right]$", RED),
                             (r"$]$", INK)], size=21)
    boxes = [
        (0.03, GREEN, "the unconditional model", ["U-Net asked with no digit", "weight 1", "a score, in noise units (step 1)"]),
        (0.215, GREEN, "two implicit classifiers", ["U-Net asked with a, with b,", "minus no digit; weight 1 each",
                                                     "the product of experts (step 4)"]),
        (0.40, ORANGE, "pair-aware push", ["log-odds of 'two clear, right'", "s = 10 (steps 3 and 5)",
                                           f"pasted fused 4 & 7, t = 489: {g['pair|fused|25']['push_over_product']:.2f}",
                                           "of the product's size"]),
        (0.585, RED, "separate-label push", ["log of 'a there' + 'b there'", "s = 10 (steps 3 and 5)",
                                              f"same canvas: {g['separate|fused|25']['push_over_product']:.2f}",
                                              "of the product's size"]),
        (0.77, INK, "the cap", ["the two pushes are added,", "then scaled down if the sum", "is over 0.1 of the product's",
                                "size, per canvas (not Bayes)"]),
    ]
    for x, col, head, rows in boxes:
        fig.patches.append(FancyBboxPatch((x, 0.42), 0.17, 0.27, boxstyle="round,pad=0,rounding_size=0.006", fc="white",
                                          ec=col, lw=1.6, transform=fig.transFigure, figure=fig))
        fig.text(x + 0.01, 0.655, head, size=13.5, color=col, weight="bold")
        for k, r in enumerate(rows):
            fig.text(x + 0.01, 0.61 - 0.04 * k, r, size=12, color=INK)
    fig.text(0.03, 0.34, "What it would sample if every assumption held, with no cap:", size=14, color=INK, weight="bold")
    fig.text(0.03, 0.27, r"$p(x)\;p(a\mid x)\;p(b\mid x)\;\cdot\;\left(\frac{d}{1-d}\right)^{10}\cdot\;\sigma(z_a)^{10}\,\sigma(z_b)^{10}$",
             size=22, color=INK)
    fig.text(0.03, 0.20, "Which assumptions break: assumption 4 at every noise level (step 4, the dropped interaction term); "
             "s = 10 instead of 1 (step 3); the cap, which is a safety rule, not Bayes.", size=12.5, color=GREY)
    fig.text(0.03, 0.155, "So this is a steered sampler built from Bayes-shaped terms, not an exact sample from that product. "
             "Whether it works is read from the pictures: 285 of 320 on fresh held-out seeds.", size=12.5, color=GREY)
    fig.text(0.03, 0.08, r"Then the DDIM step, with fresh noise (eta = 1.0):   "
             r"$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\;\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\varepsilon_{\mathrm{used}}+\sigma_t\,z,\quad"
             r"\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_{\mathrm{used}}}{\sqrt{\bar\alpha_t}}$", size=15, color=INK)
    fig.savefig(HERE / "d6-the-whole-sampler-step.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    d1(); d2(); d3(); d4(); d5(); d6()
    print("wrote d1 to d6")
