"""Render each step of 02-the-maths-step-by-step.md to a PNG with matplotlib mathtext, so the maths shows where TeX does not.

Each step is a list of lines: (tex, colour). Orange marks the push and the interaction term, as on every figure of the ladder.
Run: ~/.claude/venvs/figures/bin/python scripts/make_maths.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figure_kit as K  # noqa: E402
from figure_kit import KEY, S  # noqa: E402

INK, PUSH = S["ink"], KEY["push"]
STEPS = {
    "step-1-noise-prediction-is-a-scaled-score": [
        (r"$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sigma_t\,\epsilon,\qquad \sigma_t = \sqrt{1-\bar\alpha_t}$", INK),
        (r"$\epsilon_\theta(x_t, y) = -\,\sigma_t\,\nabla_{x_t}\log p_t(x_t\mid y)$", INK)],
    "step-2-the-product-of-experts": [
        (r"$p(x\mid y_1, y_2)\ \propto\ \dfrac{p(x\mid y_1)\;p(x\mid y_2)}{p(x)}$", INK),
        (r"$\nabla\log p(x\mid y_1,y_2) = s_1 + s_2 - s_\varnothing$", INK),
        (r"$\epsilon_{\mathrm{PoE}} = \epsilon_\varnothing + \sum_i\,(\epsilon_{y_i} - \epsilon_\varnothing)$", INK)],
    "step-3-the-interaction-term": [
        (r"$I_t = \nabla_{x_t}\log p_t(x_t\mid y_1, y_2)\ -\ [\,s_1 + s_2 - s_\varnothing\,]$", PUSH),
        (r"$r_t = -\,\sigma_t\,I_t = \epsilon_{\mathrm{joint}} - \epsilon_{\mathrm{PoE}}$", PUSH)],
    "step-4-the-region-merge": [
        (r"$\tilde x^A_t = m\odot x_t + (1-m)\odot b_t,\qquad \tilde x^B_t = (1-m)\odot x_t + m\odot b_t$", INK),
        (r"$\hat\epsilon_t = m\odot\epsilon^A_t(\tilde x^A_t) + (1-m)\odot\epsilon^B_t(\tilde x^B_t)$", INK)],
    "step-5-bayes-gives-the-checker-term": [
        (r"$p_t(x_t\mid \mathrm{both}) = \dfrac{p_t(x_t)\;C_t(\mathrm{both}\mid x_t)}{p(\mathrm{both})}$", INK),
        (r"$\nabla_{x_t}\log p_t(x_t\mid\mathrm{both}) = \nabla_{x_t}\log p_t(x_t) + \nabla_{x_t}\log C_t(\mathrm{both}\mid x_t)$", INK)],
    "step-6-from-score-to-noise-the-push": [
        (r"$\hat\epsilon_t = \epsilon_{\mathrm{base}}(x_t)$", INK),
        (r"$-\ \lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t(\mathrm{both}\mid x_t)$", PUSH),
        (r"$\epsilon_{\mathrm{base}} = \mathrm{merge}(\epsilon^A_t|_{R_A},\,\epsilon^B_t|_{R_B})\ \ \mathrm{or}\ \ \epsilon_{\mathrm{PoE}}$", INK)],
    "step-7-one-sampler-step-with-the-push": [
        (r"$\hat x_0 = \dfrac{x_t - \sigma_t\,\hat\epsilon_t}{\sqrt{\bar\alpha_t}}$", INK),
        (r"$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\eta_t^2}\;\hat\epsilon_t + \eta_t\,z$", INK)],
}


def render(name, lines):
    K.reset()
    h = 0.62 * len(lines) + 0.3
    fig = plt.figure(figsize=(9.0, h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    for i, (tex, colour) in enumerate(lines):
        ax.text(0.5, 1 - (i + 0.5 + 0.25) / (len(lines) + 0.5), tex, ha="center", va="center", fontsize=17, color=colour,
                transform=ax.transAxes)
    out = K.SCENE / "maths" / name
    K.check(fig)
    fig.savefig(f"{out}.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", out.name + ".png")


for n, ls in STEPS.items():
    render(n, ls)
