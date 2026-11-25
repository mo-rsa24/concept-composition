"""Render each step's equations on the maths page to maths/<step>.png with matplotlib mathtext.

Each entry is a list of lines; mathtext has no cases or underbrace environments, so a cases block is written as one
line per case. The TeX on 02-the-maths-step-by-step.md is the same equation in full LaTeX; read both against the PDF
pages cited there. Usage: ~/.claude/venvs/figures/bin/python render_maths.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "maths")
INK, ORANGE = "#1b2430", "#c8680a"

EQ = {
    "eq01": [r"$p(\mathbf{y}\mid\mathbf{x})=\prod_{i=1}^{k}\,p(y_i\mid\mathbf{x})$"],
    "eq02": [r"$\mathcal{C}\ \geq\ \mathcal{C}_{\min}=\dfrac{1+\sum_{i=1}^{k}(|\mathcal{Y}_i|-1)}{|\mathcal{Y}|}$"],
    "eq03": [r"$G_\theta(\mathbf{z},\mathbf{y}) := \mathbf{x}_0+\int_0^1 \mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{y})\,dt$"],
    "eq04": [r"$\hat\epsilon_\theta(\mathbf{x}_t,\mathbf{y})=\epsilon_\theta(\mathbf{x}_t)+\sum_{i=1}^{k} w_i\,(\epsilon_\theta(\mathbf{x}_t,y_i)-\epsilon_\theta(\mathbf{x}_t))$"],
    "eq05": [r"$\hat{\mathbf{v}}_\theta(\mathbf{x}_t,\mathbf{y})=\mathbf{v}_\theta(\mathbf{x}_t)+\sum_{i=1}^{k} w_i\,(\mathbf{v}_\theta(\mathbf{x}_t,y_i)-\mathbf{v}_\theta(\mathbf{x}_t))$"],
    "eq06": [r"$\mathbf{v}_t(\mathbf{x}_t\mid\mathbf{y})=a_t\,\mathbf{x}_t+b_t\,\nabla\log p_t(\mathbf{x}_t\mid\mathbf{y})$"],
    "eq07": [r"$\mathcal{L}_{\mathrm{CI}}(\theta)=\mathbb{E}_{t,\mathbf{x}_t,\mathbf{y}\sim D_a}\ \Vert\,\mathbf{v}_\theta(\mathbf{x}_t,\mathbf{y})-\hat{\mathbf{v}}_\theta(\mathbf{x}_t,\mathbf{y})\,\Vert^2$",
             r"$I_t=\mathbf{v}_\theta(\mathbf{x}_t,\mathbf{y})-\hat{\mathbf{v}}_\theta(\mathbf{x}_t,\mathbf{y})$   (the ladder's name for the gap)"],
    "eq08": [r"$\mathbf{y}_{\mathbf{m}}=\mathbf{y}\odot\mathbf{m}$,   $m_i=0$ replaces $y_i$ by $\emptyset$",
             r"$p(\mathbf{m})=\pi_{\mathrm{full}}$   if $\mathbf{m}=\mathbf{1}$",
             r"$p(\mathbf{m})=\pi_{\mathrm{marg}}\cdot\frac{1}{k}$   if $\mathbf{m}=\mathbf{e}_i$, for $i\in\{1,\ldots,k\}$",
             r"$p(\mathbf{m})=\pi_{\mathrm{uncond}}$   if $\mathbf{m}=\mathbf{0}$"],
    "eq09": [r"$\mathcal{L}^{(a)}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t,\mathbf{y},\mathbf{m},\,\mathbf{x}_t\sim p_t}\left[\ \Vert\,\mathbf{v}^{(a)}_\theta(\mathbf{x}_t,t,\mathbf{y}\odot\mathbf{m})-\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_1)\,\Vert^2\ \right]$",
             r"$\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\mathbf{x}_1,\qquad \mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_1)=\mathbf{x}_1-\mathbf{x}_0,\qquad t\sim\mathcal{U}[0,1]$"],
    "eq10_11": [r"$\mathcal{L}^{(a)}_{\mathrm{CI}}(\theta)=\mathbb{E}_{t,\mathbf{y},\mathbf{x}_t\sim p_t}\left[\,\mathbf{v}^{(a)}_\theta(\mathbf{x}_t,\mathbf{y})-\hat{\mathbf{v}}^{(a)}_\theta(\mathbf{x}_t,\mathbf{y})\,\right]$   (printed with no norm)",
                r"$\mathcal{L}^{(a)}_{\mathrm{total}}(\theta)=\mathcal{L}^{(a)}_{\mathrm{FM}}(\theta)+\lambda\cdot\mathcal{L}^{(a)}_{\mathrm{CI}}(\theta)$"],
    "eq12": [r"$\mathbf{z}^{(r_0)}_\theta(\mathbf{x}_t)=\bar{\mathbf{v}}^{(r_0)}_\theta(\mathbf{x}_t)+\sum_{i=1}^{k}\left(\bar{\mathbf{v}}^{(r_i)}_\theta(\mathbf{x}_t,y_i)-\bar{\mathbf{v}}^{(r_0)}_\theta(\mathbf{x}_t)\right)$",
             r"$\mathcal{L}^{(a)}_{\mathrm{peerCI}}(\theta)=\mathbb{E}_{\mathbf{r},t,\mathbf{y},\mathbf{x}_t\sim p_t}\left[\,\mathbf{v}^{(a)}_\theta(\mathbf{x}_t,\mathbf{y})-\mathbf{z}^{(r_0)}_\theta(\mathbf{x}_t)\,\right]$   (printed with no norm)"],
    "eq13": [r"$\bar{\mathbf{v}}^{\,r}_\theta=\delta_{r,a}\,\mathbf{v}^{(a)}_\theta+(1-\delta_{r,a})\,\mathrm{StopGrad}(\mathbf{v}^{(r)}_\theta)$"],
    "eq14": [r"$\mathcal{L}^{(a)}_{\mathrm{DCFM\!-\!A}}(\theta)=\mathcal{L}^{(a)}_{\mathrm{FM}}(\theta)+\lambda\cdot\mathcal{L}^{(a)}_{\mathrm{peerCI}}(\theta)$"],
    "eq15": [r"$\mathcal{L}_{\mathrm{student}}(\theta)=\mathbb{E}_{t,r,\hat{\mathbf{x}}_t}\left[\ \Vert\,\mathbf{v}_\theta(\hat{\mathbf{x}}_t,t,\mathbf{y}\odot\mathbf{m})-\bar{\mathbf{v}}^{(r)}_\theta(\hat{\mathbf{x}}_t,\mathbf{y})\,\Vert^2\ \right]$",
             r"$\hat{\mathbf{x}}_t=\mathrm{ODEInt}(\mathbf{x}_0,\bar{\mathbf{v}}^{(r)}_\theta,\mathbf{y},0,t)$"],
    "eq16": [r"$\mathbf{z}_\theta(\hat{\mathbf{x}}_t)=\mathbf{v}_\theta(\hat{\mathbf{x}}_t)+\sum_{i=1}^{k}(\mathbf{v}_\theta(\hat{\mathbf{x}}_t,y_i)-\mathbf{v}_\theta(\hat{\mathbf{x}}_t))$",
             r"$\mathcal{L}_{\mathrm{studentCI}}(\theta)=\mathbb{E}_{t,\mathbf{y},\hat{\mathbf{x}}_t}\left[\ \Vert\,\mathbf{v}_\theta(\hat{\mathbf{x}}_t,\mathbf{y})-\mathbf{z}_\theta(\hat{\mathbf{x}}_t)\,\Vert_2^2\ \right]$"],
    "eq17": [r"$\mathcal{L}_{\mathrm{DCFM\!-\!B}}(\theta)=\mathcal{L}_{\mathrm{student}}(\theta)+\lambda\,\mathcal{L}_{\mathrm{studentCI}}(\theta)$"],
    "a1": [r"$\nabla\log p_t(\mathbf{x}\mid y_1,\ldots,y_k)=\nabla\log p_t(\mathbf{x})+\nabla\log p_t(y_1,\ldots,y_k\mid\mathbf{x})$   (Eq. 19 to 20)",
           r"$=\nabla\log p_t(\mathbf{x})+\sum_{i=1}^{k}\nabla\log p_t(y_i\mid\mathbf{x})$   (Eq. 21 to 22, using Eq. 1)",
           r"$\nabla\log\hat p_t(\mathbf{x}\mid y_1,\ldots,y_k)=\nabla\log p_t(\mathbf{x})+\sum_{i=1}^{k}w_i\,(\nabla\log p_t(\mathbf{x}\mid y_i)-\nabla\log p_t(\mathbf{x}))$   (Eq. 23 to 24)"],
    "a2": [r"$p_t(\mathbf{x}\mid\mathbf{x}_1)=\mathcal{N}(\mathbf{x}\mid\alpha_t\mathbf{x}_1,\sigma_t^2\mathbf{I}),\quad(\alpha_0,\sigma_0)=(0,1),\ (\alpha_1,\sigma_1)=(1,0)$   (Eq. 25)",
           r"$\mathbf{v}_t(\mathbf{x}\mid y)=a_t\mathbf{x}+b_t\nabla\log p_t(\mathbf{x}\mid y),\quad a_t=\frac{\dot\alpha_t}{\alpha_t},\quad b_t=(\dot\alpha_t\sigma_t-\alpha_t\dot\sigma_t)\frac{\sigma_t}{\alpha_t}$   (Eq. 26 to 27)",
           r"$\nabla\log p_t(\mathbf{x}\mid y)=\frac{\mathbf{v}_t(\mathbf{x}\mid y)-a_t\mathbf{x}}{b_t}$   (Eq. 28)",
           r"$a_t\mathbf{x}+b_t\nabla\log\hat p_t(\mathbf{x}\mid y_1,\ldots,y_k)=\mathbf{v}_t(\mathbf{x})+\sum_{i=1}^{k}w_i\,(\mathbf{v}_t(\mathbf{x}\mid y_i)-\mathbf{v}_t(\mathbf{x}))$   (Eq. 29 to 33)"],
}


def render(name, lines):
    h = 0.62 * len(lines) + 0.25
    w = max(len(l) for l in lines) * 0.052 + 1.0
    fig = plt.figure(figsize=(min(max(w, 4.0), 15.5), h), dpi=150)
    fig.patch.set_facecolor("white")
    for i, l in enumerate(lines):
        fig.text(0.02, 1 - (i + 0.62) / (len(lines) + 0.25), l, fontsize=15, color=INK, va="center", ha="left")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, v in EQ.items():
        render(k, v)
        print("wrote", k)
