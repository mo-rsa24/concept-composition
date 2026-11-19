"""d7: what the checkers' output nodes give, and how each becomes a probability and a push.

Real logits from logit_numbers.json (make_logit_numbers.py): both checkers on four pasted test canvases,
asked 4 and 7, at t = 489. Same sheet style and helpers as make_derivation_figures.py.
Usage: python make_logit_figure.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_derivation_figures import sheet, lines, clean_ax, BLUE, GREEN, RED, ORANGE, GREY, INK

HERE = Path(__file__).resolve().parent
L = json.load(open(HERE / "logit_numbers.json"))
sig = lambda z: 1 / (1 + np.exp(-np.asarray(z, float)))


def d7():
    zf = np.array(L["fused"]["sep_logits"]); zc = np.array(L["clean"]["sep_logits"])
    lf, lc = L["fused"]["pair_logit"], L["clean"]["pair_logit"]
    fig = sheet("Step 7: the checkers' outputs are logits; a sigmoid turns each into a probability on its own",
                f"Real outputs at t = {L['t']}, asked 4 and 7. The separate-label checker has ten output nodes, the pair-aware "
                "checker one. Neither uses a softmax.")
    # the output nodes
    ax = fig.add_axes([0.02, 0.50, 0.20, 0.38]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-1.6, 10.5)
    ax.text(0.02, 10.3, "separate-label: last layer", size=12, color=INK, weight="bold")
    ax.add_patch(plt.Rectangle((0.02, 1.0), 0.12, 8.5, fc="#d6efe0", ec=GREEN))
    ax.text(0.08, 5.2, "256\nhidden\nnumbers", size=10, color=GREEN, ha="center", va="center")
    for d in range(10):
        y = 9.3 - d
        hot = d in (4, 7)
        for yy in np.linspace(1.4, 9.1, 5):
            ax.plot([0.14, 0.44], [yy, y], color="#c9d1d9", lw=0.5, zorder=0)
        ax.plot([0.48], [y], "o", ms=11, mfc=ORANGE if hot else "white", mec=ORANGE if hot else GREY, zorder=2)
        ax.text(0.56, y, f"$z_{d}$ = {zf[d]:+.2f}", size=11, va="center", color=ORANGE if hot else INK,
                weight="bold" if hot else "normal")
    ax.text(0.02, -1.4, "the fused 4 and 7 canvas\nLinear 256 > 10: 2,570 weights", size=10, color=GREY)
    # sigmoid vs softmax bars
    ax2 = fig.add_axes([0.27, 0.53, 0.25, 0.33]); clean_ax(ax2)
    xi = np.arange(10)
    sm = np.exp(zf) / np.exp(zf).sum()
    ax2.bar(xi - 0.2, sig(zf), 0.4, color=ORANGE, label=f"sigmoid, each on its own: sum {sig(zf).sum():.2f}")
    ax2.bar(xi + 0.2, sm, 0.4, color=GREY, alpha=0.6, label=f"softmax over the ten: sum {sm.sum():.2f} (not used)")
    ax2.set_xticks(xi); ax2.set_xlabel("digit d", size=11, color=GREY); ax2.set_ylim(0, 0.5)
    ax2.legend(frameon=False, fontsize=10, loc="upper left")
    ax2.set_title("fused canvas: the same ten logits, two ways", size=12, color=INK, loc="left")
    # the sigmoid curve with real points
    ax3 = fig.add_axes([0.05, 0.08, 0.22, 0.33]); clean_ax(ax3)
    zz = np.linspace(-6, 6, 300)
    ax3.plot(zz, sig(zz), color=INK, lw=2)
    pts = [(zf[4], "fused: $z_4$", ORANGE, -0.9, 0.1), (zf[7], "fused: $z_7$", ORANGE, 0.35, -0.12),
           (zc[7], "apart: $z_7$", ORANGE, 0.3, -0.08), (lf, "fused: pair logit", RED, -2.6, 0.12),
           (lc, "apart: pair logit", RED, 0.3, -0.1)]
    for z, lab, c, dx, dy in pts:
        ax3.plot([z], [sig(z)], "o", color=c, ms=7)
        ax3.text(z + dx, sig(z) + dy, f"{lab}\n{z:+.2f} > {float(sig(z)):.3f}", size=9.5, color=c)
    ax3.set_xlabel("logit z", size=11, color=GREY); ax3.set_ylabel(r"$\sigma(z)$", size=12, color=GREY)
    ax3.set_title(r"$\sigma(z)=1/(1+e^{-z})$ with the real logits on it", size=12, color=INK, loc="left")
    # log sigma vs log-odds
    ax4 = fig.add_axes([0.32, 0.08, 0.20, 0.33]); clean_ax(ax4)
    ax4.plot(zz, np.log(sig(zz)), color=ORANGE, lw=2.2)
    ax4.plot(zz, zz, color=RED, lw=2.2, ls="--")
    ax4.text(-5.8, 1.6, "pair-aware uses\n" + r"$\log\frac{d}{1-d}=\ell$" + "\nslope 1 everywhere", color=RED, size=10.5)
    ax4.text(0.6, -5.6, "separate-label uses\n" + r"$\log\sigma(z)$" + "\nslope $1-\\sigma(z)$:\nflattens as it grows sure",
             color=ORANGE, size=10.5)
    ax4.set_ylim(-6.5, 4); ax4.axhline(0, color=GREY, lw=0.7)
    ax4.set_xlabel("logit", size=11, color=GREY)
    ax4.set_title("what each push takes the gradient of", size=12, color=INK, loc="left")
    lines(fig, 0.56, 0.86, [
        ("as", "Separate-label: ten logits, one sigmoid each", None, None),
        ("eq", r"$z=W\,h+b\in\mathbb{R}^{10},\qquad \sigma(z_d)=\frac{1}{1+e^{-z_d}}=p(\mathrm{digit}\ d\ \mathrm{present}\mid x_t)$",
         "h: the 256 hidden numbers; each output node's raw value is a logit, any real number", None),
        ("eq", r"$\mathcal{L}=-\sum_{d}\ \left[\,y_d\log\sigma(z_d)+(1-y_d)\log(1-\sigma(z_d))\,\right]$",
         "trained as ten yes/no questions (binary cross-entropy), so two digits can both be near 1", None),
        ("eq", r"$\frac{\partial}{\partial z}\log\sigma(z)=\frac{\partial}{\partial z}\left[-\log(1+e^{-z})\right]=\frac{e^{-z}}{1+e^{-z}}=1-\sigma(z)$",
         "so the push on each asked digit is weighted by how unsure the checker still is", None),
        ("as", r"Pair-aware: one logit $\ell$, and $d=\sigma(\ell)$", None, None),
        ("eq", r"$\frac{d}{1-d}=\frac{1/(1+e^{-\ell})}{e^{-\ell}/(1+e^{-\ell})}=e^{\ell}\qquad\Rightarrow\qquad\log\frac{d}{1-d}=\ell$",
         "d/(1 - d) is the odds; its log is the raw output itself, so the push is the gradient of the logit", None),
        ("eq", r"$\mathrm{softmax}_d(z)=e^{z_d}\ /\ \sum_k e^{z_k}$",
         "not used anywhere here: it makes the ten share 1,\nso '4 present' and '7 present' could not both be near 1", None),
    ], dy=0.103)
    fig.text(0.56, 0.05, "Read from the real numbers: at t = 489 the separate-label checker is unsure of everything\n"
             f"(on the two-4s canvas its largest logit is digit 1, {L['twin']['sep_logits'][1]:+.2f}), while the pair-aware logit\n"
             "already splits the canvases: "
             f"{lc:+.2f} for apart against {lf:+.2f} for fused.", size=12, color=GREY)
    fig.savefig(HERE / "d7-logits-sigmoids-and-odds.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    d7(); print("wrote d7")
