"""Level 0.5: the network view of DCFM, four figures in the clean cheat-sheet register, each opening one part of the overview.

  01a-stage-one-local-training   Eq. 8 to 9: the masked label, the straight path, the target x_1 - x_0
  01b-sum-rule-tip-to-tail       Eq. 5: velocities added tip to tail, on the toy's four cells and on a silo missing one
  01c-the-penalty                Eq. 10 and Eq. 12 to 13: one network called four times, the gap I_t, and which calls learn
  01d-what-crosses-between-silos DCFM-A against DCFM-B (Eq. 14 against 15 to 17, Table 3): what is sent, to whom, when

Register: white ground, thin outlines, a soft glow on the part the figure is about, faded grey for frozen or off; colours
from ladder.json's colour key. Network blocks whose architecture the paper does not state are drawn as a labelled box.
Every number comes from dcfm_numbers.json. figcheck.check refuses a figure in which a line crosses a node, box, picture
or label. Usage: ~/.claude/venvs/figures/bin/python make_network_view.py [a b c d]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import digits as D  # noqa: E402
import toy_dcfm as T  # noqa: E402
from figcheck import check  # noqa: E402

N = json.load(open(os.path.join(HERE, "dcfm_numbers.json")))
K = N["colour_key"]
BLUE, GREEN, RED, ORANGE, GREY = (K["inputs"], K["experts_and_hidden_units"], K["outputs"],
                                  K["interaction_term_penalty_gradient_push"], K["fixed_frozen_off"])
INK, FADE, FADE_EDGE = "#222222", "#eceeef", "#aab3bb"
FILL = {BLUE: "#e6eefb", GREEN: "#e2f1e7", RED: "#fbe6e6", ORANGE: "#fdeedd", GREY: FADE}


def new(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, txt, colour, size=9.5, focus=False, faded=False, gid=None, weight="normal", ls="-"):
    if focus:
        for pad, a in ((0.16, 0.10), (0.09, 0.2)):
            g = FancyBboxPatch((x - pad, y - pad), w + 2 * pad, h + 2 * pad, boxstyle="round,pad=0.02,rounding_size=0.12",
                               color=colour, alpha=a * 0.6, lw=0, zorder=1)
            g.set_gid("free")
            ax.add_patch(g)
    ec, fc = (FADE_EDGE, FADE) if faded else (colour, FILL.get(colour, "white"))
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=1.1, edgecolor=ec, facecolor=fc,
                       zorder=3, linestyle=ls)
    if gid:
        p.set_gid(gid)
    ax.add_patch(p)
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=size, color="#7b858e" if faded else INK,
                zorder=4, linespacing=1.25, weight=weight)


def arrow(ax, a, b, colour=INK, lw=1.1, ms=10, ls="-", gid=None):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=colour, linestyle=ls, shrinkA=0, shrinkB=0, zorder=2)
    if gid:
        p.set_gid(gid)
    ax.add_patch(p)


def poly(ax, pts, colour=INK, lw=1.1, ms=10):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=colour, lw=lw, zorder=2, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], colour, lw, ms)


def text(ax, x, y, s, size=9, colour=INK, ha="center", **kw):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=colour, zorder=5, linespacing=1.25, **kw)


def tile(ax, img, x, y, w, ec=BLUE):
    ax.imshow(img, extent=(x, x + w, y, y + w), zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, w, fill=False, ec=ec, lw=1.2, zorder=4))


def save(fig, name):
    check(fig, name)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, "figures", f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote figures/" + name)


# ------------------------------------------------------------------ 01a
def fig_a():
    fig, ax = new(16.5, 7.6)
    text(ax, 8.25, 7.25, "Stage I: one silo trains its expert alone on its own images (Eq. 8 and 9, p. 5)", size=13, weight="bold")
    x1, _ = D.digit(4, "green", nth=0)
    x0 = np.clip(np.random.default_rng(4).standard_normal(x1.shape) * 0.35 + 0.5, 0, 1)
    raw0 = (x0 - 0.5) / 0.35  # the noise itself, shown clipped
    ts = (0.0, 0.25, 0.5, 0.75, 1.0)
    for k, t in enumerate(ts):
        img = np.clip((1 - t) * x0 + t * x1, 0, 1)
        tile(ax, img, 0.5 + k * 1.45, 5.0, 1.0, ec=BLUE)
        text(ax, 1.0 + k * 1.45, 4.72, f"t = {t:g}", size=8.6)
        if k < 4:
            arrow(ax, (1.55 + k * 1.45, 5.5), (1.9 + k * 1.45, 5.5), colour="#9aa3ab", lw=2.4, ms=12)
    text(ax, 1.0, 6.3, "noise $x_0$", size=9)
    text(ax, 6.8, 6.3, "image $x_1$ (a real MNIST 4, tinted)", size=9)
    text(ax, 4.1, 4.25, "the straight path $x_t=(1-t)\\,x_0+t\\,x_1$; t runs from 0 (noise) to 1 (image)", size=9.2)

    # The mask table, Eq. 8.
    box(ax, 0.4, 0.35, 5.6, 3.2, None, BLUE, gid="container")
    text(ax, 3.2, 3.25, "Eq. 8: the label the network sees, $\\mathbf{y}\\odot\\mathbf{m}$", size=9.4, weight="bold")
    rows = [("$\\mathbf{m}=(1,1)$", "(4, green)", "$\\pi_{\\mathrm{full}}$"),
            ("$\\mathbf{m}=(1,0)$", "(4, $\\emptyset$)", "$\\pi_{\\mathrm{marg}}/2$"),
            ("$\\mathbf{m}=(0,1)$", "($\\emptyset$, green)", "$\\pi_{\\mathrm{marg}}/2$"),
            ("$\\mathbf{m}=(0,0)$", "($\\emptyset$, $\\emptyset$)", "$\\pi_{\\mathrm{uncond}}$")]
    for i, (m, lab, p) in enumerate(rows):
        y = 2.7 - i * 0.45
        text(ax, 1.25, y, m, size=8.8)
        text(ax, 3.05, y, lab, size=8.8)
        text(ax, 4.85, y, p, size=8.8)
    text(ax, 3.2, 0.62, "the $\\pi$ values are not stated in the paper;\nthe toy (03) uses 1/3, 1/6, 1/6, 1/3", size=8.2, style="italic")

    # Inputs into the network.
    tile(ax, np.clip(0.5 * x0 + 0.5 * x1, 0, 1), 7.0, 2.85, 0.9)
    text(ax, 7.45, 3.98, "$x_t$, t = 0.5", size=8.6)
    box(ax, 7.0, 1.95, 0.9, 0.5, "t", BLUE, size=9)
    box(ax, 6.6, 0.95, 1.7, 0.55, "$\\mathbf{y}\\odot\\mathbf{m}$", BLUE, size=9.5)
    arrow(ax, (6.02, 1.22), (6.58, 1.22))
    box(ax, 9.0, 0.9, 2.5, 3.2, "expert $v_\\theta^{(a)}$\n\nnetwork not\nstated in the paper", GREEN, size=9.6, focus=True)
    arrow(ax, (7.92, 3.3), (8.98, 3.3))
    arrow(ax, (7.92, 2.2), (8.98, 2.2))
    arrow(ax, (8.32, 1.22), (8.98, 1.22))

    # Output, target, loss.
    arrow(ax, (11.52, 3.3), (12.38, 3.3), colour=RED)
    box(ax, 12.4, 2.95, 1.75, 0.7, "predicted\nvelocity $v$", RED, size=8.8)
    diff = np.clip(0.5 + 0.5 * (x1 - x0), 0, 1)
    tile(ax, diff, 12.8, 0.95, 0.95, ec=INK)
    text(ax, 13.28, 0.6, "target $x_1-x_0$ (shown +0.5)", size=8.2)
    box(ax, 14.65, 1.75, 1.6, 1.25, "Eq. 9\n$\\Vert v-(x_1-x_0)\\Vert^2$", INK, size=9)
    poly(ax, [(14.17, 3.3), (15.45, 3.3), (15.45, 3.02)])
    poly(ax, [(13.77, 1.42), (15.45, 1.42), (15.45, 1.73)])
    text(ax, 13.45, 4.4, "+ $\\lambda$ times the local penalty\n(Eq. 10 to 11, figure 01c)", size=8.6, colour=ORANGE)
    save(fig, "01a-stage-one-local-training")


# ------------------------------------------------------------------ 01b
def field_panel(ax, cells, w, title, truth_note):
    g = np.linspace(-2, 2, 200)
    GX, GY = np.meshgrid(g, g)
    P = np.zeros_like(GX)
    t = N["worked"]["t"]
    for c in cells:
        mu = np.array(T.CENTRE[c]) * t
        P += np.exp(-((GX - mu[0]) ** 2 + (GY - mu[1]) ** 2) / (2 * T.tau2(t)))
    cs = ax.contour(GX, GY, P, levels=5, colors=["#c9d3dc"], linewidths=0.8, zorder=1)
    for coll in getattr(cs, "collections", [cs]):
        coll.set_gid("free")
    for c, (cx, cy) in T.CENTRE.items():
        here = c in cells
        ax.add_patch(Circle((cx, cy), 0.09, facecolor=GREEN if here else "white", edgecolor=GREEN if here else GREY,
                            lw=1.2, zorder=3, linestyle="-" if here else "--"))
        ax.text(cx + (0.2 if cx > 0 else -0.2), cy + 0.22, f"({c[0]},{c[1]})" + ("" if here else " missing"), fontsize=8.4,
                ha="left" if cx > 0 else "right", va="center", color=INK if here else GREY, zorder=5)
    x = np.array(N["worked"]["point_x"])
    sc = 0.42
    u, vr, vt = (np.array(w[k]) for k in ("v_uncond", "v_right", "v_top"))
    p1 = x + sc * u
    p2 = p1 + sc * (vr - u)
    p3 = p2 + sc * (vt - u)
    tr = x + sc * np.array(w["v_true_cell_2_1"])
    arrow(ax, tuple(x), tuple(tr), colour="#555555", lw=1.0, ls=":", ms=9)
    arrow(ax, tuple(x), tuple(p1), colour=GREEN, lw=1.6, ms=10)
    arrow(ax, tuple(p1), tuple(p2), colour=GREEN, lw=1.6, ms=10)
    arrow(ax, tuple(p2), tuple(p3), colour=GREEN, lw=1.6, ms=10)
    arrow(ax, tuple(x), tuple(p3), colour=GREEN, lw=2.6, ms=13, ls="--")
    dot = Circle(tuple(x), 0.05, color=INK, zorder=6)
    dot.set_gid("free")
    ax.add_patch(dot)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d0d6db")
    return x, p1, p2, p3, tr


def fig_b():
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 7.3), dpi=130)
    fig.patch.set_facecolor("white")
    four, iid = N["worked"]["eq5_four_cells"], N["worked"]["eq5_iid_silo_missing_2_1"]
    a = field_panel(axs[0], list(T.CENTRE), four, "all four cells in the data: the sum lands on the missing cell's velocity", "")
    b = field_panel(axs[1], T.SPLITS["iid"][1], iid, "one silo, (2,1) missing: the sum points somewhere else", "")
    for ax, (x, p1, p2, p3, tr), w in ((axs[0], a, four), (axs[1], b, iid)):
        ax.text(x[0] - 0.1, x[1] - 0.22, "$x_t$", fontsize=9.5, ha="right", va="center", color=INK, zorder=6)
        ax.text(p1[0] - 0.1, p1[1] + 0.14, "$v(x_t)$", fontsize=8.6, color=GREEN, ha="right", zorder=6)
        ax.text(1.95, -1.55, "thin: $v(x_t)$, then $+\\,(v(x_t,\\mathrm{right})-v)$,\nthen $+\\,(v(x_t,\\mathrm{top})-v)$, tip to tail",
                fontsize=8.2, color=GREEN, ha="right", va="center", zorder=6)
        ax.text(1.95, -1.98, f"thick dashed: the sum $\\hat v$ = ({w['v_hat_sum_rule'][0]:.2f}, {w['v_hat_sum_rule'][1]:.2f});  "
                f"gap to the dotted truth: {w['gap_norm']:.2f}", fontsize=8.2, color=GREEN, ha="right", va="center", zorder=6)
    axs[0].text(-1.95, -1.75, "dotted: the missing cell's own velocity\n(1.57, 1.38), computed on the four-cell grid", fontsize=8.0,
                color="#555555", ha="left", va="center")
    fig.suptitle("The sum rule, Eq. 5 with $w_i$ = 1: $\\hat v(x_t,\\mathbf{y})=v(x_t)+\\sum_i\\,(v(x_t,y_i)-v(x_t))$, "
                 "asked for (2,1) = right and top", fontsize=12.5, weight="bold", y=0.985)
    fig.text(0.5, 0.035, f"Exact velocities of the 2-D toy (p. 4) at the point $x_t$ = ({N['worked']['point_x'][0]}, {N['worked']['point_x'][1]}), "
             f"t = {N['worked']['t']} (t runs from 0 noise to 1 data); arrows scaled by 0.42; grey contours: where $x_t$ lies at that t. "
             "Numbers: dcfm_numbers.json, worked.eq5 rows.", ha="center", fontsize=8.8, color=INK)
    save(fig, "01b-sum-rule-tip-to-tail")


# ------------------------------------------------------------------ 01c
def penalty_panel(ax, x0, title, rows, formula, note):
    text(ax, x0 + 4.0, 7.55, title, size=11, weight="bold")
    xt_img = D.noised(D.digit(4, "green", nth=0)[0], 0.55, seed=3)
    tile(ax, xt_img, x0 + 0.2, 3.45, 1.0)
    text(ax, x0 + 0.7, 3.15, "$x_t$, t = 0.55", size=8.6)
    ys = [6.1, 4.65, 3.2, 1.75]
    for (label, who, trained, sign), y in zip(rows, ys):
        box(ax, x0 + 2.2, y - 0.5, 2.6, 1.0, f"{who}\n{label}", GREEN, size=8.6, focus=trained, faded=not trained)
        arrow(ax, (x0 + 1.22, 3.95), (x0 + 2.18, y), colour=INK, lw=0.9, ms=8)
        arrow(ax, (x0 + 4.82, y + 0.18), (x0 + 5.98, y + 0.18), colour=INK, lw=0.9, ms=8)
        if trained:
            arrow(ax, (x0 + 5.98, y - 0.2), (x0 + 4.82, y - 0.2), colour=ORANGE, lw=1.4, ms=9)
            text(ax, x0 + 5.4, y - 0.42, sign, size=8.0, colour=ORANGE)
        else:
            text(ax, x0 + 5.4, y - 0.3, "StopGrad", size=8.0, colour=GREY)
    box(ax, x0 + 6.0, 1.1, 2.6, 5.6, formula, ORANGE, size=9.0)
    text(ax, x0 + 4.0, 0.45, note, size=8.4)


def fig_c():
    fig, ax = new(18.4, 8.0)
    loc = [("joint $J=v(x_t,(4,\\mathrm{green}))$", "own expert", True, "$+2I_t$"),
           ("$M_1=v(x_t,(4,\\emptyset))$", "own expert", True, "$-2I_t$"),
           ("$M_2=v(x_t,(\\emptyset,\\mathrm{green}))$", "own expert", True, "$-2I_t$"),
           ("$U=v(x_t,(\\emptyset,\\emptyset))$", "own expert", True, "$+2I_t$")]
    form = ("$I_t=J-(M_1+M_2-U)$\n\nloss $\\lambda\\,\\Vert I_t\\Vert^2$\n\n(Eq. 10; printed\nwith no norm,\nEq. 7 has $\\Vert\\cdot\\Vert^2$)\n\n"
            "orange: the\ngradient of the\nloss with respect\nto each output")
    penalty_panel(ax, 0.2, "Local penalty, Eq. 10 to 11: all four calls are the silo's own expert", loc, form,
                  "every call learns: the joint and the marginals all move toward agreement")
    peer = [("joint $J=v^{(2)}(x_t,(4,\\mathrm{green}))$", "own expert 2", True, "$+2I_t$"),
            ("$\\bar v^{(r_1)}(x_t,(4,\\emptyset))$, $r_1$ = 2", "own expert 2", True, "$-2I_t$"),
            ("$\\bar v^{(r_2)}(x_t,(\\emptyset,\\mathrm{green}))$, $r_2$ = 1", "peer expert 1, frozen", False, ""),
            ("$\\bar v^{(r_0)}(x_t,(\\emptyset,\\emptyset))$, $r_0$ = 1", "peer expert 1, frozen", False, "")]
    form2 = ("$z=\\bar v^{(r_0)}$\n\n$+\\,\\Sigma_i\\,(\\bar v^{(r_i)}(x_t,y_i)-\\bar v^{(r_0)})$\n\n$I_t=J-z$\n\nloss $\\lambda\\,\\Vert I_t\\Vert^2$\n"
             "(Eq. 12 and 14)\n\nEq. 13: a peer's\noutput is a constant\n(StopGrad), so only\nsilo 2's calls learn")
    penalty_panel(ax, 9.6, "Peer penalty, DCFM-A, Eq. 12 to 13: routing r = (1, 2, 1) for silo 2", peer, form2,
                  "how r is drawn is not stated; at least one entry must be the silo itself (p. 5)")
    save(fig, "01c-the-penalty")


# ------------------------------------------------------------------ 01d
def silo_row(ax, x, y, labels, kinds, w=1.15, h=0.62, gap=0.25):
    out = []
    for k, (lab, kind) in enumerate(zip(labels, kinds)):
        xx = x + k * (w + gap)
        colour = {"own": GREEN, "frozen": GREY, "data": BLUE}[kind]
        box(ax, xx, y, w, h, lab, colour, size=8.2, faded=kind == "frozen", focus=kind == "own")
        out.append((xx, y, w, h))
    return out


def fig_d():
    fig, ax = new(19.6, 9.4)
    text(ax, 9.8, 9.05, "What crosses between silos, and when: DCFM-A against DCFM-B (p. 5 to 6, Table 3 on p. 9)", size=13, weight="bold")
    cols = [0.3, 4.95, 9.6, 14.25]
    heads = ["1. Stage I", "2. Exchange", "3. Retrain with the penalty", "4. Result"]
    for x, h in zip(cols, heads):
        text(ax, x + 2.1, 8.35, h, size=11, weight="bold")
    for row_y, name in ((5.0, "DCFM-A\n(Eq. 14)"), (0.9, "DCFM-B\n(Eq. 17)")):
        pass
    cm = N["printed"]["cost"]
    # DCFM-A row.
    yA = 5.0
    text(ax, 0.3, 7.75, "DCFM-A: the silos keep their own experts", size=10, colour=INK, ha="left", weight="bold")
    for k in range(3):
        box(ax, cols[0] + 0.2, yA + 1.6 - k * 0.85, 1.3, 0.62, f"silo {['1', '2', 'N'][k]}", BLUE, size=8.4)
        box(ax, cols[0] + 2.1, yA + 1.6 - k * 0.85, 1.6, 0.62, f"expert {['1', '2', 'N'][k]}", GREEN, size=8.4, focus=True)
        arrow(ax, (cols[0] + 1.52, yA + 1.91 - k * 0.85), (cols[0] + 2.08, yA + 1.91 - k * 0.85))
    text(ax, cols[0] + 2.0, yA - 0.65, "each silo trains alone;\nnothing crosses", size=8.6)
    # Exchange: a ring of three silos with models passed both ways.
    cx = cols[1]
    pos = {"1": (cx + 0.3, yA + 1.6), "2": (cx + 2.8, yA + 1.6), "N": (cx + 1.55, yA - 0.1)}
    for k, (xx, yy) in pos.items():
        box(ax, xx, yy, 1.3, 0.62, f"silo {k}", BLUE, size=8.4)
    arrow(ax, (cx + 1.62, yA + 2.0), (cx + 2.78, yA + 2.0), colour=GREEN, lw=1.3)
    arrow(ax, (cx + 2.78, yA + 1.75), (cx + 1.62, yA + 1.75), colour=GREEN, lw=1.3)
    arrow(ax, (cx + 0.75, yA + 1.58), (cx + 1.7, yA + 0.54), colour=GREEN, lw=1.3)
    arrow(ax, (cx + 1.95, yA + 0.54), (cx + 1.0, yA + 1.58), colour=GREEN, lw=1.3)
    arrow(ax, (cx + 3.35, yA + 1.58), (cx + 2.55, yA + 0.54), colour=GREEN, lw=1.3)
    arrow(ax, (cx + 2.3, yA + 0.54), (cx + 3.1, yA + 1.58), colour=GREEN, lw=1.3)
    text(ax, cx + 2.15, yA - 0.6, "all-gather of $\\theta$: each silo sends its\nexpert to every other (models, never images)", size=8.6)
    # Retrain: own expert glowing, peers frozen.
    rx = cols[2]
    for k, lab in enumerate(["1", "2", "N"]):
        yy = yA + 1.6 - k * 0.85
        text(ax, rx + 0.45, yy + 0.31, f"silo {lab}:", size=8.4)
        others = [o for o in ["1", "2", "N"] if o != lab]
        box(ax, rx + 1.0, yy, 1.25, 0.62, f"expert {lab}", GREEN, size=8.2, focus=True)
        for j, o in enumerate(others):
            box(ax, rx + 2.35 + j * 0.92, yy, 0.85, 0.62, f"{o},\nfrozen", GREY, size=7.4, faded=True)
    text(ax, rx + 2.2, yA - 0.62, f"each silo retrains its own expert against frozen copies;\nexchange and retrain R = {cm['dcfm_a_rounds_R']}",
         size=8.4)
    # Result.
    ox = cols[3]
    silo_row(ax, ox + 0.15, yA + 1.6, ["expert 1", "expert 2", "expert N"], ["own", "own", "own"])
    text(ax, ox + 2.3, yA + 0.85, "N experts, one per silo; sampling\nroutes each attribute to an expert\n(Eq. 5 with routing)", size=8.6)
    box(ax, ox + 0.15, yA - 0.6, 4.6, 0.85, f"sent per silo: {cm['table3_communication_per_client']['DCFM (DCFM-A)']}\n"
        "(R rounds, N models each; Table 3)", INK, size=8.8)

    # DCFM-B row.
    yB = 0.9
    text(ax, 0.3, 3.95, "DCFM-B: one student, distilled on a trusted server", size=10, colour=INK, ha="left", weight="bold")
    for k in range(3):
        box(ax, cols[0] + 0.2, yB + 1.9 - k * 0.85, 1.3, 0.62, f"silo {['1', '2', 'N'][k]}", BLUE, size=8.4)
        box(ax, cols[0] + 2.1, yB + 1.9 - k * 0.85, 1.6, 0.62, f"expert {['1', '2', 'N'][k]}", GREEN, size=8.4, focus=True)
        arrow(ax, (cols[0] + 1.52, yB + 2.21 - k * 0.85), (cols[0] + 2.08, yB + 2.21 - k * 0.85))
    text(ax, cols[0] + 2.0, yB - 0.55, "same Stage I", size=8.6)
    for k in range(3):
        box(ax, cx + 0.2, yB + 1.9 - k * 0.85, 1.3, 0.62, f"silo {['1', '2', 'N'][k]}", BLUE, size=8.4)
        arrow(ax, (cx + 1.52, yB + 2.21 - k * 0.85), (cx + 2.78, yB + 1.36), colour=GREEN, lw=1.3)
    box(ax, cx + 2.8, yB + 1.0, 1.4, 0.72, "trusted\nserver", INK, size=8.6)
    text(ax, cx + 2.15, yB - 0.55, "each silo uploads its expert: 1 transfer", size=8.6)
    box(ax, rx + 0.1, yB + 0.05, 3.95, 2.6, None, INK, gid="container")
    text(ax, rx + 2.05, yB + 2.4, "on the server", size=8.6, weight="bold")
    for k, lab in enumerate(["1", "2", "N"]):
        box(ax, rx + 0.3, yB + 1.55 - k * 0.7, 1.1, 0.52, f"teacher {lab}", GREY, size=7.6, faded=True)
    box(ax, rx + 2.6, yB + 0.85, 1.25, 0.8, "student\n$v_\\theta$", GREEN, size=8.6, focus=True)
    arrow(ax, (rx + 1.42, yB + 1.25), (rx + 2.58, yB + 1.25), colour=INK)
    text(ax, rx + 2.75, yB + 0.4, "$\\hat x_t$ by ODE from each\nteacher (Eq. 15 to 17)", size=7.2)
    box(ax, ox + 0.15, yB + 1.0, 1.4, 0.72, "trusted\nserver", INK, size=8.6)
    for k in range(3):
        box(ax, ox + 3.3, yB + 1.9 - k * 0.85, 1.3, 0.62, f"silo {['1', '2', 'N'][k]}", BLUE, size=8.4)
        arrow(ax, (ox + 1.57, yB + 1.36), (ox + 3.28, yB + 2.21 - k * 0.85), colour=GREEN, lw=1.3)
    text(ax, ox + 2.3, yB - 0.55, f"the student goes back: sent per silo "
         f"{cm['table3_communication_per_client']['DCFM-B']} in all", size=8.6)
    for x in cols[1:]:
        for y in (yA + 0.9, yB + 1.2):
            arrow(ax, (x - 0.42, y), (x - 0.1, y), colour="#9aa3ab", lw=3.0, ms=13)
    save(fig, "01d-what-crosses-between-silos")


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    which = sys.argv[1:] or ["a", "b", "c", "d"]
    for w in which:
        {"a": fig_a, "b": fig_b, "c": fig_c, "d": fig_d}[w]()
