"""Level 0: DCFM as one annotated overview, a digit's attribute followed from the silo that holds it to an unseen output.

Writes figures/00-overview.png and .svg. Devices from ~/.claude/skills/zoom-ladder/scripts/annotated_explainer.py
(brackets, cards, a traced window, a zoomed inset, a readable result); colours from ladder.json's colour key. Every
number comes from dcfm_numbers.json. The digits are real MNIST test images tinted (digits.py); which silo holds which
digits is illustrative. figcheck.check refuses the figure if any line crosses a node, box, picture or label.
Usage: ~/.claude/venvs/figures/bin/python make_overview.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/.claude/skills/zoom-ladder/scripts"))
from annotated_explainer import ANNOTATED, bracket, line, trace  # noqa: E402
import digits as D  # noqa: E402
from figcheck import check  # noqa: E402

N = json.load(open(os.path.join(HERE, "dcfm_numbers.json")))
K = N["colour_key"]
S = dict(ANNOTATED, cin=K["inputs"], chid=K["experts_and_hidden_units"], cout=K["outputs"], accent=K["interaction_term_penalty_gradient_push"],
         fixed=K["fixed_frozen_off"], fin="#e4edfb", fhid="#dcefe2", fout="#fbe3e3")
GREY, GREY_FILL, ORANGE = S["fixed"], "#e9ecef", S["accent"]
INK = S["ink"]


def card(ax, x, y, w, h, text, ec, fc, size=10, weight="normal", gid=None, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09", lw=1.5, edgecolor=ec, facecolor=fc,
                       zorder=3, linestyle=ls)
    if gid:
        p.set_gid(gid)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=INK, zorder=4,
                linespacing=1.3)


def glow(ax, x, y, w, h):
    for pad, a in ((0.16, 0.10), (0.09, 0.18)):
        p = FancyBboxPatch((x - pad, y - pad), w + 2 * pad, h + 2 * pad, boxstyle="round,pad=0.02,rounding_size=0.15",
                           color="#7fd39a", alpha=a, lw=0, zorder=2)
        p.set_gid("free")
        ax.add_patch(p)


def tile(ax, img, x, y, w, ec):
    ax.imshow(img, extent=(x, x + w, y, y + w), zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, w, fill=False, ec=ec, lw=1.6, zorder=4))


def text(ax, x, y, s, size=9, color=INK, ha="center", **kw):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, zorder=5, linespacing=1.25, **kw)


def main():
    fig, ax = plt.subplots(figsize=(20.6, 12.0), dpi=120)
    ax.set_xlim(0, 20.6)
    ax.set_ylim(0, 12.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(S["bg"])

    text(ax, 10.3, 11.55, "DCFM: silos that never share images train experts, swap the models, retrain them with a penalty, "
         "then compose a digit and a colour no silo holds", size=14.5, weight="bold")
    for x0, x1, name in ((0.3, 3.6, "Data silos"), (3.9, 6.9, "Local experts"), (7.2, 8.95, "Model exchange"),
                         (9.2, 13.4, "Retraining with the penalty"), (13.7, 17.4, "Composed sampling"), (17.7, 20.35, "Output")):
        bracket(ax, x0, x1, 10.2, name, S)

    # Data silos: real MNIST test digits, tinted.
    lanes = {1: 8.55, 2: 6.05}
    silo_digits = {1: [(7, "red"), (2, "red"), (9, "cyan")], 2: [(4, "green"), (4, "blue"), (0, "yellow")]}
    captions = {1: "red digits, but no 4", 2: "4s, but never red"}
    for a, yc in lanes.items():
        card(ax, 0.3, yc - 1.05, 3.3, 2.1, None, S["cin"], S["fin"])
        text(ax, 1.95, yc + 0.78, f"silo {a} of 10", size=10.5, weight="bold")
        for j, (d, c) in enumerate(silo_digits[a]):
            img, _ = D.digit(d, c, nth=j)
            tile(ax, img, 0.55 + j * 1.0, yc - 0.42, 0.82, S["cin"])
        text(ax, 1.95, yc - 0.78, captions[a], size=9)
    cm = N["printed"]["colored_mnist"]
    text(ax, 1.95, 4.35, f"{cm['combinations']} digit-colour pairs over {cm['clients_n']} silos;\n"
         f"half of them on no silo (C = {cm['total_coverage_C']});\nnon-IID split: Dirichlet $\\alpha$ = {cm['dirichlet_alpha_non_iid']}"
         f" (p. 6 to 7).\nWhich silo holds which digits\nis illustrative.", size=8.6)

    # Local experts (Stage I).
    for a, yc in lanes.items():
        card(ax, 4.0, yc - 0.7, 2.8, 1.4, f"expert {a}   $v_\\theta^{{({a})}}$\nnetwork not stated\nin the paper", S["chid"], S["fhid"], size=9.8)
        line(ax, [(3.62, yc), (3.98, yc)], S["edge"])
    text(ax, 5.4, 4.35, "Stage I, each silo alone (Eq. 9, 11):\nlabels sometimes masked to one\nattribute or none, a straight path\n"
         "from noise to image, plus $\\lambda$ times\nthe local penalty ($\\lambda$ not stated)", size=8.6)

    # Model exchange.
    card(ax, 7.6, 5.35, 0.95, 3.9, None, INK, "white", ls="--")
    ax.text(8.075, 7.3, "all-gather of $\\theta$: every silo\nreceives every expert", rotation=90, ha="center", va="center",
            fontsize=9.2, color=INK, zorder=5, linespacing=1.2)
    for yc in lanes.values():
        line(ax, [(6.82, yc), (7.58, yc)], S["edge"])
        line(ax, [(8.57, yc), (9.28, yc)], S["edge"])
    text(ax, 8.075, 4.75, "models cross,\nimages never\n(Constraint 1, p. 2)", size=8.4)

    # Retraining (DCFM-A): frozen peer copy on the left, the silo's own expert glowing on the right.
    for a, yc in lanes.items():
        peer = 2 if a == 1 else 1
        card(ax, 9.3, yc - 0.7, 1.9, 1.4, f"expert {peer}\nfrozen copy", GREY, GREY_FILL, size=9.6)
        glow(ax, 11.45, yc - 0.7, 1.85, 1.4)
        card(ax, 11.45, yc - 0.7, 1.85, 1.4, f"expert {a}\ntrained again\n$v_\\theta^{{({a})}}$", S["chid"], S["fhid"], size=9.6)
    text(ax, 11.3, 7.3, "penalty (Eq. 12 to 14): each silo's joint prediction is\npulled onto a sum that uses its peer's frozen expert",
         size=8.6, color=ORANGE, weight="bold")
    # The traced window: silo 2's 4, half drawn, at the penalty.
    four, _ = D.digit(4, "green", nth=0)
    xt = D.noised(four, 0.55, seed=3)
    tile(ax, xt, 12.15, 4.2, 0.75, S["cin"])
    ax.add_patch(Rectangle((12.15 + 0.30, 4.2 + 0.33), 0.16, 0.16, fill=False, ec=S["cout"], lw=1.5, zorder=6))
    text(ax, 11.3, 4.55, "$x_t$: silo 2's 4, half drawn\n(t = 0.55; t runs from 0 noise\nto 1 image)", size=8.4, ha="right")

    # The inset: inside the penalty at one x_t, arrows tip to tail from the toy's worked point.
    ix, iy, iw, ih = 6.0, 0.3, 7.4, 3.4
    frame = FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0.02,rounding_size=0.1", lw=1.5, edgecolor=S["chid"],
                           facecolor="#f4faf5", zorder=1)
    frame.set_gid("container")
    ax.add_patch(frame)
    for xx in (11.45, 13.3):
        ax.plot([xx, xx], [lanes[2] - 0.72, iy + ih + 0.02], color=S["chid"], lw=1.0, ls=":", zorder=1)
    text(ax, ix + 0.15, iy + ih - 0.2, "Inside the penalty, at one half-drawn picture $x_t$", size=9.6, color=S["chid"], weight="bold", ha="left")
    w = N["worked"]["inset_iid_label_1_2"]
    sc = 0.8
    o = np.array([12.55, 3.05])
    u, vl, vb, j = (np.array(w[k]) for k in ("v_uncond", "v_left", "v_bottom", "joint"))
    p1 = o + sc * u
    p2 = p1 + sc * (vl - u)
    p3 = p2 + sc * (vb - u)
    pj = o + sc * j
    line(ax, [tuple(o), tuple(p1)], GREY, lw=2.0, ms=12)
    line(ax, [tuple(p1), tuple(p2)], GREY, lw=2.0, ms=12)
    line(ax, [tuple(p2), tuple(p3)], S["chid"], lw=2.0, ms=12)
    line(ax, [tuple(o), tuple(pj)], S["chid"], lw=3.0, ms=14)
    line(ax, [tuple(p3), tuple(pj)], ORANGE, lw=2.4, ms=13)
    ax.plot(*o, "o", color=INK, ms=5, zorder=6)
    trace(ax, (12.15 + 0.375, 4.2), tuple(o + np.array([-0.01, 0.06])), S["cout"])
    text(ax, 12.88, 2.5, "frozen:\n$\\bar v(x_t)$", size=8.0, color=GREY)
    text(ax, 10.55, 3.07, "frozen: $+(\\bar v(x_t,y_1)-\\bar v)$", size=8.0, color=GREY, ha="right")
    text(ax, 11.95, 1.75, "own:\n$+(v(x_t,y_2)-\\bar v)$\nthe sum $z$ ends here", size=7.8, color=S["chid"], ha="left")
    text(ax, 10.52, 1.2, "own joint $v(x_t,\\mathbf{y})$", size=8.2, color=S["chid"], ha="right", weight="bold")
    text(ax, 11.3, 0.72, "$I_t$, |$I_t$| = " + f"{w['I_t_norm']:.2f}" + ":\npenalty pulls\nthe tips together", size=8.0, color=ORANGE, ha="left", weight="bold")
    text(ax, 7.85, 2.5, "$z=\\bar v^{(r_0)}+\\sum_i(\\bar v^{(r_i)}(x_t,y_i)-\\bar v^{(r_0)})$\n"
         "penalty $=\\Vert v^{(a)}(x_t,\\mathbf{y})-z\\Vert^2$  (Eq. 12)", size=8.6)
    text(ax, 7.85, 0.68, "arrow lengths: the 2-D toy's IID silo at\none point, label bottom-left, scale 0.8\n(dcfm_numbers.json, inset row)", size=7.8, style="italic")

    # Composed sampling.
    card(ax, 13.75, 5.25, 3.6, 4.1, None, INK, "white", gid="container")
    text(ax, 15.55, 8.95, "unseen pair: a 4 in red", size=10.5, weight="bold")
    text(ax, 15.55, 8.15, "$\\hat v(x_t)=\\bar v(x_t)+[v^{(2)}(x_t,4)-\\bar v]$\n$+\\,[v^{(1)}(x_t,\\mathrm{red})-\\bar v]$  (Eq. 5, $w_i$ = 1)",
         size=9.0)
    text(ax, 15.55, 7.35, "solver and step count not stated", size=8.2, style="italic")
    red4, _ = D.digit(4, "red", nth=5)
    for k, (tt, lab) in enumerate(((0.0, "t = 0"), (0.5, "t = 0.5"), (0.85, "t = 0.85"))):
        tile(ax, D.noised(red4, tt, seed=9), 13.95 + k * 1.15, 5.95, 0.85, S["cin"] if k < 2 else S["cout"])
        text(ax, 14.375 + k * 1.15, 5.6, lab, size=8.2)
    for k in range(2):
        line(ax, [(14.83 + k * 1.15, 6.375), (15.07 + k * 1.15, 6.375)], S["edge"], ms=9)
    line(ax, [(13.32, lanes[1]), (13.73, lanes[1])], S["edge"])
    line(ax, [(13.32, lanes[2] + 0.5), (13.73, lanes[2] + 0.5)], S["edge"])
    text(ax, 13.52, lanes[1] + 0.22, "red", size=8.2)
    text(ax, 13.52, lanes[2] + 0.72, "4", size=8.2)

    # Output and the readable result.
    line(ax, [(17.37, 6.375), (17.88, 6.375)], S["edge"])
    tile(ax, red4, 17.95, 5.7, 1.35, S["cout"])
    text(ax, 19.75, 6.375, "a red 4", size=10, weight="bold")
    text(ax, 19.0, 5.15, "real MNIST 4, tinted:\nillustrative, not a DCFM sample", size=8.0, style="italic")
    t1 = cm["table1"]
    rows = [("DFD", t1["DFD"][1]), ("DDM", t1["DDM"][1]), ("FedFlow", t1["FedFlow"][1]), ("DCFM-B", t1["DCFM-B"][1]), ("DCFM-A", t1["DCFM-A"][1])]
    text(ax, 19.0, 9.6, "FID on pairs no silo saw,\nIID split (Table 1, p. 7;\nlower is better)", size=8.6, weight="bold")
    for k, (name, v) in enumerate(rows):
        y = 8.75 - k * 0.36
        text(ax, 18.55, y, name, size=8.2, ha="right")
        col = S["cout"] if name.startswith("DCFM") else "#c3cad1"
        ax.add_patch(Rectangle((18.65, y - 0.12), v * 0.045, 0.24, facecolor=col, edgecolor=INK, lw=0.6, zorder=3))
        text(ax, 18.72 + v * 0.045, y, f"{v:.2f}", size=8.0, ha="left")

    # Two cards: DCFM-B and the cost.
    tc = N["printed"]["cost"]
    card(ax, 13.75, 1.95, 6.6, 1.75, "DCFM-B, the other variant: every silo uploads its expert to a\n"
         "trusted server, which distils one student on the experts' own\nsamples with the same penalty (Eq. 15 to 17) and sends it back:\n"
         f"{tc['table3_communication_per_client']['DCFM-B']} transfers per silo, against R x N for DCFM-A "
         f"(R = {tc['dcfm_a_rounds_R'].split(' (')[0]} rounds; Table 3, p. 9)", GREY, GREY_FILL, size=8.7)
    t4 = tc["table4_per_step"]
    b0 = t4["Baselines (FL, DDM, DFD)"]
    card(ax, 13.75, 0.3, 6.6, 1.45, "Cost per training step, batch B (Table 4, p. 9):\n"
         f"baselines {b0['fwd']} forward and {b0['bwd']} backward;\n"
         f"DCFM-A {t4['DCFM-A']['fwd']} forward and {t4['DCFM-A']['bwd']} backward;\n"
         f"DCFM-B {t4['DCFM-B']['fwd']} forward and {t4['DCFM-B']['bwd']} backward, on one server",
         GREY, "white", size=8.7)

    # Bottom left: what the penalty assumes.
    card(ax, 0.3, 0.3, 5.45, 3.4, "What the penalty assumes (Eq. 1, p. 2):\n"
         "given the image, the attributes are\nindependent, so the joint prediction\n"
         "should equal the unconditional one\nplus one difference per attribute.\n"
         "The penalty trains that in, so anything\nthat needs both attributes at once\n"
         "(one changing how the other looks)\nis what it removes.", S["chid"], "#f4faf5", size=9.2)

    check(fig, "00-overview")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, "figures", f"00-overview.{ext}"), bbox_inches="tight", facecolor=S["bg"])
    print("wrote figures/00-overview.png and .svg")


if __name__ == "__main__":
    main()
