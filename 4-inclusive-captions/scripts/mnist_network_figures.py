"""Two figures for the MNIST stage, in the zoom ladder's recorded style (cheat-sheet devices: thin
outlines, white space, boxed formulas; colour key blue inputs, green hidden units, red outputs,
orange for the push, grey for what is fixed).

  the-label-conditional-unet.png   the network: canvas and label in, predicted noise out
  one-sampling-step-of-the-product.png
                                   one reverse step: three passes of the same network, the guided
                                   sum, and the corrector's 20 Langevin steps in their own lane

Sizes on the canvas path are real (1 x 32 x 64, halved twice). Channel widths are not settled
(round 4 left them to the smoke run), so they are written c, 2c, 4c. The noise tiles are real
Gaussian noise; the digit tiles are pasted MNIST digits, marked as illustration.

Usage:
    python3 -I scripts/mnist_network_figures.py --mnist <dir with train-images-idx3-ubyte> --out <dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_training_sheets import Canvas, load  # noqa: E402

BLUE, GREEN, RED, ORANGE, GREY, INK = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88", "#20313c"
FILL = {BLUE: "#eaf0fb", GREEN: "#eaf6ee", RED: "#fdeeee", ORANGE: "#fdf1e4", GREY: "#f1f3f5"}


def box(ax, x, y, w, h, colour, text=None, size=9, glow=False):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.08, y - 0.08), w + 0.16, h + 0.16,
                                    boxstyle="round,pad=0.02", color=colour, alpha=0.15, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=FILL[colour],
                                edgecolor=colour, lw=1.3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, a, b, colour=GREY, lw=1.1, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=lw, color=colour,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def tile(ax, img, x, y, w, h, title=None, colour=INK, cmap="gray"):
    ax.imshow(img, cmap=cmap, extent=(x, x + w, y, y + h), interpolation="nearest", zorder=3)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fill=False, edgecolor=colour,
                                lw=1.2, zorder=4))
    if title:
        ax.text(x + w / 2, y + h + 0.15, title, ha="center", va="bottom", fontsize=9, color=colour)


def formula(ax, x, y, text, colour=INK, size=11):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=colour, lw=0.9))


def unet_figure(cv, out):
    fig, ax = plt.subplots(figsize=(14, 7.6))
    noisy = np.clip(0.5 * cv.two(7, 8) / 255 + 0.5 * np.random.default_rng(17).standard_normal((32, 64)),
                    -1.5, 1.5)
    tile(ax, noisy, 0.0, 4.2, 2.0, 1.0, "noisy canvas x_t\n1 x 32 x 64", BLUE)
    tile(ax, np.random.default_rng(3).standard_normal((32, 64)), 11.2, 4.2, 2.0, 1.0,
         "predicted noise\n1 x 32 x 64", RED)

    # (x, y, height, label, glow): down, bottleneck, up.
    blocks = [(3.0, 3.6, 2.2, "32 x 64\nc channels", False), (4.6, 2.3, 1.6, "16 x 32\n2c", False),
              (6.2, 1.2, 1.0, "8 x 16\n4c", True), (7.8, 2.3, 1.6, "16 x 32\n2c", False),
              (9.4, 3.6, 2.2, "32 x 64\nc channels", False)]
    for x, y, h, t, g in blocks:
        box(ax, x, y, 0.55, h, GREEN, glow=g)
        ax.text(x + 0.27, y - 0.2, t, ha="center", va="top", fontsize=8.5, color=INK)
    for (x, y, h, _, _), (x2, y2, h2, _, _) in zip(blocks, blocks[1:]):
        arrow(ax, (x + 0.6, y + h / 2), (x2 - 0.05, y2 + h2 / 2), GREY)
    arrow(ax, (3.6, 5.6), (9.35, 5.6), GREY, lw=0.9, ls="--")
    arrow(ax, (5.2, 3.7), (7.75, 3.7), GREY, lw=0.9, ls="--")
    ax.text(6.47, 5.85, "skip connections carry each level across the U", ha="center", fontsize=9,
            color=GREY)
    arrow(ax, (2.05, 4.7), (2.95, 4.7), BLUE)
    arrow(ax, (10.0, 4.7), (11.15, 4.7), RED)

    # The condition path, in its own lane below the U.
    tile(ax, cv.one(7), 0.0, -0.1, 1.6, 0.8, 'label "7" (id 7)', BLUE)
    box(ax, 2.1, -0.15, 2.7, 0.9, BLUE, "label table, 56 rows:\n0-9, 45 joint labels, blank", size=8.5)
    box(ax, 2.1, -1.55, 2.7, 0.8, BLUE, "noise level t\nto a time vector", size=8.5)
    box(ax, 5.6, -0.85, 1.4, 0.9, BLUE, "add the two\nvectors", size=8.5, glow=True)
    arrow(ax, (1.65, 0.3), (2.05, 0.3), BLUE)
    arrow(ax, (4.85, 0.3), (5.55, -0.3), BLUE)
    arrow(ax, (4.85, -1.15), (5.55, -0.5), BLUE)
    for x, y, h, _, _ in blocks:
        arrow(ax, (6.3, 0.1), (x + 0.27, y - 0.75), BLUE, lw=0.8, rad=0.12 if x > 6.3 else -0.12)
    ax.text(7.4, -0.75, "the summed vector enters every block", fontsize=9, color=BLUE)

    formula(ax, 6.6, 6.9, r"one network, one label at a time:  $\varepsilon_\theta(x_t,\ t,\ \mathrm{label})$"
            r"   ·   training loss  $\|\varepsilon - \varepsilon_\theta(x_t, t, \mathrm{label})\|^2$")
    ax.text(6.6, -2.05, "Canvas sizes are real. Channel widths c, 2c, 4c are not settled yet; the "
            "smoke run fixes them. The digit tiles are pasted MNIST digits (illustration).",
            ha="center", fontsize=9, color=GREY)
    ax.set_xlim(-0.3, 13.5), ax.set_ylim(-2.3, 7.4), ax.set_aspect("equal"), ax.axis("off")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def step_figure(cv, out):
    fig, ax = plt.subplots(figsize=(16, 8.4))
    rng = np.random.default_rng(17)
    xt = rng.standard_normal((32, 64))
    tile(ax, xt, 0.0, 4.0, 2.0, 1.0, "canvas at level k\n(seed 17's noise)", BLUE)

    rows = [(6.0, "blank", GREY), (4.5, '"7"', BLUE), (3.0, '"8"', BLUE)]
    for y, lab, col in rows:
        box(ax, 3.0, y - 0.3, 2.4, 0.8, GREEN, f"same network\nlabel {lab}", size=9)
        arrow(ax, (2.05, 4.5), (2.95, y + 0.1), BLUE, lw=0.9)
        arrow(ax, (5.45, y + 0.1), (6.25, y + 0.1), RED)
        ax.text(6.35, y + 0.1, {"blank": r"$\varepsilon_\varnothing$", '"7"': r"$\varepsilon_7$",
                                '"8"': r"$\varepsilon_8$"}[lab], fontsize=14, va="center", color=RED)

    # The guided sum as arrows tip to tail.
    o = np.array([8.2, 3.2])
    e0, e7, e8 = np.array([1.0, 0.2]), np.array([1.3, 0.9]), np.array([1.5, -0.4])
    p1 = o + e0
    p2 = p1 + 3 * (e7 - e0)
    p3 = p2 + 3 * (e8 - e0)
    arrow(ax, tuple(o), tuple(p1), GREY, lw=2.2)
    arrow(ax, tuple(p1), tuple(p2), BLUE, lw=2.2)
    arrow(ax, tuple(p2), tuple(p3), BLUE, lw=2.2)
    arrow(ax, tuple(o), tuple(p3), RED, lw=2.6)
    ax.text(*(o + e0 / 2 + [0, -0.35]), "blank", fontsize=9, color=GREY, ha="center")
    ax.text(*((p1 + p2) / 2 + [-0.75, 0.1]), r"$3(\varepsilon_7-\varepsilon_\varnothing)$", fontsize=10,
            color=BLUE, ha="center")
    ax.text(*((p2 + p3) / 2 + [0.95, 0.15]), r"$3(\varepsilon_8-\varepsilon_\varnothing)$", fontsize=10,
            color=BLUE, ha="center")
    ax.text(*((o + p3) / 2 + [0.3, -0.45]), "the product's prediction", fontsize=9.5, color=RED,
            ha="center")
    formula(ax, 11.0, 7.0, r"$\hat\varepsilon = \varepsilon_\varnothing + 3(\varepsilon_7-\varepsilon_\varnothing)"
            r" + 3(\varepsilon_8-\varepsilon_\varnothing)$", RED)

    tile(ax, cv.two(7, 8), 14.6, 4.0, 2.0, 1.0, "canvas at level k-1\n(illustration)", BLUE)
    arrow(ax, (13.2, 4.5), (14.55, 4.5), RED, lw=1.4)
    ax.text(13.85, 4.75, "one reverse\nstep", fontsize=8.5, color=INK, ha="center", va="bottom")

    # The corrector, in its own lane below.
    box(ax, 3.0, 0.0, 10.2, 1.25, ORANGE)
    ax.text(8.1, 0.95, "corrector rows only: before stepping down, 20 Langevin steps at the same level",
            ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.text(8.1, 0.38, r"repeat 20 times:  $x \leftarrow x + \frac{\eta}{2}\,s(x) + \sqrt{\eta}\,z$,"
            r"   $s(x) = -\hat\varepsilon(x)/\sigma_k$,   $\eta = 0.05\,\sigma_k^2$,   $z$ fresh noise",
            ha="center", fontsize=10.5, color=INK)
    arrow(ax, (12.4, 1.3), (12.4, 2.4), ORANGE, lw=1.1)
    arrow(ax, (3.6, 2.4), (3.6, 1.3), ORANGE, lw=1.1)
    ax.text(16.6, 0.6, "200 levels x (3 passes, plus\n20 x 3 on corrector rows):\n"
            "about 21 times the passes", fontsize=9, color=GREY, ha="right", va="center")
    ax.set_xlim(-0.3, 16.9), ax.set_ylim(-0.4, 7.5), ax.set_aspect("equal"), ax.axis("off")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    x, y = load(args.mnist)
    cv = Canvas(x, y, np.random.default_rng(9))
    args.out.mkdir(parents=True, exist_ok=True)
    unet_figure(cv, args.out / "the-label-conditional-unet.png")
    step_figure(cv, args.out / "one-sampling-step-of-the-product.png")
    print("wrote", args.out / "the-label-conditional-unet.png")
    print("wrote", args.out / "one-sampling-step-of-the-product.png")


if __name__ == "__main__":
    main()
