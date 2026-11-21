"""One picture for the stock take: the two routes that try to stop the product of two labels from fusing,
side by side, in the zoom ladder's recorded colours (blue inputs, green the network, red outputs, orange
the thing each route changes).

  left   inclusive labels (this chat's route): change what each label is TRAINED on
  right  classifier guidance (the other session's route): leave training alone, PUSH the drawing

The digit tiles are real MNIST digits pasted by hand (illustration), not model outputs.

Usage:
    python3 -I scripts/two_routes_figure.py --mnist <dir with train-images-idx3-ubyte> --out <png>
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
import mnist_training_sheets as m  # noqa: E402

BLUE, GREEN, RED, ORANGE, GREY, INK = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88", "#20313c"
FILL = {BLUE: "#eaf0fb", GREEN: "#eaf6ee", RED: "#fdeeee", ORANGE: "#fdf1e4", GREY: "#f1f3f5"}


def box(ax, x, y, w, h, col, text, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", facecolor=FILL[col], edgecolor=col, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight="bold" if bold else "normal")


def arrow(ax, a, b, col=GREY):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=col))


def tile(ax, img, x, y, w, h, title, col):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(x, x + w, y, y + h), zorder=3)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fill=False, edgecolor=col, lw=2, zorder=4))
    ax.text(x + w / 2, y - 0.18, title, ha="center", va="top", fontsize=9, color=col)


def fused(cv):
    c = np.zeros((32, 64), np.uint8)
    c[2:30, 18:46] = np.maximum(cv.digit(7), cv.digit(8))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    x, y = m.load(a.mnist)
    cv = m.Canvas(x, y, np.random.default_rng(3))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    titles = ["This chat: inclusive labels\nchange what each label is trained on",
              "The other session: classifier guidance\nleave training alone, push the drawing"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 10), ax.set_ylim(0, 9.2), ax.axis("off")
        ax.set_title(t, fontsize=13, weight="bold", color=INK, loc="left")

    # Left: the label rule is the thing that changes.
    L = axes[0]
    tile(L, cv.two(7, 8), 0.2, 6.6, 2.2, 1.1, "a training canvas holding 7 and 8", BLUE)
    box(L, 3.2, 6.3, 6.4, 1.7, ORANGE, 'trained under "7 and 8" always,\nand ALSO under "7" and "8"\nas often as the dial says (0 to 1)', 10, True)
    arrow(L, (2.45, 7.15), (3.15, 7.15), ORANGE)
    box(L, 3.2, 4.3, 6.4, 1.3, GREEN, "one small network trained from scratch\n15 copies: 5 dial settings x 3 seeds", 10)
    arrow(L, (6.4, 6.25), (6.4, 5.65))
    box(L, 3.2, 2.4, 6.4, 1.3, GREEN, 'draw "7" x "8" (the product)\nplain, and with 20 extra nudges a step', 10)
    arrow(L, (6.4, 4.25), (6.4, 3.75))
    tile(L, cv.two(7, 8), 3.4, 0.4, 2.2, 1.1, "hoped for: two digits", GREEN)
    tile(L, fused(cv), 7.0, 0.4, 2.2, 1.1, "the failure: one fused digit", RED)
    arrow(L, (6.4, 2.35), (4.6, 1.6)), arrow(L, (6.4, 2.35), (8.0, 1.6))
    L.text(0.2, 4.9, "question:\ndoes training a pair\nunder each single label\nstop the fusing?", fontsize=10, color=INK, va="center")

    # Right: the push at drawing time is the thing that changes.
    R = axes[1]
    tile(R, cv.two(7, 8), 0.2, 6.6, 2.2, 1.1, "training canvases (labels fixed)", BLUE)
    box(R, 3.2, 6.3, 6.4, 1.7, GREY, "trained once, then frozen", 10)
    arrow(R, (2.45, 7.15), (3.15, 7.15))
    box(R, 0.1, 2.1, 2.8, 1.9, ORANGE, "a checker learns:\nreward two\nseparate things,\npenalise one\nfused thing", 9.5, True)
    box(R, 3.2, 2.4, 6.4, 1.3, GREEN, 'draw "7" x "8", and at every step add\nthe checker\'s push (its gradient)', 10)
    arrow(R, (6.4, 6.25), (6.4, 3.75), GREY)
    arrow(R, (2.95, 3.05), (3.15, 3.05), ORANGE)
    tile(R, cv.two(7, 8), 3.4, 0.4, 2.2, 1.1, "hoped for: two digits", GREEN)
    tile(R, fused(cv), 7.0, 0.4, 2.2, 1.1, "the failure: one fused digit", RED)
    arrow(R, (6.4, 2.35), (4.6, 1.6)), arrow(R, (6.4, 2.35), (8.0, 1.6))
    R.text(0.2, 5.0, "question:\ncan a push at\ndrawing time\nstop the fusing?", fontsize=10, color=INK, va="center")

    fig.text(0.5, 0.005, "Orange marks what each route changes. Both share the same failure (one fused digit) and the "
             "same success (two separate digits). Digit tiles are pasted by hand, not model outputs.",
             ha="center", fontsize=10, color=GREY)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
