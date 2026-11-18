"""Two figures for round 8 of the MNIST stage, the bar, in the zoom ladder's recorded style
(cheat-sheet devices, boxed formulas; blue inputs, green the hoped-for outcome, red failure,
orange in between, grey for fixed rules). No data: every number is a threshold written before
anything runs.

  the-bar-as-a-ruler.png         0 to 256 seeds, with the dial-0 ceiling (12), the null line (39)
                                 and the support line (128), and the zones between them
  the-order-the-checks-run.png   the checks in the order step 4 runs them, ending in the verdict

Usage:
    python3 -I scripts/mnist_bar_figures.py --out <dir>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

BLUE, GREEN, RED, ORANGE, GREY, INK = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88", "#20313c"
FILL = {BLUE: "#eaf0fb", GREEN: "#eaf6ee", RED: "#fdeeee", ORANGE: "#fdf1e4", GREY: "#f1f3f5"}


def box(ax, x, y, w, h, colour, text, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=FILL[colour],
                                edgecolor=colour, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight="bold" if bold else "normal")


def arrow(ax, a, b, colour=GREY, text=None, side=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=colour))
    if text:
        ax.text((a[0] + b[0]) / 2 + side, (a[1] + b[1]) / 2, text, fontsize=9.5, color=colour,
                ha="left" if side >= 0 else "right", va="center")


def ruler(out):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    zones = [(0, 39, RED, "null"),
             (39, 128, ORANGE, "inconclusive"),
             (128, 256, GREEN, "supported")]
    for a, b, c, t in zones:
        ax.add_patch(Rectangle((a, 0), b - a, 1, facecolor=FILL[c], edgecolor=c, lw=1.2))
        ax.text((a + b) / 2, 0.5, t, ha="center", va="center", fontsize=11, color=c, weight="bold")
    ax.add_patch(Rectangle((0, -0.55), 12, 0.4, facecolor=FILL[BLUE], edgecolor=BLUE, lw=1.2))
    ax.annotate("dial 0 must stay at 12 or fewer (5%)", (12, -0.35), xytext=(30, -0.35),
                fontsize=10.5, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE))
    for v in (12, 39, 128, 256):
        ax.plot([v, v], [1.0, 1.12], color=INK, lw=1)
        ax.text(v, 1.17, str(v), ha="center", va="bottom", fontsize=11, color=INK)
    ax.text(0, -1.0, "null: no counting setting averages 39 or more · supported: 128 or more in all three trainings", fontsize=10.5, color=INK)
    ax.text(0, 1.55, "Corrected 7 × 8 product: seeds called \"two separate right digits\", out of 256",
            fontsize=12.5, weight="bold", color=INK)
    ax.set_xlim(-5, 262), ax.set_ylim(-1.2, 1.8)
    ax.axis("off")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def order(out):
    fig, ax = plt.subplots(figsize=(13, 9.6))
    box(ax, 0.5, 8.4, 5.2, 1.0, BLUE, "1. Can the judge be trusted?\n95% per call on pasted tiles, 58 of 64 eye calls", size=9.5)
    box(ax, 0.5, 6.6, 5.2, 1.0, BLUE, "2. Does dial 0 already compose?\nplain 7 × 8 at dial 0: 12 or fewer of 256")
    box(ax, 0.5, 4.6, 5.2, 1.2, BLUE, "3. Which dial settings count?\n\"7\" and \"8\" alone within 5 points of\ntheir training lines, all three trainings")
    box(ax, 0.5, 2.6, 5.2, 1.0, BLUE, "4. Read the corrected 7 × 8 product\nat every counting setting")
    arrow(ax, (3.1, 8.35), (3.1, 7.65), GREY, "yes", 0.1)
    arrow(ax, (3.1, 6.55), (3.1, 5.85), GREY, "yes", 0.1)
    arrow(ax, (3.1, 4.55), (3.1, 3.65), GREY)

    box(ax, 7.2, 8.4, 4.6, 1.0, RED, "stop and ask:\nfix or swap the judge")
    box(ax, 7.2, 6.6, 4.6, 1.0, RED, "stop and show the counts:\nlower the share, or another pair")
    arrow(ax, (5.75, 8.9), (7.15, 8.9), RED)
    ax.text(6.45, 9.1, "no", color=RED, fontsize=9.5, ha="center")
    arrow(ax, (5.75, 7.1), (7.15, 7.1), RED)
    ax.text(6.45, 7.3, "no", color=RED, fontsize=9.5, ha="center")

    box(ax, 7.2, 4.1, 4.6, 1.2, GREY, "floors, of 256, dial 0 / 0.1 / 0.3 / 0.6 / 1\n\"7\": 244 / 231 / 217 / 207 / 199\n\"8\": 244 / 230 / 214 / 203 / 196", size=9.5)
    arrow(ax, (5.75, 5.2), (7.15, 4.9), GREY)

    box(ax, 0.2, 0.4, 3.6, 1.2, GREEN, "supported\n≥ 128 in all three trainings", bold=True)
    box(ax, 4.2, 0.4, 3.6, 1.2, ORANGE, "inconclusive\nanything between", bold=True)
    box(ax, 8.2, 0.4, 3.6, 1.2, RED, "null\nno setting averages ≥ 39", bold=True)
    for x in (2.0, 6.0, 10.0):
        arrow(ax, (3.1, 2.55), (x, 1.65), GREY)

    ax.text(12.4, 3.1, "beside the verdict, not deciding it:\n· plain 7 × 8 also ≥ 128?\n  then the corrector is not needed\n"
            "· corrected 4 × 7 ≥ 39 in all three,\n  and ≥ 3× its dial-0 count?\n  then it reaches an unseen pair",
            fontsize=9.5, color=INK, va="center")
    ax.set_xlim(0, 16.2), ax.set_ylim(0, 9.8)
    ax.axis("off")
    ax.set_title("The order step 4 runs its checks", fontsize=13, weight="bold", color=INK, loc="left")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ruler(args.out / "the-bar-as-a-ruler.png")
    order(args.out / "the-order-the-checks-run.png")
    print("wrote two figures to", args.out)


if __name__ == "__main__":
    main()
