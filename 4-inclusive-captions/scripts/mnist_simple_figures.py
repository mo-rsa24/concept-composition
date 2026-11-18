"""Three one-idea figures for rounds 6 and 7 of the MNIST stage, kept to as few words as possible.

  simple-why-not-a-flat-90.png       the "7" label's training line against the flat 90% bar, with 20
                                     real training draws labelled "7" at dial 1
  simple-what-a-leak-looks-like.png  the "7" training line and two MADE-UP example dots: one on the
                                     line (fine), one far below it (a leak). No data.
  simple-how-the-judge-calls-a-tile.png
                                     three pasted canvases, the ink patches found on each (real flood
                                     fill), and the call; the digit readings are what was pasted

Every tile is real MNIST digits pasted by hand. No model has been trained.

Usage:
    python3 -I scripts/mnist_simple_figures.py --mnist <dir with train-images-idx3-ubyte> --out <dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mnist_training_sheets as m  # noqa: E402

BLUE, RED, GREEN, GREY, INK = "#1f5fd0", "#c0262d", "#1d7a3e", "#6b7a88", "#20313c"
DIALS = list(m.DIALS)
SEVEN = [100.0, 94.9, 89.7, 85.5, 82.4]      # from mnist_leak_read_figures.py, 400,000 draws


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9d0d6")
    ax.tick_params(colors=GREY, labelsize=10)
    ax.set_xticks(DIALS)
    ax.set_xticklabels([f"{d:g}" for d in DIALS])
    ax.set_xlabel("dial", fontsize=11, color=INK)


def frame(ax, colour, lw=2.4):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour), s.set_linewidth(lw)


def why_not_flat(cv, ylab, out):
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    clean(ax)
    ax.axhline(90, color=GREY, lw=1.5, ls=(0, (4, 3)))
    ax.text(1.0, 90.6, "flat 90% bar", color=GREY, fontsize=11, va="bottom", ha="center")
    ax.plot(DIALS, SEVEN, color=BLUE, lw=2.5, marker="o", ms=8)
    for d, v in zip(DIALS, SEVEN):
        ax.annotate(f"{v:g}%", (d, v), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=11, color=INK)
    ax.set_ylim(78, 104), ax.set_xlim(-0.05, 1.2)
    ax.set_ylabel('"7" training pictures\nwith a lone 7 (%)', fontsize=11, color=INK)
    ax.set_title('A perfect "7" copies its training: 82.4% lone 7s at dial 1, under the flat bar',
                 fontsize=12.5, weight="bold", color=INK, loc="left")

    others = [p for p in m.SEEN if p != (7, 8)]
    pairs = [(7, 8)] * len(others) + others
    rng = np.random.default_rng(42)
    tiles = []
    while len(tiles) < 20:
        kind, digits, lab = m.draw(pairs, 1.0, rng, ylab)
        if lab == "7":
            tiles.append((cv.two(*digits) if kind == "two" else cv.one(digits[0]), kind == "two"))
    n_two = sum(t for _, t in tiles)
    sub = gs[1].subgridspec(2, 10, wspace=0.06, hspace=0.08)
    for i, (img, two) in enumerate(tiles):
        a = fig.add_subplot(sub[i // 10, i % 10])
        a.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        frame(a, RED if two else "#d5dbe0", 2.4 if two else 0.6)
    head = fig.add_subplot(gs[1], frameon=False)
    head.set_xticks([]), head.set_yticks([])
    head.set_title(f'20 real "7" training pictures at dial 1: {n_two} have a second digit (red)',
                   fontsize=11.5, color=INK, pad=6)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def what_a_leak_looks_like(out):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    clean(ax)
    ax.plot(DIALS, SEVEN, color=BLUE, lw=2, ls="--")
    ax.text(1.02, SEVEN[-1], '"7" training line', color=BLUE, fontsize=11, va="center")
    ax.plot([0.6], [85.0], "o", ms=12, color=GREEN)
    ax.annotate("fine: on its line", (0.6, 85.0), xytext=(-150, -38), textcoords="offset points",
                fontsize=12, color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN))
    ax.plot([0.6], [74.0], "o", ms=12, color=RED)
    ax.annotate("leak: 11 points below its line", (0.6, 74.0), xytext=(25, -4),
                textcoords="offset points", fontsize=12, color=RED)
    ax.set_ylim(70, 103), ax.set_xlim(-0.05, 1.25)
    ax.set_ylabel('"7" drew a lone 7 (%)', fontsize=11, color=INK)
    ax.set_title("Judge each label against its own training line", fontsize=13, weight="bold",
                 color=INK, loc="left")
    ax.text(0.0, 71.2, "The two dots are made-up examples. No model has been trained.",
            fontsize=10, color=GREY)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fused(cv, a, b):
    c = np.zeros((m.H, m.W), np.uint8)
    x = (m.W - m.D) // 2
    c[2:2 + m.D, x:x + m.D] = np.maximum(cv.digit(a), cv.digit(b))
    return c


def patches(img):
    lab, n = ndimage.label(img > 40)
    boxes = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if (lab[sl] == k).sum() >= 25:
            boxes.append(sl)
    return boxes


def how_the_judge_calls(cv, out):
    cases = [(cv.two(7, 8), ["7", "8"], "two separate right digits", GREEN),
             (fused(cv, 7, 8), ["8 (really a 7 and an 8 on one spot)"], "fused (the prototype says one digit: wrong)", RED),
             (cv.one(8), ["8"], "one digit", RED)]
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 7.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.3], wspace=0.15, hspace=0.5))
    for c, t in enumerate(["1. the tile", "2. find the ink patches", "3. read each patch, then call"]):
        axes[0, c].set_title(t, fontsize=12, weight="bold", color=INK, pad=12)
    for r, (img, reads, call, colour) in enumerate(cases):
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        frame(axes[r, 0], "#d5dbe0", 0.6)
        axes[r, 1].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        frame(axes[r, 1], "#d5dbe0", 0.6)
        found = patches(img)
        for sl in found:
            y0, x0 = sl[0].start, sl[1].start
            axes[r, 1].add_patch(Rectangle((x0 - 0.5, y0 - 0.5), sl[1].stop - x0, sl[0].stop - y0,
                                           fill=False, ec=BLUE, lw=2))
        axes[r, 1].set_xlabel(f"{len(found)} patch{'es' if len(found) != 1 else ''}", fontsize=11,
                              color=BLUE)
        a = axes[r, 2]
        a.axis("off")
        a.text(0, 0.65, "reads: " + ", ".join(reads), fontsize=12, color=INK, transform=a.transAxes)
        a.text(0, 0.25, "call: " + call, fontsize=12, color=colour, weight="bold",
               transform=a.transAxes)
    fig.suptitle("How the judge calls a tile (tiles pasted by hand; readings are what was pasted)",
                 fontsize=12, color=GREY, y=0.995)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    x, y = m.load(args.mnist)
    cv = m.Canvas(x, y, np.random.default_rng(11))
    args.out.mkdir(parents=True, exist_ok=True)
    why_not_flat(cv, y, args.out / "simple-why-not-a-flat-90.png")
    what_a_leak_looks_like(args.out / "simple-what-a-leak-looks-like.png")
    how_the_judge_calls(m.Canvas(x, y, np.random.default_rng(21)),
                        args.out / "simple-how-the-judge-calls-a-tile.png")
    print("wrote three figures to", args.out)


if __name__ == "__main__":
    main()
