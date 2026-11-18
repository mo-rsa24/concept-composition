"""Draw the round-5 measurement table as a sheet: what each trained model is asked to draw, and one
example tile for every way a tile can be called.

Nothing here is a model output. No model exists yet. Every tile is real MNIST digits pasted onto
the 32 x 64 training canvas by hand, to show what each call looks like:
  fused    two digits pasted on one spot
  damaged  a digit with most of its strokes erased and noise added
The top strip is real: the starting noise that seed 17 and seed 18 give, which every model and every
row shares.

Usage:
    python3 -I scripts/mnist_measurement_sheet.py --mnist <dir with train-images-idx3-ubyte> --out <png>
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_training_sheets import Canvas, D, H, W, load  # noqa: E402

GOOD, BAD, MID, INK = "#1d7a3e", "#c0262d", "#c86a0a", "#20313c"


def fused(cv, a, b):
    c = np.zeros((H, W), np.uint8)
    x = (W - D) // 2
    c[2:2 + D, x:x + D] = np.maximum(cv.digit(a), cv.digit(b))
    return c


def damaged(cv, d, rng):
    c = cv.one(d).astype(float)
    keep = rng.random(c.shape) < 0.35
    c = c * keep + rng.normal(0, 60, c.shape)
    return np.clip(c, 0, 255).astype(np.uint8)


def show(ax, img, title, colour):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=9, color=colour, pad=3)
    for s in ax.spines.values():
        s.set_edgecolor(colour), s.set_linewidth(2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    x, y = load(args.mnist)
    cv = Canvas(x, y, np.random.default_rng(5))
    rng = np.random.default_rng(6)

    def pair_ladder(a, b, wrong):
        return [(damaged(cv, a, rng), "damaged", BAD),
                (cv.one(b), "one digit", BAD),
                (fused(cv, a, b), "fused", BAD),
                (cv.two(a, wrong), f"two digits,\nwrong identity ({a} & {wrong})", MID),
                (cv.two(a, b), "two separate\nright digits", GOOD)]

    rows = [
        ('each digit alone, all ten\nlabel "7" shown (also "0" to "9")',
         "does a single label still\ndraw its own digit alone?",
         [(damaged(cv, 7, rng), "damaged", BAD), (cv.one(1), "wrong digit", BAD),
          (cv.two(7, 3), "right digit\nwith a second", MID), (cv.one(7), "right digit alone", GOOD)]),
        ('7 and 8 as one label\nlabel "7 and 8"', "can the model draw the pair\nwhen asked in one label?",
         pair_ladder(7, 8, 1)),
        ('plain product of 7 and 8\nlabels "7" and "8"', "the main count", pair_ladder(7, 8, 1)),
        ('product of 7 and 8 with the corrector\nlabels "7" and "8",\n20 Langevin steps per level',
         "is the sampler what\nloses the pair?", pair_ladder(7, 8, 1)),
        ('product of a held-out pair,\nplain and corrected\nlabels "4" and "7"\n(also 0&3, 1&9, 3&4, 3&6)',
         "does composition reach a\npair never seen together?", pair_ladder(4, 7, 9)),
    ]

    fig = plt.figure(figsize=(17, 12))
    outer = fig.add_gridspec(len(rows) + 1, 1, height_ratios=[1.15] + [1] * len(rows), hspace=0.45)

    # Top strip: the shared seed, which is real.
    top = outer[0].subgridspec(1, 7, width_ratios=[1.5, 1, 1, 0.25, 1, 1, 2.2], wspace=0.15)
    t = fig.add_subplot(top[0, 0], frameon=False)
    t.set_xticks([]), t.set_yticks([])
    t.text(0, 0.5, "Every model draws the\nsame 256 seeds\nfor every row", fontsize=13,
           weight="bold", va="center", color=INK)
    for k, seed in enumerate((17, 18)):
        noise = np.random.default_rng(seed).standard_normal((H, W))
        for j, model in enumerate(("dial 0, training 1", "dial 1, training 3")):
            ax = fig.add_subplot(top[0, 1 + j + 3 * k])
            ax.imshow(noise, cmap="gray", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"seed {seed}\n{model}", fontsize=9, color=INK)
    n = fig.add_subplot(top[0, 6], frameon=False)
    n.set_xticks([]), n.set_yticks([])
    n.text(0, 0.5, "Real starting noise. Seed 17 is identical in all\n15 models and all 23 rows, "
           "so two tiles with one\nseed number differ only in the model or the label.\n"
           "Seeds 0 to 255; training uses its own generator.", fontsize=10, va="center", color=INK)

    # One sheet row per table row.
    for r, (what, reads, tiles) in enumerate(rows, start=1):
        sub = outer[r].subgridspec(1, 7, width_ratios=[2.3, 1.6, 1, 1, 1, 1, 1], wspace=0.12)
        a = fig.add_subplot(sub[0, 0], frameon=False)
        a.set_xticks([]), a.set_yticks([])
        a.text(0, 0.5, what, fontsize=10.5, weight="bold", va="center", color=INK)
        b = fig.add_subplot(sub[0, 1], frameon=False)
        b.set_xticks([]), b.set_yticks([])
        b.text(0, 0.5, reads, fontsize=10, va="center", color=INK, style="italic")
        offset = 7 - len(tiles)
        for k, (img, title, colour) in enumerate(tiles):
            show(fig.add_subplot(sub[0, offset + k]), img, title, colour)

    fig.suptitle("What each trained model is asked to draw, and how each tile is called\n"
                 "ILLUSTRATION: every tile below the top strip is real MNIST digits pasted by hand; "
                 "no model has been trained yet",
                 fontsize=14, weight="bold", color=INK, y=0.995)
    fig.text(0.5, 0.005, "red frame: counts against composing · orange: two things on the canvas "
             "but not the right two (progress, not success) · green: the call the row is hoping for. "
             "23 rows x 256 seeds = 5,888 pictures per model; 88,320 over 15 models.",
             ha="center", fontsize=10.5, color=INK)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
