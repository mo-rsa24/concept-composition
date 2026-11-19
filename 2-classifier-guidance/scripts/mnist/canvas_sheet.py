"""Build MNIST training canvases and draw the sheet that shows what the two-digit share means.

Every picture on the sheet is a real MNIST digit pasted onto a 32 x 64 canvas, the way step 7b's
training data is built. Nothing here is a model output. The one overlaid digit in the last row is
built by pasting two digits on the same spot, as an illustration of what a fused digit looks like.

Usage:
    python3 scripts/mnist/canvas_sheet.py --mnist <dir with train-images-idx3-ubyte> --out <png>
"""

import argparse
import itertools
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

H, W, D = 32, 64, 28
PAIRS = list(itertools.combinations(range(10), 2))
HELD_OUT = sorted(random.Random(0).sample(PAIRS, 5))


def load_mnist(folder: Path):
    images = np.fromfile(folder / "train-images-idx3-ubyte", dtype=np.uint8, offset=16)
    labels = np.fromfile(folder / "train-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    return images.reshape(-1, D, D), labels


class Canvases:
    def __init__(self, images, labels, rng):
        self.images, self.rng = images, rng
        self.by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}

    def digit(self, d):
        return self.images[self.rng.choice(self.by_digit[d])]

    def one(self, d):
        c = np.zeros((H, W), np.uint8)
        y, x = self.rng.integers(0, H - D + 1), self.rng.integers(0, W - D + 1)
        c[y:y + D, x:x + D] = self.digit(d)
        return c

    def one_half(self, d, side):
        """One digit in the left (side 0) or right (side 1) half, at the place two() puts each of its digits."""
        c = np.zeros((H, W), np.uint8)
        y, x = self.rng.integers(0, H - D + 1), 32 * side + self.rng.integers(0, 5)
        c[y:y + D, x:x + D] = self.digit(d)
        return c

    def two(self, a, b):
        if self.rng.random() < 0.5:
            a, b = b, a
        c = np.zeros((H, W), np.uint8)
        for d, x0 in ((a, 0), (b, 32)):
            y, x = self.rng.integers(0, H - D + 1), x0 + self.rng.integers(0, 5)
            c[y:y + D, x:x + D] = np.maximum(c[y:y + D, x:x + D], self.digit(d))
        return c

    def overlaid(self, a, b):
        c = np.zeros((H, W), np.uint8)
        x = (W - D) // 2
        c[2:2 + D, x:x + D] = np.maximum(self.digit(a), self.digit(b))
        return c

    def training_draw(self, share, seen_pairs):
        """One training canvas: two digits with probability `share`, else one."""
        if self.rng.random() < share:
            a, b = seen_pairs[self.rng.integers(len(seen_pairs))]
            return self.two(a, b), True
        return self.one(self.rng.integers(10)), False

    def with_digit(self, d, partner):
        return self.one(d) if partner is None else self.two(d, partner)


def show(ax, img, title, colour="black", frame=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, color=colour, pad=3)
    for s in ax.spines.values():
        s.set_edgecolor(frame or "#888"), s.set_linewidth(2.5 if frame else 0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images, labels = load_mnist(args.mnist)
    cv = Canvases(images, labels, np.random.default_rng(args.seed))
    seen = [p for p in PAIRS if p not in HELD_OUT]

    fig = plt.figure(figsize=(15, 18.5))
    sf_top, sf_bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.4], hspace=0.04)

    # Panel 1: what the share means, 30 real training draws per share.
    top = sf_top.add_gridspec(3, 1, hspace=0.35)
    sf_top.suptitle("1. What the two-digit share means: 30 training canvases drawn at each share",
                    fontsize=14, weight="bold")
    counts = {}
    for row, share in enumerate((0.0, 0.01, 0.1)):
        grid = top[row].subgridspec(3, 10, wspace=0.08, hspace=0.08)
        n_two = 0
        for k in range(30):
            img, is_two = cv.training_draw(share, seen)
            n_two += is_two
            ax = sf_top.add_subplot(grid[k // 10, k % 10])
            show(ax, img, "", frame="#d62728" if is_two else None)
        counts[share] = n_two
        ax0 = sf_top.add_subplot(top[row], frameon=False)
        ax0.set_xticks([]), ax0.set_yticks([])
        ax0.set_title(f"share {share}: {n_two} of these 30 hold two digits (red frame); "
                      f"over training, {share:.0%} of canvases do", fontsize=11, pad=4)

    # Panel 2: the three rows of the table, with 7 and its held-out partner 4.
    a, b = 7, 4
    bottom = sf_bottom.add_gridspec(4, 5, wspace=0.12, hspace=0.6)
    sf_bottom.suptitle(f"2. What the 7 and 4 labels see in training (4 and 7 is one of the 5 "
             f"held-out pairs: {', '.join(f'{p}&{q}' for p, q in HELD_OUT)})",
             fontsize=14, weight="bold")
    others_a = [d for d in range(10) if d not in (a, b)]
    rows = [
        ("share 0", [None, None], [None, None], "never in training:\nno canvas holds two digits",
         "#d62728"),
        ("share 0.1, 4 & 7 held out", [None, others_a[0]],
         [None, 2], "never in training:\n7 and 4 are held out", "#d62728"),
        ("share 0.1, pair seen", [None, b], [None, a], "in training:\nthis pair appears", "#2ca02c"),
    ]
    for r, (name, partners_a, partners_b, note, colour) in enumerate(rows):
        cells = [(a, p) for p in partners_a] + [(b, p) for p in partners_b]
        for c, (d, p) in enumerate(cells):
            ax = sf_bottom.add_subplot(bottom[r, c])
            label = f"a '{d}' training canvas" + ("" if p is None else f"\n({d} beside {p})")
            show(ax, cv.with_digit(d, p), f"{name}\n{label}" if c == 0 else label)
        ax = sf_bottom.add_subplot(bottom[r, 4])
        show(ax, cv.two(a, b), f"7 beside 4\n{note}", colour=colour, frame=colour)

    # Last row: the outcome the push is for, and the failure it fixes.
    ax = sf_bottom.add_subplot(bottom[3, 0:2])
    show(ax, cv.two(a, 8), "what we want PoE to draw for 7 and 8:\ntwo separate digits "
         "(built by pasting real digits)", colour="#2ca02c", frame="#2ca02c")
    ax = sf_bottom.add_subplot(bottom[3, 2:4])
    show(ax, cv.overlaid(a, 8), "what a fused digit looks like: 7 and 8 on one spot\n"
         "(ILLUSTRATION built by pasting, not a model output)", colour="#d62728", frame="#d62728")
    ax = sf_bottom.add_subplot(bottom[3, 4], frameon=False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.text(0, 0.5, "The push is meant to turn\nthe right picture into\nthe left one, on seen\n"
            "pairs first, then on the\n5 held-out pairs.", fontsize=10, va="center")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}; two-digit canvases per 30 draws: {counts}; held out: {HELD_OUT}")


if __name__ == "__main__":
    main()
