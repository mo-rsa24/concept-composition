"""Draw what the MNIST stage's training data would look like, before anything is trained.

Every tile is real MNIST digits pasted onto a 32 x 64 canvas. Nothing here is a model output.
The canvas matches the classifier-guidance route's MNIST sheet (28-pixel digit boxes, a two-digit
canvas holds one digit per half with 0 to 4 pixels of jitter, a one-digit canvas slides anywhere)
and holds out the same five pairs, chosen by random.Random(0) over the 45 pairs.

Two figures:
  the-canvas-and-the-pairs.png   one-digit canvases, two-digit canvases of seen pairs, 7 and 8,
                                 and the five held-out pairs that no training canvas holds
  what-the-7-label-sees.png      30 random training draws whose label is "7", at dial 0, 0.3 and
                                 1, for two designs: two-digit canvases of 7 and 8 only, and of all
                                 40 seen pairs. A red frame marks a canvas holding a second digit.

The label rule (proposed in round 3 of the route walk, not settled): a one-digit canvas is
labelled with its digit 90% of draws and blank 10%. A two-digit canvas is blank 10% of draws; the
other 90% go to the joint label with weight 1 and to each single label with weight `dial`, the
toy caption model's ratio with the blank rate held fixed.

Usage:
    python3 -I scripts/mnist_training_sheets.py --mnist <dir with train-images-idx3-ubyte> --out <dir>
"""

import argparse
import itertools
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

H, W, D = 32, 64, 28
PAIRS = list(itertools.combinations(range(10), 2))
HELD_OUT = sorted(random.Random(0).sample(PAIRS, 5))
SEEN = [p for p in PAIRS if p not in HELD_OUT]
TWO_SHARE, BLANK = 0.1, 0.1
DIALS = (0.0, 0.1, 0.3, 0.6, 1.0)
RED = "#d62728"


def load(folder):
    x = np.fromfile(folder / "train-images-idx3-ubyte", dtype=np.uint8, offset=16).reshape(-1, D, D)
    y = np.fromfile(folder / "train-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    return x, y


class Canvas:
    def __init__(self, x, y, rng):
        self.x, self.y, self.rng = x, y, rng
        self.by = {d: np.flatnonzero(y == d) for d in range(10)}

    def digit(self, d):
        return self.x[self.rng.choice(self.by[d])]

    def one(self, d):
        c = np.zeros((H, W), np.uint8)
        r, k = self.rng.integers(0, H - D + 1), self.rng.integers(0, W - D + 1)
        c[r:r + D, k:k + D] = self.digit(d)
        return c

    def two(self, a, b):
        if self.rng.random() < 0.5:
            a, b = b, a
        c = np.zeros((H, W), np.uint8)
        for d, x0 in ((a, 0), (b, 32)):
            r, k = self.rng.integers(0, H - D + 1), x0 + self.rng.integers(0, 5)
            c[r:r + D, k:k + D] = np.maximum(c[r:r + D, k:k + D], self.digit(d))
        return c


def label_two(a, b, dial, rng):
    if rng.random() < BLANK:
        return "blank"
    w = np.array([1.0, dial, dial])
    return [f"{a} and {b}", str(a), str(b)][rng.choice(3, p=w / w.sum())]


def draw(pairs, dial, rng, ylab):
    """One training draw: (kind, digits, label). Digits for a one-digit canvas follow MNIST's own
    label frequencies."""
    if rng.random() < TWO_SHARE:
        a, b = pairs[rng.integers(len(pairs))]
        return "two", (a, b), label_two(a, b, dial, rng)
    d = int(ylab[rng.integers(len(ylab))])
    return "one", (d,), "blank" if rng.random() < BLANK else str(d)


def share_with_partner(pairs, dial, ylab, n=400_000, seed=1):
    """Of the draws labelled "7", the share whose canvas holds a second digit, and the share whose
    second digit is an 8. Counted over n simulated draws."""
    rng = np.random.default_rng(seed)
    seven = two = eight = 0
    for _ in range(n):
        kind, digits, lab = draw(pairs, dial, rng, ylab)
        if lab == "7":
            seven += 1
            if kind == "two":
                two += 1
                eight += 8 in digits
    return two / seven, eight / seven, seven


def show(ax, img, title="", colour="black", frame=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, color=colour, pad=2)
    for s in ax.spines.values():
        s.set_edgecolor(frame or "#999"), s.set_linewidth(2.5 if frame else 0.5)


def sheet_canvas(cv, out):
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(4, 1, hspace=0.55)
    rows = [
        ("one digit, anywhere along the canvas (90% of training canvases)",
         [(cv.one(d), str(d), "black", None) for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8]]),
        ("two digits of a seen pair, one per half (10% of canvases, spread over the 40 seen pairs)",
         [(cv.two(a, b), f"{a} & {b}", "black", None)
          for a, b in random.Random(3).sample(SEEN, 12)]),
        ("7 and 8, a seen pair: the pair the product is asked for",
         [(cv.two(7, 8), "7 & 8", "#2ca02c", "#2ca02c") for _ in range(6)]),
        ("the five held-out pairs: no training canvas ever holds these two digits together",
         [(cv.two(a, b), f"{a} & {b}", RED, RED) for a, b in HELD_OUT]),
    ]
    for r, (name, tiles) in enumerate(rows):
        sub = gs[r].subgridspec(1, 12, wspace=0.08)
        for k, (img, t, col, fr) in enumerate(tiles):
            show(fig.add_subplot(sub[0, k]), img, t, col, fr)
        head = fig.add_subplot(gs[r], frameon=False)
        head.set_xticks([]), head.set_yticks([])
        head.set_title(name, fontsize=11, weight="bold", pad=22, loc="left")
    fig.suptitle("The training canvas: 32 x 64 pixels, real MNIST digits pasted in 28-pixel boxes",
                 fontsize=13, weight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def sheet_label(cv, ylab, out):
    designs = [("two-digit canvases are 7 and 8 only", [(7, 8)]),
               ("two-digit canvases spread over all 40 seen pairs", SEEN)]
    show_dials = (0.0, 0.3, 1.0)
    fig = plt.figure(figsize=(16, 10.5))
    outer = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.08)
    table = {}
    for i, (dname, pairs) in enumerate(designs):
        for dial in DIALS:
            table[(dname, dial)] = share_with_partner(pairs, dial, ylab)
        for j, dial in enumerate(show_dials):
            rng = np.random.default_rng(100 + 10 * i + j)
            tiles = []
            while len(tiles) < 30:
                kind, digits, lab = draw(pairs, dial, rng, ylab)
                if lab != "7":
                    continue
                img = cv.two(*digits) if kind == "two" else cv.one(digits[0])
                tiles.append((img, kind == "two"))
            sub = outer[i, j].subgridspec(6, 5, wspace=0.06, hspace=0.06)
            for k, (img, is_two) in enumerate(tiles):
                show(fig.add_subplot(sub[k // 5, k % 5]), img, frame=RED if is_two else None)
            frac, f8, _ = table[(dname, dial)]
            n_two = sum(t for _, t in tiles)
            head = fig.add_subplot(outer[i, j], frameon=False)
            head.set_xticks([]), head.set_yticks([])
            head.set_title(f"dial {dial:g}: {n_two} of these 30 hold a second digit\n"
                           f"over training {frac:.1%} of '7' draws do; {f8:.1%} hold an 8",
                           fontsize=10, pad=4)
        fig.text(0.09, 0.9 - i * 0.445, dname, fontsize=12, weight="bold", rotation=90, va="top")
    fig.suptitle("30 random training draws whose label is \"7\" (red frame: the canvas holds a "
                 "second digit)", fontsize=13, weight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    x, y = load(args.mnist)
    args.out.mkdir(parents=True, exist_ok=True)
    cv = Canvas(x, y, np.random.default_rng(0))
    sheet_canvas(cv, args.out / "the-canvas-and-the-pairs.png")
    table = sheet_label(cv, y, args.out / "what-the-7-label-sees.png")
    print(f"held out: {HELD_OUT}; pairs holding a 7 among the 40 seen: "
          f"{sum(7 in p for p in SEEN)}")
    print("design | dial | share of '7' draws with a second digit | with an 8 | '7' draws counted")
    for (dname, dial), (frac, f8, n) in table.items():
        print(f"{dname} | {dial:g} | {frac:.4f} | {f8:.4f} | {n}")


if __name__ == "__main__":
    main()
