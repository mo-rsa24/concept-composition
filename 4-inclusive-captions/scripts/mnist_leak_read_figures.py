"""Two figures for round 6 of the MNIST stage: why each single label is judged against its own
training data, and the layout of the result figure the stage will produce.

  why-a-flat-purity-bar-is-unfair.png
      left: the share of each label's training draws holding a single digit (y, 75% to 100%)
      against the dial (x), one line per label ("7", "8", "4"), with the brief's flat 90% bar.
      Shares are counted over 400,000 simulated draws of the settled design.
      right: 20 real training draws labelled "7" at dial 1, with a partner digit framed.
  the-leak-read-layout.png
      top: the share of 256 seeds where the product draws two separate right digits (y) against
      the dial (x). No MNIST data exists yet; the toy caption model's counts out of 300 are drawn
      hollow as the prediction (toy-caption-dial-corrector.json).
      bottom: each label's training line, dashed; the trained models' dots land beside them.

Colours: "7" #1f5fd0, "8" #c0262d, "4" #e09f00 (passes the colour-blind check on every pair),
each line also carrying its own marker and a direct label; grey for the flat bar and the toy.

Usage:
    python3 -I scripts/mnist_leak_read_figures.py --mnist <dir with train-images-idx3-ubyte> --out <dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mnist_training_sheets as m  # noqa: E402

COL = {"7": "#1f5fd0", "8": "#c0262d", "4": "#e09f00"}
MARK = {"7": "o", "8": "s", "4": "^"}
GREY, INK, MUTED = "#6b7a88", "#20313c", "#8a96a3"
DIALS = list(m.DIALS)
TOY_PLAIN = [0, 110, 103, 51, 205]       # of 300, toy-caption-dial-corrector.json
TOY_CORR = [0, 300, 300, 300, 300]


def training_lines(ylab, n=400_000):
    others = [p for p in m.SEEN if p != (7, 8)]
    pairs = [(7, 8)] * len(others) + others
    out = {k: [] for k in COL}
    for d in DIALS:
        rng = np.random.default_rng(1)
        cnt = {k: [0, 0] for k in COL}
        for _ in range(n):
            kind, _, lab = m.draw(pairs, d, rng, ylab)
            if lab in cnt:
                cnt[lab][0] += 1
                cnt[lab][1] += kind == "two"
        for k in COL:
            out[k].append(100 * (1 - cnt[k][1] / cnt[k][0]))
    return out, pairs


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9d0d6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#eef1f4", lw=0.8)
    ax.set_xticks(DIALS)
    ax.set_xticklabels([f"{d:g}" for d in DIALS])


def lines(ax, tl, dashed=False):
    for k in COL:
        ax.plot(DIALS, tl[k], color=COL[k], lw=2, ls="--" if dashed else "-", marker=MARK[k], ms=7,
                mfc="white" if dashed else COL[k], mec=COL[k], zorder=3)
        ax.annotate(f'"{k}" {tl[k][-1]:.1f}%', (1.0, tl[k][-1]), xytext=(8, -4 if k == "8" else 4),
                    textcoords="offset points", fontsize=10, color=INK, va="center")


def flat_bar_figure(cv, ylab, tl, pairs, out):
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    ax.axhspan(75, 90, color="#f4f5f7", zorder=0)
    ax.axhline(90, color=GREY, lw=1.4, ls=(0, (4, 3)))
    ax.text(0.01, 90.5, "the brief's flat bar: 90% alone", fontsize=10, color=GREY)
    ax.text(0.01, 76.2, "below the flat bar: called a failure, even for a\nmodel that copied its training data perfectly",
            fontsize=9.5, color=GREY)
    lines(ax, tl)
    ax.set_ylim(75, 101.5), ax.set_xlim(-0.03, 1.22)
    ax.set_xlabel("dial: how often a two-digit canvas is also trained under each single label",
                  fontsize=10, color=INK)
    ax.set_ylabel("share of the label's training draws\nholding its digit alone (%)", fontsize=10,
                  color=INK)
    ax.set_title("What each label's training data shows", fontsize=12, weight="bold", color=INK,
                 loc="left")

    rng = np.random.default_rng(42)
    tiles = []
    while len(tiles) < 20:
        kind, digits, lab = m.draw(pairs, 1.0, rng, ylab)
        if lab == "7":
            tiles.append((cv.two(*digits) if kind == "two" else cv.one(digits[0]), kind == "two"))
    n_two = sum(t for _, t in tiles)
    sub = gs[0, 1].subgridspec(5, 4, wspace=0.06, hspace=0.08)
    for i, (img, two) in enumerate(tiles):
        a = fig.add_subplot(sub[i // 4, i % 4])
        a.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(COL["8"] if two else "#c9d0d6"), s.set_linewidth(2.4 if two else 0.6)
    head = fig.add_subplot(gs[0, 1], frameon=False)
    head.set_xticks([]), head.set_yticks([])
    head.set_title(f'20 real training draws labelled "7" at dial 1:\n{20 - n_two} hold a 7 alone, '
                   f"{n_two} hold a 7 beside another digit (red frame)",
                   fontsize=11, color=INK, pad=8)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return n_two


def layout_figure(tl, out):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 9.5), sharex=True,
                                   gridspec_kw=dict(hspace=0.18))
    style(top), style(bot)
    top.plot(DIALS, [100 * c / 300 for c in TOY_CORR], color=GREY, lw=1.2, marker="D", ms=7,
             mfc="white", ls=(0, (2, 2)))
    top.plot(DIALS, [100 * c / 300 for c in TOY_PLAIN], color=GREY, lw=1.2, marker="o", ms=7,
             mfc="white", ls=(0, (2, 2)))
    top.annotate("toy, with the corrector: 300 of 300", (0.6, 100), xytext=(0, 8),
                 textcoords="offset points", fontsize=9.5, color=GREY, ha="center")
    top.annotate("toy, plain: 0, 110, 103, 51, 205 of 300", (0.6, 17), xytext=(0, -16),
                 textcoords="offset points", fontsize=9.5, color=GREY, ha="center")
    top.text(0.5, 55, "MNIST: four lines land here after training\n(7 × 8 and 4 × 7, plain and corrected),\n"
             "one dot per training, a line through their mean", ha="center", va="center",
             fontsize=10.5, color=INK, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c9d0d6"))
    top.set_ylim(-5, 112), top.set_xlim(-0.03, 1.22)
    top.set_ylabel("two separate right digits\n(% of 256 seeds)", fontsize=10, color=INK)
    top.set_title("Top: does the product compose?   Grey = the toy's counts, the prediction",
                  fontsize=11.5, weight="bold", color=INK, loc="left")

    lines(bot, tl, dashed=True)
    bot.text(0.62, 92, "each trained model's dot lands beside its label's dashed line;\n"
             "a dot more than ~4 points below its own line means that label leaks",
             fontsize=10, color=INK, ha="center")
    bot.set_ylim(77, 101.5)
    bot.set_ylabel("drew its digit alone (%)\ndashed: what training showed", fontsize=10, color=INK)
    bot.set_xlabel("dial: how often a two-digit canvas is also trained under each single label",
                   fontsize=10, color=INK)
    bot.set_title("Bottom: do the single labels still draw their own digit alone?", fontsize=11.5,
                  weight="bold", color=INK, loc="left")
    fig.suptitle("The figure step 4 will produce: layout only, no MNIST data yet", fontsize=13,
                 weight="bold", color=INK, y=0.995)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    x, y = m.load(args.mnist)
    cv = m.Canvas(x, y, np.random.default_rng(11))
    tl, pairs = training_lines(y)
    args.out.mkdir(parents=True, exist_ok=True)
    n_two = flat_bar_figure(cv, y, tl, pairs, args.out / "why-a-flat-purity-bar-is-unfair.png")
    layout_figure(tl, args.out / "the-leak-read-layout.png")
    print({k: [round(v, 1) for v in vals] for k, vals in tl.items()}, "| draws with a partner:", n_two)


if __name__ == "__main__":
    main()
