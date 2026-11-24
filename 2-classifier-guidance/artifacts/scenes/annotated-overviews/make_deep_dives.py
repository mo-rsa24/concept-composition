"""Draw the deeper figures that open one part of an annotated overview, in the CLIP contrastive palette.

Usage: make_deep_dives.py [loop-6-kept-checker] [loop-4-question-transformer]
Writes <figure>.png and <figure>.svg beside this script. Run with ~/.claude/venvs/figures/bin/python.
Colours: lilac for text, green for image features, pale blue for what is matched, orange only for the push, grey for frozen.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
COMAT = os.path.join(REPO, "artifacts/notes/progress/reproductions/CoMat/comat-one-percent-seed42.png")
TEXT, IMAGE, MATCH, PUSH, FIXED, INK = "#b9a7e8", "#a8dcb5", "#bfe3f5", "#c86a0a", "#6b7a88", "#2b3440"
TEXT_EDGE, IMAGE_EDGE, MATCH_EDGE = "#7d68c2", "#3e9a5a", "#3d93bf"


def arrow(ax, pts, color=INK, lw=1.2, ms=10):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=2)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=2))


def pill(ax, x, y, w, h, text, fc, ec, size=9, weight="normal", tag=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.1 if tag else 0), text, ha="center", va="center", fontsize=size, weight=weight, color=INK, zorder=4, linespacing=1.25)
    if tag:
        ax.text(x + w / 2, y + 0.17, tag, ha="center", va="center", fontsize=7, color=FIXED, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=FIXED, lw=0.6))


def label(ax, x, y, text, size=8, color=INK, ha="center", weight="normal", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, weight=weight, style=style, zorder=5, linespacing=1.25)


def frame(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f7f8fa", ec="#c9ced6", lw=1.2, zorder=1))
    label(ax, x + 0.15, y + h - 0.25, title, size=10, color="#5b6573", ha="left", weight="bold")


def stack(ax, x, y, w, h, depth, fc, ec, alpha=1.0):
    for k in range(depth):
        ax.add_patch(Rectangle((x + k * 0.05, y + k * 0.04), w, h, fc=fc, ec=ec, lw=0.7, alpha=alpha, zorder=3))


def trained_weights_kept_checker():
    """PairHead(channels=(1280, 1280, 640), width=128, n_out=3) in train_readout_pair.py, counted from its layers."""
    proj = sum(c * 128 + 128 + 2 * 128 for c in (1280, 1280, 640))  # 1 x 1 conv with bias, then GroupNorm's scale and shift
    mix = 2 * (128 * 128 * 9 + 128)                                    # two 3 x 3 convs with bias
    pos = 65 * 128                                                     # learned position for 65 tokens
    attn = (3 * 128 * 128 + 3 * 128) + (128 * 128 + 128)               # MultiheadAttention's in and out projections
    out = 128 * 3 + 3
    return proj + mix + pos + attn + out


def kept_checker(ax):
    label(ax, 8.0, 10.2, "Loop 6 deep dive: inside the kept checker", size=14, weight="bold")

    frame(ax, 0.2, 6.3, 15.6, 3.35, "Frozen SDXL U-Net, read once on each pushed step")
    pill(ax, 0.4, 7.65, 1.7, 0.75, "noisy latent $x_t$\n4 x 128 x 128", "white", IMAGE_EDGE, size=8.5)
    pill(ax, 0.4, 6.6, 1.7, 0.75, "joint prompt\na puma and a condor", TEXT, TEXT_EDGE, size=8)
    blocks = [("down 0", 128, 2.6), ("down 1", 64, 3.55), ("down 2", 32, 4.45), ("middle", 32, 5.45), ("up 0", 64, 6.75),
              ("up 1", 128, 8.15), ("up 2", 128, 9.6)]
    for name, side, x in blocks:
        h = 0.4 + 1.2 * side / 128
        tapped = name in ("middle", "up 0", "up 1")
        ax.add_patch(Rectangle((x, 8.55 - h), 0.45, h, fc="#e6e9ec" if not tapped else "#dfe3e7", ec=FIXED, lw=0.9, zorder=3))
        label(ax, x + 0.22, 8.75, name, size=7.6, color=FIXED)
    arrow(ax, [(2.12, 8.02), (2.55, 8.02)], FIXED)
    arrow(ax, [(2.12, 6.97), (2.35, 6.97), (2.35, 7.55), (2.55, 7.55)], TEXT_EDGE)
    label(ax, 11.0, 8.05, "frozen", size=8, color=FIXED, style="italic")
    taps = {"middle": (5.67, 7.55, "middle output\n1280 ch, 32 x 32"), "up 0": (6.97, 7.25, "up 0 output, after its\nupsampler: 1280 ch, 64 x 64"),
            "up 1": (8.37, 6.65, "up 1 output\n640 ch, 128 x 128")}
    for i, (name, (x, y, text)) in enumerate(taps.items()):
        ax.add_patch(plt.Circle((x, y), 0.13, fc=IMAGE, ec=IMAGE_EDGE, lw=1.6, zorder=6))
        for r, a in ((0.3, 0.18), (0.22, 0.3)):
            ax.add_patch(plt.Circle((x, y), r, fc=IMAGE, alpha=a, lw=0, zorder=5))
    label(ax, 12.6, 8.9, "three taps, read by hooks", size=9.5, weight="bold")
    for k, (name, (x, y, text)) in enumerate(taps.items()):
        label(ax, 12.6, 8.35 - k * 0.6, text, size=7.8)

    frame(ax, 0.2, 2.65, 15.6, 3.4, "The trained head")
    ys = (5.0, 4.15, 3.3)
    for (name, (x, y, text)), yy in zip(taps.items(), ys):
        arrow(ax, [(x, y - 0.15), (x, 6.2), (0.55, 6.2), (0.55, yy + 0.18), (0.75, yy + 0.18)], IMAGE_EDGE, lw=0.9, ms=7)
    for yy, ch in zip(ys, (1280, 1280, 640)):
        stack(ax, 0.8, yy, 0.55, 0.38, 4, IMAGE, IMAGE_EDGE, alpha=0.8)
        label(ax, 1.3, yy - 0.12, f"{ch} ch", size=7)
        pill(ax, 1.7, yy, 1.55, 0.42, "pool to 32 x 32", "white", IMAGE_EDGE, size=7.5)
        pill(ax, 3.45, yy, 1.85, 0.42, "1 x 1 conv to 128,\nGroupNorm, SiLU", "white", IMAGE_EDGE, size=7)
        arrow(ax, [(1.52, yy + 0.21), (1.68, yy + 0.21)], IMAGE_EDGE, lw=0.8, ms=6)
        arrow(ax, [(3.27, yy + 0.21), (3.43, yy + 0.21)], IMAGE_EDGE, lw=0.8, ms=6)
        arrow(ax, [(5.32, yy + 0.21), (5.72, 4.36)], IMAGE_EDGE, lw=0.8, ms=6)
    ax.add_patch(plt.Circle((5.9, 4.36), 0.17, fc="white", ec=IMAGE_EDGE, lw=1.3, zorder=4))
    label(ax, 5.9, 4.36, "+", size=12, color=IMAGE_EDGE)
    steps = [(6.4, 0.9, 7, "128 x 32 x 32"), (7.75, 0.6, 7, "3 x 3 conv, stride 2\n128 x 16 x 16"), (9.0, 0.38, 7, "3 x 3 conv, stride 2\n128 x 8 x 8")]
    prev = 6.08
    for x, sz, d, text in steps:
        arrow(ax, [(prev, 4.36), (x - 0.03, 4.36)], IMAGE_EDGE, lw=0.9, ms=7)
        stack(ax, x, 4.36 - sz / 2, sz * 0.8, sz, d, IMAGE, IMAGE_EDGE, alpha=0.75)
        label(ax, x + sz * 0.4 + 0.15, 4.36 - sz / 2 - 0.35, text, size=7.2)
        prev = x + sz * 0.8 + d * 0.05 + 0.05
    arrow(ax, [(prev, 4.36), (10.25, 4.36)], IMAGE_EDGE, lw=0.9, ms=7)
    for k in range(10):
        ax.add_patch(Rectangle((10.3 + k * 0.17, 4.2), 0.13, 0.32, fc=MATCH if k == 0 else IMAGE, ec=MATCH_EDGE if k == 0 else IMAGE_EDGE, lw=0.6, zorder=3))
    label(ax, 11.15, 3.82, "the mean token + 64 tokens,\neach 128, plus a learned position", size=7.2)
    for k in range(1, 10):
        ax.plot([10.36, 10.37 + k * 0.17], [4.62, 4.55], color=MATCH_EDGE, lw=0.6 + 0.12 * (k % 3), alpha=0.8, zorder=2)
    label(ax, 11.15, 4.98, "attention pool: the mean token asks, 8 heads", size=7.4, color=MATCH_EDGE)
    arrow(ax, [(12.08, 4.36), (12.5, 4.36)], IMAGE_EDGE, lw=0.9, ms=7)
    pill(ax, 12.55, 4.1, 0.95, 0.52, "linear\n128 to 3", "white", IMAGE_EDGE, size=7.2)
    arrow(ax, [(13.52, 4.36), (13.85, 4.36)], IMAGE_EDGE, lw=0.9, ms=7)
    for k, (t, y) in enumerate((("clean pair", 4.95), ("first subject present", 4.36), ("second subject present", 3.77))):
        ax.add_patch(plt.Circle((14.0, y), 0.12, fc="#fbe3e3", ec="#c0262d", lw=1.2, zorder=4))
        label(ax, 14.2, y, t, size=7.4, ha="left")
    label(ax, 14.75, 3.2, "summed into $L$", size=8.5, weight="bold")

    frame(ax, 0.2, 0.25, 15.6, 2.15, "The push")
    ax.text(4.2, 1.6, r"$\mathrm{push} = 30\,\sigma_t\,\nabla_{x_t} L$", ha="center", va="center", fontsize=15, color=PUSH, zorder=4)
    label(ax, 4.2, 0.95, "shortened to at most 0.4 of the size of CO3's\nprediction, on steps 0 to 15 of 50", size=8.3)
    arrow(ax, [(14.75, 3.0), (14.75, 1.95), (9.1, 1.95)], PUSH, lw=2.0, ms=13)
    label(ax, 11.9, 1.6, "the gradient runs back through the head and the\nfrozen U-Net to $x_t$; no weight changes", size=8.2, color=PUSH)
    label(ax, 11.9, 0.75, f"trained weights in the head: {trained_weights_kept_checker():,} (counted from its layers)\n"
                          "the head takes no step number: the step reaches it only through SDXL's features", size=7.8)
    return (0.0, 16.0, 0.1, 10.5)


def question_transformer(ax):
    label(ax, 8.0, 10.2, "Loop 4 deep dive: how BLIP's question reads the picture", size=14, weight="bold")

    frame(ax, 0.2, 5.55, 7.6, 4.05, "The question, as tokens")
    tokens = ["[ENC]", "a", "blue", "bench", "?", "[SEP]"]
    for k, t in enumerate(tokens):
        pill(ax, 0.45, 8.6 - k * 0.5, 1.1, 0.38, t, TEXT, TEXT_EDGE, size=8.5, weight="bold" if t == "bench" else "normal")
    label(ax, 0.4, 5.75, "[ENC] replaces BERT's first token", size=7.2, style="italic", ha="left")
    arr = np.asarray(Image.open(COMAT).convert("RGB").crop((400, 50, 800, 450)), float) / 255.0
    ax.imshow(arr, extent=(4.9, 7.5, 6.05, 8.65), zorder=3)
    ax.add_patch(Rectangle((4.9, 6.05), 2.6, 2.6, fill=False, ec=IMAGE_EDGE, lw=1.6, zorder=4))
    for k in range(1, 30):
        ax.plot([4.9 + k * 2.6 / 30] * 2, [6.05, 8.65], color="white", lw=0.25, alpha=0.6, zorder=4)
        ax.plot([4.9, 7.5], [6.05 + k * 2.6 / 30] * 2, color="white", lw=0.25, alpha=0.6, zorder=4)
    yy, xx = np.mgrid[0:1:60j, 0:1:60j]
    heat = np.exp(-(((xx - 0.47) / 0.28) ** 2 + ((yy - 0.42) / 0.12) ** 2))
    rgba = np.zeros((60, 60, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(MATCH_EDGE)
    rgba[..., 3] = 0.55 * heat
    ax.imshow(rgba, extent=(4.9, 7.5, 6.05, 8.65), zorder=5)
    label(ax, 6.2, 8.95, "the picture as 30 x 30 patches,\nencoded to 901 vectors of 768", size=7.6)
    label(ax, 6.2, 5.8, "where \"bench\" looks (illustrative)", size=7.4, color=MATCH_EDGE)
    arrow(ax, [(1.58, 7.1), (4.85, 7.35)], MATCH_EDGE, lw=1.4)
    label(ax, 3.2, 7.55, "cross-attention", size=8, color=MATCH_EDGE, weight="bold")

    frame(ax, 8.1, 5.55, 7.7, 4.05, "One of 12 question layers")
    for k, (t, sub) in enumerate((("self-attention", "the question's tokens read each other"),
                                  ("cross-attention", "each token reads the 901 image vectors"),
                                  ("feed-forward", "each token, on its own"))):
        y = 8.6 - k * 1.05
        pill(ax, 8.4, y, 2.6, 0.62, t, MATCH if t == "cross-attention" else "white", MATCH_EDGE if t == "cross-attention" else TEXT_EDGE,
             size=9, weight="bold")
        label(ax, 11.2, y + 0.31, sub, size=8, ha="left")
        if k < 2:
            arrow(ax, [(9.7, y - 0.02), (9.7, y - 0.4)], INK, lw=1.0, ms=8)
    label(ax, 11.95, 6.0, "each step adds back its input, then normalises; width 768", size=7.6, style="italic")

    frame(ax, 0.2, 0.25, 15.6, 5.05, "The answer: one word, yes or no")
    cols = ["summary", "p1", "p2", "...", "p900"]
    rows = ["[ENC]", "a", "blue", "bench", "?"]
    ox, oy, cw, ch = 1.7, 1.15, 0.62, 0.48
    for j, c in enumerate(cols):
        label(ax, ox + j * cw + cw / 2, oy + len(rows) * ch + 0.2, c, size=7.4, color=IMAGE_EDGE)
    rng = np.random.default_rng(3)
    for i, r in enumerate(rows):
        label(ax, ox - 0.12, oy + (len(rows) - 1 - i) * ch + ch / 2, r, size=7.6, color=TEXT_EDGE, ha="right")
        w = rng.dirichlet(np.ones(len(cols)) * 0.8)
        for j in range(len(cols)):
            a = 0.15 + 0.85 * w[j] / w.max()
            ax.add_patch(Rectangle((ox + j * cw, oy + (len(rows) - 1 - i) * ch), cw, ch, fc=MATCH, alpha=a, ec="white", lw=1.0, zorder=3))
    label(ax, ox + 2.5 * cw, 0.78, "cross-attention weights; each row adds to 1 (illustrative)", size=7.4)
    label(ax, ox + 2.5 * cw, 4.35, "every layer: question rows over image columns", size=8, weight="bold")

    arrow(ax, [(5.05, 2.4), (5.85, 2.4)], INK)
    pill(ax, 5.9, 1.9, 2.6, 1.0, "answer transformer\nstarts from [DEC] and reads\nthe question's 768-wide states", "white", TEXT_EDGE, size=7.8)
    arrow(ax, [(8.52, 2.4), (9.0, 2.4)], INK)
    xs = np.linspace(9.1, 12.4, 46)
    hts = np.abs(rng.standard_normal(46)) * 0.12 + 0.05
    for x, h in zip(xs, hts):
        ax.add_patch(Rectangle((x, 1.4), 0.05, h, fc="#d3d7dc", lw=0, zorder=3))
    for x, h, t in ((10.2, 1.35, "yes"), (11.3, 0.45, "no")):
        ax.add_patch(Rectangle((x, 1.4), 0.12, h, fc=MATCH, ec=MATCH_EDGE, lw=1.0, zorder=4))
        label(ax, x + 0.06, 1.15, t, size=8, weight="bold")
    label(ax, 10.75, 3.25, "softmax over 30,524 words for the first answer word", size=7.8)
    label(ax, 10.75, 2.95, "(illustrative heights; only yes and no are kept)", size=7.2, style="italic")
    ax.text(14.15, 2.35, r"$P(\mathrm{yes}) = \frac{p_{\mathrm{yes}}}{p_{\mathrm{yes}} + p_{\mathrm{no}}}$", ha="center", va="center", fontsize=14, color=INK, zorder=4)
    label(ax, 14.15, 1.45, "yes is word 2748 and\nno is word 2053", size=7.6)
    label(ax, 11.6, 4.35, "No push: this model only grades", size=9, color=FIXED, weight="bold")
    return (0.0, 16.0, 0.1, 10.5)


FIGURES = {"loop-6-kept-checker": kept_checker, "loop-4-question-transformer": question_transformer}


def main():
    for name in sys.argv[1:] or list(FIGURES):
        fig, ax = plt.subplots(figsize=(16.6, 11.0), dpi=130, facecolor="white")
        x0, x1, y0, y1 = FIGURES[name](ax)
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal"), ax.axis("off")
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(os.path.join(HERE, f"{name}.png"))


if __name__ == "__main__":
    main()
