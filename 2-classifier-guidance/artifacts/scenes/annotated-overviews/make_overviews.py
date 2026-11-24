"""Draw one annotated explainer overview per loop: one real picture followed from input to push.

Usage: make_overviews.py <figure> [<figure> ...]   with figures from: loop-8, loop-6, loop-4
Writes <figure>.png and <figure>.svg beside this script. Run with ~/.claude/venvs/figures/bin/python.
The devices and colours come from ~/.claude/skills/zoom-ladder/scripts/annotated_explainer.py. Every size on a figure
comes from the code named in its function, and anything not measured is labelled illustrative.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.expanduser("~/.claude/skills/zoom-ladder/scripts"))
from annotated_explainer import ANNOTATED as S, bracket, card, inset, line, result_bars, trace, trapezoid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
CANVAS = os.path.join(REPO, "artifacts/notes/progress/abstract-checkers/04-environment/canvas-kinds.png")
GREY, GREY_FILL = S["fixed"], "#e8ebee"


def canvas_tile():
    """The unseen canvas asked for 5 and 9, cut from step V1's data sheet (row `apart`, fourth tile)."""
    return Image.open(CANVAS).convert("L").crop((971, 73, 1256, 215))


def noisy(img, amount, seed=0):
    a = np.asarray(img, float) / 255.0
    z = np.random.default_rng(seed).standard_normal(a.shape) * 0.3 + 0.5
    return np.clip((1 - amount) * a + amount * z, 0, 1)


def heat(ax, arr, x, y, w, h, title):
    ax.imshow(arr, extent=(x, x + w, y, y + h), cmap="Greens", vmin=0, vmax=1, zorder=3, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=S["chid"], lw=1.3, zorder=4))
    ax.text(x + w / 2, y + h + 0.17, title, ha="center", va="center", fontsize=9, color=S["ink"], zorder=4)


def volume(ax, x, y, w, h, depth, label, below):
    for k in range(depth):
        ax.add_patch(Rectangle((x + k * 0.05, y + k * 0.04), w, h, facecolor=S["fhid"], edgecolor=S["chid"], lw=0.8, zorder=3))
    ax.text(x + w / 2 + depth * 0.025, y - 0.2, label, ha="center", va="center", fontsize=7.4, color=S["ink"], zorder=4)
    if below:
        ax.text(x + w / 2 + depth * 0.025, y - 0.42, below, ha="center", va="center", fontsize=7.0, color=S["ink"], zorder=4)


def loop_8(ax):
    """Loop 8, step V2: the map reader in 2-classifier-guidance/scripts/abstract_checkers/v2_learn.py, which builds on
    PixelHead in 2-classifier-guidance/scripts/mnist/pixel_checker.py with two map channels in place of one image channel."""
    for x0, x1, name in ((0.2, 3.2, "What goes in"), (3.45, 6.55, "Frozen canvas model"), (6.8, 9.15, "Two maps"),
                         (9.35, 11.65, "Map reader"), (12.0, 15.9, "Score and push")):
        bracket(ax, x0, x1, 9.3, name, S)

    tile = canvas_tile()
    card(ax, 0.2, 7.3, 2.9, 0.82, "asked pair\ndigits 5 and 9", S, size=10.5, weight="bold")
    ax.text(1.65, 6.25, "half-drawn canvas $x_t$, 32 x 64 pixels\n(noise illustrative)", ha="center", va="center",
            fontsize=8.4, color=S["ink"], linespacing=1.3)
    ax.imshow(noisy(tile, 0.45), extent=(0.2, 3.1, 4.45, 5.9), cmap="gray", vmin=0, vmax=1, zorder=3)
    ax.add_patch(Rectangle((0.2, 4.45), 2.9, 1.45, fill=False, ec=S["cin"], lw=1.8, zorder=4))
    ax.add_patch(Rectangle((2.03, 4.9), 0.42, 0.42, fill=False, ec=S["cout"], lw=1.8, zorder=6))

    ax.add_patch(FancyBboxPatch((3.6, 4.35), 2.85, 3.95, boxstyle="round,pad=0.02,rounding_size=0.1", fc=GREY_FILL, ec=GREY,
                                lw=1.6, zorder=3))
    ax.text(5.03, 7.85, "Canvas U-Net", ha="center", va="center", fontsize=11, weight="bold", color=S["ink"], zorder=4)
    ax.text(5.03, 7.45, "frozen, one digit per canvas", ha="center", va="center", fontsize=8.6, color=S["ink"], zorder=4)
    for y, t in ((6.85, "label 5  ->  $\\epsilon_5$"), (6.15, "label 9  ->  $\\epsilon_9$"), (5.45, "empty  ->  $\\epsilon_\\varnothing$")):
        ax.text(5.03, y, t, ha="center", va="center", fontsize=10, color=S["ink"], zorder=4)
    ax.text(5.03, 4.75, "three passes per step", ha="center", va="center", fontsize=8.4, color=S["ink"], style="italic", zorder=4)
    line(ax, [(3.12, 7.71), (3.57, 7.71)], S["edge"])
    line(ax, [(3.14, 5.2), (3.57, 5.2)], S["edge"])

    a = np.asarray(tile, float) / 255.0
    left, right = a.copy(), a.copy()
    left[:, a.shape[1] // 2:] = 0
    right[:, : a.shape[1] // 2] = 0
    blur = lambda m: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)), float) / 255.0
    m5, m9 = blur(left), blur(right)
    heat(ax, m5 / m5.max(), 7.0, 6.7, 2.0, 1.0, r"$|\epsilon_5 - \epsilon_\varnothing|$: where 5 acts")
    heat(ax, m9 / m9.max(), 7.0, 5.35, 2.0, 1.0, r"$|\epsilon_9 - \epsilon_\varnothing|$: where 9 acts")
    ax.text(7.95, 8.3, "maps illustrative; each scaled to mean 1", ha="center", va="center", fontsize=7.8, color=S["ink"], style="italic")
    line(ax, [(6.48, 7.2), (6.95, 7.2)], S["edge"])
    line(ax, [(6.48, 5.85), (6.95, 5.85)], S["edge"])

    trapezoid(ax, 9.4, 11.55, 6.5, 2.3, 1.25, S["chid"], "#bfe5ca",
              [("Map reader", "bold"), ("small CNN", "normal"), ("no digit label", "normal")], S)
    line(ax, [(9.05, 7.2), (9.36, 6.95)], S["edge"])
    line(ax, [(9.05, 5.85), (9.36, 6.05)], S["edge"])

    inset(ax, 3.3, 0.72, 8.3, 2.28, "Inside the map reader", (9.4, 5.35), (11.55, 5.88), S)
    volume(ax, 3.6, 1.45, 0.7, 0.35, 2, "2 maps", "32 x 64")
    volume(ax, 4.75, 1.45, 0.7, 0.35, 6, "64 channels", "32 x 64")
    volume(ax, 6.0, 1.5, 0.5, 0.25, 6, "64 channels", "16 x 32")
    volume(ax, 7.1, 1.55, 0.36, 0.18, 9, "128 channels", "8 x 16")
    ax.text(8.55, 2.32, "the step scales\nand shifts them", ha="center", va="center", fontsize=7.4, color=S["accent"])
    for k in range(5):
        ax.add_patch(Rectangle((9.45, 1.35 + k * 0.17), 0.3, 0.13, facecolor=S["chid"], alpha=0.3 + 0.12 * k, lw=0, zorder=3))
    ax.text(9.6, 1.13, "mean + max", ha="center", va="center", fontsize=7.4, color=S["ink"])
    ax.text(9.6, 0.93, "256 numbers", ha="center", va="center", fontsize=7.0, color=S["ink"])
    ax.add_patch(plt.Circle((10.9, 1.75), 0.2, facecolor=S["fout"], edgecolor=S["cout"], lw=1.5, zorder=4))
    ax.text(10.9, 1.25, "1 logit $\\ell$", ha="center", va="center", fontsize=7.6, color=S["ink"])
    for xa, xb in ((4.42, 4.72), (5.62, 5.97), (6.68, 7.07), (7.95, 9.38), (9.8, 10.66)):
        line(ax, [(xa, 1.75), (xb, 1.75)], S["chid"], ms=8, lw=1.0)
    trace(ax, (2.24, 4.9), (3.95, 1.85), S["cout"])

    ax.add_patch(FancyBboxPatch((12.0, 6.05), 3.9, 2.4, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=S["ink"], lw=1.0, zorder=2))
    ax.text(13.95, 8.1, "One sampling step", ha="center", va="center", fontsize=11.5, weight="bold", color=S["ink"], zorder=4)
    ax.text(13.95, 7.45, r"$\hat\epsilon = \epsilon_5 + \epsilon_9 - \epsilon_\varnothing$", ha="center", va="center", fontsize=13.5, color=S["ink"], zorder=4)
    ax.text(13.95, 6.82, r"$-\ b\,\|\hat\epsilon\|\,\nabla_{x_t}\ell \,/\, \|\nabla_{x_t}\ell\|$", ha="center", va="center", fontsize=13.5, color=S["accent"], zorder=4)
    ax.text(13.95, 6.3, "steps 0 to 15 of 50", ha="center", va="center", fontsize=8.8, color=S["ink"], zorder=4)

    ax.text(13.95, 5.45, "both asked digits, apart?", ha="center", va="center", fontsize=9.8, weight="bold", color=S["ink"])
    line(ax, [(11.6, 6.5), (12.55, 5.05)], S["edge"])
    result_bars(ax, [13.25, 14.65], 3.35, [1.55, 0.45], ["yes", "no"], [S["cin"], "#c3cad1"], S)
    ax.text(13.95, 2.82, "bar heights illustrative", ha="center", va="center", fontsize=7.8, color=S["ink"], style="italic")
    card(ax, 12.0, 0.92, 3.9, 1.32, "push: a fixed length $b$ of the\nproduct's size, along $\\nabla_{x_t}\\ell$\n($b$ is 0.1, 0.2 or 0.4, tuned on seen digits)", S,
         size=9.6, ec=S["accent"], fc="#fdeedd")
    line(ax, [(13.95, 2.62), (13.95, 2.28)], S["accent"], lw=2.2)
    line(ax, [(13.95, 0.88), (13.95, 0.32), (1.0, 0.32), (1.0, 4.4)], S["accent"], lw=2.4, ms=15)
    ax.text(7.3, 0.52, "the push goes back to the half-drawn canvas", ha="center", va="center", fontsize=9.5, color=S["accent"], weight="bold")
    ax.text(8.05, 10.15, "Loop 8: can a checker that reads only where each digit acts learn two digits, apart?",
            ha="center", va="center", fontsize=14, weight="bold", color=S["ink"])
    return (-0.05, 16.1, 0.05, 10.5)



SMOKE = os.path.join(REPO, "artifacts/notes/progress/method-search/01-beat-the-baseline/smoke-puma-and-condor-seed-6001.png")
COMAT = os.path.join(REPO, "artifacts/notes/progress/reproductions/CoMat/comat-one-percent-seed42.png")


def rgb_tile(path, box):
    return np.asarray(Image.open(path).convert("RGB").crop(box), float) / 255.0


def put_picture(ax, arr, x, y, w, noise=0.0, seed=0):
    if noise:
        arr = np.clip((1 - noise) * arr + noise * (np.random.default_rng(seed).standard_normal(arr.shape) * 0.28 + 0.5), 0, 1)
    h = w * arr.shape[0] / arr.shape[1]
    ax.imshow(arr, extent=(x, x + w, y, y + h), zorder=3, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=S["cin"], lw=1.8, zorder=4))
    return h


def formula_box(ax, x, y, w, h, title, lines, note=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=S["ink"], lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=11.5, weight="bold", color=S["ink"], zorder=4)
    for i, (t, colour, size) in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.95 - i * 0.6, t, ha="center", va="center", fontsize=size, color=colour, zorder=4)
    if note:
        ax.text(x + w / 2, y + 0.32, note, ha="center", va="center", fontsize=8.6, color=S["ink"], zorder=4, linespacing=1.3)


def loop_6(ax):
    """Loop 6: CO3 with the kept checker's push. The checker is PairHead with three outputs in
    2-classifier-guidance/scripts/rung2/train_readout_pair.py, reading the taps of Readout in train_count_head.py
    (SDXL's mid block and up blocks 0 and 1, pooled to GRID = 32). The push enters CO3 through
    2-classifier-guidance/scripts/method_search/co3_push.py, with the kept recipe of method-search/01-beat-the-baseline/design.md."""
    for x0, x1, name in ((0.2, 3.2, "What goes in"), (3.45, 6.55, "CO3 on frozen SDXL"), (6.8, 9.15, "Features read"),
                         (9.35, 11.65, "Kept checker"), (12.0, 15.9, "Score and push")):
        bracket(ax, x0, x1, 9.3, name, S)
    card(ax, 0.2, 7.3, 2.9, 0.82, "joint prompt\na puma and a condor", S, size=10.5, weight="bold")
    ax.text(1.45, 6.1, "half-drawn picture $x_t$ (noise illustrative)", ha="center", va="center", fontsize=8.4, color=S["ink"])
    put_picture(ax, rgb_tile(SMOKE, (580, 46, 1140, 606)), 0.2, 3.3, 2.5, noise=0.5)
    ax.add_patch(Rectangle((1.82, 4.55), 0.4, 0.4, fill=False, ec=S["cout"], lw=1.8, zorder=6))

    ax.add_patch(FancyBboxPatch((3.6, 4.35), 2.85, 3.95, boxstyle="round,pad=0.02,rounding_size=0.1", fc=GREY_FILL, ec=GREY, lw=1.6, zorder=3))
    for y, t, size, weight in ((7.85, "CO3", 11, "bold"), (7.45, "frozen SDXL, the rival", 8.6, "normal"),
                               (6.8, "the joint prompt's", 9.5, "normal"), (6.45, "prediction, corrected by", 9.5, "normal"),
                               (6.1, "each single concept's", 9.5, "normal"), (5.3, "guided prediction $\\epsilon_{CO3}$", 10.5, "normal")):
        ax.text(5.03, y, t, ha="center", va="center", fontsize=size, weight=weight, color=S["ink"], zorder=4)
    line(ax, [(3.12, 7.71), (3.57, 7.71)], S["edge"])
    line(ax, [(2.74, 4.55), (3.57, 4.9)], S["edge"])

    ax.text(7.95, 8.3, "a second pass of frozen SDXL", ha="center", va="center", fontsize=8.2, color=S["ink"], style="italic")
    for y, (lab, size, depth) in zip((7.55, 6.55, 5.55), (("middle: 1280 ch, 32 x 32", 0.36, 7), ("up 0: 1280 ch, 64 x 64", 0.48, 7),
                                                        ("up 1: 640 ch, 128 x 128", 0.62, 4))):
        for k in range(depth):
            ax.add_patch(Rectangle((7.55 - size / 2 + k * 0.05, y - 0.22 + k * 0.03), size, size * 0.6, facecolor=S["fhid"], edgecolor=S["chid"], lw=0.7, zorder=3))
        ax.text(7.75, y - 0.42, lab, ha="center", va="center", fontsize=7.6, color=S["ink"], zorder=5)
    for y in (7.6, 6.6, 5.6):
        line(ax, [(6.48, y), (7.05, y)], S["edge"])

    trapezoid(ax, 9.4, 11.55, 6.3, 2.4, 1.25, S["chid"], "#bfe5ca",
              [("Kept checker", "bold"), ("readout head", "normal"), ("2,000 updates, 243 pairs", "normal")], S)
    line(ax, [(8.45, 7.6), (9.36, 6.95)], S["edge"])
    line(ax, [(8.45, 6.6), (9.36, 6.35)], S["edge"])
    line(ax, [(8.45, 5.6), (9.36, 5.75)], S["edge"])

    inset(ax, 3.3, 0.72, 8.3, 2.28, "Inside the kept checker", (9.4, 5.1), (11.55, 5.68), S)
    for k, y in enumerate((2.25, 1.75, 1.25)):
        ax.add_patch(Rectangle((3.55, y - 0.17), 0.36, 0.34, facecolor=S["fhid"], edgecolor=S["chid"], lw=0.8, zorder=3))
        ax.add_patch(Rectangle((4.25, y - 0.12), 0.5, 0.24, facecolor="white", edgecolor=S["chid"], lw=0.8, zorder=3))
        line(ax, [(3.93, y), (4.23, y)], S["chid"], ms=7, lw=0.9)
        line(ax, [(4.77, y), (5.22, 1.75 + (y - 1.75) * 0.25)], S["chid"], ms=7, lw=0.9)
    ax.text(3.73, 0.9, "each pooled\nto 32 x 32", ha="center", va="center", fontsize=7.0, color=S["ink"])
    ax.text(4.5, 0.9, "1 x 1 conv\nto 128", ha="center", va="center", fontsize=7.0, color=S["ink"])
    ax.add_patch(plt.Circle((5.38, 1.75), 0.15, facecolor="white", edgecolor=S["chid"], lw=1.2, zorder=4))
    ax.text(5.38, 1.75, "+", ha="center", va="center", fontsize=11, color=S["chid"], zorder=5)
    line(ax, [(5.55, 1.75), (5.9, 1.75)], S["chid"], ms=8, lw=1.0)
    volume(ax, 5.95, 1.55, 0.42, 0.3, 6, "two 3 x 3 convs,", "stride 2: 8 x 8")
    line(ax, [(6.75, 1.75), (7.1, 1.75)], S["chid"], ms=8, lw=1.0)
    for k in range(9):
        ax.add_patch(Rectangle((7.15 + k * 0.16, 1.62), 0.12, 0.26, facecolor=S["chid"] if k else S["accent"], alpha=0.45 if k else 0.85, lw=0, zorder=3))
    ax.text(7.87, 1.25, "64 tokens + 1 mean token", ha="center", va="center", fontsize=7.0, color=S["ink"])
    line(ax, [(8.62, 1.75), (9.0, 1.75)], S["chid"], ms=8, lw=1.0)
    ax.add_patch(FancyBboxPatch((9.05, 1.45), 1.0, 0.6, boxstyle="round,pad=0.02", fc="white", ec=S["chid"], lw=1.0, zorder=3))
    ax.text(9.55, 1.75, "attention\npool, 8 heads", ha="center", va="center", fontsize=7.0, color=S["ink"], zorder=4)
    line(ax, [(10.1, 1.75), (10.45, 1.75)], S["chid"], ms=8, lw=1.0)
    for k, y in enumerate((2.15, 1.75, 1.35)):
        ax.add_patch(plt.Circle((10.7, y), 0.13, facecolor=S["fout"], edgecolor=S["cout"], lw=1.3, zorder=4))
    ax.text(11.1, 1.75, "3\nlogits", ha="center", va="center", fontsize=7.2, color=S["ink"])
    trace(ax, (2.02, 4.55), (3.73, 2.42), S["cout"])

    formula_box(ax, 12.0, 5.75, 3.9, 2.7, "One sampling step",
                [(r"$\hat\epsilon = \epsilon_{CO3} - 30\,\sigma_t\,\nabla_{x_t} L$", S["accent"], 13.5),
                 (r"$L = \ell_{pair} + \ell_{puma} + \ell_{condor}$", S["ink"], 12.5)],
                note="the push is shortened to at most 0.4 of\n$\\|\\epsilon_{CO3}\\|$, on steps 0 to 15 of 50")
    line(ax, [(11.6, 6.3), (12.55, 5.05)], S["edge"])
    result_bars(ax, [12.7, 13.95, 15.2], 3.35, [1.45, 1.2, 0.95], ["clean pair", "puma", "condor"], [S["cin"], S["cin"], S["cin"]], S, width=0.62)
    ax.text(13.95, 5.15, "three answers, summed into $L$", ha="center", va="center", fontsize=9.5, weight="bold", color=S["ink"])
    ax.text(13.95, 2.82, "bar heights illustrative", ha="center", va="center", fontsize=7.8, color=S["ink"], style="italic")
    card(ax, 12.0, 0.92, 3.9, 1.32, "measured in step M1: 7 faithful pairs\nof 96 with the push, 11 for CO3 alone,\non the 12 tuning pairs", S, size=9.2,
         ec=S["accent"], fc="#fdeedd")
    line(ax, [(13.95, 0.88), (13.95, 0.32), (1.0, 0.32), (1.0, 3.25)], S["accent"], lw=2.4, ms=15)
    ax.text(7.3, 0.52, "the push goes back to the half-drawn picture", ha="center", va="center", fontsize=9.5, color=S["accent"], weight="bold")
    ax.text(8.05, 10.15, "Loop 6: CO3 with the kept checker's push, Paper 2's current method", ha="center", va="center",
            fontsize=14, weight="bold", color=S["ink"])
    return (-0.05, 16.1, 0.05, 10.5)


def loop_4(ax):
    """Loop 4: how T2I-CompBench's BLIP-VQA scores a picture, from ~/PhD/replications/t2i-compbench/BLIPvqa_eval:
    BLIP_vqa.py splits the prompt into spaCy noun phrases, asks one question per phrase and multiplies the answers;
    configs/vqa.yaml sets ViT 'base' at 480 pixels; models/blip.py builds ViT-B/16 with 12 layers of width 768; the question
    and answer transformers are BERT with 12 layers and cross-attention; inference 'vqa_prob' reads the yes probability.
    The targets are rows 1 to 3 of artifacts/notes/progress/scoreboard.json."""
    for x0, x1, name in ((0.2, 3.2, "What goes in"), (3.45, 5.95, "Split the prompt"), (6.2, 10.6, "BLIP-VQA"),
                         (10.85, 12.85, "Yes or no"), (13.1, 15.9, "Score and targets")):
        bracket(ax, x0, x1, 9.3, name, S)
    card(ax, 0.2, 7.2, 2.9, 1.0, "prompt\na blue bench\nand a green bowl", S, size=9.8, weight="bold")
    ax.text(1.5, 6.25, "finished picture, CoMat, seed 42", ha="center", va="center", fontsize=8.4, color=S["ink"])
    put_picture(ax, rgb_tile(COMAT, (400, 50, 800, 450)), 0.2, 3.3, 2.6)
    ax.add_patch(Rectangle((1.0, 4.35), 0.42, 0.42, fill=False, ec=S["cout"], lw=1.8, zorder=6))

    card(ax, 3.55, 7.35, 2.3, 0.62, "a blue bench?", S, size=10.5, weight="bold")
    card(ax, 3.55, 6.35, 2.3, 0.62, "a green bowl?", S, size=10.5, weight="bold")
    ax.text(4.7, 5.85, "one question per noun\nphrase, found by spaCy", ha="center", va="center", fontsize=8.2, color=S["ink"], style="italic")
    line(ax, [(3.12, 7.7), (3.52, 7.66)], S["edge"])
    line(ax, [(3.12, 7.55), (3.52, 6.75)], S["edge"])

    trapezoid(ax, 6.3, 8.1, 4.6, 2.3, 1.3, S["chid"], "#bfe5ca",
              [("Image", "bold"), ("transformer", "bold"), ("ViT-B/16", "normal"), ("12 layers", "normal")], S)
    line(ax, [(2.84, 4.6), (6.27, 4.6)], S["edge"])
    for y0, y1, title, sub in ((6.35, 7.95, "Question transformer", "BERT, 12 layers;\nreads the patches by\ncross-attention"),
                               (3.95, 5.25, "Answer transformer", "reads the first word:\nyes?")):
        ax.add_patch(FancyBboxPatch((8.4, y0), 2.15, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#dff1e5", ec=S["chid"], lw=1.5, zorder=3))
        ax.text(9.47, y1 - 0.25, title, ha="center", va="center", fontsize=9.2, weight="bold", color=S["ink"], zorder=4)
        ax.text(9.47, (y0 + y1) / 2 - 0.15, sub, ha="center", va="center", fontsize=7.9, color=S["ink"], zorder=4, linespacing=1.25)
    line(ax, [(5.88, 7.66), (8.37, 7.45)], S["edge"])
    line(ax, [(5.88, 6.66), (8.37, 6.95)], S["edge"])
    line(ax, [(8.12, 4.95), (8.25, 4.95), (8.25, 6.65), (8.37, 6.65)], S["chid"])
    line(ax, [(9.47, 6.32), (9.47, 5.28)], S["edge"])

    inset(ax, 3.3, 0.72, 7.3, 2.28, "Inside the image transformer", (6.3, 3.45), (8.1, 3.95), S)
    ax.add_patch(Rectangle((3.6, 1.05), 1.35, 1.35, facecolor="white", edgecolor=S["chid"], lw=1.0, zorder=3))
    for k in range(1, 10):
        ax.plot([3.6 + k * 1.35 / 10] * 2, [1.05, 2.4], color="#b9d8c2", lw=0.5, zorder=4)
        ax.plot([3.6, 4.95], [1.05 + k * 1.35 / 10] * 2, color="#b9d8c2", lw=0.5, zorder=4)
    ax.add_patch(Rectangle((3.6 + 3 * 0.135, 1.05 + 5 * 0.135), 0.135, 0.135, facecolor=S["cout"], zorder=5))
    ax.text(4.27, 0.88, "480 px: 30 x 30 patches", ha="center", va="center", fontsize=7.2, color=S["ink"], zorder=5)
    line(ax, [(5.02, 1.72), (5.45, 1.72)], S["chid"], ms=8, lw=1.0)
    for k in range(6):
        ax.add_patch(Rectangle((5.55 + k * 0.17, 1.27), 0.12, 0.9, facecolor=S["chid"], alpha=0.35 + 0.1 * k, lw=0, zorder=3))
    ax.text(6.0, 1.05, "12 layers", ha="center", va="center", fontsize=7.2, color=S["ink"])
    line(ax, [(6.6, 1.72), (7.0, 1.72)], S["chid"], ms=8, lw=1.0)
    for k in range(8):
        ax.add_patch(Rectangle((7.1 + k * 0.17, 1.4), 0.12, 0.62, facecolor=S["chid"], alpha=0.3 + 0.08 * k, lw=0, zorder=3))
    ax.text(9.75, 1.72, "901 vectors of 768:\none per patch,\nplus one summary", ha="center", va="center", fontsize=7.4, color=S["ink"])
    trace(ax, (1.21, 4.35), (3.6 + 3.5 * 0.135, 1.05 + 5.5 * 0.135), S["cout"])

    line(ax, [(10.58, 4.6), (11.0, 4.6), (11.0, 4.3)], S["edge"])
    result_bars(ax, [11.35, 12.4], 3.35, [1.1, 0.8], ["bench?", "bowl?"], [S["cin"], S["cin"]], S, width=0.56)
    ax.text(11.85, 5.0, "P(yes) per question", ha="center", va="center", fontsize=9.2, weight="bold", color=S["ink"])
    ax.text(11.85, 2.82, "bar heights illustrative", ha="center", va="center", fontsize=7.6, color=S["ink"], style="italic")

    formula_box(ax, 13.1, 5.8, 2.8, 2.6, "The picture's score",
                [(r"$P(\mathrm{yes}\mid \mathrm{bench}) \times P(\mathrm{yes}\mid \mathrm{bowl})$", S["ink"], 10.5)],
                note="then averaged over every\npicture in the subset")
    line(ax, [(12.75, 4.3), (13.6, 5.75)], S["edge"])
    ax.add_patch(FancyBboxPatch((13.1, 1.6), 2.8, 3.4, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fbe3e3", ec=S["cout"], lw=1.5, zorder=3))
    for y, t, size, weight in ((4.65, "Targets to beat", 10.5, "bold"), (4.15, "printed, 0 to 1, higher wins", 8.0, "normal"),
                               (3.55, "Color: CoMat 0.7827", 9.6, "normal"), (3.0, "Shape: ToMe 0.6051", 9.6, "normal"),
                               (2.45, "Texture: ToMe 0.6894", 9.6, "normal"), (1.9, "scoreboard.json, rows 1 to 3", 7.4, "normal")):
        ax.text(14.5, y, t, ha="center", va="center", fontsize=size, weight=weight, color=S["ink"], zorder=4)
    ax.add_patch(FancyBboxPatch((3.3, 0.12), 12.6, 0.42, boxstyle="round,pad=0.02", fc="#f1f2f4", ec=GREY, lw=1.0, ls="--", zorder=2))
    ax.text(9.6, 0.33, "No push here: a scorer never steers, because a scorer that moved the picture could no longer grade it",
            ha="center", va="center", fontsize=9.2, color=S["ink"], zorder=3)
    ax.text(8.05, 10.15, "Loop 4: how a picture earns its BLIP-VQA score, which sets the targets", ha="center", va="center",
            fontsize=14, weight="bold", color=S["ink"])
    return (-0.05, 16.1, 0.05, 10.5)


FIGURES = {"loop-8": loop_8, "loop-6": loop_6, "loop-4": loop_4}


def main():
    for name in sys.argv[1:] or list(FIGURES):
        fig, ax = plt.subplots(figsize=(16.6, 11.0), dpi=130, facecolor=S["bg"])
        ax.set_facecolor(S["bg"])
        x0, x1, y0, y1 = FIGURES[name](ax)
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal"), ax.axis("off")
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", facecolor=S["bg"])
        plt.close(fig)
        print(os.path.join(HERE, f"{name}.png"))


if __name__ == "__main__":
    main()
