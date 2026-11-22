"""The supervisor slides for the MNIST classifier-guidance experiment (plan 24, route step 7b).

Drawn in the form of the inclusive-captions set (4-inclusive-captions/artifacts/decks/supervisor-meeting/): a bold
title of at most ten words, one grey line saying what rows, columns and cells are, real canvases, one thin
outline colour, a count line under each cell, one orange "Look at:" line. No graphs, axes, bars or plots.
1920 x 1080, nothing below 13 pt.

Tile rules (also stated on each slide's grey line):
  1  per share, the first 20 training canvases the training generator draws (Canvases.training_draw, seed = the row's number 0, 1, 2), in the
     order drawn, kept so that the number holding two digits is the share times 20 rounded up: the first
     ceil(share x 20) two-digit canvases and the first 20 - that one-digit canvases, merged in drawing order
  2  one real call of the share-0.01 model: the first clean 4-and-7 canvas of the held-out-pair test canvases (MNIST
     test digits, rng 11), noised to step 25 of 50 (t = 489) with noise seed 3, asked for the digit 4 (called the noise level here, t)
Every count is read from its source at run time (sources/train_log-share-*.json, the checkpoint's own arguments and
weights) and every split is checked to add up; the check prints at the end.

    cd /home/molef/PhD/concept-composition
    /home/molef/micromamba/envs/sdxl-local/bin/python -I 2-classifier-guidance/artifacts/decks/supervisor-meeting/make_slides.py [1 2]
"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[2] / "scripts/mnist"))
INK, GREY, ORANGE, OUTLINE, BLUE, RED = "#1f2a33", "#6b7a88", "#c8680a", "#1d6b3a", "#1f5fd0", "#c0262d"
W, H = 1920, 1080
CHECKS = []


def check(what, ok):
    CHECKS.append((what, bool(ok)))
    assert ok, what


def sheet(title, line):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 42, title, ha="center", va="center", size=30, weight="bold", color=INK)
    ax.text(W / 2, H - 100, line, ha="center", va="center", size=15, color=GREY, linespacing=1.4)
    return fig, ax


def look_at(ax, s):
    ax.text(W / 2, 48, "Look at: " + s, ha="center", va="center", size=19, weight="bold", color=ORANGE)


def canvas(ax, a, x, y, w, ec="#b8c0c8", lw=0.8, rgb=False):
    """A 32 x 64 canvas, its lower left corner at (x, y), width w, height w / 2."""
    ax.imshow(a, cmap=None if rgb else "gray", vmin=None if rgb else 0, vmax=None if rgb else 255,
              extent=(x, x + w, y, y + w / 2), interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((x, y), w, w / 2, fc="none", ec=ec, lw=lw, zorder=3))


def pct(n, total):
    v = 100 * n / total
    return f"{v:.0f}%" if v == 0 or v >= 10 or v == int(v) else f"{v:.1f}%"


def short(n):
    """Three significant figures with k or M, for display only; the checks use the exact numbers."""
    if n >= 1e6:
        return f"{n / 1e6:.3g}M" if n < 1e7 else f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.3g}k"
    return str(n)


def count(n, total):
    return f"{short(n)} of {short(total)} ({pct(n, total)})"


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=100)
    fig.savefig(HERE / f"{name}.svg")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ slide 1: the training canvases and the share
def slide1():
    from canvas_sheet import Canvases as C
    from canvas_unet import SEEN, load_mnist
    mn = Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw")
    images, labels = load_mnist(mn)
    shares = [0.0, 0.01, 0.1]
    fig, ax = sheet("Share: the fraction of training canvases that hold two digits",
                    "Rows: the share the digit model was trained at. Cells: 20 real training canvases, in the order drawn, the ones "
                    "with two digits outlined;\nthe number outlined is the share times 20, rounded up. Canvases are 32 × 64, rebuilt at every step from MNIST's 60k training digits, not stored.")
    tw, th, gap = 140, 70, 6
    per_row, rows_per_block = 10, 2
    x0 = 330
    top = H - 180
    pitch = 246
    for r, s in enumerate(shares):
        log = json.load(open(HERE / "sources" / f"train_log-share-{s}.json"))
        two = log[-1]["two_digit_canvases"]
        examples = 40000 * 128
        draws = examples - two                    # a two-digit canvas gives one training example per digit
        one = draws - two
        check(f"share {s}: the log ends at step 40,000", log[-1]["step"] == 40000)
        check(f"share {s}: two-digit canvases are the share of all canvases within 0.05 percentage points",
              abs(two / draws - s) < 0.0005)
        check(f"share {s}: one-digit + two-digit canvases add up to all canvases", one + two == draws)
        # the tiles
        rng = np.random.default_rng(r)
        cv = C(images, labels, rng)
        need2 = math.ceil(round(s * 20, 9)); need1 = 20 - need2
        got2, got1, seq = 0, 0, []
        while got2 < need2 or got1 < need1:
            c, is2 = cv.training_draw(s, SEEN)
            if is2 and got2 < need2:
                seq.append((c, True)); got2 += 1
            elif (not is2) and got1 < need1:
                seq.append((c, False)); got1 += 1
        check(f"share {s}: {need2} two-digit tiles of 20", sum(t[1] for t in seq) == need2 and len(seq) == 20)
        ytop = top - r * pitch
        ax.text(x0 - 40, ytop - (rows_per_block * th + gap) / 2, f"share {s:g}", ha="right", va="center", size=24,
                weight="bold", color=INK)
        for j, (c, is2) in enumerate(seq):
            col, row = j % per_row, j // per_row
            canvas(ax, c, x0 + col * (tw + gap), ytop - th - row * (th + gap), tw,
                   ec=OUTLINE if is2 else "#b8c0c8", lw=3.0 if is2 else 0.8)
        yb = ytop - rows_per_block * th - gap - 28
        ax.text(x0 + (per_row * tw + (per_row - 1) * gap) / 2, yb,
                f"{count(two, draws)} canvases drawn in training hold two digits", ha="center", va="center", size=19, weight="bold", color=ORANGE)
        ax.text(x0 + (per_row * tw + (per_row - 1) * gap) / 2, yb - 30,
                f"{short(one)} hold one digit · {short(two)} hold two · {short(draws)} drawn in all, over 40k steps of 128 examples",
                ha="center", va="center", size=15, color=GREY)
    ax.text(W / 2, 116, "Five pairs never share a training canvas: 0 & 3, 1 & 9, 3 & 4, 3 & 6, 4 & 7. Two-digit canvases hold one digit in each half,\n"
            "from the other 40 pairs; one-digit canvases put the digit anywhere.", ha="center", va="center", size=15, color=INK, linespacing=1.45)
    look_at(ax, "at share 0 no canvas holds two digits; at 0.1 one canvas in ten does.")
    save(fig, "01-the-training-canvases-and-the-share")


# ------------------------------------------------------------------ slide 2: the digit model
def slide2():
    import torch
    from canvas_sheet import Canvases as C, D, H as CH, W as CW
    from canvas_unet import ABAR
    from checker import STEPS, load_unet
    ck_path = REPO / "2-classifier-guidance/data/mnist-canvas/24-mnist-one-model/arch/models/share-0.01/unet.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    unet = load_unet(ck_path, "cpu")
    weights = sum(p.numel() for p in unet.parameters())
    steps, batch = ck["args"]["steps"], ck["args"]["batch"]
    check("the checkpoint is at its final step", ck["step"] == steps)
    mn = Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw")
    timg = np.fromfile(mn / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
    tlbl = np.fromfile(mn / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    clean = C(timg, tlbl, np.random.default_rng(11)).two(4, 7)
    t = STEPS[25].view(1); a = ABAR[t]
    noise = torch.randn(1, 1, CH, CW, generator=torch.Generator().manual_seed(3))
    x0 = torch.from_numpy(clean).float().div(127.5).sub(1)[None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    with torch.no_grad():
        eps = unet(xt, t, torch.tensor([4]))
    to8 = lambda v: ((v.clamp(-1, 1) + 1) * 127.5).numpy()[0, 0]
    eps8 = ((eps.clamp(-3, 3) + 3) / 6 * 255).numpy()[0, 0]
    check("the model's weights count matches the saved checkpoint's", weights == sum(v.numel() for v in ck["ema"].values()))
    print(f"slide 2: weights {weights:,}, call at t = {int(t)}, noise estimate mean {eps.mean():.3f} sd {eps.std():.3f}")

    fig, ax = sheet("The digit model: a noisy canvas, a level and a digit in, noise out",
                    "One real call of the share-0.01 model on a 32 × 64 canvas, asked for the digit 4. "
                    "Each picture is what goes in or out; the grey line under it says its size.")
    cw_ = 440; ch_ = cw_ / 2
    ycen = 550
    # training canvas -> noisy canvas -> model -> noise estimate
    xs = [70, 590]
    canvas(ax, to8(x0), xs[0], ycen - ch_ / 2, cw_, ec=GREY, lw=1.5)
    ax.text(xs[0] + cw_ / 2, ycen + ch_ / 2 + 36, "training canvas", ha="center", size=19, weight="bold", color=INK)
    ax.text(xs[0] + cw_ / 2, ycen - ch_ / 2 - 28, "1 × 32 × 64, pixels from −1 to 1", ha="center", size=15, color=GREY)
    canvas(ax, to8(xt), xs[1] + 40, ycen - ch_ / 2, cw_, ec=GREY, lw=1.5)
    ax.text(xs[1] + 40 + cw_ / 2, ycen + ch_ / 2 + 36, "noisy canvas", ha="center", size=19, weight="bold", color=INK)
    ax.text(xs[1] + 40 + cw_ / 2, ycen - ch_ / 2 - 28, f"noise added up to level 489 of 999", ha="center", size=15, color=GREY)
    arrow = dict(arrowstyle="-|>", color=GREY, lw=2.2, mutation_scale=22)
    ax.annotate("", xy=(xs[1] + 40 - 10, ycen), xytext=(xs[0] + cw_ + 10, ycen), arrowprops=arrow)
    ax.text((xs[0] + cw_ + xs[1] + 30) / 2, ycen + 24, "+ noise", ha="center", size=16, color=GREY)
    # the model box
    mx, mw, mh = 1150, 300, 330
    ax.add_patch(FancyBboxPatch((mx, ycen - mh / 2), mw, mh, boxstyle="round,pad=0,rounding_size=10", fc="white", ec=INK, lw=2.6))
    ax.text(mx + mw / 2, ycen + 105, "digit model", ha="center", size=24, weight="bold", color=INK)
    ax.text(mx + mw / 2, ycen + 62, "a U-Net", ha="center", size=17, color=GREY)
    ax.text(mx + mw / 2, ycen - 40, "grids 32×64, 16×32, 8×16\n64 and 128 channels", ha="center", size=15, color=GREY, linespacing=1.5)
    ax.text(mx + mw / 2, ycen - 100, f"{weights:,} weights", ha="center", size=17, weight="bold", color=INK)
    ax.annotate("", xy=(mx - 10, ycen), xytext=(xs[1] + 40 + cw_ + 10, ycen), arrowprops=arrow)
    # step and digit boxes on top
    for bx, head, sub in ((mx - 120, "noise level", "0 to 999"), (mx + 150, "digit", "0 to 9, or none")):
        ax.add_patch(FancyBboxPatch((bx, ycen + mh / 2 + 70), 260, 100, boxstyle="round,pad=0,rounding_size=10", fc="white", ec=GREY, lw=1.6))
        ax.text(bx + 130, ycen + mh / 2 + 135, head, ha="center", size=20, weight="bold", color=INK)
        ax.text(bx + 130, ycen + mh / 2 + 98, sub, ha="center", size=16, color=GREY)
        ax.annotate("", xy=(bx + 130 if bx < mx else mx + mw - 70, ycen + mh / 2 + 6), xytext=(bx + 130, ycen + mh / 2 + 66),
                    arrowprops=arrow)
    ax.text(mx + 290, ycen + mh / 2 + 188, "asked here for the digit 4", ha="center", size=15, color=GREY)
    # output
    ox = 1560
    canvas(ax, eps8, ox, ycen - 80, 320, ec=GREY, lw=1.5)
    ax.annotate("", xy=(ox - 10, ycen), xytext=(mx + mw + 10, ycen), arrowprops=arrow)
    ax.text(ox + 160, ycen + 80 + 36, "noise estimate", ha="center", size=19, weight="bold", color=INK)
    ax.text(ox + 160, ycen - 80 - 28, "1 × 32 × 64, one value per pixel", ha="center", size=15, color=GREY)
    ax.text(W / 2, 290, f"Trained for {short(steps)} steps of {batch} examples = {short(steps * batch)}, rebuilt at every step from MNIST's 60k training digits.",
            ha="center", size=19, color=INK)
    ax.text(W / 2, 240, "The label is replaced by “none” on 10% of examples, so the same model also answers with no digit asked. "
            "The sampler asks it three times per step.", ha="center", size=15, color=GREY)
    look_at(ax, "the digit goes in beside the noise level, and what comes out is the noise, not a cleaner canvas.")
    save(fig, "02-the-digit-model")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2"]
    if "1" in want:
        slide1()
    if "2" in want:
        slide2()
    print(f"\n{sum(ok for _, ok in CHECKS)} of {len(CHECKS)} checks passed:")
    for what, ok in CHECKS:
        print(("  ok  " if ok else "  FAIL"), what)
