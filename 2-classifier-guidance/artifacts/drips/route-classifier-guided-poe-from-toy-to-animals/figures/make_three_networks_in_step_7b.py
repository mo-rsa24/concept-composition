"""Zoom ladder level 0.5: the three networks of step 7b, one row each, so they are never confused.

Row 1 is the pair-aware checker (the claim), row 2 the separate-label control, row 3 the judge,
which only scores. Each row reads left to right: what goes in, the network as one box, what comes
out, what it is used for.

What is real and what is not:
  - every canvas is real MNIST training digits pasted by scripts/mnist/canvas_sheet.py. The 7 beside 8
    canvas is the fifth canvas of mnist-judge-ladder.png's top row (Canvases seed 1), the same one
    judge-network-view.png reads; the training examples in row 1 come from Canvases seed 2;
  - the noisy canvas is that canvas scaled to -1..1 and mixed with real Gaussian noise (torch seed 0)
    at alpha_bar = 0.5, x_t = sqrt(0.5) x + sqrt(0.5) eps, a level picked so the digits still show;
  - rows 1 and 2 are not trained: their outputs are empty boxes, and their heads are drawn as
    illustrative shapes, with no sizes;
  - row 3's reads (7 at 0.9998, 8 at 0.9728) and weight count are copied from
    judge_network_view_numbers.json (prototype weights), and the blob crops are recomputed here with
    judge_sheet.py's own blobs and crop28;
  - the toy counts (172 of 300 against at most 10 of 300) are from
    report/classifier-guided-poe/a1-exact-toy-classifier-push.md.

The figure is refused (nothing written) by the earlier script's check if any line crosses a label,
an image or a node it is not joined to, or if two labels overlap.

Usage:
    python make_three_networks_in_step_7b.py [--mnist <raw dir>]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_judge_network_view as base  # noqa: E402
from canvas_sheet import Canvases, load_mnist  # noqa: E402  (path added by base)
from judge_sheet import CONFIDENT, blobs, crop28  # noqa: E402

INPUT, HIDDEN, OUTPUT, OFF = base.INPUT, base.HIDDEN, base.OUTPUT, base.OFF
TEXT, MUTED, EDGE = base.TEXT, base.MUTED, base.EDGE
PUSH = "#c86a0a"  # ladder.json colour for backward signals and the push
W, H = 2400, 1470
base.W, base.H = W, H  # the base Fig and check read these
ALPHA_BAR = 0.5
tint = base._tint


def build(mnist: Path):
    images, labels = load_mnist(mnist)
    cv = Canvases(images, labels, np.random.default_rng(1))
    ladder = [cv.one(7), cv.overlaid(7, 8), cv.two(7, 1), cv.two(7, 7), cv.two(7, 8)]
    canvas, fused = ladder[4], ladder[1]
    ex = Canvases(images, labels, np.random.default_rng(2))
    examples = dict(pos78=ex.two(7, 8), pos35=ex.two(3, 5), fused=ex.overlaid(7, 8), one=ex.one(7),
                    wrong=ex.two(7, 7))
    x0 = torch.tensor(canvas, dtype=torch.float32) / 255.0 * 2 - 1
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0))
    xt = ALPHA_BAR ** 0.5 * x0 + (1 - ALPHA_BAR) ** 0.5 * eps
    noisy = ((xt.clamp(-1, 1) + 1) / 2).numpy()
    found = sorted(blobs(canvas), key=lambda b: b[1].min())
    crops = [crop28(canvas, ys, xs)[0, 0].numpy() for ys, xs in found]
    rec = json.loads((HERE / "judge_network_view_numbers.json").read_text())
    reads = [(b["top_digit"], b["top_probability"]) for b in rec["blobs"]]
    return dict(canvas=canvas, noisy=noisy, fused=fused, examples=examples, crops=crops, reads=reads,
                call=rec["call"], total_weights=rec["total_weights"])


def image(F, name, img, x, y, px, edge, vmax=255):
    h, w = img.shape
    F.ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest",
                extent=(x, x + w * px, y + h * px, y), zorder=4)
    F.ax.add_patch(mp.Rectangle((x, y), w * px, h * px, fill=False, ec=edge, lw=1.4, zorder=5))
    F.rect(name, x, y, x + w * px, y + h * px)
    return x + w * px, y + h * px


def chip(F, name, x0, y0, x1, y1, s, color=INPUT):
    F.ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=8",
                                     fc=tint(color, 0.9), ec=color, lw=1.2, zorder=3))
    F.rect("box-" + name, x0, y0, x1, y1)
    F.text((x0 + x1) / 2, (y0 + y1) / 2, s, 13, color, bold=True, inside=True)


def panel(F, name, x0, y0, x1, y1, color, dashed=False, glow=False, fill="white"):
    if glow:
        for k, a in ((10, 0.05), (6, 0.08), (3, 0.12)):
            F.ax.add_patch(mp.Rectangle((x0 - k, y0 - k), x1 - x0 + 2 * k, y1 - y0 + 2 * k, fc=color,
                                        alpha=a, lw=0, zorder=1))
    F.ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=10",
                                     fc=fill, ec=color, lw=1.4 if not dashed else 1.4,
                                     ls=(0, (6, 4)) if dashed else "-", zorder=2))
    F.rect("box-" + name, x0, y0, x1, y1)


def arrow(F, x0, x1, y, color=OFF, lw=2.6, allow=()):
    F.line([(x0, y), (x1, y)], color, lw, allow=set(allow), arrow=True)


def unet_with_head(F, tag, x, yc, n_out):
    """The frozen U-Net as five grey bars (down, bottom, up) and a small green head reading its
    decoder side. Illustrative shapes: no sizes, since nothing here is trained yet."""
    hs = [120, 82, 50, 82, 120]
    for i, h in enumerate(hs):
        F.ax.add_patch(mp.Rectangle((x + i * 34, yc - h / 2), 18, h, fc=tint(OFF, 0.8), ec=OFF, lw=0.9,
                                    zorder=3))
    for i in range(2):  # skip links, faint
        a, b = x + 9 + i * 34, x + 9 + (4 - i) * 34
        top = yc - hs[i] / 2 - 10 - i * 8
        F.ax.plot([a, a, b, b], [yc - hs[i] / 2, top, top, yc - hs[4 - i] / 2], color=tint(OFF, 0.4),
                  lw=0.8, zorder=2)
    F.rect(f"{tag}-unet", x, yc - 60 - 30, x + 4 * 34 + 18, yc + 60)
    F.text(x - 12, yc + 82, "MNIST U-Net, frozen", 12, OFF, ha="left", bold=True)
    F.text(x - 12, yc + 102, "the head reads its decoder", 11, MUTED, ha="left")
    # the head
    hx, ox = x + 270, x + 370
    hy = [yc - 45, yc - 15, yc + 15, yc + 45]
    F.line([(x + 4 * 34 + 26, yc), (hx - 30, yc)], OFF, 2.0, allow={f"{tag}-unet"}, arrow=True)
    for i, y in enumerate(hy):
        F.circle(f"{tag}-h{i}", hx, y, 10, HIDDEN)
    oys = [yc] if n_out == 1 else [yc - 30, yc + 30]
    for k, y in enumerate(oys):
        F.circle(f"{tag}-o{k}", ox, y, 13, OUTPUT, glow=True)
        for i, hy_ in enumerate(hy):
            a, b = np.array([hx, hy_]), np.array([ox, y])
            u = (b - a) / np.linalg.norm(b - a)
            F.line([tuple(a + u * 10), tuple(b - u * 13)], EDGE, 0.7, allow={f"{tag}-h{i}", f"{tag}-o{k}"})
    F.text(hx + 80, yc + 82, "small head, learnt", 12, HIDDEN, bold=True)
    F.text(hx + 80, yc + 102, "illustrative: no size set yet", 11, MUTED)
    return ox + 13, oys


def row_title(F, y, num, title, note):
    F.ax.plot([40, W - 40], [y - 28, y - 28], color=tint(OFF, 0.6), lw=1.0, zorder=1)
    t = F.text(40, y, f"{num}   {title}", 21, ha="left", bold=True)
    F.fig.canvas.draw()
    bb = t.get_window_extent(F.fig.canvas.get_renderer())
    x_end = F.ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    F.text(x_end + 24, y + 1, note, 14, MUTED, ha="left")


def column_heads(F, y):
    for x, s in ((60, "what goes in"), (650, "the network, as one box"), (1270, "what comes out"),
                 (1730, "what it is used for")):
        F.text(x, y, s.upper(), 12, MUTED, ha="left", bold=True)


def draw(N):
    F = base.Fig()
    ax = F.ax
    F.text(40, 42, "Step 7b has three networks with three different jobs", 30, ha="left", bold=True)
    F.text(40, 86, "Rows 1 and 2 push the sampler while it draws; row 3 only scores finished canvases. "
           "Rows 1 and 2 are not trained yet, so their outputs are empty.", 16, MUTED, ha="left")
    column_heads(F, 130)

    # ================= Row 1: the pair-aware checker =================
    row_title(F, 190, "1", "The pair-aware checker: the claim",
              "the MNIST twin of SDXL's cat × dog checker, which asks \"is this becoming a clean cat and dog?\"")
    yc = 335
    rx, ry = image(F, "r1-canvas", N["noisy"], 60, yc - 80, 5, INPUT, vmax=1)
    F.text(60, ry + 20, "the whole noisy canvas, 32 × 64", 13, INPUT, ha="left", bold=True)
    F.text(60, ry + 42, f"real Gaussian noise at ᾱ = {ALPHA_BAR} (half signal, half noise)", 11, MUTED,
           ha="left")
    chip(F, "r1-step", 395, yc - 70, 585, yc - 30, "step t")
    chip(F, "r1-pair", 395, yc - 10, 585, yc + 30, "asked pair (7, 8)")
    arrow(F, 593, 643, yc)
    ax.add_patch(mp.FancyBboxPatch((650, yc - 115), 520, 260, boxstyle="round,pad=0,rounding_size=12",
                                   fill=False, ec=tint(TEXT, 0.5), lw=1.0, zorder=1))
    out_x, oys = unet_with_head(F, "r1", 680, yc - 20, 1)
    arrow(F, 1180, 1262, yc - 20, OFF, allow={"box-r1-out"})
    panel(F, "r1-out", 1270, yc - 60, 1450, yc + 20, OUTPUT)
    F.text(1360, yc - 32, "ONE number", 14, OUTPUT, bold=True, inside=True)
    F.text(1360, yc - 6, "not trained yet", 12, MUTED, inside=True)
    F.text(1270, yc + 48, "log-odds that the canvas is becoming", 12, TEXT, ha="left")
    F.text(1270, yc + 68, "two clear, right digits: a 7 and an 8, apart", 12, TEXT, ha="left")
    arrow(F, 1458, 1722, yc - 20, PUSH, allow={"box-r1-out", "box-r1-use"})
    F.text(1590, yc - 44, "its gradient, ∇ log c", 12, PUSH, bold=True)
    panel(F, "r1-use", 1730, yc - 100, 2360, yc + 70, PUSH, glow=True)
    F.text(1755, yc - 72, "the push, added at every sampler step", 14, TEXT,
           ha="left", bold=True, inside=True)
    F.text(2045, yc - 20, r"$\epsilon_7 + \epsilon_8 - \epsilon_{\rm null} - s\,\sigma_t\,\nabla \log c$",
           22, PUSH, inside=True)
    F.text(2045, yc + 36, "the product's prediction, then the checker's push; s sets its strength",
           11, MUTED, inside=True)

    # trained on: real example canvases
    ty = 495
    F.text(60, ty, "Trained on", 14, TEXT, ha="left", bold=True)
    F.text(200, ty, "(examples pasted by canvas_sheet.py; the real training set is not built yet)", 11,
           MUTED, ha="left")
    F.text(60, ty + 32, "positives: a separate 7 and 8, and other seen pairs", 12, HIDDEN, ha="left",
           bold=True)
    F.text(640, ty + 32, "negatives: everything that is not two clear, right digits", 12, OUTPUT,
           ha="left", bold=True)
    ex, px, y0 = N["examples"], 2.5, ty + 52
    tiles = [("pos78", 60, "a 7 and an 8, apart", HIDDEN), ("pos35", 250, "3 and 5, a seen pair", HIDDEN),
             ("fused", 640, "fused 7 and 8", OUTPUT), ("one", 830, "one digit", OUTPUT),
             ("wrong", 1020, "wrong pair: 7 beside 7", OUTPUT)]
    for key, x, cap, col in tiles:
        image(F, f"ex-{key}", ex[key], x, y0, px, col)
        F.text(x, y0 + 80 + 18, cap, 11, col, ha="left")
    panel(F, "r1-own", 1210, y0, 1560, y0 + 80, OUTPUT, dashed=True)
    F.text(1385, y0 + 28, "the plain product's own draws", 12, OUTPUT, bold=True, inside=True)
    F.text(1385, y0 + 54, "(none drawn yet: the U-Net is not trained)", 11, MUTED, inside=True)

    # ================= Row 2: the separate-label control =================
    row_title(F, 690, "2", "The separate-label control",
              "the same kind of head, asked two questions about one digit each")
    yc = 830
    rx, ry = image(F, "r2-canvas", N["noisy"], 60, yc - 80, 5, INPUT, vmax=1)
    F.text(60, ry + 20, "the same noisy canvas", 13, INPUT, ha="left", bold=True)
    F.text(60, ry + 42, "no pair goes in", 11, MUTED, ha="left")
    chip(F, "r2-step", 395, yc - 20, 585, yc + 20, "step t")
    arrow(F, 593, 643, yc)
    ax.add_patch(mp.FancyBboxPatch((650, yc - 115), 520, 260, boxstyle="round,pad=0,rounding_size=12",
                                   fill=False, ec=tint(TEXT, 0.5), lw=1.0, zorder=1))
    out_x, oys = unet_with_head(F, "r2", 680, yc - 20, 2)
    F.text(1360, yc - 105, "TWO numbers", 14, OUTPUT, bold=True)
    for k, (q, y) in enumerate((("is there a 7?", oys[0]), ("is there an 8?", oys[1]))):
        F.line([(out_x + 8, y), (1262, y)], OFF, 1.6, allow={f"box-r2-out{k}"}, arrow=True)
        panel(F, f"r2-out{k}", 1270, y - 28, 1450, y + 28, OUTPUT)
        F.text(1360, y - 9, q, 13, OUTPUT, bold=True, inside=True)
        F.text(1360, y + 13, "not trained yet", 11, MUTED, inside=True)
    F.line([(1458, yc - 20), (1722, yc - 20)], PUSH, 2.6, allow={"box-r2-use"}, arrow=True)
    F.line([(1458, oys[0]), (1490, oys[0]), (1490, oys[1]), (1458, oys[1])], PUSH, 1.6,
           allow={"box-r2-out0", "box-r2-out1"})
    F.text(1600, yc - 44, "both pushed up", 12, PUSH, bold=True)
    panel(F, "r2-use", 1730, yc - 105, 2360, yc + 130, PUSH)
    F.text(1755, yc - 78, "the control push, and why it cannot fix blending", 14, TEXT, ha="left",
           bold=True, inside=True)
    image(F, "r2-fused", N["fused"], 1755, yc - 50, 2.5, OUTPUT)
    F.text(1755, yc + 50, "a fused 7 and 8 (pasted)", 11, OUTPUT, ha="left", inside=True)
    for i, s in enumerate(("A fused 7 and 8 can score high on both,",
                           "so pushing both up can make a better",
                           "blend, and still not two digits.")):
        F.text(1940, yc - 38 + i * 24, s, 13, TEXT, ha="left", inside=True)
    F.text(1755, yc + 84, "On the exact toy: at most 10 of 300 crisp pairs with this push,", 12, TEXT,
           ha="left", inside=True)
    F.text(1755, yc + 106, "against 172 of 300 with the pair-aware push (strength 1).", 12, TEXT,
           ha="left", inside=True)

    # ================= Row 3: the judge =================
    row_title(F, 1030, "3", "The judge: scoring only",
              "it never pushes, and never shares training data with rows 1 and 2")
    yc = 1175
    rx, ry = image(F, "r3-canvas", N["canvas"], 60, yc - 80, 5, INPUT)
    F.text(60, ry + 20, "a FINISHED canvas, never a noisy one", 13, INPUT, ha="left", bold=True)
    F.text(60, ry + 42, "the same 7 beside 8, before any noise", 11, MUTED, ha="left")
    chip(F, "r3-pair", 395, yc - 20, 585, yc + 20, "asked pair (7, 8)")
    arrow(F, 593, 643, yc)
    ax.add_patch(mp.FancyBboxPatch((650, yc - 115), 520, 260, boxstyle="round,pad=0,rounding_size=12",
                                   fill=False, ec=tint(TEXT, 0.5), lw=1.0, zorder=1))
    panel(F, "r3-cut", 665, yc - 100, 885, yc + 70, OFF, dashed=True, fill="none")
    F.text(680, yc - 80, "RULE, NOT LEARNT", 10, OFF, ha="left", bold=True, inside=True)
    F.text(680, yc - 60, "cut each blob of ink", 12, TEXT, ha="left", bold=True, inside=True)
    for i, crop in enumerate(N["crops"]):
        image(F, f"r3-crop{i}", crop, 690 + i * 95, yc - 35, 2.5, INPUT, vmax=1)
        F.text(690 + i * 95 + 35, yc + 50, f"{'left' if i == 0 else 'right'} blob", 11, INPUT, inside=True)
    F.text(775, yc + 100, "28 × 28 each", 11, MUTED)
    # the digit classifier, small
    arrow(F, 891, 913, yc - 10, OFF, lw=2.0)
    vx = base.volume(F, "r3-v1", 918, yc - 10, 50, 1, HIDDEN)
    vx = base.volume(F, "r3-v2", vx + 14, yc - 10, 36, 16, HIDDEN)
    vx = base.volume(F, "r3-v3", vx + 14, yc - 10, 24, 32, HIDDEN)
    ox = 1130
    oy = [yc - 52, yc - 24, yc + 4, yc + 32]
    for k, y in enumerate(oy):
        F.circle(f"r3-o{k}", ox, y, 9, OUTPUT, glow=(k == 2))
    F.text(ox, yc + 52, "⋮", 13, OUTPUT)
    for k, y in enumerate(oy):
        a, b = np.array([vx + 2, yc - 10]), np.array([ox, y])
        u = (b - a) / np.linalg.norm(b - a)
        F.line([tuple(a), tuple(b - u * 9)], EDGE, 0.7, allow={"r3-v3", f"r3-o{k}"})
    F.text(1030, yc + 82, "digit classifier, learnt", 12, HIDDEN, bold=True)
    F.text(1030, yc + 102, f"on single digits; {N['total_weights']:,} weights", 11, MUTED)
    arrow(F, 1180, 1262, yc - 10, OFF)
    F.text(1270, yc - 100, "a digit and how sure, per blob", 13, OUTPUT, ha="left", bold=True)
    (d0, p0), (d1, p1) = N["reads"]
    F.text(1270, yc - 72, f"left blob: {d0} at {p0:.3f}", 13, OUTPUT, ha="left")
    F.text(1270, yc - 48, f"right blob: {d1} at {p1:.3f}", 13, OUTPUT, ha="left")
    panel(F, "r3-ladder", 1270, yc - 28, 1650, yc + 72, OFF, dashed=True, fill="none")
    F.text(1285, yc - 10, "RULE, NOT LEARNT: the ladder", 10, OFF, ha="left", bold=True, inside=True)
    F.text(1285, yc + 14, f"both sure (at least {CONFIDENT}), one 7 and one 8", 11, MUTED, ha="left",
           inside=True)
    F.text(1285, yc + 44, f"one call: {N['call']}", 14, OUTPUT, ha="left", bold=True, inside=True)
    arrow(F, 1658, 1722, yc + 22, OFF, allow={"box-r3-ladder", "box-r3-use"})
    panel(F, "r3-use", 1730, yc - 75, 2360, yc + 110, OFF)
    F.text(1755, yc - 48, "counting results only", 14, TEXT, ha="left", bold=True, inside=True)
    for i, s in enumerate(("It never pushes the sampler.",
                           "It never shares training data with rows 1 and 2,",
                           "so the push cannot learn to fool its own scorer.")):
        F.text(1755, yc - 12 + i * 26, s, 13, TEXT, ha="left", inside=True)
    F.text(1755, yc + 82, "row 3's reads are the prototype judge's (judge-network-view.png)", 11,
           MUTED, ha="left", inside=True)

    # ================= Footer =================
    F.text(40, 1370, "Every canvas is real MNIST training digits pasted by canvas_sheet.py; the 7 beside 8 "
           "is the fifth canvas of mnist-judge-ladder.png's top row (Canvases seed 1), and rows 1 and 2 "
           f"see it with noise added at ᾱ = {ALPHA_BAR}.", 12, MUTED, ha="left")
    F.text(40, 1396, "Illustrative: the heads in rows 1 and 2 (no sizes, nothing trained). Row 3's numbers "
           "come from judge_network_view_numbers.json; the toy counts from "
           "report/classifier-guided-poe/a1-exact-toy-classifier-push.md.", 12, MUTED, ha="left")
    F.text(40, 1422, "Drawn by make_three_networks_in_step_7b.py.", 12, MUTED, ha="left")
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path,
                    default=Path("/home/molef/PhD/replications/original-superdiff/data/MNIST/raw"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    N = build(args.mnist)
    F = draw(N)
    base.check(F)
    F.fig.savefig(HERE / "three-networks-in-step-7b.png", dpi=100)
    F.fig.savefig(HERE / "three-networks-in-step-7b.svg")
    print("wrote three-networks-in-step-7b.png and .svg; judge reads", N["reads"], "call", N["call"])


if __name__ == "__main__":
    main()
