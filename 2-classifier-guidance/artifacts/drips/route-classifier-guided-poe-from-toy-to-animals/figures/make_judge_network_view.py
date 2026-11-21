"""Zoom ladder level 0.5: the step 7b judge drawn as a network, with its two rules around it.

Every number on the figure comes from a run in this script: the canvas is the fifth canvas of
mnist-judge-ladder.png ("7 beside 8"), rebuilt with judge_sheet.py's own functions and the same
random seed; the blobs, crops, layer sizes, weight counts, probabilities and test accuracy come from
the prototype weights. The numbers are written to judge_network_view_numbers.json beside the PNG.

Style: journey 13's recorded choice, cheat-sheet devices (thin outlines, a representative fan of
arrows, the top output glowing, boxed formulas, white space) in the ladder colour key.

The figure is refused (nothing written) if any line crosses a label, a node or a volume it is not
joined to, or if two labels overlap.

Usage:
    python make_judge_network_view.py [--mnist <raw dir>] [--weights <pt>]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "scripts" / "mnist"))
from canvas_sheet import Canvases, load_mnist  # noqa: E402
from judge_sheet import CONFIDENT, INK, MIN_BLOB, blobs, crop28, judge, judge_net  # noqa: E402

# The ladder colour key (journey 13 ladder.json), one meaning per colour.
INPUT, HIDDEN, OUTPUT, OFF = "#1f5fd0", "#1d7a3e", "#c0262d", "#6b7a88"
TEXT, MUTED, EDGE = "#1f2933", "#6b7a88", "#9aa5b1"
W, H = 2500, 990


class Fig:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W), self.ax.set_ylim(H, 0), self.ax.axis("off")
        self.segments, self.circles, self.rects = [], [], []

    def text(self, x, y, s, size=13, color=TEXT, ha="center", va="center", bold=False, inside=False):
        t = self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                         weight="bold" if bold else "normal", zorder=6)
        t.set_gid("inside" if inside else "label")
        return t

    def line(self, pts, color=EDGE, lw=1.0, allow=(), arrow=False, ls="-", z=2):
        for a, b in zip(pts[:-1], pts[1:]):
            self.segments.append((np.array(a, float), np.array(b, float), set(allow)))
        xs, ys = zip(*pts)
        if arrow:
            self.ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=z,
                         solid_capstyle="butt")
            self.ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                             mutation_scale=14 + lw * 2, shrinkA=0, shrinkB=0))
        else:
            self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z)

    def circle(self, name, cx, cy, r, color, glow=False):
        if glow:
            for k, a in ((2.0, 0.08), (1.6, 0.14), (1.3, 0.22)):
                self.ax.add_patch(mp.Circle((cx, cy), r * k, color=color, alpha=a, lw=0, zorder=3))
        self.ax.add_patch(mp.Circle((cx, cy), r, facecolor=_tint(color, 0.82), edgecolor=color,
                                    lw=1.6 if glow else 1.0, zorder=4))
        self.circles.append((name, cx, cy, r))

    def rect(self, name, x0, y0, x1, y1):
        self.rects.append((name, x0, y0, x1, y1))


def _tint(hexcol, f):
    c = np.array(matplotlib.colors.to_rgb(hexcol))
    return tuple(c + (1 - c) * f)


def volume(F, name, x, yc, side, channels, color, img=None):
    """A stack of feature maps: the face is the spatial size, the depth the channel count."""
    d = 4 + channels * 0.62
    dx, dy = d * 0.8, -d * 0.6
    yt, yb = yc - side / 2, yc + side / 2
    ax = F.ax
    edge, fill = color, _tint(color, 0.86)
    top = [(x, yt), (x + dx, yt + dy), (x + side + dx, yt + dy), (x + side, yt)]
    sidep = [(x + side, yt), (x + side + dx, yt + dy), (x + side + dx, yb + dy), (x + side, yb)]
    ax.add_patch(mp.Polygon(top, closed=True, fc=_tint(color, 0.75), ec=edge, lw=0.9, zorder=3))
    ax.add_patch(mp.Polygon(sidep, closed=True, fc=_tint(color, 0.68), ec=edge, lw=0.9, zorder=3))
    for k in range(1, min(channels, 8)):
        f = k / min(channels, 8)
        ax.plot([x + side + dx * f] * 2, [yt + dy * f, yb + dy * f], color=edge, lw=0.4, zorder=3)
        ax.plot([x + dx * f, x + side + dx * f], [yt + dy * f] * 2, color=edge, lw=0.4, zorder=3)
    if img is None:
        ax.add_patch(mp.Rectangle((x, yt), side, side, fc=fill, ec=edge, lw=1.0, zorder=4))
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(x, x + side, yb, yt),
                  interpolation="nearest", zorder=4)
        ax.add_patch(mp.Rectangle((x, yt), side, side, fill=False, ec=edge, lw=1.4, zorder=5))
    F.rect(name, x, yt + dy, x + side + dx, yb)
    return x + side + dx


def numbers(mnist: Path, weights: Path):
    images, labels = load_mnist(mnist)
    net = judge_net()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    # The ladder sheet's first panel, in the same order and seed, so canvas 5 is its "7 beside 8".
    cv = Canvases(images, labels, np.random.default_rng(1))
    ladder = [cv.one(7), cv.overlaid(7, 8), cv.two(7, 1), cv.two(7, 7), cv.two(7, 8)]
    canvas = ladder[4]
    asked = (7, 8)
    call, reads = judge(net, canvas, asked, "cpu")
    found = blobs(canvas)
    blob_rows = []
    for ys, xs in found:
        crop = crop28(canvas, ys, xs)
        p = torch.softmax(net(crop), 1)[0].detach().numpy()
        blob_rows.append(dict(x0=int(xs.min()), x1=int(xs.max()), y0=int(ys.min()), y1=int(ys.max()),
                              pixels=int(len(ys)), crop=crop[0, 0].numpy(), probs=p,
                              top=int(p.argmax()), top_p=float(p.max())))
    blob_rows.sort(key=lambda b: b["x0"])  # left blob first
    # Layer output shapes, read from a forward pass, not typed.
    shapes, x = [], blob_rows[1]["crop"][None, None]
    x = torch.tensor(x)
    shapes.append(tuple(x.shape[1:]))
    for m in net:
        x = m(x)
        shapes.append((type(m).__name__, tuple(x.shape[1:])))
    params = [(type(m).__name__, sum(q.numel() for q in m.parameters())) for m in net]
    test_imgs = np.fromfile(mnist / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, 28, 28)
    test_lbls = np.fromfile(mnist / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    with torch.no_grad():
        pred = net(torch.tensor(test_imgs[:, None], dtype=torch.float32) / 255.0).argmax(1).numpy()
    return dict(canvas=canvas, asked=asked, call=call, blobs=blob_rows, shapes=shapes, params=params,
                total_params=sum(p for _, p in params), test_acc=float((pred == test_lbls).mean()),
                n_train=len(images), n_test=len(test_imgs))


def draw(N):
    F = Fig()
    ax = F.ax
    left, right = N["blobs"]
    shp = {i: s for i, s in enumerate(N["shapes"])}
    # shapes: 0 input, 1 conv, 2 relu, 3 pool, 4 conv, 5 relu, 6 pool, 7 flatten, 8 lin, 9 relu, 10 lin
    c1, p1, c2, p2 = shp[1][1], shp[3][1], shp[4][1], shp[6][1]
    flat, hid, out = shp[7][1][0], shp[8][1][0], shp[10][1][0]
    prm = [p for _, p in N["params"]]

    F.text(40, 48, "The step 7b judge: one small network, with two rules around it", 30, ha="left",
           bold=True)
    F.text(40, 95, "A blob of ink goes in as a 28 × 28 grey picture; ten probabilities come out, "
           "one per digit. The rules (grey dashed boxes) are written by hand, not learnt.",
           16, MUTED, ha="left")

    # ---- Rule A: find and cut each blob ---------------------------------------------------
    ax.add_patch(mp.FancyBboxPatch((40, 150), 410, 670, boxstyle="round,pad=0,rounding_size=12",
                                   fill=False, ec=OFF, lw=1.4, ls=(0, (6, 4)), zorder=1))
    F.text(62, 178, "RULE, NOT LEARNT", 11, OFF, ha="left", bold=True, inside=True)
    F.text(62, 208, "1. Find each blob of ink", 15, ha="left", bold=True, inside=True)
    F.text(85, 243, f"the canvas, {N['canvas'].shape[0]} × {N['canvas'].shape[1]}, asked for "
           f"{N['asked'][0]} and {N['asked'][1]}", 12, INPUT, ha="left", inside=True)
    cx0, cy0, px = 85, 262, 5
    ch, cw = N["canvas"].shape
    ax.imshow(N["canvas"], cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(cx0, cx0 + cw * px, cy0 + ch * px, cy0), zorder=4)
    ax.add_patch(mp.Rectangle((cx0, cy0), cw * px, ch * px, fill=False, ec=INPUT, lw=1.6, zorder=5))
    for b in (left, right):
        ax.add_patch(mp.Rectangle((cx0 + b["x0"] * px, cy0 + b["y0"] * px), (b["x1"] - b["x0"] + 1) * px,
                                  (b["y1"] - b["y0"] + 1) * px, fill=False, ec="#f2c14e", lw=1.4, zorder=5))
    F.rect("canvas", cx0, cy0, cx0 + cw * px, cy0 + ch * px)
    F.text(85, 482, f"flood fill joins touching ink pixels\n(brighter than {INK} of 255, 8 neighbours);\n"
           f"a patch of {MIN_BLOB} pixels or more is a blob.\nThis canvas has {len(N['blobs'])} "
           f"(yellow boxes).", 12, MUTED, ha="left", inside=True)
    F.text(62, 562, "2. Cut each blob to 28 × 28", 15, ha="left", bold=True, inside=True)
    F.text(85, 598, "longer side scaled to 20 pixels, centred\nby centre of mass, like an MNIST digit",
           12, MUTED, ha="left", inside=True)
    crops = {}
    for name, b, x0 in (("left blob", left, 95), ("right blob", right, 280)):
        ax.imshow(b["crop"], cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                  extent=(x0, x0 + 126, 760, 634), zorder=4)
        ax.add_patch(mp.Rectangle((x0, 634), 126, 126, fill=False, ec=INPUT, lw=1.4, zorder=5))
        F.rect(name, x0, 634, x0 + 126, 760)
        F.text(x0 + 63, 786, name, 12, INPUT, inside=True)
        crops[name] = x0

    # ---- The network, run on the right blob ------------------------------------------------
    yc = 470
    s28 = 6.0
    x_in = 540
    r_in = volume(F, "v-in", x_in, yc, 28 * s28, 1, INPUT, img=right["crop"])
    F.line([(crops["right blob"] + 126, 697), (500, 697), (500, yc), (x_in, yc)], OFF, 1.6,
           allow={"right blob", "v-in"}, arrow=True)
    # kernel window on the input face and the cell it makes on the first conv's face
    win = (x_in + 10 * s28, yc - 84 + 8 * s28)
    ax.add_patch(mp.Rectangle(win, 3 * s28, 3 * s28, fill=False, ec="#f2c14e", lw=1.6, zorder=6))

    x_c1 = 790
    r_c1 = volume(F, "v-c1", x_c1, yc, c1[1] * s28, c1[0], HIDDEN)
    cell = (x_c1 + 11 * s28, yc - c1[1] * s28 / 2 + 9 * s28)
    ax.add_patch(mp.Rectangle(cell, s28, s28, fc="#f2c14e", ec="#b8860b", lw=0.8, zorder=6))
    for a, b in (((win[0] + 3 * s28, win[1]), (cell[0], cell[1])),
                 ((win[0] + 3 * s28, win[1] + 3 * s28), (cell[0], cell[1] + s28))):
        F.line([a, b], "#b8860b", 0.8, allow={"v-in", "v-c1"}, z=6)
    x_p1 = 1040
    r_p1 = volume(F, "v-p1", x_p1, yc, p1[1] * s28, p1[0], HIDDEN)
    x_c2 = 1200
    r_c2 = volume(F, "v-c2", x_c2, yc, c2[1] * s28, c2[0], HIDDEN)
    x_p2 = 1385
    r_p2 = volume(F, "v-p2", x_p2, yc, p2[1] * s28, p2[0], HIDDEN)
    for a, b, na, nb in ((r_in, x_c1, "v-in", "v-c1"), (r_c1, x_p1, "v-c1", "v-p1"),
                         (r_p1, x_c2, "v-p1", "v-c2"), (r_c2, x_p2, "v-c2", "v-p2")):
        F.line([(a + 8, yc + 30), (b - 8, yc + 30)], EDGE, 3.0, allow={na, nb}, arrow=True)

    # flatten bar
    fx, fy0, fy1 = 1530, yc - 150, yc + 150
    ax.add_patch(mp.Rectangle((fx, fy0), 18, fy1 - fy0, fc=_tint(HIDDEN, 0.8), ec=HIDDEN, lw=1.0,
                              zorder=4))
    for k in range(1, 12):
        ax.plot([fx, fx + 18], [fy0 + k * 25] * 2, color=HIDDEN, lw=0.4, zorder=4)
    F.rect("flat", fx, fy0, fx + 18, fy1)
    F.line([(r_p2 + 8, yc + 30), (fx - 8, yc + 30)], EDGE, 3.0, allow={"v-p2", "flat"}, arrow=True)

    # hidden column of 128, drawn as 7 circles and a gap
    hx, rr = 1660, 12
    hy = [yc - 150, yc - 105, yc - 60, yc - 15, yc + 60, yc + 105, yc + 150]
    for i, y in enumerate(hy):
        F.circle(f"h{i}", hx, y, rr, HIDDEN)
    F.text(hx, yc + 24, "⋮", 16, HIDDEN, inside=True)
    for i, y0 in enumerate(np.linspace(fy0 + 15, fy1 - 15, 5)):
        for j in (0, 2, 3, 4, 6):
            if (i + j) % 2:
                continue
            p = np.array([hx, hy[j]]) - np.array([fx + 18, y0])
            end = np.array([hx, hy[j]]) - p / np.linalg.norm(p) * rr
            F.line([(fx + 18, y0), tuple(end)], EDGE, 0.7, allow={"flat", f"h{j}"})

    # output column of 10
    ox, orr, oy0, ostep = 1840, 14, yc - 153, 34
    top = right["top"]
    for k in range(out):
        F.circle(f"o{k}", ox, oy0 + k * ostep, orr, OUTPUT, glow=(k == top))
        F.text(ox, oy0 + k * ostep, str(k), 11, OUTPUT, inside=True, bold=(k == top))
        pk = right["probs"][k]
        F.text(ox + 26, oy0 + k * ostep, f"{pk:.3f}", 12 if k == top else 11,
               OUTPUT if k == top or pk >= 0.01 else MUTED, ha="left", bold=(k == top))
    for i, y in enumerate(hy):
        for k in range(out):
            if (i * 3 + k) % 4:
                continue
            a = np.array([hx, y]); b = np.array([ox, oy0 + k * ostep]); u = (b - a) / np.linalg.norm(b - a)
            F.line([tuple(a + u * rr), tuple(b - u * orr)], EDGE, 0.7, allow={f"h{i}", f"o{k}"})

    # what made each part, and its learnt weight count, above the part
    ops = [(x_in + 84, "input", "1 grey channel", INPUT),
           (x_c1 + 84, f"conv 3 × 3, pad 1, ReLU", f"{prm[0]:,} weights", HIDDEN),
           (x_p1 + 42, "max pool 2", "no weights", HIDDEN),
           (x_c2 + 42 + 10, "conv 3 × 3, pad 1, ReLU", f"{prm[3]:,} weights", HIDDEN),
           (x_p2 + 21 + 10, "max pool 2", "no weights", HIDDEN),
           (fx + 9, "flatten", "no weights", HIDDEN),
           (hx, "linear, ReLU", f"{prm[7]:,} weights", HIDDEN),
           (ox, "linear, softmax", f"{prm[9]:,} weights", OUTPUT)]
    for i, (x, a, b, col) in enumerate(ops):
        y = 228 if i % 2 else 268
        F.text(x, y, a, 12, col, bold=True)
        F.text(x, y + 20, b, 11, MUTED)
    sizes = [(x_in + 84, "28 × 28 × 1", INPUT), (x_c1 + 84, f"{c1[1]} × {c1[2]} × {c1[0]}", HIDDEN),
             (x_p1 + 42, f"{p1[1]} × {p1[2]} × {p1[0]}", HIDDEN),
             (x_c2 + 52, f"{c2[1]} × {c2[2]} × {c2[0]}", HIDDEN),
             (x_p2 + 31, f"{p2[1]} × {p2[2]} × {p2[0]}", HIDDEN),
             (fx + 9, f"{flat:,}", HIDDEN), (hx, f"{hid}", HIDDEN), (ox, f"{out} logits", OUTPUT)]
    for i, (x, s, col) in enumerate(sizes):
        F.text(x, 668 if i % 2 == 0 else 694, s, 13, col, bold=True)
    F.text(x_in + 84, 720, "this run: the right blob. The left blob\ngoes through the same weights next.",
           11, MUTED)
    F.text(x_c1 + 84 + 40, 360, "one 3 × 3 window → one cell", 11, "#b8860b")

    # size arithmetic, boxed
    ax.add_patch(mp.Rectangle((560, 760), 900, 70, fill=False, ec=TEXT, lw=0.8, zorder=2))
    F.rect("box-size", 560, 760, 1460, 830)
    F.text(1010, 782, f"conv:  (in − 3 + 2 × 1) / 1 + 1 = in, so 28 stays 28 and 14 stays 14", 13,
           inside=True)
    F.text(1010, 810, f"pool:  28 / 2 = 14,  14 / 2 = 7        flatten:  7 × 7 × 64 = {flat:,}", 13,
           inside=True)
    # softmax, boxed, with a soft glow
    for k, a in ((10, 0.05), (6, 0.08), (3, 0.12)):
        ax.add_patch(mp.Rectangle((1600 - k, 750 - k), 360 + 2 * k, 90 + 2 * k, fc=OUTPUT, alpha=a, lw=0,
                                  zorder=1))
    ax.add_patch(mp.Rectangle((1600, 750), 360, 90, fc="white", ec=OUTPUT, lw=1.0, zorder=2))
    F.rect("box-softmax", 1600, 750, 1960, 840)
    F.text(1780, 782, r"$p_k = e^{z_k} \,/\, (e^{z_0} + e^{z_1} + \cdots + e^{z_9})$", 17, OUTPUT, inside=True)
    F.text(1780, 822, "z: the 10 logits; the 10 p add up to 1", 11, MUTED, inside=True)

    # ---- Rule B: the ladder ----------------------------------------------------------------
    bx = 2030
    ax.add_patch(mp.FancyBboxPatch((bx, 150), 440, 670, boxstyle="round,pad=0,rounding_size=12",
                                   fill=False, ec=OFF, lw=1.4, ls=(0, (6, 4)), zorder=1))
    F.line([(ox + 80, oy0 + top * ostep), (bx - 6, oy0 + top * ostep)], OFF, 1.6,
           allow=set(), arrow=True)
    tx = bx + 22
    F.text(tx, 178, "RULE, NOT LEARNT", 11, OFF, ha="left", bold=True, inside=True)
    F.text(tx, 208, "3. One call per canvas", 15, ha="left", bold=True, inside=True)
    F.text(tx, 248, f"right blob reads {right['top']}, top probability {right['top_p']:.3f}", 12,
           OUTPUT, ha="left", inside=True)
    F.text(tx, 274, f"left blob reads {left['top']}, top probability {left['top_p']:.3f}", 12,
           OUTPUT, ha="left", inside=True)
    F.text(tx, 308, f"sure means a top probability of {CONFIDENT} or more", 12, MUTED, ha="left",
           inside=True)
    F.text(tx, 342, "the ladder, best call at the top:", 12, TEXT, ha="left", inside=True)
    rows = [("two clear, right digits", f"2 blobs, both sure, one {N['asked'][0]} and one {N['asked'][1]}"),
            ("two digits, wrong identity", "2 blobs, any other reads"),
            ("fused", "1 blob, not sure"),
            ("one digit", "1 blob, sure"),
            ("damaged", "no blob, or 3 blobs or more")]
    for i, (call, cond) in enumerate(rows):
        y = 385 + i * 62
        hit = call == N["call"]
        if hit:
            ax.add_patch(mp.FancyBboxPatch((tx - 8, y - 20), 404, 52,
                                           boxstyle="round,pad=0,rounding_size=6",
                                           fc=_tint(OUTPUT, 0.88), ec=OUTPUT, lw=1.0, zorder=2))
        F.text(tx, y, call, 13, OUTPUT if hit else TEXT, ha="left", bold=True, inside=True)
        F.text(tx + 14, y + 22, cond, 11, MUTED, ha="left", inside=True)
    F.text(tx, 720, "this canvas:", 13, TEXT, ha="left", inside=True)
    F.text(tx, 748, N["call"], 16, OUTPUT, ha="left", bold=True, inside=True)

    # ---- Footer --------------------------------------------------------------------------
    F.text(40, 880, f"Learnt with cross-entropy on MNIST's {N['n_train']:,} single training digits "
           f"(labels 0 to 9): 2 passes, Adam at 1e-3, batch 256; {N['total_params']:,} learnt weights. "
           f"It reads {N['test_acc']:.1%} of the {N['n_test']:,} test digits right. It never saw a "
           f"canvas or two digits while it learnt.", 13, TEXT, ha="left")
    F.text(40, 910, "The canvas is the last canvas of the top row of mnist-judge-ladder.png "
           "(7 beside 8), rebuilt with judge_sheet.py's functions; every probability is read from "
           "the prototype weights by make_judge_network_view.py.", 12, MUTED, ha="left")
    F.text(40, 940, "Colour: blue is what goes in, green the learnt layers, red what comes out; "
           "grey dashed boxes are rules written by hand.", 12, MUTED, ha="left")
    return F


def seg_hits_rect(a, b, r, pad=1.0):
    x0, y0, x1, y1 = r[0] + pad, r[1] + pad, r[2] - pad, r[3] - pad
    if x1 <= x0 or y1 <= y0:
        return False
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def seg_hits_circle(a, b, c, r):
    d = b - a
    t = np.clip(np.dot(c - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
    return np.linalg.norm(a + t * d - c) < r - 0.5


def check(F):
    F.fig.canvas.draw()
    rend = F.fig.canvas.get_renderer()
    inv = F.ax.transData.inverted()
    texts = []
    for i, t in enumerate(F.ax.texts):
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        texts.append((f"text {t.get_text()[:30]!r}", min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                      t.get_gid()))
    bad = []
    for a, b, allow in F.segments:
        for name, *r, _ in texts:
            if seg_hits_rect(a, b, r):
                bad.append(f"line {a}->{b} crosses {name}")
        for name, *r in F.rects:
            if name not in allow and seg_hits_rect(a, b, r):
                bad.append(f"line {a}->{b} crosses {name}")
        for name, cx, cy, rr in F.circles:
            if name not in allow and seg_hits_circle(a, b, np.array([cx, cy]), rr):
                bad.append(f"line {a}->{b} crosses {name}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            p, q = texts[i][1:5], texts[j][1:5]
            if p[0] < q[2] - 1 and q[0] < p[2] - 1 and p[1] < q[3] - 1 and q[1] < p[3] - 1:
                bad.append(f"{texts[i][0]} overlaps {texts[j][0]}")
        name, *r, gid = texts[i]
        for rn, *rr in F.rects:
            if gid != "inside" or not rn.startswith("box-"):
                if r[0] < rr[2] and rr[0] < r[2] and r[1] < rr[3] and rr[1] < r[3]:
                    bad.append(f"{name} overlaps {rn}")
        if r[0] < 0 or r[2] > W or r[1] < 0 or r[3] > H:
            bad.append(f"{name} runs off the canvas")
    if bad:
        raise SystemExit("refused, nothing written:\n" + "\n".join(bad))
    print(f"crossing check: 0 crossings over {len(F.segments)} segments, {len(texts)} labels, "
          f"{len(F.rects)} boxes, {len(F.circles)} nodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", type=Path, default=Path("/home/molef/Playground/data/MNIST/raw"))
    ap.add_argument("--weights", type=Path, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    N = numbers(args.mnist, args.weights)
    F = draw(N)
    check(F)
    F.fig.savefig(HERE / "judge-network-view.png", dpi=100)
    F.fig.savefig(HERE / "judge-network-view.svg")
    record = dict(
        canvas="mnist-judge-ladder.png top row, fifth canvas (7 beside 8), Canvases seed 1",
        asked=list(N["asked"]), call=N["call"], confident_at=CONFIDENT,
        blobs=[dict(side=s, box_x=[b["x0"], b["x1"]], box_y=[b["y0"], b["y1"]], pixels=b["pixels"],
                    top_digit=b["top"], top_probability=round(b["top_p"], 4),
                    probabilities=[round(float(p), 4) for p in b["probs"]])
               for s, b in zip(("left", "right"), N["blobs"])],
        layer_output_shapes=[str(s) for s in N["shapes"]],
        weights_per_layer=N["params"], total_weights=N["total_params"],
        test_accuracy=round(N["test_acc"], 4), n_train=N["n_train"], n_test=N["n_test"],
        weights_file=str(args.weights))
    (HERE / "judge_network_view_numbers.json").write_text(json.dumps(record, indent=1))
    print("wrote judge-network-view.png, .svg and judge_network_view_numbers.json;",
          f"call {N['call']}; reads", [(b["top"], round(b["top_p"], 4)) for b in N["blobs"]],
          f"test accuracy {N['test_acc']:.4f}")


if __name__ == "__main__":
    main()
