"""The inclusive-labels MNIST stage in the classifier-guidance results-figures layout (r1 to r4).

r1  from CFG to the product and the label dial: the equations, how a two-digit canvas is labelled at each
    dial, one real tile per row and dial with its count of 256, the three trainings, the verdict
r2  the U-Net and its label: canvas in, noise estimate out, the label's table row added to the step
    embedding and fed to every residual block
r3  the product in one sampler step: the same U-Net three times (blank, 7, 8), the product, the DDPM
    step, then the corrector's Langevin moves in their own lane
r4  the judge: the CNN that calls every tile, its rules, and the ladder of calls

Layout and drawing helpers copied from
2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/results-figures/make_results_figures.py
and its network-view helper (Fig, crosses, node, column, fan), unchanged except that figures save here.
Every number from mnist_stage_numbers.json (make_mnist_stage_numbers.py); every picture a real tile from
data/mnist/samples or a real tensor from mnist_stage_arrays.npz. Each figure refuses to save if a line
crosses a box, circle or label it does not connect.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I make_mnist_stage_figures.py [r1 r2 r3 r4]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
REPO = HERE.parents[3]
N = json.load(open(HERE / "mnist_stage_numbers.json"))
A = np.load(HERE / "mnist_stage_arrays.npz")
SAMPLES = REPO / "data/mnist/samples"
# the ladder colour key (4-inclusive-captions/artifacts/scenes/how-the-toy-relabelling-experiment-works/ladder.json)
BLUE, GREEN, RED, ORANGE, GREY = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88"
FADE, INK = "#eceff2", "#1f2a33"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})


# ------------------------------------------------------------------ helpers (copied, see the docstring)
class Fig:
    def __init__(self, w, h, title, sub):
        self.fig = plt.figure(figsize=(w / 50, h / 50))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.rects, self.segs, self.texts = {}, [], []
        self.text(20, h - 28, title, size=22, weight="bold", color=INK, ha="left", id="title")
        self.text(20, h - 58, sub, size=13, color=GREY, ha="left", id="sub")

    def box(self, id, x, y, w, h, label="", fc="white", ec=GREY, lc=INK, size=12, glow=False, lw=1.2, dashed=False):
        if glow:
            self.ax.add_patch(FancyBboxPatch((x - 4, y - 4), w + 8, h + 8, boxstyle="round,pad=0,rounding_size=8",
                                             fc=ec, ec="none", alpha=0.12))
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=ec, lw=lw,
                                         ls="--" if dashed else "-"))
        if label:
            self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", size=size, color=lc)
        self.rects[id] = (x, y, x + w, y + h)

    def img(self, id, a, x, y, w, h, ec=BLUE, cmap="gray"):
        self.ax.imshow(a, cmap=cmap, vmin=0, vmax=255, extent=(x, x + w, y, y + h), interpolation="nearest", zorder=2)
        self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=2, zorder=3)); self.rects[id] = (x, y, x + w, y + h)

    def text(self, x, y, s, id=None, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("size", 11)
        t = self.ax.text(x, y, s, **kw); self.texts.append((id or f"t{len(self.texts)}", t))

    def arrow(self, pts, a, b, color=GREY, lw=1.4, head=True):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=1)
        if head:
            self.ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                                                            mutation_scale=14), zorder=1)
        for p, q in zip(pts, pts[1:]):
            self.segs.append((p, q, {a, b}))

    def check_and_save(self, name):
        self.fig.canvas.draw()
        inv = self.ax.transData.inverted()
        boxes = dict(self.rects)
        for id, t in self.texts:
            bb = t.get_window_extent().transformed(inv)
            boxes[id] = (bb.x0, bb.y0, bb.x1, bb.y1)
        bad = []
        for p, q, ok in self.segs:
            for id, (x0, y0, x1, y1) in boxes.items():
                if id in ok or any(id.startswith(o + ".") for o in ok):
                    continue
                if crosses(p, q, (x0 + 1.5, y0 + 1.5, x1 - 1.5, y1 - 1.5)):
                    bad.append((id, p, q))
        if bad:
            raise SystemExit(f"{name}: {len(bad)} crossing(s), not saved: {bad[:5]}")
        OUT.mkdir(exist_ok=True)
        for ext in ("png", "svg"):
            self.fig.savefig(OUT / f"{name}.{ext}", dpi=100 if ext == "png" else None)
        plt.close(self.fig)
        print(f"wrote {name}: {len(self.segs)} segments, {len(boxes)} boxes and labels, 0 crossings")


def crosses(p, q, r):
    x0, y0, x1, y1 = r
    if x1 <= x0 or y1 <= y0:
        return False
    (ax, ay), (bx, by) = p, q
    for i in range(41):
        f = i / 40
        x, y = ax + f * (bx - ax), ay + f * (by - ay)
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False


def node(f, id, x, y, r=14, fc="white", ec=GREEN, glow=False, label=None, lc=INK, size=10):
    if glow:
        f.ax.add_patch(Circle((x, y), r + 6, fc=ec, ec="none", alpha=0.15))
    f.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.2, zorder=2))
    if label is not None:
        f.ax.text(x, y, label, ha="center", va="center", size=size, color=lc, zorder=3)
    f.rects[id] = (x - r, y - r, x + r, y + r)


def column(f, prefix, x, ys, **kw):
    for k, y in enumerate(ys):
        node(f, f"{prefix}{k}", x, y, **kw)
    return [(f"{prefix}{k}", x, y) for k, y in enumerate(ys)]


def fan(f, left, right, r=14, color=GREY, lw=0.8):
    for i, (a, xa, ya) in enumerate(left):
        for b, xb, yb in (right[i % len(right)], right[(i + 1) % len(right)]):
            f.arrow([(xa + r, ya), (xb - r, yb)], a, b, color=color, lw=lw, head=False)


# ------------------------------------------------------------------ r1: equations, label dial, results
DIALS = ["0", "0.1", "0.3", "0.6", "1"]
FILE = {"0": "0", "0.1": "0.1", "0.3": "0.3", "0.6": "0.6", "1": "1"}
ROWS = [("7 alone", '"7" alone', "single"), ("8 alone", '"8" alone', "single"),
        ("7 and 8", '"7 and 8"\nthe joint label', "pair"), ("7 x 8 plain", "7 × 8\nproduct", "pair"),
        ("7 x 8 corrected", "7 × 8 product\n+ corrector", "pair"), ("4 x 7 corrected", "4 × 7 product\n+ corrector", "pair")]
TRAIN = 1
K7 = "'7' share of a 7-and-8 canvas's non-blank draws"
S8 = "share of '7' pictures that also show an 8"
KJ = "joint share of a 7-and-8 canvas's labelled draws"
K7S = "'7' share of a 7-and-8 canvas's labelled draws"
KS = "share of '7' draws holding two digits"


def SH(d):
    """A setting named by its share, the dial in brackets."""
    return f"{100 * N['shares'][d][S8]:.1f}% (dial {d})"


def bar_of(row, dial):
    b = N["bar"]
    if row in ("7 alone", "8 alone"):
        return b["purity floor"][row[0]][dial]
    if row == "7 x 8 corrected":
        return b["support min (judge, of 256)"]
    if row == "4 x 7 corrected":
        return b["unseen min"]
    return None


def r1():
    V = N["verdict"]
    sm = N["sampler"]
    f = Fig(1960, 1780, "From CFG to the product, and what the share of \"7\" pictures that also show an 8 does to it",
            f"MNIST digits on 32 × 64 canvases, one U-Net per setting and training; training {TRAIN} drawn in every column, "
            "the same starting noise in every cell; counts of 256 seeds from the stage's judge.")
    # the equation chain
    f.box("e1", 60, 1560, 520, 90, r"$\hat\varepsilon = \varepsilon_{\varnothing} + w\,(\varepsilon_{c} - \varepsilon_{\varnothing})$",
          fc="white", ec=INK, lc=INK, size=19)
    f.text(320, 1665, "1  classifier-free guidance, one label c", id="e1l", color=INK, size=12, weight="bold")
    f.box("e2", 690, 1560, 560, 90, r"$\varepsilon_{\mathrm{PoE}} = \varepsilon_{\varnothing} + 3\,[(\varepsilon_{7} - \varepsilon_{\varnothing}) + (\varepsilon_{8} - \varepsilon_{\varnothing})]$",
          fc="white", ec=RED, lc=RED, size=19)
    f.text(970, 1665, f"2  the product of \"7\" and \"8\", w = {sm['guidance']:g} ({sm['product line']})", id="e2l",
           color=RED, size=12, weight="bold")
    f.box("e3", 1360, 1560, 540, 90, r"$z \leftarrow z + \eta\, s + \sqrt{2\eta}\,\xi,\quad s=-\varepsilon_{\mathrm{PoE}}/\sqrt{1-\bar\alpha}$",
          fc="white", ec=ORANGE, lc=ORANGE, size=17, glow=True)
    f.text(1630, 1665, f"3  the corrector: {sm['corrector steps per level']} steps per level, η = 0.05 (1 − ᾱ)",
           id="e3l", color=ORANGE, size=12, weight="bold")
    f.arrow([(580, 1605), (690, 1605)], "e1", "e2", color=INK)
    f.arrow([(1250, 1605), (1360, 1605)], "e2", "e3", color=INK)
    f.text(980, 1515, "No checker and no push in this stage: what changes between columns is only how the training "
                      "canvases were labelled.", id="nopush", color=INK, size=13)
    # one box per dial: the labelling rule
    x0, cw = 200, 345
    rowy = [1020, 880, 740, 600, 460, 320]
    for ri, (_, name, _) in enumerate(ROWS):
        f.text(100, rowy[ri] + 47, name, id=f"rl{ri}", color=INK, size=12.5, weight="bold")
    for ci, d in enumerate(DIALS):
        x = x0 + ci * cw
        by = N["canvases"]["by dial"][d]
        lab = d == "0"
        col = GREY if lab else GREEN
        sh = N["shares"][d]
        f.box(f"h{ci}", x, 1140, cw - 20, 345, "", fc="white", ec=col)
        f.text(x + (cw - 20) / 2, 1458, f"{100 * sh[S8]:.1f}% of \"7\" pictures also show an 8", id=f"hn{ci}",
               color=INK if lab else GREEN, size=13, weight="bold")
        f.text(x + (cw - 20) / 2, 1432, f"dial {d} · " + ("labelled" if lab else "relabelled"), id=f"hd{ci}",
               color=GREY, size=10)
        # 40 real "7"-labelled training canvases, the ones that also hold an 8 outlined
        g = A[f"grid/{d}"]
        hot = set(N["r1 header grids"][d]["outlined positions"])
        gw, gh, gx, gy = 36, 18, x + 14, 1405
        for q in range(40):
            cx_, cy_ = gx + (q % 8) * (gw + 3), gy - (q // 8 + 1) * (gh + 3)
            f.ax.imshow(g[q], cmap="gray", vmin=0, vmax=255, extent=(cx_, cx_ + gw, cy_, cy_ + gh), interpolation="nearest", zorder=2)
            f.ax.add_patch(Rectangle((cx_, cy_), gw, gh, fc="none", ec=ORANGE if q in hot else "#c9d1d9",
                                     lw=2.2 if q in hot else 0.6, zorder=3))
        f.rects[f"grid{ci}"] = (gx, gy - 5 * (gh + 3), gx + 8 * (gw + 3), gy)
        f.text(x + (cw - 20) / 2, 1280, f"40 \"7\"-labelled training canvases; {sh['outlined of 40']} also hold an 8 (outlined)",
               id=f"gl{ci}", color=GREY, size=9.5)
        # how a 7-and-8 canvas's labelled draws divide
        bw, bx, byy = 300, x + 12, 1200
        segs = [(sh[KJ], "#d6efe0", GREEN, '"7 and 8"'), (sh[K7S], "#e6eefb", BLUE, '"7"'), (sh[K7S], "#fbe3e4", RED, '"8"')]
        xx = bx
        for frac, fc_, ec_, name_ in segs:
            if frac > 0:
                f.ax.add_patch(Rectangle((xx, byy), bw * frac, 22, fc=fc_, ec=ec_, lw=1, zorder=2))
                if bw * frac > 40:
                    f.ax.text(xx + bw * frac / 2, byy + 11, f"{name_} {100 * frac:.0f}%", ha="center", va="center", size=8.5,
                              color=ec_, zorder=3)
                xx += bw * frac
        f.rects[f"lb{ci}"] = (bx, byy, bx + bw, byy + 22)
        f.text(x + (cw - 20) / 2, 1172, "how a 7-and-8 canvas's labelled draws divide", id=f"lbl{ci}", color=GREY, size=9.5)
        npz = np.load(SAMPLES / f"dial-{FILE[d]}-seed-{TRAIN}.npz")
        for ri, (row, _, kind) in enumerate(ROWS):
            key = f"{d}|{TRAIN}|{row}"
            c = N["cells"][key]
            b = bar_of(row, d)
            ok = b is not None and c["count"] >= b
            f.img(f"i{ci}{ri}", npz[row][c["tile seed"]], x + 15, rowy[ri], 190, 95, ec=GREEN if ok else "#c9d1d9")
            f.text(x + 262, rowy[ri] + 60, f"{c['count']}/256", id=f"n{ci}{ri}", color=GREEN if ok else INK, size=13,
                   weight="bold" if ok else "normal")
            f.text(x + 262, rowy[ri] + 32, f"seed {c['tile seed']}", id=f"sd{ci}{ri}", color=GREY, size=9.5)
        st = V["settings"][f"{float(d)}"]
        three = (f"\"7\" {', '.join(map(str, st['seven_alone']))} · "
                 f"7×8+c {', '.join(map(str, st['corrected_7x8_two']))}")
        f.text(x + (cw - 20) / 2, 280, "three trainings: " + three, id=f"t3{ci}", color=GREY, size=10)
        f.text(x + (cw - 20) / 2, 252, "counting setting" if st["counts"] else "not a counting setting",
               id=f"cs{ci}", color=GREEN if st["counts"] else RED, size=11.5, weight="bold")
    s06, s1 = V["settings"]["0.6"], V["settings"]["1.0"]
    pf = N["bar"]["purity floor"]
    means = V["mean corrected 7 x 8 at counting settings"]
    f.text(980, 205, f"Verdict {V['verdict']}: no counting setting's corrected 7 × 8 reaches a three-training mean of "
                     f"{N['bar']['null if mean below (judge, of 256)']} of 256 ({SH('0')}, {SH('0.1')}, {SH('0.3')}: "
                     f"{', '.join(f'{v:g}' for v in means.values())}).", id="foot1", color=INK, size=12.5, weight="bold")
    f.text(980, 175, f"{SH('0.6')} and {SH('1')} do not count: \"7\" alone draws {min(s06['seven_alone'])} to {max(s06['seven_alone'])} "
                     f"against its line of {pf['7']['0.6']}, and {min(s1['seven_alone'])} to {max(s1['seven_alone'])} against "
                     f"{pf['7']['1']}; \"8\" alone at {SH('1')} draws {min(s1['eight_alone'])} to {max(s1['eight_alone'])} against "
                     f"{pf['8']['1']}.", id="foot2", color=GREY, size=12)
    f.text(980, 145, "Green: at or over the line written before the run (single labels: their purity line; corrected 7 × 8: "
                     f"{N['bar']['support min (judge, of 256)']}; corrected 4 × 7: {N['bar']['unseen min']}). The joint label "
                     "and the plain product have no line.", id="foot3", color=GREY, size=12)
    f.text(980, 115, "Counts: single labels, the right digit alone; every other row, two separate right digits. "
                     "Each tile is the first seed whose call is that cell's most common call.", id="foot4", color=GREY, size=12)
    eye = N["eye calls, corrected 7 x 8, first 64 seeds"]
    e03 = [eye[f"dial-0.3-seed-{s}-7x8-corrected"] for s in (1, 2, 3)]
    f.text(980, 85, f"The plan reads the verdict by eye on the first 64 seeds: {SH('0.3')} gives {', '.join(map(str, e03))} of 64 "
                    f"(mean {np.mean(e03):.1f}, null under {N['bar']['eye null mean']}).", id="foot5", color=GREY, size=12)
    f.check_and_save("r1-from-cfg-to-the-product-and-the-label-dial")


# ------------------------------------------------------------------ r2: the U-Net and its label
def r2():
    nw = N["network"]
    f = Fig(1700, 960, "The U-Net and its label",
            f"{nw['parameters']:,} trained weights, one model per dial and training. Noisy canvas, step and label in; "
            "its estimate of the noise out.")
    st = N["one step (dial 1, training 1, CPU)"]
    f.img("xt", A["step/x_k"], 30, 380, 220, 110)
    f.text(140, 515, r"$x_k$: the noisy canvas, $1\times32\times64$", id="xtl", color=BLUE, size=12)
    (h0, w0, c0), (h1, w1, c1), (h2, w2, c2) = nw["levels"]
    xs = [360, 480, 600, 720, 840, 960, 1080]
    ys = [500, 460, 420, 380]
    labs = [f"{c0}\n{h0}×{w0}", f"{c1}\n{h1}×{w1}", f"{c2}\n{h2}×{w2}", f"{c2}\nattention", f"{c2}\n{h2}×{w2}",
            f"{c1}\n{h1}×{w1}", f"{c0}\n{h0}×{w0}"]
    cols = []
    for i, x in enumerate(xs):
        c = column(f, f"u{i}_", x, ys, glow=i == 3, fc="#d6efe0" if i == 3 else "white")
        cols.append(c)
        f.text(x, ys[-1] - 48, labs[i], id=f"ul{i}", color=GREEN, size=10.5)
    for a, b in zip(cols, cols[1:]):
        fan(f, a, b, color=GREEN)
    f.text(480, 280, "down", id="dn", color=GREY, size=11)
    f.text(960, 280, "up, with skips", id="upl", color=GREY, size=11)
    f.arrow([(250, 440), (360 - 14, 440)], "xt", "u0_1")
    f.img("out", A["step/eps product"], 1260, 380, 220, 110, ec=RED)
    f.arrow([(1080 + 14, 440), (1260, 440)], "u6_1", "out", color=RED)
    f.text(1370, 515, r"$\varepsilon$: the noise estimate, $1\times32\times64$", id="ol", color=RED, size=12)
    # the label and the step, from above
    node(f, "l7", 360, 800, r=20, fc="#e6eefb", ec=BLUE, label="7", lc=BLUE, size=14)
    f.text(360, 840, "the label", id="ll", color=BLUE, size=12)
    f.box("tab", 440, 780, 260, 40, f"its row of the {nw['label table'][0]}-row table, {nw['label table'][1]}",
          fc="#e6eefb", ec=BLUE, lc=BLUE, size=11)
    f.arrow([(380, 800), (440, 800)], "l7", "tab", color=BLUE)
    node(f, "k", 360, 700, r=20, fc="#e6eefb", ec=BLUE, label="k", lc=BLUE, size=14)
    f.text(360, 660, "the step", id="kl", color=BLUE, size=12)
    f.box("temb", 440, 680, 260, 40, f"step embedding, {nw['step embedding width']}", fc="#e6eefb", ec=BLUE, lc=BLUE, size=11)
    f.arrow([(380, 700), (440, 700)], "k", "temb", color=BLUE)
    f.box("sum", 780, 720, 50, 50, "+", fc="white", ec=INK, lc=INK, size=20)
    f.arrow([(700, 800), (805, 800), (805, 770)], "tab", "sum", color=BLUE)
    f.arrow([(700, 700), (805, 700), (805, 720)], "temb", "sum", color=BLUE)
    f.box("e", 900, 725, 180, 40, f"label + step, {nw['step embedding width']}", fc="#e6eefb", ec=BLUE, lc=BLUE, size=12)
    f.arrow([(830, 745), (900, 745)], "sum", "e", color=BLUE)
    f.arrow([(1080, 745), (1120, 745), (1120, 630), (360, 630)], "e", "bus.none", color=BLUE, head=False)
    for i, x in enumerate(xs):
        f.arrow([(x, 630), (x, 500 + 14)], "bus", f"u{i}_0", color=BLUE)
    f.text(1150, 700, f"added inside every residual block\n({nw['residual blocks']} blocks), after its first convolution",
           id="every", color=BLUE, size=11, ha="left")
    f.box("pred", 1000, 140, 640, 80, "trained to predict the noise: loss = mean $(\\hat\\varepsilon-\\varepsilon)^2$\n"
          f"{nw['training']['steps']:,} steps of {nw['training']['batch']}, learning rate {nw['training']['learning rate']}, "
          f"EMA {nw['training']['EMA']}", fc="white", ec=ORANGE, lc=INK, size=12.5, glow=True)
    f.text(850, 60, f"Two residual blocks per level, three levels ({c0}, {c1}, {c2} channels), self-attention at the bottom; "
                    "the label enters through the same channel shift as the step, never as a separate input.",
           id="foot", color=GREY, size=12)
    f.check_and_save("r2-the-network-and-its-label")


# ------------------------------------------------------------------ r3: the product in one step
def r3():
    st = N["one step (dial 1, training 1, CPU)"]
    nm = st["norms"]
    f = Fig(1750, 980, "The product in one sampler step",
            f"Dial 1, training 1, at step {st['k']} (ᾱ = {st['abar']}). One noisy canvas goes into the same U-Net three times, "
            "with the blank label, \"7\" and \"8\". Real tensors, run on CPU.")
    f.img("xt", A["step/x_k"], 30, 400, 200, 100)
    f.text(130, 525, r"$x_k$", id="xtl", color=BLUE, size=15)
    labs = [("∅", "blank", "step/eps blank", r"$\varepsilon_{\varnothing}$", 760),
            ("7", "7", "step/eps 7", r"$\varepsilon_{7}$", 600), ("8", "8", "step/eps 8", r"$\varepsilon_{8}$", 440)]
    for k, (sym, name, key, tex, y) in enumerate(labs):
        a = column(f, f"m{k}a_", 400, [y + 30, y - 10], fc=FADE, ec=GREY, r=12)
        b = column(f, f"m{k}b_", 480, [y + 30, y - 10], fc=FADE, ec=GREY, r=12)
        fan(f, a, b, r=12)
        node(f, f"lab{k}", 440, y + 75, r=15, fc="#e6eefb", ec=BLUE, label=sym, lc=BLUE, size=12)
        f.arrow([(440, y + 60), (440, y + 42)], f"lab{k}", f"m{k}a_0.none", color=BLUE)
        f.img(f"eps{k}", A[key], 560, y - 30, 160, 80, ec=RED)
        f.arrow([(492, y + 10), (560, y + 10)], f"m{k}b_0", f"eps{k}", color=RED)
        f.text(760, y + 10, tex, id=f"et{k}", color=RED, size=15, ha="left")
        f.arrow([(230, 450), (300, 450), (300, y + 30), (400 - 12, y + 30)], "xt", f"m{k}a_0", color=GREY)
    f.text(440, 860, "the same U-Net, run 3 times", id="ul", color=GREY, size=12)
    f.box("prod", 870, 560, 470, 70, r"$\varepsilon_{\varnothing} + 3\,[(\varepsilon_7-\varepsilon_{\varnothing}) + (\varepsilon_8-\varepsilon_{\varnothing})]$",
          fc="white", ec=RED, lc=RED, size=16)
    for k, y in ((0, 760), (1, 600), (2, 440)):
        f.arrow([(810, y + 10), (840, y + 10), (840, 595), (870, 595)], f"et{k}", "prod", color=RED)
    f.text(1105, 660, f"|ε∅| = {nm['eps blank']}; |ε7 − ε∅| = {nm['eps 7 - eps blank']}, |ε8 − ε∅| = {nm['eps 8 - eps blank']}, "
                      f"cosine {st['cosine between the two label differences']}", id="norms", color=RED, size=11)
    f.img("pe", A["step/eps product"], 1400, 555, 160, 80, ec=RED)
    f.arrow([(1340, 595), (1400, 595)], "prod", "pe", color=RED)
    f.text(1480, 660, r"$\varepsilon_{\mathrm{PoE}}$", id="pel", color=RED, size=15)
    f.box("ddpm", 1390, 400, 180, 50, "DDPM step to k − 1", fc="white", ec=INK, lc=INK, size=12)
    f.arrow([(1480, 555), (1480, 450)], "pe", "ddpm", color=INK)
    f.img("xn", A["step/x_k-1"], 1400, 260, 160, 80, ec=BLUE)
    f.arrow([(1480, 400), (1480, 340)], "ddpm", "xn", color=INK)
    f.text(1610, 300, r"$x_{k-1}$", id="xnl", color=BLUE, size=14, ha="left")
    # the corrector lane
    f.box("corr", 840, 150, 470, 70, r"$z \leftarrow z + \eta\,s + \sqrt{2\eta}\,\xi,\;\; s=-\varepsilon_{\mathrm{PoE}}/\sqrt{1-\bar\alpha_{k-1}}$",
          fc="white", ec=ORANGE, lc=ORANGE, size=15, glow=True)
    f.text(1075, 120, f"corrected rows only: {N['sampler']['corrector steps per level']} such steps at level k − 1, "
                      f"η = {st['corrector eta at step 99']}", id="corrl", color=ORANGE, size=11.5)
    f.arrow([(1400, 300), (1360, 300), (1360, 185), (1310, 185)], "xn", "corr", color=ORANGE)
    f.img("xc", A["step/x_k-1 corrected"], 560, 145, 160, 80, ec=ORANGE)
    f.arrow([(840, 185), (720, 185)], "corr", "xc", color=ORANGE)
    f.text(640, 245, "after the corrector", id="xcl", color=ORANGE, size=11.5)
    f.text(875, 55, "The tensors are real; the canvas is the r1 corrected 7 × 8 tile (seed 0) noised to step 100 by hand, "
                    "so the picture shows the mechanism, not one chain.", id="foot", color=GREY, size=12)
    f.check_and_save("r3-the-product-in-one-step")


# ------------------------------------------------------------------ r4: the judge
def r4():
    J = N["judge"]
    L = J["layers"]
    f = Fig(1750, 900, "The judge: what the counts in r1 count",
            f"A {J['parameters']:,}-weight CNN trained on 28 × 28 MNIST crops (shared with subproject 2, weights pinned by sha256). "
            "It reads each digit-sized blob of a finished tile, never the whole canvas.")
    # the tile and its blobs
    npz = np.load(SAMPLES / "dial-1-seed-1.npz")
    c = N["cells"]["1|1|7 x 8 corrected"]
    f.img("tile", npz["7 x 8 corrected"][c["tile seed"]], 30, 560, 220, 110)
    f.text(140, 695, "a finished tile", id="tl", color=BLUE, size=12)
    f.box("blob", 30, 400, 220, 110, "", fc="white", ec=GREY)
    f.text(140, 470, f"blobs of {J['min blob']}+ pixels", id="b1", color=INK, size=11.5)
    f.text(140, 440, f"one {J['split width']}+ pixels wide is cut\nat its least-ink column", id="b2", color=INK, size=10.5)
    f.arrow([(140, 560), (140, 510)], "tile", "blob")
    f.box("crop", 30, 260, 220, 80, f"each blob cropped to\n{J['crop']} × {J['crop']}, centred", fc="#e6eefb", ec=BLUE,
          lc=BLUE, size=11.5)
    f.arrow([(140, 400), (140, 340)], "blob", "crop", color=BLUE)
    # the CNN as layers of circles with real sizes
    xs = [380, 480, 560, 660, 760, 840, 940, 1040]
    names = [f"conv {L[0]['output'][0]}\n{L[0]['output'][1]}×{L[0]['output'][2]}",
             f"conv {L[1]['output'][0]}\n{L[1]['output'][1]}×{L[1]['output'][2]}",
             f"pool\n{L[2]['output'][1]}×{L[2]['output'][2]}",
             f"conv {L[3]['output'][0]}\n{L[3]['output'][1]}×{L[3]['output'][2]}",
             f"conv {L[4]['output'][0]}\n{L[4]['output'][1]}×{L[4]['output'][2]}",
             f"pool\n{L[5]['output'][1]}×{L[5]['output'][2]}",
             f"flatten\n{L[6]['output'][0]:,}", f"dense\n{L[7]['output'][0]}"]
    ys = [520, 480, 440, 400]
    cols = []
    for i, x in enumerate(xs):
        cc = column(f, f"j{i}_", x, ys, fc=FADE if "pool" in names[i] or "flatten" in names[i] else "white")
        cols.append(cc)
        f.text(x, ys[-1] - 50, names[i], id=f"jl{i}", color=GREEN, size=10.5)
    for a, b in zip(cols, cols[1:]):
        fan(f, a, b, color=GREEN)
    f.arrow([(250, 300), (330, 300), (330, 460), (380 - 14, 460)], "crop", "j0_1", color=BLUE)
    outs = []
    oys = list(np.linspace(640, 300, 11))
    for d, y in enumerate(oys):
        lab = str(d) if d < 10 else "two on\none spot"
        node(f, f"o{d}", 1170, y, r=13, fc="#fbe3e4" if d == 10 else "white", ec=RED, glow=d == 10,
             label=str(d) if d < 10 else "", lc=RED, size=10)
        outs.append((f"o{d}", 1170, y))
    f.text(1205, oys[-1], "two digits\non one spot", id="o10l", color=RED, size=10, ha="left")
    for i, (a, xa, ya) in enumerate(cols[-1]):
        for b, xb, yb in outs[i * 3:i * 3 + 4]:
            f.arrow([(xa + 14, ya), (xb - 13, yb)], a, b, color=GREY, lw=0.6, head=False)
    f.text(1170, 680, f"{J['classes']} classes", id="ocl", color=RED, size=12)
    # the rules
    f.box("rule", 1300, 560, 420, 110, "", fc="white", ec=ORANGE, glow=True)
    f.text(1510, 640, "a blob is fused if it reads as \"two on one spot\",", id="r1", color=INK, size=11)
    f.text(1510, 615, f"or, on a one-blob tile, no class reaches {J['confident']:.2f}", id="r2", color=INK, size=11)
    f.text(1510, 585, "(single-label rows read the ten digits only)", id="r3", color=GREY, size=10.5)
    # the ladder
    ladder = J["pair calls (ladder)"]
    f.text(1510, 500, "the ladder for a two-digit row, worst to best", id="ldl", color=INK, size=12, weight="bold")
    for i, name in enumerate(ladder):
        y = 460 - i * 52
        best = i == len(ladder) - 1
        f.box(f"lad{i}", 1340, y - 18, 340, 36, name, fc="#e3f1e8" if best else "white", ec=GREEN if best else GREY,
              lc=GREEN if best else INK, size=11.5, glow=best)
        if i:
            f.arrow([(1510, y + 34), (1510, y + 18)], f"lad{i - 1}", f"lad{i}", color=GREY)
    f.text(1510, 160, "single-label rows: damaged, wrong digit,\nright digit with a second, right digit alone",
           id="single", color=GREY, size=11)
    f.text(875, 70, "A count of 256 is how many of the 256 seeds reach the top of the ladder: \"two separate right digits\" "
                    "for a two-digit row, \"right digit alone\" for a single label.", id="foot", color=GREY, size=12)
    f.check_and_save("r4-the-judge")


# ------------------------------------------------------------------ r5: the product in one step, at each share
R5A = np.load(HERE / "r5_trajectories.npz") if (HERE / "r5_trajectories.npz").exists() else {}


def r5():
    R = N["r5 trajectories"]
    seed = R["seed"]
    f = Fig(2000, 1080, "The product at each share: one seed, step by step",
            f"7 × 8 plain product, training 1 of each setting, seed {seed} in every row. The starting noise and every "
            "step's noise are seeded by the seed and the step alone, so every row starts from the same noise.")
    ys = [880, 710, 540, 370, 200]
    levels = R["levels"]
    # the arrows' shared scale: the longest tip-to-tail sum over rows maps to 190 pixels
    vecs = {}
    for d in DIALS:
        d7, d8 = R5A[f"{d}/d7"].ravel(), R5A[f"{d}/d8"].ravel()
        u = d7 / np.linalg.norm(d7)
        w_ = d8 - (d8 @ u) * u
        w_ = w_ / (np.linalg.norm(w_) or 1)
        v1, v2 = 3 * np.array([np.linalg.norm(d7), 0.0]), 3 * np.array([d8 @ u, d8 @ w_])
        sm_ = v1 + v2                                       # turn the pair so their sum points right
        c_, s_ = sm_ / (np.linalg.norm(sm_) or 1)
        rot = np.array([[c_, s_], [-s_, c_]])
        vecs[d] = (rot @ v1, rot @ v2)
    pts = [p for a, b in vecs.values() for p in (a, a + b)]
    xs_, ys_ = [abs(p[0]) for p in pts], [abs(p[1]) for p in pts]
    scale = min(190 / max(xs_), 55 / max(ys_))
    f.text(400, 975, "(a) at step 100: the two label pushes, 3 (ε7 − ε∅) and 3 (ε8 − ε∅), tip to tail, one scale", id="ha",
           color=ORANGE, size=11.5, weight="bold")
    f.text(1240, 975, "(b) the plain product's canvas, the same seed throughout", id="hb", color=INK, size=11.5, weight="bold")
    f.text(1905, 975, "(c) the judge", id="hc", color=INK, size=11.5, weight="bold")
    for r, d in enumerate(DIALS):
        y = ys[r]
        st = R["by dial"][d]
        a = st["at step 100"]
        sh = N["shares"][d][S8]
        f.text(30, y + 45, f"{100 * sh:.1f}%", id=f"sh{r}", color=INK, size=16, weight="bold", ha="left")
        f.text(30, y + 15, f"dial {d}", id=f"dl{r}", color=GREY, size=10, ha="left")
        # (a) the arrows
        v1, v2 = vecs[d]
        o = np.array([200.0, y + 40])
        p1, p2 = o + v1 * scale, o + (v1 + v2) * scale
        f.ax.annotate("", xy=tuple(p1), xytext=tuple(o), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, mutation_scale=12))
        f.ax.annotate("", xy=tuple(p2), xytext=tuple(p1), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=12))
        f.ax.annotate("", xy=tuple(p2), xytext=tuple(o), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3,
                                                                        ls="--", mutation_scale=10))
        f.text(420, y + 70, f"|ε∅| {a['|eps blank|']}   |ε7 − ε∅| {a['|eps 7 - eps blank|']}   |ε8 − ε∅| {a['|eps 8 - eps blank|']}",
               id=f"na{r}", color=INK, size=10, ha="left")
        f.text(420, y + 42, f"cosine {a['cosine']}    summed push |3 (Δ7 + Δ8)| {a['|3 (sum of the two differences)|']}",
               id=f"nb{r}", color=ORANGE, size=10, ha="left")
        # (b) the canvases
        for i, k in enumerate(levels + ["stored"]):
            img = R5A[f"{d}/final stored"] if k == "stored" else R5A[f"{d}/k{k}"]
            f.img(f"c{r}{i}", img, 820 + i * 142, y, 130, 65, ec=GREEN if k == 0 else ("#9aa5b1" if k == "stored" else "#c9d1d9"))
            if r == len(DIALS) - 1:
                f.text(820 + i * 142 + 65, y - 18, {0: "finished", "stored": "stored sample"}.get(k, f"k = {k}"),
                       id=f"kl{i}", color=GREY, size=10)
        # (c) the call
        call = N["r5 calls of the recaptured canvas"][d]
        f.box(f"call{r}", 1905 - 80, y + 20, 160, 40, call, fc="#e3f1e8" if call == "two separate right digits" else "white",
              ec=GREEN if call == "two separate right digits" else GREY, lc=INK, size=9)
        f.text(1905, y + 5, f"stored sample: {N['r5 calls'][d]}", id=f"sc{r}", color=GREY, size=8.5)
    f.text(30, 140, "", id="pad")
    rr = [R["by dial"][d]["at step 100"]["|eps blank|"] / R["by dial"][d]["at step 100"]["|3 (sum of the two differences)|"] for d in DIALS]
    RAT = (min(rr), max(rr))
    f.text(980, 120, f"Blue: 3 (ε7 − ε∅); red: 3 (ε8 − ε∅), drawn in the plane of the two with their true lengths and angle; "
                     f"dashed orange: their sum, what the labels add to ε∅, which is {RAT[0]:.1f} to {RAT[1]:.0f} times longer than it.",
           id="foot1", color=GREY, size=11.5)
    agree = all(R["by dial"][d]["this seed: largest pixel difference from the stored sample"] == 0 for d in DIALS)
    mx = max(R["by dial"][d]["this seed: largest pixel difference from the stored sample"] for d in DIALS)
    f.text(980, 90, (f"Rerun with sample.py's draw() loop on the GPU; the finished canvases equal the stored samples for seed {seed}"
                     if agree else f"Rerun with sample.py's draw() loop on the GPU, compiled as sample.py does; the finished canvases "
                     f"differ from the stored samples (bf16 sampling does not repeat exactly; up to {mx} of 255 in a pixel), so the stored "
                     f"sample is shown last; the judge calls both the same in every row."),
           id="foot2", color=GREY, size=11.5)
    f.text(980, 60, f"Seed rule: the lowest seed whose call at 0% (dial 0, training 1) is that model's most common call "
                    f"(\"{N['cells']['0|1|7 x 8 plain']['majority call']}\"); the same seed in every row, never picked per row.",
           id="foot3", color=GREY, size=11.5)
    f.check_and_save("r5-the-product-in-one-step-at-each-share")


# ------------------------------------------------------------------ r6: what goes wrong and what emerges
CALL_FILL = {"damaged": "#e9ecef", "one digit": "#cfd6dd", "fused": "#f6dcc2", "two digits with a wrong identity": "#e6eefb",
             "two separate right digits": "#cfe9d8"}


def r6_earlier():
    R6 = N["r6 calls, training 1"]
    calls = N["judge"]["pair calls (ladder)"]
    f = Fig(1960, 1080, "What goes wrong, and what emerges, as the share rises",
            "The 7 × 8 product of training 1 at each share, its 256 seeds sorted by the judge's call. Each cell: the first seed "
            "with that call, its count of 256 plain, and the corrected product's count beneath.")
    x0, cw = 200, 300
    ys = [820, 660, 500, 340, 180]
    for c, name in enumerate(calls):
        f.text(x0 + c * cw + 110, 935, name, id=f"cn{c}", color=GREEN if c == len(calls) - 1 else INK, size=11.5, weight="bold")
    f.text(x0 + 5 * cw + 105, 935, "the 256 seeds by call", id="barh", color=INK, size=11.5, weight="bold")
    for r, d in enumerate(DIALS):
        y = ys[r]
        sh = N["shares"][d][S8]
        f.text(40, y + 55, f"{100 * sh:.1f}%", id=f"sh{r}", color=INK, size=16, weight="bold", ha="left")
        f.text(40, y + 25, f"dial {d}", id=f"dl{r}", color=GREY, size=10, ha="left")
        npz = np.load(SAMPLES / f"dial-{FILE[d]}-seed-1.npz")
        plain, corr = R6[d]["7 x 8 plain"], R6[d]["7 x 8 corrected"]
        for c, name in enumerate(calls):
            x = x0 + c * cw
            pc, cc = plain[name], corr[name]
            best = name == "two separate right digits"
            if pc["first seed"] is None:
                f.box(f"e{r}{c}", x, y + 10, 190, 95, "none", fc="white", ec="#dfe3e8", lc=GREY, size=12, dashed=True)
            else:
                f.img(f"i{r}{c}", npz["7 x 8 plain"][pc["first seed"]], x, y + 10, 190, 95,
                      ec=GREEN if best else "#c9d1d9")
            f.text(x + 240, y + 75, f"{pc['count']}", id=f"np{r}{c}", color=GREEN if best and pc["count"] else INK,
                   size=16, weight="bold")
            f.text(x + 240, y + 40, f"corrected {cc['count']}", id=f"nc{r}{c}", color=GREY, size=9.5)
            if pc["first seed"] is not None:
                f.text(x + 95, y - 5, f"seed {pc['first seed']}", id=f"sd{r}{c}", color=GREY, size=9)
        # the stacked bar: plain above, corrected below
        bx, bw = x0 + 5 * cw - 10, 230
        for j, (kind, yy, hh) in enumerate((("7 x 8 plain", y + 62, 26), ("7 x 8 corrected", y + 28, 14))):
            xx = bx
            for name in calls:
                n = R6[d][kind][name]["count"]
                if n:
                    f.ax.add_patch(Rectangle((xx, yy), bw * n / 256, hh, fc=CALL_FILL[name],
                                             ec=GREEN if name == "two separate right digits" else "#9aa5b1", lw=0.8))
                    xx += bw * n / 256
            f.rects[f"bar{r}{j}"] = (bx, yy, bx + bw, yy + hh)
        f.text(bx + bw / 2, y + 12, "plain (top), corrected (thin)", id=f"bl{r}", color=GREY, size=8.5)
    sh0, sh1 = N["shares"]["0"], N["shares"]["1"]
    j78 = [R6[d]["7 and 8, three trainings"][0] for d in DIALS]
    f.text(980, 105, f"The second change: as the share rises the joint label \"7 and 8\" is trained less, "
                     f"{100 * sh0[KJ]:.0f}% of a 7-and-8 canvas's labelled draws at 0% down to {100 * sh1[KJ]:.0f}% at "
                     f"{100 * sh1[S8]:.1f}%,", id="foot1", color=INK, size=12)
    f.text(980, 78, f"and \"7 and 8\" itself draws two separate right digits on {', '.join(map(str, j78))} of 256 "
                    f"(training 1, from 0% to {100 * sh1[S8]:.1f}%).", id="foot2", color=INK, size=12)
    f.text(980, 45, "Calls from scores.csv; columns in the judge's ladder order, worst to best (judge_checks.py). "
                    "\"none\": no seed of 256 got that call.", id="foot3", color=GREY, size=11)
    f.check_and_save("earlier/r6-what-goes-wrong-and-what-emerges")


# ------------------------------------------------------------------ r6: one row per share (the main results figure)
SHORT = {"damaged": "damaged", "one digit": "one", "fused": "fused", "two digits with a wrong identity": "wrong",
         "two separate right digits": "✓"}


def r6():
    R6 = N["r6 calls, training 1"]
    W_ = N["what training saw"]
    f = Fig(1960, 1180, "What training saw, what the 7 × 8 product draws, and how often it composes",
            f"One model per row (training 1). Each model trained on {W_['canvases per model']:,} canvases built fresh from "
            f"{W_['from']}; counts below are the expected draws per model, from canvases.py's rule.")
    ys = [930, 750, 570, 390, 210]
    f.text(130, 1060, "the share", id="h1", color=GREY, size=11.5)
    f.text(470, 1060, "what training saw (draws per model)", id="h2", color=GREY, size=11.5)
    f.text(1180, 1060, "what the 7 × 8 product draws: seeds 0 to 7, the same seeds in every row", id="h3", color=GREY, size=11.5)
    f.text(1800, 1060, "two separate right digits", id="h4", color=INK, size=11.5, weight="bold")
    for r, d in enumerate(DIALS):
        y = ys[r]
        sh = N["shares"][d]
        ex = sh["expected draws per model"]
        # 1. the share
        f.text(130, y + 75, f"{100 * sh[S8]:.1f}%", id=f"s{r}", color=INK, size=26, weight="bold")
        f.text(130, y + 35, "of \"7\" training pictures\nalso show an 8", id=f"sl{r}", color=GREY, size=10)
        f.text(130, y + 5, f"dial {d}", id=f"sd{r}", color=GREY, size=9)
        # 2. the table
        f.box(f"tb{r}", 300, y - 5, 340, 125, "", fc="white", ec="#d5dbe1")
        lines = [('"7"', f"{ex[chr(34) + '7' + chr(34)]:,}"),
                 ("   of which also hold an 8", f"{ex[chr(34) + '7' + chr(34) + ' whose canvas also holds an 8']:,}"),
                 ('"8"', f"{ex[chr(34) + '8' + chr(34)]:,}"), ('"7 and 8"', f"{ex[chr(34) + '7 and 8' + chr(34)]:,}"),
                 ("blank", f"{ex['blank']:,}")]
        for i, (nm, v) in enumerate(lines):
            yy = y + 98 - i * 23
            hot = i == 1
            f.text(318, yy, nm, id=f"tn{r}{i}", color=INK if hot else GREY, size=10.5, ha="left",
                   weight="bold" if hot else "normal")
            f.text(625, yy, v, id=f"tv{r}{i}", color=INK if hot else GREY, size=10.5, ha="right",
                   weight="bold" if hot else "normal")
        f.arrow([(640, y + 57), (700, y + 57)], f"tb{r}", f"i{r}0", color="#c3cbd3")
        # 3. the first eight seeds
        npz = np.load(SAMPLES / f"dial-{FILE[d]}-seed-1.npz")
        first = R6[d]["first 8 seeds, plain"]
        for q in range(8):
            x = 705 + q * 120
            ok = first[q] == "two separate right digits"
            f.img(f"i{r}{q}", npz["7 x 8 plain"][q], x, y + 30, 110, 55, ec=GREEN if ok else "#c9d1d9")
            f.text(x + 55, y + 12, SHORT[first[q]], id=f"m{r}{q}", color=GREEN if ok else GREY,
                   size=13 if ok else 9.5, weight="bold" if ok else "normal")
        f.arrow([(1665, y + 57), (1715, y + 57)], f"i{r}7", f"res{r}", color="#c3cbd3")
        # 4. the result
        plain = R6[d]["7 x 8 plain"]["two separate right digits"]["count"]
        corr = R6[d]["7 x 8 corrected"]["two separate right digits"]["count"]
        hit = plain >= N["bar"]["support min (judge, of 256)"]
        f.text(1800, y + 75, f"{plain} of 256", id=f"res{r}", color=GREEN if hit else INK, size=24, weight="bold")
        f.text(1800, y + 40, f"corrected: {corr}", id=f"rc{r}", color=GREY, size=11)
        f.text(1800, y + 15, "three trainings: " + ", ".join(map(str, R6[d]["7 x 8 plain, three trainings"])),
               id=f"r3{r}", color=GREY, size=9.5)
    sh0, sh1 = N["shares"]["0"], N["shares"]["1"]
    j78 = [R6[d]["7 and 8, three trainings"][0] for d in DIALS]
    pf = N["bar"]["purity floor"]
    s06, s1 = N["verdict"]["settings"]["0.6"], N["verdict"]["settings"]["1.0"]
    f.text(980, 125, f"As the share rises, the joint label is trained less: its share of a 7-and-8 canvas's labelled draws falls "
                     f"from {100 * sh0[KJ]:.0f}% to {100 * sh1[KJ]:.0f}%, and \"7 and 8\" itself draws the pair on "
                     f"{j78[0]}, then down to {j78[-1]}, of 256.", id="foot1", color=INK, size=12)
    f.text(980, 95, f"At {100 * N['shares']['0.6'][S8]:.1f}% and {100 * sh1[S8]:.1f}% the single labels fall below the purity line "
                    f"written before the run (\"7\" alone {min(s06['seven_alone'])} against {pf['7']['0.6']}; "
                    f"{min(s1['seven_alone'])} against {pf['7']['1']}), so those rows do not count and the verdict is "
                    f"{N['verdict']['verdict']}.", id="foot2", color=INK, size=12)
    f.text(980, 62, f"Marks under the tiles are the judge's calls (✓ two separate right digits; one: one digit; wrong: two digits, "
                    f"wrong identity). A count would be green at {N['bar']['support min (judge, of 256)']} or more, the line "
                    "written for the corrected product.", id="foot3", color=GREY, size=10.5)
    f.check_and_save("r6-one-row-per-share")


# ------------------------------------------------------------------ r7: how the product fails, and what replaces it
def r7():
    R6 = N["r6 calls, training 1"]
    pair = ["0", "1"]
    present = [c for c in N["judge"]["pair calls (ladder)"]
               if any(R6[d]["7 x 8 plain"][c]["count"] for d in pair)]
    cw = 440
    f = Fig(150 + cw * len(present) + 40, 720, "How the product fails, and what replaces it",
            "The 7 × 8 plain product of training 1 at the lowest and highest share; columns are the judge's calls that occur, "
            "worst to best; each cell shows the first three seeds with that call and its count of 256.")
    ys = [430, 160]
    for c, name in enumerate(present):
        best = name == "two separate right digits"
        f.text(150 + c * cw + cw / 2 - 20, 600, name, id=f"c{c}", color=GREEN if best else INK, size=12.5, weight="bold")
    for r, d in enumerate(pair):
        y = ys[r]
        sh = N["shares"][d]
        f.text(70, y + 95, f"{100 * sh[S8]:.1f}%", id=f"s{r}", color=INK, size=22, weight="bold")
        f.text(70, y + 60, "of \"7\" pictures\nalso show an 8", id=f"sl{r}", color=GREY, size=9.5)
        f.text(70, y + 30, f"dial {d}", id=f"sd{r}", color=GREY, size=9)
        npz = np.load(SAMPLES / f"dial-{FILE[d]}-seed-1.npz")
        for c, name in enumerate(present):
            x = 150 + c * cw
            best = name == "two separate right digits"
            n = R6[d]["7 x 8 plain"][name]["count"]
            seeds = R6[d]["first three seeds per call, plain"][name]
            if not seeds:
                f.box(f"e{r}{c}", x, y + 70, 400, 70, "none", fc="white", ec="#dfe3e8", lc=GREY, size=12, dashed=True)
            for q, sd in enumerate(seeds):
                f.img(f"i{r}{c}{q}", npz["7 x 8 plain"][sd], x + q * 135, y + 70, 125, 62, ec=GREEN if best else "#c9d1d9")
                f.text(x + q * 135 + 62, y + 55, f"seed {sd}", id=f"sd{r}{c}{q}", color=GREY, size=8.5)
            bw = 400 * n / 256
            f.ax.add_patch(Rectangle((x, y + 5), max(bw, 1), 26, fc="#cfe9d8" if best else "#e4e7eb",
                                     ec=GREEN if best else "#9aa5b1", lw=1))
            f.rects[f"b{r}{c}"] = (x, y + 5, x + max(bw, 1), y + 31)
            f.text(x + max(bw, 1) + 8 if bw < 340 else x + bw - 8, y + 18, f"{n}", id=f"n{r}{c}",
                   color=GREEN if best and n else INK, size=13, weight="bold", ha="left" if bw < 340 else "right")
    f.text(f.ax.get_xlim()[1] / 2, 60, "Bars are drawn to one scale, the full width being all 256 seeds.", id="foot",
           color=GREY, size=11)
    f.check_and_save("r7-how-the-product-fails-and-what-replaces-it")


if __name__ == "__main__":
    which = sys.argv[1:] or ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]
    for name in which:
        globals()[name]()
