"""Level 0.5 of the digit-map checker ladder: the network view, figures dc0 to dc7.

Every size and value comes from ../numbers/numbers.json and the example canvases in ../numbers/examples/. The real maps
in dc2 and the real gradient in dc7 come from ../numbers/real/count-explained.npz (job 94108: the frozen U-Net and the
saved count checker run on the same example canvases with the same display noise).

Run: ~/.claude/venvs/figures/bin/python -I make_network_view.py   (prints "0 crossings" per figure)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_kit import (COUNT, COUNT_FILL, FADED_FILL, FADED_TEXT, GREY, HIDDEN, HIDDEN_FILL, IDENT, IDENT_FILL,  # noqa: E402
                        INK, INPUT, INPUT_FILL, ORANGE, ORANGE_FILL, OUTPUT, OUTPUT_FILL, WHITE, Fig)

N = json.loads((HERE.parent / "numbers" / "numbers.json").read_text())
EX = HERE.parent / "numbers" / "examples"
REAL = np.load(HERE.parent / "numbers" / "real" / "count-explained.npz")
MAP_TOP = 10.0          # dc2's colour scale runs 0 to MAP_TOP; pixels above it are drawn at MAP_TOP and counted on the figure


def real(key, name="apart_1_3", t=480):
    i, j = list(REAL["names"]).index(name), list(REAL["timesteps"]).index(t)
    return REAL[key][i, j]


def ex(name, t=None):
    return np.load(EX / (f"{name}-clean.npy" if t is None else f"{name}-t{t}.npy"))


def node(f, x, y, w, h, lines, color=INK, fill=WHITE, stroke=GREY, size=14, oid=None, lw=1.2, weight="normal", dash=None):
    """A box with centred lines of text inside; returns every id an arrow touching it may cross."""
    bid = f.card(x, y, w, h, fill, stroke, lw, 8, oid, dash)
    ids = [bid]
    gap = size * 1.45
    top = y + h / 2 - gap * (len(lines) - 1) / 2
    for i, s in enumerate(lines):
        ids.append(f.text(x + w / 2, top + i * gap, s, size, color, weight if i == 0 else "normal", "center", "center"))
    return ids


def title(f, s, sub):
    f.text(30, 42, s, 24, INK, "bold")
    f.text(30, 74, sub, 14, GREY)


def footer(f, s, y=None):
    f.text(30, y or f.H - 22, s, 12, GREY)


# ---------------------------------------------------------------------------------------------------- dc0
def dc0():
    f = Fig("dc0-where-the-checkers-sit", 1640, 820)
    title(f, "Where the two checkers sit in one sampler step",
          "The frozen digit model runs three times on the noisy canvas. A small trained checker reads two maps made from those runs, and its gradient becomes a push.")
    img = f.image(ex("apart_1_3", 616), 40, 300, 200, 100, oid="xt")
    f.text(140, 288, "noisy canvas $x_t$, 1 × 32 × 64", 14, INPUT, ha="center", oid="xt-label")
    f.text(400, 128, "frozen U-Net, three calls", 13, GREY, ha="center", oid="unet-head")
    u = [node(f, 300, y, 210, 56, [s, "frozen"], GREY, FADED_FILL, GREY, 13, oid=f"u{i}")
         for i, (y, s) in enumerate(((150, "asked digit a"), (250, "asked digit b"), (350, "asked nothing (label 10)")))]
    for i, ids in enumerate(u):
        yy = 178 + 100 * i
        f.route([(240, 350), (270, 350), (270, yy), (300, yy)], GREY, 1.2, ids + [img], f"in{i}") if i != 2 else \
            f.route([(240, 350), (270, 350), (270, yy), (300, yy)], GREY, 1.2, ids + [img], f"in{i}")
    f.line(510, 178, 590, 178, GREY, allowed=u[0], name="o0")
    f.line(510, 278, 590, 278, GREY, allowed=u[1], name="o1")
    f.line(510, 378, 590, 378, GREY, allowed=u[2], name="o2")
    f.line(590, 178, 590, 378, GREY, name="bus")
    prod = node(f, 640, 140, 300, 76, [r"$\varepsilon_{\rm product} = \varepsilon_a + \varepsilon_b - \varepsilon_\varnothing$",
                                       "the product of the two digit experts"], OUTPUT, WHITE, OUTPUT, 14, oid="prod")
    f.arrow(590, 178, 640, 178, GREY, allowed=prod, name="to-prod")
    maps = node(f, 640, 330, 220, 76, ["two maps $m_a$, $m_b$", "2 × 32 × 64"], HIDDEN, HIDDEN_FILL, HIDDEN, 14, oid="maps")
    f.arrow(590, 368, 640, 368, GREY, allowed=maps, name="to-maps")
    f.halo(920, 320, 250, 96, HIDDEN)
    chk = node(f, 920, 320, 250, 96, ["checker: count or identity", "481,665 trained weights", "never told a or b"],
               HIDDEN, WHITE, HIDDEN, 14, oid="chk", lw=1.6, weight="bold")
    f.arrow(860, 368, 920, 368, HIDDEN, allowed=maps + chk, name="m-to-chk")
    logit = node(f, 1230, 338, 150, 60, ["one logit $\\ell$"], OUTPUT, OUTPUT_FILL, OUTPUT, 15, oid="logit")
    f.arrow(1170, 368, 1230, 368, HIDDEN, allowed=chk + logit, name="chk-to-logit")
    # backward lane
    f.route([(1305, 398), (1305, 560), (140, 560), (140, 610)], ORANGE, 2.0, logit + ["grad"], "lane")
    f.text(720, 548, "backward: $\\nabla_{x_t}\\ell$ runs back through the checker, the two maps and all three U-Net calls",
           13, ORANGE, ha="center", oid="lane-label")
    grad = node(f, 40, 610, 200, 60, ["$\\nabla_{x_t}\\ell$, 1 × 32 × 64", "same shape as the canvas"], ORANGE, ORANGE_FILL,
                ORANGE, 13, oid="grad")
    push = node(f, 380, 600, 560, 80, [r"push $= \Vert\varepsilon_{\rm product}\Vert \; b \; \nabla_{x_t}\ell \, / \, \Vert\nabla_{x_t}\ell\Vert$",
                                       "b = 0.4: the push is 0.4 of the product's length"], ORANGE, WHITE, ORANGE, 14, oid="push")
    f.arrow(240, 640, 380, 640, ORANGE, 2.0, grad + push, "grad-to-push")
    comb = node(f, 1100, 600, 300, 80, [r"$\hat\varepsilon = \varepsilon_{\rm product} - $ push", "what the sampler uses"],
                OUTPUT, OUTPUT_FILL, OUTPUT, 14, oid="comb")
    f.arrow(940, 640, 1100, 640, ORANGE, 2.0, push + comb, "push-to-comb")
    f.route([(940, 178), (1480, 178), (1480, 640), (1400, 640)], OUTPUT, 1.4, prod + comb, "prod-to-comb")
    ddim = node(f, 1100, 712, 300, 56, ["DDIM step to $x_{t-1}$, the next canvas"], INK, WHITE, INK, 14, oid="ddim")
    f.arrow(1250, 680, 1250, 712, INK, allowed=comb + ddim, name="to-ddim")
    footer(f, "Grey: frozen. Green: the trained checker. Red: what the networks output. Orange: the backward signal and the push.",
           f.H - 42)
    footer(f, "The push acts at sampler steps 15 to 25 of 50 (timesteps 693 to 489). On digits 5 to 9 it gives 43 clean pairs "
              "of 320 against 0 for the plain product (job 94134).")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc1
def dc1():
    steps = N["noise"]["shown_in_strip"]
    sched = {r["t"]: r for r in N["noise"]["schedule"]}
    f = Fig("dc1-the-canvas-and-its-noise", 1560, 640)
    title(f, "The canvas, and what the noise leaves of it",
          "Two real training canvases for the asked pair 1 and 3, noised at four of the sixteen timesteps the checkers train on.")
    x0, w, h, gap = 330, 200, 100, 30
    cols = [("clean $x_0$", None)] + [(f"t = {t}", t) for t in steps]
    for c, (head, t) in enumerate(cols):
        cx = x0 + c * (w + gap) + w / 2
        f.text(cx, 132, head, 15, INK, "bold", "center", oid=f"h{c}")
        if t is not None:
            r = sched[t]
            f.text(cx, 156, f"ᾱ = {r['abar']:.4f}", 13, GREY, ha="center", oid=f"a{c}")
            f.text(cx, 176, f"√ᾱ = {r['sqrt_abar']:.3f},  √(1−ᾱ) = {r['sqrt_one_minus_abar']:.3f}", 12, GREY, ha="center", oid=f"s{c}")
    rows = [("apart_1_3", "apart: shows 1 | 3", "count 1, identity 1"), ("wrong_1_3", "wrong: shows 3 | 2", "count 1, identity 0")]
    for r, (name, l1, l2) in enumerate(rows):
        y = 200 + r * 150
        f.text(300, y + 44, l1, 15, INPUT, "bold", "right", oid=f"r{r}a")
        f.text(300, y + 68, l2, 13, GREY, ha="right", oid=f"r{r}b")
        for c, (_, t) in enumerate(cols):
            f.image(ex(name, t), x0 + c * (w + gap), y, w, h, oid=f"im{r}{c}")
    f.box(330, 520, 520, 64, WHITE, INK, 1.0, 8, "formula")
    f.text(590, 560, r"$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$", 22, INK, ha="center", oid="ftext")
    f.text(880, 545, "Linear betas 1e-4 to 0.02 over T = 1000. The 16 timesteps 712 to 480 give the toy SDXL's", 13, INK, oid="n1")
    f.text(880, 565, "signal share at its first 16 sampler steps. One noise draw ε (numpy seed 0) is shared", 13, INK, oid="n2")
    f.text(880, 585, "across the columns, so only the level changes; the checker draws its own noise with torch.", 13, INK, oid="n3")
    footer(f, "Pixels run from −1 (background) to +1 (ink). The checker never sees these pictures: it sees the two maps the frozen U-Net makes from them (dc2).")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc2
def map_tile(f, x, y, w, h, oid, label, m):
    f.image(m, x, y, w, h, cmap="Greens", vmin=0, vmax=MAP_TOP, frame=HIDDEN, oid=oid)
    f.text(x + w / 2, y - 10, label, 12, HIDDEN, ha="center", oid=oid + "-t1")
    return int((m > MAP_TOP).sum())


def dc2():
    f = Fig("dc2-where-the-two-maps-come-from", 1640, 700)
    title(f, "Where the two maps come from",
          "Each map shows where asking for one digit changes the frozen model's noise guess. The checker gets the two maps and never the digit labels.")
    img = f.image(ex("apart_1_3", 480), 40, 300, 200, 100, oid="xt")
    f.text(140, 288, "$x_t$ at t = 480", 14, INPUT, ha="center", oid="xt-l")
    calls = [("asked a = 1", "ea", r"$\varepsilon_a$"), ("asked b = 3", "eb", r"$\varepsilon_b$"),
             ("asked nothing (10)", "e0", r"$\varepsilon_\varnothing$")]
    outs = []
    for i, (s, oid, sym) in enumerate(calls):
        y = 160 + 120 * i
        u = node(f, 310, y, 200, 56, ["U-Net, frozen", s], GREY, FADED_FILL, GREY, 13, oid=f"u-{oid}")
        f.route([(240, 350), (275, 350), (275, y + 28), (310, y + 28)], GREY, 1.2, u + [img], f"in-{oid}")
        e = node(f, 570, y + 4, 150, 48, [sym + ",  1 × 32 × 64"], OUTPUT, WHITE, OUTPUT, 13, oid=f"o-{oid}")
        f.arrow(510, y + 28, 570, y + 28, GREY, allowed=u + e, name=f"to-{oid}")
        outs.append(e)
    da = node(f, 800, 170, 200, 60, [r"$\varepsilon_a - \varepsilon_\varnothing$", "what asking for a changes"], HIDDEN, WHITE, HIDDEN, 13, oid="da")
    db = node(f, 800, 300, 200, 60, [r"$\varepsilon_b - \varepsilon_\varnothing$", "what asking for b changes"], HIDDEN, WHITE, HIDDEN, 13, oid="db")
    f.arrow(720, 188, 800, 196, GREY, allowed=outs[0] + da, name="ea-da")
    f.arrow(720, 308, 800, 326, GREY, allowed=outs[1] + db, name="eb-db")
    f.route([(720, 428), (760, 428), (760, 216), (800, 216)], GREY, 1.2, outs[2] + da, "e0-da")
    f.arrow(760, 344, 800, 344, GREY, allowed=db, name="e0-db")
    nm = [f.card(1060, 196, 250, 136, HIDDEN_FILL, HIDDEN, 1.2, 8, "norm")]
    nm.append(f.text(1185, 236, r"$m = 2048 \cdot \frac{|d|}{\sum_{\rm pixels} |d|}$", 18, HIDDEN, ha="center", va="center", oid="norm-f"))
    nm.append(f.text(1185, 290, "absolute value, then scaled", 13, HIDDEN, ha="center", va="center", oid="norm-a"))
    nm.append(f.text(1185, 312, "so each map adds to 2,048", 13, HIDDEN, ha="center", va="center", oid="norm-b"))
    f.route([(1000, 200), (1030, 200), (1030, 250), (1060, 250)], HIDDEN, 1.2, da + nm, "da-norm")
    f.route([(1000, 330), (1030, 330), (1030, 290), (1060, 290)], HIDDEN, 1.2, db + nm, "db-norm")
    m = real("maps")
    over_a = map_tile(f, 1380, 150, 200, 100, "ma", f"$m_a$, channel 1, largest {m[0].max():.1f}", m[0])
    over_b = map_tile(f, 1380, 290, 200, 100, "mb", f"$m_b$, channel 2, largest {m[1].max():.1f}", m[1])
    f.route([(1310, 250), (1345, 250), (1345, 200), (1380, 200)], HIDDEN, 1.2, nm + ["ma"], "to-ma")
    f.route([(1310, 290), (1345, 290), (1345, 340), (1380, 340)], HIDDEN, 1.2, nm + ["mb"], "to-mb")
    f.text(1480, 420, "stacked: the checker's input,", 13, HIDDEN, ha="center", oid="st1")
    f.text(1480, 440, "2 × 32 × 64", 13, HIDDEN, "bold", ha="center", oid="st2")
    f.text(40, 540, "What follows from the definition", 15, INK, "bold", oid="w0")
    f.text(40, 566, "• A map's mean over the 2,048 pixels is 1, so its total size is divided out and only its shape is kept.", 13, INK, oid="w1")
    f.text(40, 588, "• The digit labels a and b enter only through which two U-Net calls are made. Swapping the asked pair swaps the channels.", 13, INK, oid="w2")
    f.text(40, 610, "• In training the three calls run under torch.no_grad (v2_learn.py line 106), so no gradient reaches the U-Net there.", 13, INK, oid="w3")
    f.text(40, 632, f"• The two tiles are the real maps of this canvas at t = 480. White is 0 and dark green {MAP_TOP:.0f} or more "
                    f"({over_a} pixels of $m_a$ and {over_b} of $m_b$ lie above {MAP_TOP:.0f}).", 13, INK, oid="w4")
    footer(f, "Source: v2_learn.py lines 85-98 (maps), canvas_unet.py line 35 (label 10 means no digit). Model: the share-0.0 canvas U-Net. "
              "Real maps: numbers/real/count-explained.npz, job 94108.")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc3
def dc3():
    L = {r["layer"]: r for r in N["head"]["layers"]}
    f = Fig("dc3-the-checker-network", 1760, 720)
    title(f, "The checker: 481,665 weights from two maps to one number",
          "The same network for both branches. Each volume's face is its grid (height × width) and its depth is its channels.")
    S = 3.0  # pixels per grid cell on the face

    def vol(x, c, h, w, color, fill, oid, depth):
        top = 380 - h * S / 2
        f.volume(x, top, w * S, h * S, depth, fill, color, oid=oid)
        return top, x + w * S + depth

    specs = [(40, 2, 32, 64, HIDDEN, HIDDEN_FILL, "v0", 6, ["two maps", "2 × 32 × 64"]),
             (300, 64, 32, 64, HIDDEN, HIDDEN_FILL, "v1", 20, ["conv 3×3, SiLU", "64 × 32 × 64", f"{L['stem conv 1']['weights']:,} w"]),
             (580, 64, 16, 32, HIDDEN, HIDDEN_FILL, "v2", 20, ["conv 3×3 stride 2, SiLU", "64 × 16 × 32", f"{L['stem conv 2']['weights']:,} w"]),
             (790, 128, 8, 16, HIDDEN, HIDDEN_FILL, "v3", 30, ["conv 3×3 stride 2, SiLU", "128 × 8 × 16", f"{L['stem conv 3']['weights']:,} w"]),
             (960, 128, 8, 16, HIDDEN, HIDDEN_FILL, "v4", 30, ["c1, conv 3×3, SiLU", "then scale and shift", "128 × 8 × 16", f"{L['c1']['weights']:,} w"]),
             (1130, 128, 8, 16, HIDDEN, HIDDEN_FILL, "v5", 30, ["c2, conv 3×3, SiLU", "128 × 8 × 16", f"{L['c2']['weights']:,} w"])]
    ends = []
    for x, c, h, w, col, fill, oid, d, lab in specs:
        top, right = vol(x, c, h, w, col, fill, oid, d)
        for i, s in enumerate(lab):
            f.text(x + w * S / 2, 380 + h * S / 2 + 26 + 20 * i, s, 13, col if i == 0 else INK, "bold" if i == 0 else "normal",
                   "center", oid=f"{oid}-l{i}")
        ends.append((x, right))
    for i in range(len(ends) - 1):
        f.arrow(ends[i][1] + 6, 380, ends[i + 1][0] - 6, 380, GREY, 1.4, name=f"a{i}")
    # pool and MLP
    pool = node(f, 1300, 340, 120, 80, ["mean + max", "over 128", "positions: 256"], HIDDEN, WHITE, HIDDEN, 13, oid="pool")
    f.arrow(ends[-1][1] + 6, 380, 1300, 380, GREY, 1.4, allowed=pool, name="to-pool")
    mlp = node(f, 1470, 330, 140, 100, ["MLP", "272 → 256 → 1", f"{L['mlp 1']['weights'] + L['mlp 2']['weights']:,} w"],
               HIDDEN, WHITE, HIDDEN, 13, oid="mlp")
    f.arrow(1420, 380, 1470, 380, GREY, 1.4, allowed=pool + mlp, name="to-mlp")
    out = node(f, 1650, 350, 80, 60, ["logit $\\ell$"], OUTPUT, OUTPUT_FILL, OUTPUT, 14, oid="out")
    f.arrow(1610, 380, 1650, 380, GREY, 1.4, allowed=mlp + out, name="to-out")
    # time path
    tf = node(f, 840, 120, 310, 56, ["t → 16 time features", "sin, cos of 8 frequencies × log-SNR/10"], INPUT, INPUT_FILL, INPUT, 13, oid="tf")
    film = node(f, 940, 200, 200, 50, ["scale γ, shift β: 128 each", f"linear 16 → 256, {L['film']['weights']:,} w"], INPUT, WHITE, INPUT, 12, oid="film")
    f.arrow(990, 176, 990, 200, INPUT, allowed=tf + film, name="tf-film")
    f.arrow(1010, 250, 1010, 326, INPUT, allowed=film + ["v4"], name="film-v4")
    f.route([(1150, 148), (1540, 148), (1540, 330)], INPUT, 1.2, tf + mlp, "tf-mlp")
    f.text(1345, 138, "the same 16 features join the MLP's input", 12, INPUT, ha="center", oid="tf-mlp-l")
    f.box(40, 560, 620, 100, WHITE, INK, 1.0, 8, "arith")
    f.text(350, 592, r"$n_{\rm out} = \lfloor (n_{\rm in} + 2p - k)/s \rfloor + 1$", 20, INK, ha="center", oid="ar0")
    f.text(350, 628, "k = 3, p = 1, s = 2:  (32 + 2 − 3)/2 → 15 → 16;  (64 + 2 − 3)/2 → 31 → 32", 13, INK, ha="center", oid="ar1")
    f.text(700, 590, r"Scale and shift on every channel at all 128 positions: $h \leftarrow h\,(1+\gamma) + \beta$.", 13, INK, oid="t1")
    f.text(700, 614, "The pooled 256 numbers plus the 16 time features make the MLP's 272 inputs.", 13, INK, oid="t2")
    f.text(700, 638, f"Total: {N['head']['weights']:,} weights, counted from the code and equal to the smoke stage's count.", 13, INK, oid="t3")
    footer(f, "Source: PixelHead in scripts/mnist/pixel_checker.py lines 33-56, first conv swapped to 2 input channels in v2_learn.py lines 77-82.")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc4
def dc4():
    vals = np.array(N["time_features"]["values"])
    sched = N["noise"]["schedule"]
    f = Fig("dc4-the-timestep-as-sixteen-numbers", 1500, 760)
    title(f, "The timestep enters as sixteen numbers",
          "Each row is one of the 16 training timesteps; each column one feature. Real values from checker.py's time_features.")
    x0, y0, cw, ch = 300, 140, 34, 28
    f.image(vals, x0, y0, 16 * cw, 16 * ch, cmap="RdBu_r", vmin=-1, vmax=1, frame=INPUT, oid="heat")
    for i in range(16):
        f.text(x0 - 10, y0 + ch * i + ch / 2, f"t = {sched[i]['t']},  ᾱ = {sched[i]['abar']:.4f}", 11, INK, ha="right",
               va="center", oid=f"row{i}")
    f.text(x0 + 4 * cw, y0 - 14, "sin, 8 frequencies", 12, INPUT, ha="center", oid="c-sin")
    f.text(x0 + 12 * cw, y0 - 14, "cos, 8 frequencies", 12, INPUT, ha="center", oid="c-cos")
    for k in (0, 3, 8, 15):
        f.text(x0 + cw * k + cw / 2, y0 + 16 * ch + 18, f"{vals[0, k]:+.2f}", 10, GREY, ha="center", oid=f"v0-{k}")
    f.text(x0 + 8 * cw, y0 + 16 * ch + 40, "row t = 712's values under columns 1, 4, 9 and 16", 11, GREY, ha="center", oid="vnote")
    # colour bar labels
    bar = np.linspace(1, -1, 50)[:, None]
    f.image(bar, x0 + 16 * cw + 30, y0, 18, 16 * ch, cmap="RdBu_r", vmin=-1, vmax=1, frame=GREY, oid="bar")
    f.text(x0 + 16 * cw + 54, y0 + 10, "+1", 11, INK, oid="b1")
    f.text(x0 + 16 * cw + 54, y0 + 16 * ch, "−1", 11, INK, oid="b2")
    fx = 960
    f.box(fx, 150, 500, 90, WHITE, INK, 1.0, 8, "f1")
    f.text(fx + 250, 186, r"$f_k = e^{3k/7}\cdot \log\frac{\bar\alpha_t}{1-\bar\alpha_t}\,/\,10,\;\; k = 0..7$", 18, INK, ha="center", oid="f1t")
    f.text(fx + 250, 222, "features: sin f_0 .. sin f_7, cos f_0 .. cos f_7", 13, INK, ha="center", oid="f1u")
    f.text(fx, 290, f"log-SNR runs from {sched[0]['log_snr']:.2f} at t = 712 to {sched[-1]['log_snr']:.2f} at t = 480.", 13, INK, oid="l1")
    f.text(fx, 314, "The low frequencies change slowly across the window and the", 13, INK, oid="l2")
    f.text(fx, 338, "high ones wrap round, so nearby timesteps get distinct rows.", 13, INK, oid="l3")
    f.text(fx, 400, "Where the 16 numbers go", 15, INK, "bold", oid="g0")
    f.text(fx, 428, "1. A linear layer turns them into a scale γ and a shift β", 13, INPUT, oid="g1")
    f.text(fx, 450, "    for each of c1's 128 channels: h ← h (1 + γ) + β.", 13, INPUT, oid="g2")
    f.text(fx, 478, "2. They are appended to the 256 pooled numbers,", 13, INPUT, oid="g3")
    f.text(fx, 500, "    making the MLP's 272 inputs.", 13, INPUT, oid="g4")
    f.text(fx, 540, "Training draws one of the 16 timesteps per example, so one", 13, INK, oid="h1")
    f.text(fx, 562, "network learns all 16 noise levels and is told which it is on.", 13, INK, oid="h2")
    footer(f, "Source: time_features, scripts/mnist/checker.py lines 69-73, run in numpy by numbers/make_numbers.py.")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc5
def chip(f, x, y, w, text, color, fill, oid):
    f.box(x, y, w, 28, fill, color, 1.2, 14, oid, "card")
    f.text(x + w / 2, y + 15, text, 13, color, "bold", "center", "center", oid=oid + "-t")


def dc5():
    B = N["branches"]
    f = Fig("dc5-two-checkers-two-label-sets", 1640, 760)
    title(f, "Two checkers, the same network, two label sets",
          "The five canvas kinds for the asked pair 1 and 3, real training canvases. Only the label column differs between the branches.")
    kinds = [("apart_1_3", "apart", "1 | 3"), ("overlaid_1_3", "overlaid", "1 on 3"), ("same_half_1_3", "same half", "1, 3 right"),
             ("single_1_3", "single", "3 alone"), ("wrong_1_3", "wrong", "3 | 2")]
    x0, w, gap = 210, 220, 36
    f.text(190, 290, "COUNT", 16, COUNT, "bold", "right", oid="rc")
    f.text(190, 312, "two digits apart?", 12, COUNT, ha="right", oid="rc2")
    f.text(190, 352, "IDENTITY", 16, IDENT, "bold", "right", oid="ri")
    f.text(190, 374, "both asked digits?", 12, IDENT, ha="right", oid="ri2")
    for c, (name, k, shows) in enumerate(kinds):
        x = x0 + c * (w + gap)
        info = N["examples"][name]
        f.text(x + w / 2, 132, k, 16, INK, "bold", "center", oid=f"k{c}")
        f.text(x + w / 2, 152, f"shows {shows}", 12, GREY, ha="center", oid=f"s{c}")
        f.image(ex(name), x, 162, w, w / 2, oid=f"im{c}")
        chip(f, x + 50, 284, w - 100, f"label {info['count_label']}", COUNT, COUNT_FILL, f"cc{c}")
        if info["identity_label"] is None:
            f.text(x + w / 2, 363, "not used", 13, FADED_TEXT, ha="center", va="center", oid=f"ci{c}")
        else:
            chip(f, x + 50, 348, w - 100, f"label {info['identity_label']}", IDENT, IDENT_FILL, f"ci{c}")
    ct, it = B["count"]["train_pairs_0_to_3"], B["identity"]["train_pairs_0_to_3"]
    ch, ih = B["count"]["test_pairs_holding_4"], B["identity"]["test_pairs_holding_4"]
    f.box(40, 430, 760, 270, WHITE, COUNT, 1.4, 10, "cbox", "card")
    f.text(60, 464, "COUNT branch", 16, COUNT, "bold", oid="cb0")
    f.text(60, 494, f"Trains on the 6 pairs among digits 0 to 3: {ct['canvases']:,} canvases,", 13, INK, oid="cb1")
    f.text(60, 516, f"{ct['label_1']:,} label 1 (6,000 apart, 1,004 wrong) and {ct['label_0']:,} label 0.", 13, INK, oid="cb2")
    f.text(60, 546, f"Scored on the 4 pairs holding digit 4: {ch['canvases']} canvases, {ch['label_1']} label 1.", 13, INK, oid="cb3")
    f.text(60, 576, "What it has to see: where ink sits, two blobs one per half.", 13, INK, oid="cb4")
    f.text(60, 598, "Which digits they are does not change the label.", 13, INK, oid="cb5")
    f.text(60, 636, "Bar: balanced accuracy at least 0.75 on pairs holding 4,", 13, COUNT, oid="cb6")
    f.text(60, 658, "and within 0.05 of the trained pairs.", 13, COUNT, oid="cb7")
    f.box(840, 430, 760, 270, WHITE, IDENT, 1.4, 10, "ibox", "card")
    f.text(860, 464, "IDENTITY branch", 16, IDENT, "bold", oid="ib0")
    f.text(860, 494, f"Trains on apart against wrong only: {it['canvases']:,} canvases,", 13, INK, oid="ib1")
    f.text(860, 516, f"{it['label_1']:,} apart and {it['label_0']:,} wrong. Each batch is 128 of each, so each", 13, INK, oid="ib2")
    f.text(860, 538, "wrong canvas is drawn about 510 times over 4,000 updates.", 13, INK, oid="ib3")
    f.text(860, 568, f"Scored on pairs holding 4: {ih['canvases']} canvases, {ih['label_1']} apart and {ih['label_0']} wrong.", 13, INK, oid="ib4")
    f.text(860, 598, "What it has to see: which digit each blob is. Both kinds", 13, INK, oid="ib5")
    f.text(860, 620, "have the same layout, two blobs apart.", 13, INK, oid="ib6")
    f.text(860, 658, "Bar: balanced accuracy at least 0.75 on pairs holding 4.", 13, IDENT, oid="ib7")
    footer(f, "Counts from the rebuilt canvases (data hash f70b0fb3...) with v2_learn.py's inner_splits and relabel. Canvases holding a 4 are dropped from training.")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc6
def dc6():
    f = Fig("dc6-training-backprop-into-the-weights", 1700, 760)
    title(f, "Training: the error flows back into the checker's weights",
          "One update. The forward pass runs left to right; the gradient of the loss runs right to left in the orange lane and stops at the weights.")
    unet = node(f, 40, 200, 170, 76, ["U-Net, three calls", "frozen, torch.no_grad"], FADED_TEXT, FADED_FILL, GREY, 13, oid="unet")
    maps = node(f, 260, 200, 140, 76, ["maps", "2 × 32 × 64"], HIDDEN, HIDDEN_FILL, HIDDEN, 14, oid="maps")
    f.arrow(210, 238, 260, 238, GREY, allowed=unet + maps, name="u-m")
    L = {r["layer"]: r["weights"] for r in N["head"]["layers"]}
    stem = L["stem conv 1"] + L["stem conv 2"] + L["stem conv 3"]
    layers = [("stem", "3 convs", f"{stem:,} w"), ("c1", "conv", f"{L['c1']:,} w"), ("scale, shift", "from t", f"{L['film']:,} w"),
              ("c2", "conv", f"{L['c2']:,} w"), ("pool", "mean, max", "0 w"), ("MLP", "272 → 256 → 1", f"{L['mlp 1'] + L['mlp 2']:,} w")]
    xs, prev = [], maps
    for i, (a, b, c) in enumerate(layers):
        x = 450 + i * 150
        f.halo(x, 200, 120, 76, HIDDEN) if c != "0 w" else None
        ids = node(f, x, 200, 120, 76, [a, b, c], HIDDEN, WHITE, HIDDEN, 13, oid=f"L{i}")
        f.arrow((xs[-1][0] + 120) if xs else 400, 238, x, 238, HIDDEN, allowed=prev + ids, name=f"f{i}")
        xs.append((x, ids))
        prev = ids
    lg = node(f, 1360, 205, 110, 66, ["logit $\\ell$"], OUTPUT, OUTPUT_FILL, OUTPUT, 15, oid="lg")
    f.arrow(1320, 238, 1360, 238, HIDDEN, allowed=prev + lg, name="f-lg")
    loss = node(f, 1510, 190, 160, 96, ["loss L", "binary", "cross-entropy"], OUTPUT, WHITE, OUTPUT, 13, oid="loss")
    f.arrow(1470, 238, 1510, 238, OUTPUT, allowed=lg + loss, name="lg-loss")
    yb = node(f, 1490, 330, 200, 84, ["label y", "COUNT: 1 if apart or wrong", "IDENTITY: 1 if apart"], INK, WHITE, INK, 12, oid="yb")
    f.arrow(1590, 330, 1590, 286, INK, allowed=yb + loss, name="y-loss")
    # backward lane
    lane_y = 480
    f.route([(1560, 286), (1560, 300), (1475, 300), (1475, lane_y), (450, lane_y)], ORANGE, 2.0, loss, "lane")
    f.text(1385, lane_y - 12, r"$\partial L/\partial \ell = \sigma(\ell) - y$", 16, ORANGE, ha="center", oid="dl")
    for i, (x, ids) in enumerate(xs):
        if layers[i][2] == "0 w":
            continue
        f.arrow(x + 60, lane_y, x + 60, 276, ORANGE, 1.4, allowed=ids, name=f"g{i}")
    f.text(460, lane_y + 30, "Each weight gets ∂L/∂W by the chain rule: the lane's signal times that layer's own slope.", 13, ORANGE, oid="cr")
    f.text(260, lane_y - 12, "the lane stops here: no gradient", 12, FADED_TEXT, oid="stop1")
    f.text(260, lane_y + 8, "into the maps or the U-Net", 12, FADED_TEXT, oid="stop2")
    adam = node(f, 450, 560, 520, 64, ["Adam, learning rate 0.001: every trained weight moves", "a little against its ∂L/∂W"],
                ORANGE, ORANGE_FILL, ORANGE, 13, oid="adam")
    f.box(1020, 560, 560, 64, WHITE, INK, 1.0, 8, "bce")
    f.text(1300, 600, r"$L = -\,[\,y \log\sigma(\ell) + (1-y)\log(1-\sigma(\ell))\,]$", 18, INK, ha="center", oid="bcet")
    f.text(40, 680, "Per update: 256 canvases (128 label 1, 128 label 0), one of the 16 timesteps and a fresh noise draw per canvas. 4,000 updates.", 13, INK, oid="b1")
    f.text(40, 704, "The two branches run this same loop on the same network shape; only y and the canvas pool differ, so they end with different weights.", 13, INK, oid="b2")
    footer(f, "Source: v2_learn.py train(), lines 110-133. The weight counts are those of dc3; 'stem' is its three convs together.")
    f.write(HERE)


# ---------------------------------------------------------------------------------------------------- dc7
def dc7():
    f = Fig("dc7-the-push-gradient-into-the-canvas", 1700, 820)
    title(f, "The push: the same gradient, carried on into the noisy canvas",
          "With the trained weights held fixed, the logit's gradient runs back through the checker, the maps and the three U-Net calls to x_t.")
    img = f.image(ex("apart_1_3", 480), 40, 190, 200, 100, oid="xt")
    f.text(140, 178, "$x_t$, 1 × 32 × 64", 14, INPUT, ha="center", oid="xt-l")
    unet = node(f, 300, 200, 190, 80, ["U-Net, three calls", "a, b and nothing", "weights fixed"], GREY, FADED_FILL, GREY, 13, oid="unet")
    f.arrow(240, 240, 300, 240, GREY, allowed=unet + [img], name="x-u")
    mapsn = node(f, 550, 200, 190, 80, ["two maps", "|ε_a − ε_∅|, |ε_b − ε_∅|", "each scaled to 2,048"], HIDDEN, HIDDEN_FILL, HIDDEN, 13, oid="maps")
    f.arrow(490, 240, 550, 240, GREY, allowed=unet + mapsn, name="u-m")
    chk = node(f, 800, 200, 200, 80, ["checker", "count or identity", "weights fixed"], HIDDEN, WHITE, HIDDEN, 13, oid="chk")
    f.arrow(740, 240, 800, 240, HIDDEN, allowed=mapsn + chk, name="m-c")
    lg = node(f, 1060, 210, 120, 60, ["logit $\\ell$"], OUTPUT, OUTPUT_FILL, OUTPUT, 15, oid="lg")
    f.arrow(1000, 240, 1060, 240, HIDDEN, allowed=chk + lg, name="c-lg")
    lane_y = 400
    f.route([(1120, 270), (1120, lane_y), (155, lane_y), (155, 500)], ORANGE, 2.2, lg + ["grad"], "lane")
    for x, s1, s2 in ((1000, r"$\partial\ell/\partial\ell = 1$", "start"), (770, r"$\partial\ell/\partial m$", "2 × 32 × 64"),
                      (520, r"through $|\cdot|$ and the scaling", "into each ε"), (270, "through all three", "U-Net calls")):
        f.text(x, lane_y + 30, s1, 14, ORANGE, ha="center", oid=f"ln{x}a")
        f.text(x, lane_y + 52, s2, 12, ORANGE, ha="center", oid=f"ln{x}b")
    grad = node(f, 40, 500, 230, 64, [r"$g = \nabla_{x_t}\ell$", "one number per pixel of $x_t$"], ORANGE, ORANGE_FILL, ORANGE, 13, oid="grad")
    push = node(f, 340, 492, 620, 80, [r"$\hat\varepsilon = \varepsilon_{\rm product} - \Vert\varepsilon_{\rm product}\Vert \; b \; g/\Vert g\Vert$",
                                       "b = 0.4 for one checker: the push's length is fixed at 0.4 of the product's"],
                ORANGE, WHITE, ORANGE, 14, oid="push")
    f.arrow(270, 532, 340, 532, ORANGE, 2.0, allowed=grad + push, name="grad-push")
    f.text(1010, 516, "Subtracting along g raises ℓ, because the", 13, INK, oid="s1")
    f.text(1010, 538, "sampler moves x against ε̂. Only the direction", 13, INK, oid="s2")
    f.text(1010, 560, "of g is used; its size is replaced by the fixed length.", 13, INK, oid="s3")
    # comparison table
    ty = 620
    f.text(40, ty, "Two backward passes", 15, INK, "bold", oid="t0")
    rows = [("", "training (dc6)", "push (this figure)"),
            ("starts from", "the loss L, with a label y", "the logit ℓ, no label"),
            ("passes through", "the checker only", "the checker, the maps and three U-Net calls"),
            ("ends at", "every trained weight", "the noisy canvas $x_t$"),
            ("changes", "the weights (Adam step)", "the noise prediction, never a weight")]
    for r, (a, b, c) in enumerate(rows):
        y = ty + 30 + 24 * r
        wgt = "bold" if r == 0 else "normal"
        f.text(40, y, a, 13, GREY, wgt, oid=f"t{r}a")
        f.text(220, y, b, 13, INK, wgt, oid=f"t{r}b")
        f.text(620, y, c, 13, ORANGE if r else INK, wgt, oid=f"t{r}c")
    g = real("grad")[0]
    top = float(np.percentile(np.abs(g), 99))
    f.image(g, 1250, 190, 200, 100, cmap="RdBu_r", vmin=-top, vmax=top, frame=ORANGE, oid="g-real")
    f.text(1350, 178, "real g on this canvas, t = 480", 14, ORANGE, ha="center", oid="g-real-h")
    f.text(1350, 312, f"count checker; red raises ℓ, blue lowers it", 12, ORANGE, ha="center", oid="g-real-a")
    f.text(1350, 332, f"colour ±{top:.2f} per pixel,  ‖g‖ = {np.linalg.norm(g):.2f},  ℓ = {float(real('logit')):.2f}", 12, ORANGE,
           ha="center", oid="g-real-b")
    footer(f, "Code: make_push in scripts/abstract_checkers/v3_compose.py lines 50-66, used at sampler steps 15 to 25 of 50 (job 94134). "
              "Real g: numbers/real/count-explained.npz, job 94108.")
    f.write(HERE)


if __name__ == "__main__":
    which = sys.argv[1:] or ["dc0", "dc1", "dc2", "dc3", "dc4", "dc5", "dc6", "dc7"]
    for name in which:
        globals()[name]()
