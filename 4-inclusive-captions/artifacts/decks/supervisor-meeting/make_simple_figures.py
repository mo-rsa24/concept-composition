"""The supervisor set as slides: one point per figure, in the form of the toy results card's
the-training-set-with-and-without-relabelling.png (bold one-sentence title, one grey line saying what rows,
columns and cells are, a grid of real tiles with the ones that matter outlined, one short line of counts under
each cell, one orange "Look at:" sentence). Figures 2 to 4 are diagrams with the same title, grey line and
"Look at:" structure. The fuller versions are in detailed/, drawn by make_supervisor_figures.py, whose data
loaders, palette and Fig class this script imports.

Tile rules (stated on each figure's grey line):
  figure 1  8 real training pictures per cell, drawn so the number outlined is the pool's share x 8, rounded
  5, 6, 7   the first 8 samples of training 1 (toy seed 20261022; MNIST training 1, seeds 0 to 7)
  4         the first sample of each call in the stated model's samples
Every count is read from the result files at run time (toy-caption-replace-sweep.json, toy-caption-threshold.json,
counts.json, scores.csv).

    cd /home/molef/PhD/concept-composition
    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python -I \
        4-inclusive-captions/artifacts/decks/supervisor-meeting/make_simple_figures.py [0 1a 1b 2 3 4 5 5b 6 6b 6c 7]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_supervisor_figures as base  # noqa: E402
from make_supervisor_figures import (APART, ALONE, BOTH, CALLS, GREY, INK, LINE, ORANGE, PRODUCT, RIG, SEEDS,  # noqa: E402
                                     TWO, WITH2, ZA, ZR, Fig, b, mn3, mn_pct, mn_tiles, mnist_pool, mnist_step,
                                     pick, toy_pool, toy_pct, toy_rows, toy_share, u8, u8e)
import numpy as np  # noqa: E402

base.OUT = HERE
TXT, SMALL = 14, 13          # body and smallest text; nothing below 12 pt
OUTLINE = BOTH               # the one outline colour: a picture that shows both things, apart


def med(v):
    return int(np.median(v))


def pct(n, total):
    """A percentage computed from its own count and total: one decimal under 10%, whole above."""
    v = 100 * n / total
    return f"{v:.0f}%" if v == 0 or v >= 10 or v == int(v) else f"{v:.1f}%"


def count(n, total):
    return f"{n:,} of {total:,} ({pct(n, total)})"


def thousands(n):
    """An approximate count in thousands for MNIST's large numbers: 6,496 -> ≈6.5k, 64,956 -> ≈65k, 768,000 -> ≈768k."""
    if n == 0:
        return "0"
    v = n / 1000
    return f"\u2248{v:.1f}k".replace(".0k", "k") if v < 10 else f"\u2248{v:.0f}k"


def kcount(n, total, p=None):
    return f"{thousands(n)} of {thousands(total)} ({p or pct(n, total)})"


def slide(title, line, h=540):
    """A Fig with the model figure's head: bold title and one grey line, both centred."""
    f = Fig(title, line, h=h)
    for (id_, tx), y in zip(f.texts[:2], (f.h - 20, f.h - 48)):
        tx.set_x(f.w / 2); tx.set_y(y); tx.set_ha("center")
    return f


def look_at(f, s, y=16):
    f.text(f.w / 2, y, "Look at: " + s, size=16, weight="bold", color=ORANGE)


def tile(f, id_, kind, a, x, y, w, ok):
    ec, lw = (OUTLINE, 2.2) if ok else ("#b8c0c8", 0.6)
    if kind == "toy":
        f.rgb(id_, a, x, y, w, ec=ec, lw=lw)
    else:
        f.gray(id_, a, x, y, w, w / 2, ec=ec, lw=lw)


def grid(f, rows, cols, cells, top, x0, tw, per_row, col_pitch, row_pitch, label_x, count_lines=2, head_y=None, n=8,
         col_x=None, label_size=16, first_gap=6, line_gap=4):
    """cells[r][c] = (kind, tiles, oks, [count lines]); 8 tiles per cell in rows of per_row."""
    gap = 6
    th = tw / 2
    cw = per_row * tw + (per_row - 1) * gap
    nrows = (n + per_row - 1) // per_row
    xs = col_x or [x0 + c * col_pitch for c in range(len(cols))]
    ch = nrows * th + (nrows - 1) * gap
    for c, head in enumerate(cols):
        f.text(xs[c] + cw / 2, head_y if head_y else top + (26 if "\n" in head else 16), head,
               size=16 if "\n" in head else 17, weight="bold", linespacing=1.3)
    for r, lab in enumerate(rows):
        ytop = top - r * row_pitch
        name, col = lab if isinstance(lab, tuple) else (lab, INK)
        f.text(label_x, ytop - ch / 2, name, ha="right", size=label_size, weight="bold", color=col)
        for c in range(len(cols)):
            kind, tiles, oks, lines = cells[r][c]
            cx = xs[c]
            for j in range(len(tiles)):
                tile(f, f"g{r}{c}{j}", kind, tiles[j], cx + (j % per_row) * (tw + gap),
                     ytop - th - (j // per_row) * (th + gap), tw, oks[j])
            yy = ytop - ch - 4
            for k, (txt, kw) in enumerate(lines):
                sz = kw.get("size", TXT)
                yy -= sz * 0.62 + (line_gap if k else first_gap)
                f.text(cx + cw / 2, yy, txt, **kw)
                yy -= sz * 0.62
    return ch


# ------------------------------------------------------------------ 0: what relabelling means
def label_chip(f, id_, x, yc, w, text, new=False):
    f.box(id_, x, yc - 15, w, 30, "", ec=OUTLINE if new else GREY, fc="#f4f6f8", lw=2.0 if new else 1.2)
    f.text(x + w / 2, yc, text, size=15)


def fig0():
    f = slide("Relabelling: the same picture, also trained under each single label",
              "One real training picture per testbed, and the labels it is trained under.")
    hy = 448
    f.text(170, hy, "a training picture", size=17, weight="bold")
    f.text(475, hy, "without relabelling", size=17, weight="bold")
    f.text(760, hy, "with relabelling", size=17, weight="bold")
    # MNIST: the first canvas labelled "7 and 8" that ShareCanvasMaker (share 0.2, seed 0) draws
    x, lab, holds = base.cv.ShareCanvasMaker(0.2, device="cpu", seed=0).batch(2000)
    j78 = base.cv.joint_id(7, 8)
    canvas = u8(x[(lab == j78).nonzero()[0, 0].item(), 0].numpy())
    # toy: the first picture filed under "a red circle and a blue square" in the replacing design (k = 182, seed 20261022)
    tx, _ = toy_caption_pool(182, base.m.BOTH)
    rows = [("MNIST", "mn", canvas, 330, "\u201c7 and 8\u201d", ["\u201c7\u201d", "\u201c8\u201d"], 110, 80,
             "a canvas like this is also\ntrained under each digit"),
            ("toy", "toy", tx[0], 140, "\u201ca red circle and a blue square\u201d",
             ["\u201ca red circle\u201d", "\u201ca blue square\u201d"], 250, 150,
             "the same picture is also\ntrained under each caption")]
    for name, kind, pic, yc, joint, singles, wj, ws, note in rows:
        f.text(52, yc, name, ha="right", size=18, weight="bold")
        if kind == "mn":
            f.gray(f"pic{name}", pic, 64, yc - 53, 212, 106, ec=OUTLINE, lw=2.2)
        else:
            f.rgb(f"pic{name}", pic, 64, yc - 53, 212, ec=OUTLINE, lw=2.2)
        x1 = 475 - wj / 2
        label_chip(f, f"wo{name}", x1, yc, wj, joint)
        f.arrow([(276, yc), (x1, yc)], f"pic{name}", f"wo{name}")
        x2 = 620
        label_chip(f, f"w0{name}", x2, yc + 44, wj, joint)
        for i, sgl in enumerate(singles):
            label_chip(f, f"w{i + 1}{name}", x2, yc - 44 * i, ws, sgl, new=True)
        f.arrow([(170, yc + 53), (170, yc + 76), (x2 + 20, yc + 76), (x2 + 20, yc + 59)], f"pic{name}", f"w0{name}")
        f.text(786, yc - 22, note, ha="left", size=TXT, color=OUTLINE, linespacing=1.3)
    toy_k, seven = 182, base.MIX["7"]
    f.text(480, 56, f"The share is how many of a single label's pictures are relabelled ones: for example "
                    f"{count(toy_k, 2000)} in the toy, or {kcount(round(0.2 * seven), seven)} \u201c7\u201d draws in MNIST.",
           size=TXT, color=GREY)
    look_at(f, "the picture does not change; only the labels it is trained under.")
    f.save("00-what-relabelling-means")


# ------------------------------------------------------------------ 1a and 1b: the training sets
def toy_caption_pool(k, cap, seed=SEEDS[0]):
    """The pictures filed under caption id cap in the replacing design at k, and whether each holds both shapes."""
    x, c = base.rep.corpus_replace(k, np.random.default_rng(seed))
    x = x[c == cap].permute(0, 2, 3, 1).reshape(-1, base.t.D).cpu().numpy()
    both = np.ones(len(x), bool) if cap == base.m.BOTH else np.r_[np.zeros(len(x) - k, bool), np.ones(k, bool)]
    return x, both


def mnist_label_pool(share, label, n=6000, seed=0):
    x, lab, holds = base.cv.ShareCanvasMaker(share, device="cpu", seed=seed).batch(n)
    sel = (lab == label)
    return x[sel][:, 0].numpy(), (holds[sel][:, 1] >= 0).numpy()


def cell_from_pool(kind, x, both, share, seed):
    idx = pick(both, 8, share, seed=seed) if 0 < share < 1 else np.arange(8)
    tiles = [x[i] if kind == "toy" else u8(x[i]) for i in idx]
    return tiles, [bool(both[i]) for i in idx]


def fig1a():
    ks = [0, 182, 1000]
    f = slide("Toy: relabelling puts both-shapes pictures under each single caption",
              "Rows: the caption trained under. Columns: the share. Each cell: 8 real training pictures.")
    caps = [(base.m.CIRCLE, "\u201ca red circle\u201d"), (base.m.SQUARE, "\u201ca blue square\u201d"),
            (base.m.BOTH, "\u201ca red circle and\na blue square\u201d")]
    cells = []
    for r, (cap, _) in enumerate(caps):
        row = []
        for c, k in enumerate(ks):
            x, both = toy_caption_pool(k, cap)
            sh = both.mean()
            tiles, oks = cell_from_pool("toy", x, both, sh, seed=10 * r + c)
            row.append(("toy", tiles, oks, [(count(int(both.sum()), len(x)) + " show both", dict(size=TXT))]))
        cells.append(row)
    grid(f, [lab for _, lab in caps], [f"{k:,} relabelled of 2,000\n({pct(k, 2000)})" for k in ks], cells,
         top=432, x0=176, tw=56, per_row=4, col_pitch=262, row_pitch=150, label_x=156)
    look_at(f, "the single-caption rows go from 0 to 1,000 of 2,000 both-shapes pictures (0% to 50%); every pool stays 2,000.")
    f.save("01a-the-toy-training-set")


def fig1b():
    ss = [0.0, 0.2, 0.8]
    f = slide("MNIST: the share puts 7-and-8 canvases under \u201c7\u201d and \u201c8\u201d",
              "Rows: the label trained under. Columns: the share. Each cell: 8 real training canvases.")
    j78 = base.cv.joint_id(7, 8)
    labs = [(7, "\u201c7\u201d", base.MIX["7"]), (8, "\u201c8\u201d", base.MIX["8"]), (j78, "\u201c7 and 8\u201d", base.MIX["7 and 8"])]
    cells = []
    for r, (lab, _, n) in enumerate(labs):
        row = []
        for c, s_ in enumerate(ss):
            x, both = mnist_label_pool(s_, lab)
            sh = 1.0 if lab == j78 else s_
            tiles, oks = cell_from_pool("mn", x, both, sh, seed=10 * r + c)
            nb = n if lab == j78 else round(n * s_)
            what = "show both" if lab == j78 else ("show an 8" if lab == 7 else "show a 7")
            row.append(("mn", tiles, oks, [(f"{count(nb, n)} {what}", dict(size=TXT))]))
        cells.append(row)
    grid(f, [name for _, name, _ in labs], [f"{thousands(round(s_ * base.MIX['7']))} of {thousands(base.MIX['7'])} \u201c7\u201d draws\nshow an 8 ({pct(round(s_ * base.MIX['7']), base.MIX['7'])})" for s_ in ss], cells,
         top=432, x0=176, tw=56, per_row=4, col_pitch=262, row_pitch=150, label_x=156)
    look_at(f, "\u201c7\u201d and \u201c8\u201d gain 7-and-8 canvases; \u201c7 and 8\u201d is trained the same at every share.")
    f.save("01b-the-mnist-training-set")


# ------------------------------------------------------------------ 2 and 3: diagrams
def chip(f, id_, x, y, w, h, title, sub):
    f.box(id_, x, y, w, h, "", ec=GREY, lw=1.4)
    f.text(x + w / 2, y + h * 0.64, title, size=TXT, weight="bold")
    f.text(x + w / 2, y + h * 0.28, sub, size=SMALL, color=GREY)


def fig2():
    f = slide("The toy: a small latent diffusion model, captioned by id",
              "One real training step of the 50% model.")
    yc = 290
    f.rgb("pic", RIG["r1000/train/picture"], 16, yc - 40, 160, ec=GREY, lw=1.6)
    f.text(96, yc + 62, "training picture", size=TXT, weight="bold")
    f.box("enc", 206, yc - 36, 88, 72, "VAE\nencoder", size=TXT)
    f.arrow([(176, yc), (206, yc)], "pic", "enc")
    f.box("den", 420, yc - 75, 170, 150, "", lw=2.2)
    f.text(505, yc + 14, "denoiser", size=18, weight="bold")
    f.text(505, yc - 18, "fully connected", size=SMALL, color=GREY)
    f.arrow([(294, yc), (420, yc)], "enc", "den")
    f.text(357, yc + 24, "noisy latent", size=TXT, weight="bold")
    f.text(357, yc - 22, "128 numbers", size=SMALL, color=GREY)
    chip(f, "step", 400, yc + 115, 96, 60, "step", "1 to 200")
    chip(f, "cap", 514, yc + 115, 96, 60, "caption id", "0, 1, 2 or 3")
    f.arrow([(448, yc + 115), (448, yc + 75)], "step", "den")
    f.arrow([(562, yc + 115), (562, yc + 75)], "cap", "den")
    f.box("dec", 716, yc - 36, 88, 72, "VAE\ndecoder", size=TXT)
    f.arrow([(590, yc), (716, yc)], "den", "dec")
    f.text(653, yc + 24, "its guess of the", size=TXT, weight="bold")
    f.text(653, yc + 6, "clean latent", size=TXT, weight="bold")
    f.text(653, yc - 22, "128 numbers", size=SMALL, color=GREY)
    f.rgb("out", RIG["r1000/train/guess decoded"], 834, yc - 40, 110, ec=GREY, lw=1.6)
    f.arrow([(804, yc), (834, yc)], "dec", "out")
    f.text(889, yc + 50, "picture", size=TXT, weight="bold")
    f.text(480, 132, "Four captions: 0 “a red circle”, 1 “a blue square”, "
                     "2 “a red circle and a blue square”, 3 empty.", size=16)
    f.text(480, 100, "Each picks one row of a learned table. 6,000 training pictures per model.", size=TXT, color=GREY)
    look_at(f, "the caption goes in as an id beside the step; the denoiser never sees a word.", y=40)
    f.save("02-the-toy-setup")


def fig3():
    one, pair, xk, e, z, K, ab, mse, width = mnist_step()
    f = slide("MNIST: a U-Net estimates the noise from canvas, step and label",
              "One real call of the 0% model on a 32 × 64 canvas; no VAE.")
    yc = 290
    f.gray("x0", u8(one), 30, yc - 40, 160, 80, ec=GREY)
    f.text(110, yc + 60, "training canvas", size=TXT, weight="bold")
    f.gray("xk", u8(xk), 250, yc - 40, 160, 80, ec=GREY)
    f.arrow([(190, yc), (250, yc)], "x0", "xk")
    f.text(220, yc + 14, "+ noise", size=SMALL, color=GREY)
    f.text(330, yc + 60, "noisy canvas", size=TXT, weight="bold")
    f.box("unet", 470, yc - 75, 190, 150, "", lw=2.2)
    f.text(565, yc + 14, "U-Net", size=18, weight="bold")
    f.text(565, yc - 18, "3 levels, 48 to 192 channels", size=SMALL, color=GREY)
    f.arrow([(410, yc), (470, yc)], "xk", "unet")
    chip(f, "step", 460, yc + 115, 96, 60, "step", "1 to 200")
    chip(f, "lab", 574, yc + 115, 96, 60, "label", "e.g. “7”")
    f.arrow([(508, yc + 115), (508, yc + 75)], "step", "unet")
    f.arrow([(622, yc + 115), (622, yc + 75)], "lab", "unet")
    f.gray("eps", u8e(e), 750, yc - 40, 160, 80, ec=GREY)
    f.arrow([(660, yc), (750, yc)], "unet", "eps")
    f.text(830, yc + 60, "noise estimate", size=TXT, weight="bold")
    f.text(480, 132, "56 labels: the 10 digits, 45 joint labels “a and b”, and blank.", size=16)
    f.text(480, 100, f"Each model trains on {thousands(base.CANVASES)} canvases, built fresh from MNIST's training digits.",
           size=TXT, color=GREY)
    look_at(f, "the label goes in beside the step, and what comes out is the noise, not a cleaner canvas.", y=40)
    f.save("03-the-mnist-setup")


# ------------------------------------------------------------------ outcome words, the same on every slide
# Plain words for the scorer's and the judge's calls. Each cell's split uses training 1, the training its tiles
# come from, and must add up to its total.
TOY_SPLIT = {"clean pair": ["red and blue apart"], "fused": ["purple or overlaid"],
             "one thing only": ["red only", "blue only"], "other": ["nothing readable"]}
MN_SPLIT = {"clean pair": [TWO], "fused": ["fused"], "one thing only": ["one digit"],
            "other": ["two digits with a wrong identity", "damaged"]}
MN_ALONE = {"a 7 with a second digit": [WITH2], "the 7 alone": ["right digit alone"], "other": ["wrong digit", "damaged"]}
TOY_ALONE = {"clean pair": ["red and blue apart"], "the circle alone": ["red only"],
             "other": ["blue only", "purple or overlaid", "nothing readable"]}
WORDS = ["clean pair", "fused", "one thing only", "other"]


def split(counts, groups, total):
    out = {name: sum(counts.get(c, 0) for c in cs) for name, cs in groups.items()}
    assert sum(out.values()) == total, (out, total)
    return out


def toy_counts1(k, who):
    rows = THRESH0 if k == "0" else base.SWEEP[k]
    return [r for r in rows if r["seed"] == SEEDS[0]][0]["counts"][who]


def mn_counts1(s, row):
    return base.MC[f"{s}|1|{row}"]


THRESH0 = base.THRESH0


def lines_product(sp, total, big=False):
    rest = " \u00b7 ".join(f"{sp[w]} {w}" for w in WORDS[1:])
    return [(f"{count(sp['clean pair'], total)} show a clean pair", dict(size=20 if big else 12, weight="bold", color=ORANGE)),
            (rest, dict(size=13 if big else 12, color=GREY))]


def lines_alone(sp, total, first, others, big=False):
    rest = " \u00b7 ".join(f"{sp[w]} {w}" for w in others)
    phrase = ("show a clean pair" if first == "clean pair" else
              ("show " + first if big else first.replace("a 7 with ", "with ")))
    return [(f"{count(sp[first], total)} {phrase}", dict(size=20 if big else 12, weight="bold", color=ORANGE)),
            (rest, dict(size=13 if big else 12, color=GREY))]


# ------------------------------------------------------------------ 4
def first_with_call(calls, want):
    return next(i for i, c in enumerate(calls) if c in want)


def fig4():
    f = slide("The product makes one picture, counted as a pair or not",
              "The rule, one toy sample from noise to finished, and the four outcomes every count uses.")
    f.box("eq", 170, 414, 620, 50, "", ec=PRODUCT, lw=2.0)
    f.text(480, 439, r"$\varepsilon \;=\; \varepsilon_{\varnothing} \;+\; g\,\left[(\varepsilon_A-\varepsilon_{\varnothing}) "
                     r"+ (\varepsilon_B-\varepsilon_{\varnothing})\right],\quad g=3$", size=21, color=PRODUCT)
    f.text(480, 398, "A and B: the two single captions; \u2205: the empty caption. 200 reverse steps.", size=TXT, color=GREY)
    levels = base.TOYNUM["replacing design (main)"]["reference model, replacing 1,000 (seed 20261022)"]["levels recorded"]
    trj = RIG["r1000/traj/product"]
    q = b.read(trj[-1])[0].index(APART)
    show = [200, 100, 50, 25, 0]
    for j, L in enumerate(show):
        cx = 150 + j * 140
        f.rgb(f"s{j}", trj[levels.index(L)][q], cx, 300, 110, ec=OUTLINE if L == 0 else LINE, lw=2.2 if L == 0 else 0.6)
        f.text(cx + 55, 288, "noise" if L == 200 else ("finished" if L == 0 else f"step {L}"), size=TXT, color=GREY)
        if j:
            f.arrow([(cx - 26, 327), (cx - 4, 327)], f"s{j - 1}", f"s{j}")
    # one real example of each outcome: the first product sample with that call, training 1, searching the shares
    # from the largest down (toy: the sweep and the 0% file; MNIST: scores.csv)
    toy_src = [(k, (ZA if k == "0" else ZR)[f"{k}_{SEEDS[0]}_product"]) for k in ("1000", "750", "462", "182", "49", "20", "5", "1", "0")]
    toy_ex = {}
    for w in WORDS:
        for k, tl in toy_src:
            cl = b.read(tl)[0]
            hit = [i for i, c in enumerate(cl) if c in TOY_SPLIT[w]]
            if hit:
                toy_ex[w] = tl[hit[0]]; break
    mn_ex = {}
    for w in WORDS:
        for s_ in ("0.4", "0.2", "0.1", "0", "0.8"):
            cl = [CALLS[(s_, 1, "7 x 8 plain", q_)] for q_ in range(256)]
            hit = [i for i, c in enumerate(cl) if c in MN_SPLIT[w]]
            if hit:
                mn_ex[w] = mn_tiles(s_, 1)["7 x 8 plain"][hit[0]]; break
    defs = {"clean pair": "both things, apart", "fused": "both merged into one shape",
            "one thing only": "only one of the two", "other": "unreadable or wrong"}
    f.text(150, 186, "toy", ha="right", size=16, weight="bold")
    f.text(150, 108, "MNIST", ha="right", size=16, weight="bold")
    for i, w in enumerate(WORDS):
        x = 180 + i * 190
        f.text(x + 60, 252, w, size=17, weight="bold", color=ORANGE if w == "clean pair" else INK)
        f.text(x + 60, 232, defs[w], size=SMALL, color=GREY)
        ok = w == "clean pair"
        if w in toy_ex:
            f.rgb(f"te{i}", toy_ex[w], x, 156, 120, ec=OUTLINE if ok else LINE, lw=2.2 if ok else 0.6)
        if w in mn_ex:
            f.gray(f"me{i}", mn_ex[w], x, 78, 120, 60, ec=OUTLINE if ok else LINE, lw=2.2 if ok else 0.6)
    look_at(f, "only a clean pair counts: both things there and apart, out of 300 toy or 256 MNIST samples per model.", y=24)
    f.save("04-making-and-counting-one-picture")
    return {w: (w in toy_ex, w in mn_ex) for w in WORDS}


# ------------------------------------------------------------------ 5, 6, 7: grids of samples
def toy_cell(k, who, big=False):
    z = ZA if k == "0" else ZR
    tiles = z[f"{k}_{SEEDS[0]}_{who}"][:8]
    oks = [c == APART for c in b.read(tiles)[0]]
    c1 = toy_counts1(k, who)
    if who == "product":
        lines = lines_product(split(c1, TOY_SPLIT, 300), 300, big)
    else:
        lines = lines_alone(split(c1, TOY_ALONE, 300), 300, "clean pair", ["the circle alone", "other"], big)
    return ("toy", tiles, oks, lines)


def mn_cell(s, row, big=False):
    z = mn_tiles(s, 1)[row][:8]
    call = TWO if row == "7 x 8 plain" else WITH2
    oks = [CALLS[(s, 1, row, q)] == call for q in range(8)]
    c1 = mn_counts1(s, row)
    if row == "7 x 8 plain":
        lines = lines_product(split(c1, MN_SPLIT, 256), 256, big)
    else:
        lines = lines_alone(split(c1, MN_ALONE, 256), 256, "a 7 with a second digit", ["the 7 alone", "other"], big)
    return ("mn", z, oks, lines)


def fig5():
    ks = ["0", "182", "462", "1000"]
    f = slide("Toy: the product's pairs far outnumber the caption's own",
              "Cells: first 8 samples of training 1. Counts: all 300 samples of the same training.")
    cells = [[toy_cell(k, "product") for k in ks], [toy_cell(k, "circle") for k in ks]]
    grid(f, [("the product", PRODUCT), ("\u201ca red circle\u201d\nalone", ALONE)],
         [f"{int(k):,} relabelled of 2,000\n({pct(int(k), 2000)})" for k in ks], cells,
         top=444, x0=170, tw=70, per_row=2, col_pitch=202, row_pitch=212, label_x=154)
    look_at(f, "the product row fills with outlined pairs while the single caption's row stays mostly single shapes.", y=12)
    f.save("05-toy-result")


def fig6():
    ss = ["0", "0.1", "0.2", "0.4"]
    f = slide("MNIST: \u201c7\u201d alone gains pairs as fast as the product",
              "Cells: first 8 samples of training 1. Counts: all 256 seeds of the same training.")
    cells = [[mn_cell(s, "7 x 8 plain") for s in ss], [mn_cell(s, "7 alone") for s in ss]]
    grid(f, [("the product,\n7 \u00d7 8", PRODUCT), ("\u201c7\u201d alone", ALONE)],
         [f"{thousands(round(float(s) * base.MIX['7']))} of {thousands(base.MIX['7'])}\n\u201c7\u201d draws show an 8 ({mn_pct(s)})" for s in ss], cells,
         top=444, x0=170, tw=70, per_row=2, col_pitch=202, row_pitch=212, label_x=154)
    look_at(f, "the \u201c7\u201d alone row gains outlined pairs as fast as the product row.", y=12)
    f.save("06-mnist-result")


# ------------------------------------------------------------------ 5b, 6b, 6c
TOYD = base.TOYD


def toy_corrector_rows():
    rows, npz = [], {}
    for fn in ("toy-caption-corrector", "toy-caption-corrector-no-single-captions"):
        rows += base.json.load(open(TOYD / f"{fn}.json"))["results"]
        npz.update(dict(np.load(TOYD / f"{fn}-samples.npz")))
    return rows, npz


def fig5b_adding():
    """The adding-design corrector test, kept in detailed/."""
    rows, npz = toy_corrector_rows()
    pick_rows = []
    for k in (0, 50, 200, 2000):
        mine = sorted([r for r in rows if r["pictures"] == k], key=lambda r: r["seed"])
        r1 = [r for r in mine if r["seed"] == SEEDS[0]]
        pick_rows.append((r1 or mine)[0])
    f = slide("Toy: the corrector completes the pairs only where captions shared pictures",
              "Adding design, retrained for this test; one model per row, sampled three ways from the same noise.")
    labs, cells = [], []
    for r in pick_rows:
        k = r["pictures"]
        run = SEEDS.index(r["seed"]) + 1
        labs.append(f"{k:,} added ({pct(k, 2000 + k)})" + ("" if run == 1 else f"\nrun {run} (no run 1)"))
        row = []
        for mode in ("plain", "main", "stronger"):
            tiles = npz[f"{k}_{r['seed']}_{mode}"][:4]
            oks = [c == APART for c in b.read(tiles)[0]]
            row.append(("toy", tiles, oks, lines_product(split(r[mode]["product"], TOY_SPLIT, 300), 300)))
        cells.append(row)
    grid(f, labs, ["the product", "+ 5 corrector steps", "+ 20 corrector steps"], cells,
         top=436, x0=196, tw=58, per_row=4, col_pitch=256, row_pitch=92, label_x=180, n=4, label_size=14)
    f.text(480, 50, "The replacing design (slides 01a and 05) ran the corrector only on its 50% model: "
                    "218 of 300 (73%) plain, 300 of 300 (100%) with 20 steps.", size=TXT, color=GREY)
    look_at(f, "every model with both-shapes pictures under its single captions reaches 300 of 300 with 20 steps; "
               "the models without stay at 0.", y=14)
    f.save("detailed/05b-the-toy-corrector-adding-design")


REPLACE_KS = [0, 100, 200, 300, 500, 700, 1000]
CORR_SETTINGS = ["plain", "5 corrector steps", "20 corrector steps"]


def replace_corrector(k):
    """The replacing-design corrector run at k: its three trainings (seed order) and its sample file."""
    d = base.json.load(open(base.TOYRES / f"toy-replace-corrector-k{k}.json"))
    return sorted(d["results"][str(k)], key=lambda r: r["seed"]), np.load(base.TOYRES / f"toy-replace-corrector-k{k}-samples.npz"), d["bar"]


def fig5b():
    rows_all = {k: replace_corrector(k) for k in REPLACE_KS}
    reach = [(k, r["20 corrector steps"]["red and blue apart"]) for k in REPLACE_KS if k >= 200 for r in rows_all[k][0]]
    n_reach = sum(v >= 290 for _, v in reach)
    f = slide(f"Toy: the corrector completes the pair in {n_reach} of {len(reach)} trainings from 10%",
              "300 pictures drawn from each model; each caption keeps 2,000 training pictures. Shown: training 1 of 3 per row.")
    labs, cells, threes = [], [], []
    for k in REPLACE_KS:
        rows, z, _ = rows_all[k]
        r1 = rows[0]
        assert r1["seed"] == SEEDS[0]
        labs.append(count(k, 2000))
        row = []
        for st in CORR_SETTINGS:
            tiles = z[f"{k}_{r1['seed']}_{st}"][:4]
            oks = [c == APART for c in b.read(tiles)[0]]
            row.append(("toy", tiles, oks, lines_product(split(r1[st], TOY_SPLIT, 300), 300)))
        cells.append(row)
    xs = [254, 476, 698]
    grid(f, labs, ["the product", "+ 5 corrector steps", "+ 20 corrector steps"], cells,
         top=448, x0=254, tw=46, per_row=4, col_pitch=222, row_pitch=62, label_x=240, n=3, col_x=xs,
         label_size=13, head_y=470, first_gap=3, line_gap=2)
    f.text(240, 470, "\u201ca red circle\u201d pictures\nthat also show a square", ha="right", size=13, weight="bold",
           linespacing=1.2)
    look_at(f, "from 10%, 9 of 15 trainings reach at least 290 of 300 with 20 steps; at 5% none do; at 0% it stays at 0.", y=8)
    f.save("05b-the-toy-corrector")
    return {k: {st: [r[st]["red and blue apart"] for r in rows_all[k][0]] for st in CORR_SETTINGS} for k in REPLACE_KS}


CALLS_G1_FILE = HERE / "guidance-1-training-1-calls.json"
SHARES5 = ["0", "0.1", "0.2", "0.4", "0.8"]
SHARES9 = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
H9 = 800  # nine rows of tiles and two count lines need a taller canvas than 540


def g1_calls():
    """The pinned judge's calls on training 1's guidance-1 samples; scored once and cached beside the slides."""
    import json
    need = [f"{s}|{r}" for s in SHARES9 for r in ("7 x 8 plain", "7 x 8 corrected", "4 x 7 corrected", "4 alone", "7 alone")]
    have = json.load(open(CALLS_G1_FILE)) if CALLS_G1_FILE.exists() else {}
    if not all(k in have for k in need):
        import judge_checks as jc
        j, net, _ = jc.pinned_judge()
        for key in need:
            if key in have:
                continue
            s, row = key.split("|")
            z = np.load(base.REPO / f"data/mnist/share-samples-g1/share-{s}-seed-1.npz")[row]
            if row.endswith("alone"):
                have[key] = [jc.single_call(j, net, t, int(row[0])) for t in z]
            else:
                have[key] = [jc.pair_call(j, net, t, (int(row[0]), int(row[4]))) for t in z]
        json.dump(have, open(CALLS_G1_FILE, "w"))
    return have


def g1_tiles(s, row):
    return np.load(base.REPO / f"data/mnist/share-samples-g1/share-{s}-seed-1.npz")[row]


def share_label(s):
    return kcount(round(float(s) * base.MIX["7"]), base.MIX["7"], mn_pct(s))


def mn_pair_cell(tiles, calls, n=4):
    c = {k: calls.count(k) for k in set(calls)}
    oks = [x == TWO for x in calls[:n]]
    return ("mn", tiles[:n], oks, lines_product(split(c, MN_SPLIT, 256), 256))


def fig6b():
    G = g1_calls()
    f = slide("MNIST: the corrector cuts pairs at guidance 3, not at 1",
              f"Rows: \u201c7\u201d draws showing an 8, of {thousands(base.MIX['7'])} per model. Training 1, seeds 0 to 3; counts of 256.", h=H9)
    cells = []
    for s in SHARES9:
        z3 = mn_tiles(s, 1)
        row = [mn_pair_cell(z3["7 x 8 plain"], [CALLS[(s, 1, "7 x 8 plain", q)] for q in range(256)]),
               mn_pair_cell(z3["7 x 8 corrected"], [CALLS[(s, 1, "7 x 8 corrected", q)] for q in range(256)]),
               mn_pair_cell(g1_tiles(s, "7 x 8 plain"), G[f"{s}|7 x 8 plain"]),
               mn_pair_cell(g1_tiles(s, "7 x 8 corrected"), G[f"{s}|7 x 8 corrected"])]
        cells.append(row)
    xs = [180, 372, 576, 768]
    cw = 4 * 42 + 3 * 6
    grid(f, [share_label(s) for s in SHARES9], ["the product", "+ 20 corrector steps", "the product", "+ 20 corrector steps"],
         cells, top=H9 - 120, x0=180, tw=42, per_row=4, col_pitch=192, row_pitch=72, label_x=166, n=4, col_x=xs,
         label_size=13, head_y=H9 - 108)
    for x0, x1, name in ((xs[0], xs[1] + cw, "guidance 3: what the earlier slides use"),
                         (xs[2], xs[3] + cw, "guidance 1: the exact product of the two labels")):
        f.text((x0 + x1) / 2, H9 - 78, name, size=16, weight="bold")
        f.ax.plot([x0, x1], [H9 - 90, H9 - 90], color=GREY, lw=1.0)
    look_at(f, "guidance 3: the corrector lowers clean pairs from 20% up; guidance 1: it raises them at 10 to 70% (80%: 185 against 189).", y=14)
    f.save("06b-the-mnist-product-across-shares")


def fig6c():
    G = g1_calls()
    f = slide("MNIST: relabelling helps only the pair seen together in training",
              "Guidance 1, the product + 20 corrector steps, training 1. 4 and 7 never share a training canvas.", h=H9)
    cells = []
    for s in SHARES9:
        cells.append([mn_pair_cell(g1_tiles(s, "7 x 8 corrected"), G[f"{s}|7 x 8 corrected"]),
                      mn_pair_cell(g1_tiles(s, "4 x 7 corrected"), G[f"{s}|4 x 7 corrected"])])
    xs = [300, 600]
    grid(f, [share_label(s) for s in SHARES9],
         ["7 \u00d7 8: seen together", "4 \u00d7 7: never together"],
         cells, top=H9 - 120, x0=300, tw=54, per_row=4, col_pitch=258, row_pitch=72, label_x=282, n=4, col_x=xs,
         label_size=13, head_y=H9 - 92)
    look_at(f, "only the pair seen together gains clean pairs; 4 \u00d7 7 stays near 0 at every share.", y=14)
    f.save("06c-seen-together-against-never-together")
    return {s: (G[f"{s}|7 x 8 corrected"].count(TWO), G[f"{s}|4 x 7 corrected"].count(TWO), G[f"{s}|4 alone"].count(WITH2)) for s in SHARES9}


def fig7():
    f = slide("The toy product composes beyond its captions; MNIST's does not",
              "Each testbed at the product's strongest share. Cells: first 8 samples of training 1; counts: that training.")
    cells = [[toy_cell("1000", "product", big=True), mn_cell("0.4", "7 x 8 plain", big=True)],
             [toy_cell("1000", "circle", big=True), mn_cell("0.4", "7 alone", big=True)]]
    tw, gap = 84, 6
    cw = 4 * tw + 3 * gap
    grid(f, [("the product", PRODUCT), ("alone:\n\u201ca red circle\u201d\nor \u201c7\u201d", ALONE)],
         [f"toy: 1,000 relabelled\nof 2,000 ({pct(1000, 2000)})",
          f"MNIST: {kcount(round(0.4 * base.MIX['7']), base.MIX['7'])} \u201c7\u201d draws\nshow an 8"], cells,
         top=436, x0=180, tw=tw, per_row=4, col_pitch=cw + 46, row_pitch=184, label_x=162)
    f.text(480, 64, "Open: guidance 1 tests whether guidance 3 made \u201c7\u201d alone draw these pairs; training 1 is drawn "
                    "(slides 06b, 06c); trainings 2 and 3 are being finished.", size=TXT, color=GREY)
    look_at(f, "in the toy the product row has far more outlined pairs than the single row; in MNIST it has fewer.")
    f.save("07-what-it-means")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["0", "1a", "1b", "2", "3", "4", "5", "5b", "6", "6b", "6c", "7"]:
        globals()[f"fig{n}"]()
