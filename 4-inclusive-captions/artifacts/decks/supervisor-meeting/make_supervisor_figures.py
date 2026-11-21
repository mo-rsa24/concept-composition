"""Seven figures for a supervisor meeting on the inclusive-captions experiments (toy and MNIST).

01  the question and the two testbeds: real training pictures under one single caption at 0% and a middle share
02  the toy model: picture -> VAE -> 128 numbers; noisy latent + step + caption id -> denoiser -> guess -> picture
03  the MNIST model: canvas; noisy canvas + step + label -> U-Net -> noise estimate
04  making and counting one picture: the product, 200 reverse steps, the corrector, the scorer and the judge
05  toy result: separate pairs against share, product beside each single caption alone
06  MNIST result: two separate right digits against share, product beside "7" alone
07  what it means: one number pair per testbed, and what is still open

Every count is read from the result files at run time:
  toy    4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/
         toy-caption-replace-sweep.json (shares 0.05% to 50%), toy-caption-threshold.json (k = 0, the 0% row)
  MNIST  4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/
         counts.json and scores.csv
Every picture is a real tile from the experiments' sample files, a real training picture built by the experiments'
own data code (toy_caption_replace.corpus_replace, canvases.ShareCanvasMaker), or a real tensor computed here on CPU
from a saved model (toy: rig_replace1000.npz; MNIST: data/mnist/checkpoints/dial-0-seed-1.pt, the 0% share model).
Layout helpers follow make_mnist_stage_figures.py (Fig and its crossing check) and the palette of the ladder's
colour key; each diagram figure refuses to save if an arrow crosses a box, picture or label it does not connect.

    cd /home/molef/PhD/concept-composition
    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python -I \
        4-inclusive-captions/artifacts/decks/supervisor-meeting/make_supervisor_figures.py [1 2 3 4 5 6 7]
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NOTES = REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"
MNISTCODE = REPO / "poe_repair/experiments/inclusive_captions/mnist"
for p in (NOTES, MNISTCODE, REPO / "4-inclusive-captions/scripts"):
    sys.path.insert(0, str(p))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import toy_sampler_hybrids as t          # noqa: E402  the toy's pictures
import toy_blob_scorer as b              # noqa: E402  the toy's colour scorer
import toy_caption_dial_ldm as m         # noqa: E402  the toy's model code
import toy_caption_replace as rep        # noqa: E402  the replacing design's training set
import canvases as cv                    # noqa: E402  MNIST canvases
import model as md                       # noqa: E402  MNIST U-Net
import share_bar as bar                  # noqa: E402  MNIST bar, written before the run

# ------------------------------------------------------------------ palette: one colour per meaning
BLUE, GREEN, RED, ORANGE, GREY = "#1f5fd0", "#1d7a3e", "#c0262d", "#c86a0a", "#6b7a88"
INK, FADE, LINE = "#1f2a33", "#eceff2", "#c3cbd3"
PRODUCT, CORRECTED, ALONE, TRAINED, BOTH = RED, ORANGE, BLUE, GREY, GREEN
NET = INK                                   # network boxes carry no meaning colour
TITLE, SUB, LAB, ANN, FOOT = 23, 14, 13, 12, 11
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm", "font.size": LAB})

# ------------------------------------------------------------------ sources
TOYRES = REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
TOYD = REPO / "3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs"
MNRES = REPO / "4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist"
TOYNUM = json.load(open(REPO / "4-inclusive-captions/artifacts/scenes/how-the-toy-relabelling-experiment-works/toy_relabelling_numbers.json"))
MNNUM = json.load(open(REPO / "4-inclusive-captions/artifacts/scenes/how-the-inclusive-labels-mnist-stage-works/mnist_stage_numbers.json"))
SWEEP = json.load(open(TOYRES / "toy-caption-replace-sweep.json"))["results"]
THRESH0 = json.load(open(TOYRES / "toy-caption-threshold.json"))["results"]["0"]
RIG = np.load(REPO / "4-inclusive-captions/data/toy-relabelling-ladder/rig_replace1000.npz")
ZR = np.load(TOYRES / "toy-caption-replace-sweep-samples.npz")
ZA = np.load(TOYD / "toy-caption-threshold-samples.npz")
COUNTS = json.load(open(MNRES / "counts.json"))
MC, MV = COUNTS["counts"], COUNTS["verdict"]
R5 = np.load(REPO / "4-inclusive-captions/artifacts/scenes/how-the-inclusive-labels-mnist-stage-works/r5_trajectories.npz")
CALLS = {}
for r in csv.DictReader(open(MNRES / "scores.csv")):
    CALLS[(f"{float(r['share']):g}", int(r["training_seed"]), r["row"], int(r["seed"]))] = r["call"]

SEEDS = [20261022, 20261023, 20261024]
TOY_KS = ["0", "1", "5", "20", "49", "182", "462", "750", "1000"]
MN_SHARES = ["0", "0.1", "0.2", "0.4", "0.8"]
APART, TWO, WITH2 = "red and blue apart", "two separate right digits", "right digit with a second"
OUT = HERE / "detailed"


def toy_rows(k):
    """Per training (seed order): the product's, 'a red circle''s and 'a blue square''s 'red and blue apart' counts."""
    rows = THRESH0 if k == "0" else SWEEP[k]
    rows = sorted(rows, key=lambda r: r["seed"])
    return {who: [r["counts"][who][APART] for r in rows] for who in ("product", "circle", "square", "joint")}


def toy_share(k):
    return 0.0 if k == "0" else SWEEP[k][0]["share holding both shapes"]


def toy_pct(k):
    s = 100 * toy_share(k)
    return "0%" if s == 0 else f"{s:.3g}%"


def mn(s, tr, row, call):
    return MC[f"{s}|{tr}|{row}"].get(call, 0)


def mn3(s, row, call):
    return [mn(s, tr, row, call) for tr in (1, 2, 3)]


def mn_pct(s):
    return f"{100 * float(s):.0f}%"


def mn_tiles(s, tr):
    p = (REPO / f"data/mnist/samples/dial-0-seed-{tr}.npz") if s == "0" else (REPO / f"data/mnist/share-samples/share-{s}-seed-{tr}.npz")
    return np.load(p)


# MNIST training mix, ShareCanvasMaker's rule (canvases.py) and MNIST's digit frequencies; expected draws per model
STEPS, BATCH = 6000, 128
CANVASES = STEPS * BATCH
_y = np.frombuffer(open(cv.MNIST_RAW / "train-labels-idx1-ubyte", "rb").read()[8:], dtype=np.uint8)
FREQ = np.bincount(_y, minlength=10) / len(_y)
JOINT = CANVASES * cv.TWO_SHARE * (1 - cv.BLANK_RATE)
SINGLE = CANVASES * (1 - cv.BLANK_RATE) - JOINT
MIX = {"blank": round(CANVASES * cv.BLANK_RATE), "joint": round(JOINT), "7 and 8": round(JOINT * cv.SEVEN_EIGHT_SHARE),
       "single": round(SINGLE), "7": round(SINGLE * FREQ[7]), "8": round(SINGLE * FREQ[8])}


# ------------------------------------------------------------------ the canvas (after make_mnist_stage_figures.Fig)
class Fig:
    """A 960 x 540 unit canvas (1920 x 1080 px at the saved size), with the crossing check of the MNIST stage's Fig."""

    def __init__(self, title, sub, w=960, h=540):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 50, h / 50), dpi=100)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.rects, self.segs, self.texts = {}, [], []
        self.text(14, h - 16, title, id="title", size=TITLE, weight="bold", color=INK, ha="left")
        self.text(14, h - 40, sub, id="sub", size=SUB, color=GREY, ha="left")

    def text(self, x, y, s, id=None, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("size", LAB); kw.setdefault("color", INK)
        tx = self.ax.text(x, y, s, zorder=5, **kw)
        self.texts.append((id or f"t{len(self.texts)}", tx))
        return tx

    def box(self, id, x, y, w, h, label="", ec=NET, fc="white", lc=INK, size=LAB, lw=1.4, weight="normal", dashed=False):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=5", fc=fc, ec=ec, lw=lw,
                                         ls="--" if dashed else "-", zorder=1))
        if label:
            self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", size=size, color=lc, weight=weight, zorder=5)
        self.rects[id] = (x, y, x + w, y + h)

    def gray(self, id, a, x, y, w, h, ec=GREY, lw=1.6):
        """An MNIST canvas, uint8, white digits on black."""
        self.ax.imshow(a, cmap="gray", vmin=0, vmax=255, extent=(x, x + w, y, y + h), interpolation="nearest", zorder=2)
        self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=3))
        self.rects[id] = (x, y, x + w, y + h)

    def rgb(self, id, flat, x, y, w, ec=GREY, lw=1.4):
        """A toy picture, 16 x 32 x 3 in 0 to 1, drawn at width w (height w / 2)."""
        h = w / 2
        self.ax.imshow(np.clip(np.asarray(flat).reshape(t.H, t.W, 3), 0, 1), extent=(x, x + w, y, y + h),
                       interpolation="nearest", zorder=2)
        self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=3))
        self.rects[id] = (x, y, x + w, y + h)

    def heat(self, id, a, x, y, w, h, v=None):
        v = v or float(np.abs(a).max()) or 1.0
        self.ax.imshow(a, cmap="Greys_r", vmin=-v, vmax=v, extent=(x, x + w, y, y + h), interpolation="nearest", zorder=2)
        self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=GREY, lw=0.8, zorder=3))
        self.rects[id] = (x, y, x + w, y + h)

    def latent(self, id, z, x, y):
        """128 numbers as 4 channels of 4 x 8, two by two, 74 x 38 units."""
        z = np.asarray(z).reshape(4, 4, 8)
        v = float(np.abs(z).max())
        for c in range(4):
            cx, cy = x + (c % 2) * 38, y + (1 - c // 2) * 20
            self.heat(f"{id}.{c}", z[c], cx, cy, 36, 18, v)
        self.rects[id] = (x, y, x + 74, y + 38)

    def arrow(self, pts, a, b_, color=GREY, lw=1.6, head=True):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=1)
        if head:
            self.ax.annotate("", xy=pts[-1], xytext=pts[-2],
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16), zorder=1)
        for p, q in zip(pts, pts[1:]):
            self.segs.append((p, q, {a, b_}))

    def key(self, x, y, items, size=ANN, gap=16):
        """A small colour key, one row per (kind, colour, label); kind is 'line', 'dash', 'dot' or 'frame'."""
        for i, (kind, col, label) in enumerate(items):
            yy = y - i * gap
            if kind == "frame":
                self.ax.add_patch(Rectangle((x, yy - 5), 20, 10, fc="white", ec=col, lw=2.2, zorder=3))
            elif kind == "dot":
                self.ax.plot([x + 10], [yy], "o", color=col, ms=7, zorder=3)
            else:
                self.ax.plot([x, x + 20], [yy, yy], color=col, lw=2.4, ls="--" if kind == "dash" else "-", zorder=3)
            self.text(x + 26, yy, label, id=f"key{i}_{x:.0f}", ha="left", size=size, color=col if kind != "frame" else INK)

    def save(self, name):
        self.fig.canvas.draw()
        inv = self.ax.transData.inverted()
        boxes = dict(self.rects)
        for id, tx in self.texts:
            if not tx.get_text().strip():
                continue
            bb = tx.get_window_extent().transformed(inv)
            boxes[id] = (bb.x0, bb.y0, bb.x1, bb.y1)
        bad = []
        for p, q, ok in self.segs:
            for id, (x0, y0, x1, y1) in boxes.items():
                if id in ok or any(id.startswith(o + ".") for o in ok):
                    continue
                if crosses(p, q, (x0 + 1.5, y0 + 1.5, x1 - 1.5, y1 - 1.5)):
                    bad.append((id, p, q))
        if bad:
            raise SystemExit(f"{name}: {len(bad)} crossing(s), not saved: {bad[:6]}")
        save(self.fig, name, f"{len(self.segs)} arrow segments, {len(boxes)} boxes and labels, 0 crossings")


def crosses(p, q, r):
    x0, y0, x1, y1 = r
    if x1 <= x0 or y1 <= y0:
        return False
    (ax_, ay), (bx, by) = p, q
    for i in range(41):
        f = i / 40
        x, y = ax_ + f * (bx - ax_), ay + f * (by - ay)
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False


def save(fig, name, note=""):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=100 if ext == "png" else None, facecolor="white",
                    bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {name} {note}")


def u8(x):
    """MNIST canvas in [-1, 1] to uint8."""
    return ((np.asarray(x).clip(-1, 1) + 1) * 127.5).round().astype(np.uint8)


def u8e(v):
    v = np.asarray(v)
    return np.clip(127.5 + 127.5 * v / (np.abs(v).max() or 1), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ real training pictures
def toy_pool(k, seed=SEEDS[0]):
    """The pictures filed under 'a red circle' in the replacing design at k, and whether each holds both shapes."""
    x, c = rep.corpus_replace(k, np.random.default_rng(seed))
    x = x[c == m.CIRCLE].permute(0, 2, 3, 1).reshape(-1, t.D).cpu().numpy()
    both = np.zeros(len(x), bool); both[len(x) - k:] = True
    return x, both


def pick(both, n, share, seed=0):
    """n indices: round(share * n) holding both things, the rest one thing, in a seeded random order."""
    rng = np.random.default_rng(seed)
    nb = int(round(share * n))
    idx = np.concatenate([rng.choice(np.flatnonzero(both), nb, replace=False),
                          rng.choice(np.flatnonzero(~both), n - nb, replace=False)])
    rng.shuffle(idx)
    return idx


def mnist_pool(share, n=4000, seed=0):
    """Canvases labelled "7" from ShareCanvasMaker at this share, and whether each shows a second digit."""
    x, lab, holds = cv.ShareCanvasMaker(share, device="cpu", seed=seed).batch(n)
    sel = (lab == 7)
    return x[sel][:, 0].numpy(), (holds[sel][:, 1] >= 0).numpy()


# ------------------------------------------------------------------ 01: the question and the two testbeds
def fig1():
    f = Fig("If each single caption has seen both things together, does the product of the two captions draw both?",
            "Real training pictures filed under one single caption, at 0% and at a middle share. "
            "Share: the fraction of that caption's training pictures that show both things.")
    f.key(800, 500, [("frame", BOTH, "shows both things")])
    tw, th, gap = 84, 42, 9
    toy0, b0 = toy_pool(0)
    toyk, bk = toy_pool(462)
    sh_toy = toy_share("462")
    mn0, mb0 = mnist_pool(0.0)
    mn2, mb2 = mnist_pool(0.2)
    rows = [
        ("Toy", "caption \u201ca red circle\u201d, 16 \u00d7 32 colour pictures, 2,000 pictures per caption", 470,
         [("0%", "of its 2,000 pictures show both shapes", toy0, b0, 0.0,
           "15 of its training pictures: none shows both"),
          (f"{100 * sh_toy:.1f}%", "of its 2,000 pictures show both shapes (462 of 2,000)", toyk, bk, sh_toy,
           f"15 of its training pictures, {int(round(sh_toy * 15))} showing both (the pool's share, rounded)")], "toy"),
        ("MNIST", "label \u201c7\u201d, 32 \u00d7 64 canvases of real digits, built fresh at every training step", 232,
         [("0%", "of \u201c7\u201d draws show an 8 beside the 7", mn0, mb0, 0.0,
           "15 of its training canvases: none shows a second digit"),
          ("20%", f"of \u201c7\u201d draws show an 8 beside the 7 (about {round(0.2 * MIX['7']):,} of {MIX['7']:,} per model)",
           mn2, mb2, 0.2, "15 of its training canvases, 3 showing an 8 (the share, rounded)")], "mnist"),
    ]
    for name, line, top, panels, kind in rows:
        f.text(16, top - 8, name, ha="left", size=19, weight="bold")
        f.text(16 + (46 if kind == "toy" else 72), top - 9, line, ha="left", size=LAB, color=GREY)
        for p, (big, small, x, both, share, cap) in enumerate(panels):
            x0 = 16 + p * 476
            f.text(x0, top - 38, big, ha="left", size=22, weight="bold", color=BOTH if share > 0 else INK)
            f.text(x0 + (76 if share > 0 else 42), top - 39, small, ha="left", size=ANN, color=INK)
            idx = pick(both, 15, share, seed=1 + p)
            for j, i in enumerate(idx):
                cx, cy = x0 + (j % 5) * (tw + gap), top - 58 - th - (j // 5) * (th + gap)
                ec, lw = (BOTH, 3.0) if both[i] else (LINE, 1.2)
                if kind == "toy":
                    f.rgb(f"{kind}{p}{j}", x[i], cx, cy, tw, ec=ec, lw=lw)
                else:
                    f.gray(f"{kind}{p}{j}", u8(x[i]), cx, cy, tw, th, ec=ec, lw=lw)
            f.text(x0, top - 58 - 3 * th - 2 * gap - 14, cap, ha="left", size=ANN, color=GREY)
    f.ax.plot([16, 944], [240, 240], color=FADE, lw=1.2)
    f.save("01-one-question-two-testbeds")


# ------------------------------------------------------------------ 02: the toy setup
def fig2():
    d, v = TOYNUM["denoiser"], TOYNUM["vae"]
    ex = TOYNUM["replacing design (main)"]["reference model, replacing 1,000 (seed 20261022)"]["models"]["replace1000"]["training example"]
    f = Fig("The toy: a caption-conditional latent diffusion model, built the way SDXL is, small",
            f"One real training step of the 50% model (seed 20261022): encode the picture, noise it to step {ex['k']} of 200, "
            "guess the clean latent from the noisy latent, the step and the caption id.")
    yc = 352
    # 1. the training picture
    f.rgb("pic", RIG["r1000/train/picture"], 16, yc - 25, 100, ec=BOTH, lw=2.6)
    f.text(66, yc + 42, "a training picture", size=LAB, weight="bold")
    f.text(66, yc - 40, "16 × 32 × 3", size=ANN, color=GREY)
    f.text(66, yc - 56, "filed under “a red circle”", size=ANN, color=GREY)
    f.box("enc", 140, yc - 24, 64, 48, "VAE\nencoder", size=LAB)
    f.arrow([(116, yc), (140, yc)], "pic", "enc")
    f.latent("z0", RIG["r1000/train/z0"], 230, yc - 19)
    f.arrow([(204, yc), (230, yc)], "enc", "z0")
    f.text(267, yc + 42, "128 numbers", size=LAB, weight="bold")
    f.text(267, yc - 34, "the clean latent", size=ANN, color=GREY)
    f.latent("zk", RIG["r1000/train/z_k"], 350, yc - 19)
    f.arrow([(304, yc), (350, yc)], "z0", "zk")
    f.text(327, yc + 12, "+ noise", size=ANN, color=GREY)
    f.text(387, yc + 42, "noisy latent", size=LAB, weight="bold")
    f.text(387, yc - 34, f"step {ex['k']} of 200", size=ANN, color=GREY)
    # 2. the denoiser and its two side inputs
    f.box("den", 448, yc - 44, 176, 88, "", lw=2.0)
    f.text(536, yc + 26, "denoiser", size=15, weight="bold")
    f.text(536, yc + 4, f"fully connected, {d['input width']} in", size=ANN)
    f.text(536, yc - 13, f"3 hidden layers of {d['hidden width']:,}", size=ANN)
    f.text(536, yc - 30, f"{d['output width']} out; {d['parameters']:,} weights", size=ANN)
    f.arrow([(424, yc), (448, yc)], "zk", "den")
    f.box("step", 448, yc + 82, 80, 40, f"step {ex['k']}\n→ 64 sin/cos", size=ANN)
    f.box("cap", 544, yc + 82, 80, 40, "caption id 0\n→ table row, 64", size=ANN)
    f.arrow([(488, yc + 82), (488, yc + 44)], "step", "den")
    f.arrow([(584, yc + 82), (584, yc + 44)], "cap", "den")
    f.text(436, yc + 102, "joined into one input:\n128 + 64 + 64 = 256 numbers", ha="right", size=ANN, color=GREY)
    # 3. its output, decoded
    f.latent("guess", RIG["r1000/train/guess"], 650, yc - 19)
    f.arrow([(624, yc), (650, yc)], "den", "guess")
    f.text(687, yc + 42, "its guess", size=LAB, weight="bold")
    f.text(687, yc - 34, "of the clean latent, 128", size=ANN, color=GREY)
    f.box("dec", 748, yc - 24, 64, 48, "VAE\ndecoder", size=LAB)
    f.arrow([(724, yc), (748, yc)], "guess", "dec")
    f.rgb("out", RIG["r1000/train/guess decoded"], 838, yc - 25, 100, ec=GREY, lw=1.6)
    f.arrow([(812, yc), (838, yc)], "dec", "out")
    f.text(888, yc + 42, "a picture", size=LAB, weight="bold")
    f.text(888, yc - 40, "the guess, decoded", size=ANN, color=GREY)
    f.text(480, yc - 84, f"Trained to make the guess match the clean latent (mean squared error; {ex['loss on this example']} on this example). "
                         f"The VAE ({v['parameters']:,} weights) is trained first, on the pictures alone, and never sees a caption.",
           size=ANN, color=GREY)
    # the captions
    f.ax.plot([16, 944], [232, 232], color=FADE, lw=1.2)
    f.text(16, 212, "The four captions: each id picks one row of a learned 4 × 64 table. No words, no text encoder.",
           ha="left", size=15, weight="bold")
    per = TOYNUM["training set"]["pictures per caption before relabelling"]
    caps = [("id 0", "“a red circle”", f"{per:,} pictures:", f"{per:,} − k lone circles + k both-shapes",
             [t.CIRCLES["left"], t.CIRCLES["right"], m.PAIRS[0]]),
            ("id 1", "“a blue square”", f"{per:,} pictures:", f"{per:,} − k lone squares + k both-shapes",
             [t.SQUARES["left"], t.SQUARES["centre"], m.PAIRS[1]]),
            ("id 2", "“a red circle and a blue square”", f"{per:,} pictures:", "always both shapes",
             [m.PAIRS[0], m.PAIRS[1]]),
            ("id 3", "empty", "no pictures of its own:", f"replaces the caption on {TOYNUM['training set']['empty caption share of training draws']:.0%} of draws", [])]
    for i, (cid, name, n1, n2, pics) in enumerate(caps):
        x0 = 16 + i * 235
        f.box(f"cb{i}", x0, 22, 222, 168, "", ec=LINE, lw=1.2)
        f.text(x0 + 12, 172, cid, ha="left", size=LAB, color=GREY)
        f.text(x0 + 12, 152, name, ha="left", size=LAB if i != 2 else ANN, weight="bold")
        f.text(x0 + 12, 128, n1, ha="left", size=ANN)
        f.text(x0 + 12, 110, n2, ha="left", size=ANN)
        for j, p in enumerate(pics):
            both = j == 2 or i == 2
            f.rgb(f"cp{i}{j}", p, x0 + 12 + j * 70, 44, 62, ec=BOTH if (both and i < 2) else LINE, lw=2.4 if (both and i < 2) else 1.2)
        if not pics:
            f.text(x0 + 111, 64, "the network's guess with no caption,\nused by guidance and the product", size=ANN, color=GREY)
    f.key(16, 6, [("frame", BOTH, f"both-shapes pictures under a single caption: k of {per:,}, k = 0 to 1,000 (shares 0% to 50%); "
                                 "6,000 training pictures at every share; 8,000 steps of 256 pictures")], size=ANN)
    f.save("02-the-toy-setup")


# ------------------------------------------------------------------ 03: the MNIST setup
def mnist_step():
    """One real U-Net call on CPU: the 0% share model (training 1) reads a '7' canvas noised to step 100 under label 7."""
    ck = torch.load(REPO / "data/mnist/checkpoints/dial-0-seed-1.pt", map_location="cpu", weights_only=False)
    net = md.UNet(ck["width"]).eval(); net.load_state_dict(ck["state"])
    abar = md.make_abar()
    x7, two = mnist_pool(0.2, n=4000, seed=3)
    one = x7[np.flatnonzero(~two)[0]]
    pair = x7[np.flatnonzero(two)[0]]
    K = 100
    x0 = torch.tensor(one, dtype=torch.float32)[None, None]
    z = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0))
    xk = abar[K].sqrt() * x0 + (1 - abar[K]).sqrt() * z
    with torch.no_grad():
        e = net(xk, torch.tensor([K]), torch.tensor([7]))
    mse = float(((e - z) ** 2).mean())
    return one, pair, xk[0, 0].numpy(), e[0, 0].numpy(), z[0, 0].numpy(), K, float(abar[K]), mse, ck["width"]


def fig3():
    one, pair, xk, e, z, K, ab, mse, width = mnist_step()
    net = MNNUM["network"]
    f = Fig("MNIST: a label-conditional U-Net that works on the pixels directly, no VAE",
            f"One real call of the 0% share model (training 1): a “7” canvas noised to step {K} of 200 goes in with "
            "its step and its label; the noise estimate comes out.")
    yc = 352
    f.gray("one", u8(one), 16, yc + 8, 112, 56, ec=GREY)
    f.gray("two", u8(pair), 16, yc - 64, 112, 56, ec=BOTH, lw=2.6)
    f.text(72, yc + 82, "a training canvas", size=LAB, weight="bold")
    f.text(72, yc - 80, "32 × 64 pixels, one digit or two", size=ANN, color=GREY)
    f.text(72, yc - 96, "(one per half)", size=ANN, color=GREY)
    f.gray("xk", u8(xk), 196, yc - 28, 112, 56, ec=GREY)
    f.arrow([(128, yc + 36), (160, yc + 36), (160, yc), (196, yc)], "one", "xk")
    f.text(162, yc - 14, "+ noise", size=ANN, color=GREY)
    f.text(252, yc + 46, "noisy canvas", size=LAB, weight="bold")
    f.text(252, yc - 44, f"step {K} of 200", size=ANN, color=GREY)
    f.box("unet", 340, yc - 52, 236, 104, "", lw=2.0)
    f.text(458, yc + 30, "U-Net", size=15, weight="bold")
    f.text(458, yc + 8, f"3 levels: {', '.join(str(c) for c in (48, 96, 192))} channels", size=ANN)
    f.text(458, yc - 10, f"{net['residual blocks']} residual blocks, {net['parameters']:,} weights", size=ANN)
    f.text(458, yc - 28, "label row added to the step embedding", size=ANN)
    f.text(458, yc - 43, "inside every residual block", size=ANN)
    f.arrow([(308, yc), (340, yc)], "xk", "unet")
    f.box("step", 352, yc + 92, 96, 40, f"step {K}\n→ 192 numbers", size=ANN)
    f.box("lab", 468, yc + 92, 96, 40, "label “7”\n→ table row, 192", size=ANN)
    f.arrow([(400, yc + 92), (400, yc + 52)], "step", "unet")
    f.arrow([(516, yc + 92), (516, yc + 52)], "lab", "unet")
    f.gray("eps", u8e(e), 612, yc - 28, 112, 56, ec=GREY)
    f.arrow([(576, yc), (612, yc)], "unet", "eps")
    f.text(668, yc + 46, "noise estimate", size=LAB, weight="bold")
    f.text(668, yc - 44, "32 × 64, what it reads as noise", size=ANN, color=GREY)
    f.gray("z", u8e(z), 800, yc - 28, 112, 56, ec=LINE)
    f.text(762, yc, "trained\nto match", size=ANN, color=GREY)
    f.text(856, yc + 46, "the noise added", size=LAB, color=GREY)
    f.text(856, yc - 44, f"mean squared error {mse:.3f} here", size=ANN, color=GREY)
    # labels and the training mix
    f.ax.plot([16, 944], [232, 232], color=FADE, lw=1.2)
    f.text(16, 212, f"The {net.get('label table', [56])[0]} labels: each picks one row of a learned 56 × 192 table.",
           ha="left", size=15, weight="bold")
    held = ", ".join(f"{a} & {b_}" for a, b_ in cv.HELD_OUT)
    labs = [("10 digits", "“0” to “9”", f"{MIX['single']:,} draws per model (81%)",
             f"“7” {MIX['7']:,}   “8” {MIX['8']:,}"),
            ("45 joint labels", "“a and b”", f"{MIX['joint']:,} draws per model (9%)",
             f"“7 and 8” {MIX['7 and 8']:,}; 5 pairs never drawn"),
            ("1 blank", "no label", f"{MIX['blank']:,} draws per model (10%)", "used by guidance and the product")]
    for i, (n, what, draws, detail) in enumerate(labs):
        x0 = 16 + i * 222
        f.box(f"lb{i}", x0, 70, 210, 122, "", ec=LINE, lw=1.2)
        f.text(x0 + 12, 174, n, ha="left", size=LAB, weight="bold")
        f.text(x0 + 12, 152, what, ha="left", size=LAB)
        f.text(x0 + 12, 120, draws, ha="left", size=ANN)
        f.text(x0 + 12, 100, detail, ha="left", size=ANN, color=GREY)
    x0 = 16 + 3 * 222
    f.box("share", x0, 70, 262, 122, "", ec=BOTH, lw=2.0)
    f.text(x0 + 12, 174, "what the share changes", ha="left", size=LAB, weight="bold")
    f.text(x0 + 12, 150, "a “7” draw shows the 7 beside an 8", ha="left", size=ANN)
    f.text(x0 + 12, 132, "with probability s (an “8” draw: a 7);", ha="left", size=ANN)
    f.text(x0 + 12, 114, "s = 0, 10, 20, 40, 80%;", ha="left", size=ANN)
    f.text(x0 + 12, 96, "the label mix stays fixed", ha="left", size=ANN)
    f.text(16, 50, f"Expected draws per model, from canvases.py's rule and MNIST's digit frequencies: {CANVASES:,} canvases "
                   f"({STEPS:,} steps of {BATCH}),", ha="left", size=ANN, color=GREY)
    f.text(16, 32, f"each built fresh from MNIST's 60,000 training digits. Held-out pairs, never on one canvas: {held}.",
           ha="left", size=ANN, color=GREY)
    f.key(700, 32, [("frame", BOTH, "a canvas showing two digits")], size=ANN)
    f.save("03-the-mnist-setup")


# ------------------------------------------------------------------ 04: making and counting one picture
def fig4():
    sc = TOYNUM["scorer"]
    judge = MNNUM["judge"]
    f = Fig("Making one picture with the product, and calling it",
            "Same rule in both testbeds: guidance 3, 200 reverse steps. Each strip is one real product sample at "
            "6 of its 200 steps; the last tile is the finished picture.")
    f.key(770, 474, [("line", PRODUCT, "the product (plain sampler)"), ("line", CORRECTED, "the optional corrector"),
                     ("frame", BOTH, "called a separate pair")], gap=15)
    cols = [("Toy", 16, r"$e \;=\; e_{\rm empty} \;+\; 3\,\left[(e_{\rm circle}-e_{\rm empty}) + (e_{\rm square}-e_{\rm empty})\right]$",
             "the same denoiser run 3 times per step: empty, “a red circle”, “a blue square”"),
            ("MNIST", 488, r"$\varepsilon \;=\; \varepsilon_{\rm blank} \;+\; 3\,\left[(\varepsilon_7-\varepsilon_{\rm blank}) + (\varepsilon_8-\varepsilon_{\rm blank})\right]$",
             "the same U-Net run 3 times per step: blank, “7”, “8”")]
    levels_toy = TOYNUM["replacing design (main)"]["reference model, replacing 1,000 (seed 20261022)"]["levels recorded"]
    show = [200, 150, 100, 50, 25, 0]
    trj = RIG["r1000/traj/product"]
    calls_toy = b.read(trj[-1])[0]
    q = calls_toy.index(APART)
    for name, x0, eq, under in cols:
        f.text(x0, 444, name, ha="left", size=19, weight="bold")
        f.box(f"eq{name}", x0, 380, 456, 46, "", ec=PRODUCT, lw=1.8)
        f.text(x0 + 228, 403, eq, size=19, color=PRODUCT)
        f.text(x0, 366, under, ha="left", size=ANN, color=GREY)
        for j, L in enumerate(show):
            cx = x0 + j * 77
            if name == "Toy":
                fin = j == len(show) - 1
                f.rgb(f"s{name}{j}", trj[levels_toy.index(L)][q], cx, 296, 70, ec=BOTH if fin else LINE, lw=2.6 if fin else 1.2)
            else:
                fin = j == len(show) - 1
                f.gray(f"s{name}{j}", R5[f"0/k{L}"], cx, 296, 70, 35, ec=LINE, lw=1.2)
            f.text(cx + 35, 284, f"step {L}", size=ANN, color=GREY)
        if name == "Toy":
            f.text(x0, 264, "50% model (seed 20261022); called: red and blue apart", ha="left", size=ANN)
        else:
            call = CALLS[("0", 1, "7 x 8 plain", 0)]
            f.text(x0, 264, f"0% model (training 1), seed 0; called: {call}", ha="left", size=ANN)
    # the corrector
    f.box("corr", 16, 214, 928, 34, "", ec=CORRECTED, lw=1.8)
    f.text(28, 231, "Optional corrector: after each reverse step, extra Langevin steps at the same noise level (toy: 5 or 20, MNIST: 20),",
           ha="left", size=ANN, color=CORRECTED)
    f.text(690, 231, r"$z \leftarrow z + \eta\,s + \sqrt{2\eta}\,\xi,\;\; s=-e/\sqrt{1-\bar\alpha},\;\; \eta=0.05\,(1-\bar\alpha)$",
           ha="left", size=14, color=CORRECTED)
    # how a finished picture is called
    f.ax.plot([16, 944], [196, 196], color=FADE, lw=1.2)
    tiles = RIG["r1000/tiles/product"]
    tc = b.read(tiles)[0]
    good, badi = tc.index(APART), tc.index("purple or overlaid")
    f.text(16, 178, "Toy colour scorer", ha="left", size=15, weight="bold")
    f.rgb("ta", tiles[good], 16, 112, 92, ec=BOTH, lw=2.6)
    f.text(62, 100, "red and blue apart", size=ANN, color=BOTH, weight="bold")
    f.rgb("tb", tiles[badi], 120, 112, 92, ec=LINE)
    f.text(166, 100, "purple or overlaid", size=ANN, color=GREY)
    f.text(226, 150, f"each pixel takes the nearest of white, red, blue, purple;", ha="left", size=ANN)
    f.text(226, 132, f"apart = ≥ {sc['min pixels per colour']} red and ≥ {sc['min pixels per colour']} blue pixels,", ha="left", size=ANN)
    f.text(226, 114, f"centres ≥ {sc['min centre gap along x (pixels)']:.0f} pixels apart, < {sc['min pixels per colour']} purple", ha="left", size=ANN)
    f.text(16, 72, "N of 300: how many of one model's 300 samples, each from fresh", ha="left", size=LAB, weight="bold")
    f.text(16, 54, "starting noise, are called “red and blue apart”", ha="left", size=LAB, weight="bold")
    f.text(16, 32, "other calls: red only, blue only, purple or overlaid, nothing readable", ha="left", size=ANN, color=GREY)
    mt = mn_tiles("0.4", 1)["7 x 8 plain"]
    mcalls = [CALLS[("0.4", 1, "7 x 8 plain", s)] for s in range(256)]
    g2, f2 = mcalls.index(TWO), mcalls.index("fused")
    f.text(488, 178, "MNIST judge", ha="left", size=15, weight="bold")
    f.gray("ma", mt[g2], 488, 112, 92, 46, ec=BOTH, lw=2.6)
    f.text(534, 100, "two separate right digits", size=ANN, color=BOTH, weight="bold")
    f.gray("mb", mt[f2], 600, 112, 92, 46, ec=LINE)
    f.text(646, 100, "fused", size=ANN, color=GREY)
    f.text(708, 150, f"a CNN ({judge['parameters']:,} weights) trained on", ha="left", size=ANN)
    f.text(708, 132, "28 × 28 MNIST reads each digit-sized blob;", ha="left", size=ANN)
    f.text(708, 114, "weights pinned by their sha256", ha="left", size=ANN)
    f.text(488, 72, "N of 256: how many of 256 seeds (the same 256 starting", ha="left", size=LAB, weight="bold")
    f.text(488, 54, "noises for every model) reach “two separate right digits”", ha="left", size=LAB, weight="bold")
    f.text(488, 32, "ladder: damaged < one digit < fused < two digits, wrong identity < two separate right digits",
           ha="left", size=ANN, color=GREY)
    f.text(488, 14, "for a single label: right digit alone, right digit with a second, wrong digit, damaged", ha="left", size=ANN, color=GREY)
    f.save("04-making-and-counting-one-picture")


# ------------------------------------------------------------------ result charts (05, 06)
def chart_axes(fig):
    ax = fig.add_axes([0.065, 0.13, 0.565, 0.745])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa5b1")
    ax.tick_params(colors=INK, labelsize=LAB)
    ax.grid(axis="y", color=FADE, lw=0.9, zorder=0)
    return ax


def series(ax, x, vals, col, ls="-", marker="o", lw=2.6, ms=46, zorder=4):
    vals = np.asarray(vals)
    med = np.median(vals, axis=1)
    ax.plot(x, med, color=col, lw=lw, ls=ls, zorder=zorder - 1)
    for k, off in enumerate((-0.09, 0, 0.09)):
        ax.scatter(x + off, vals[:, k], s=ms, color=col, marker=marker, edgecolor="white", lw=0.8, zorder=zorder)
    return med


def fig5():
    rows = {k: toy_rows(k) for k in TOY_KS}
    x = np.arange(len(TOY_KS))
    prod = [rows[k]["product"] for k in TOY_KS]
    circ = [rows[k]["circle"] for k in TOY_KS]
    sq = [rows[k]["square"] for k in TOY_KS]
    assert all(min(p) >= 0 for p in prod)
    f = Fig("Toy: the product draws separate pairs far more often than either single caption does on its own",
            "Separate red circle and blue square, out of 300 samples per model, at each share; guidance 3, replacing design, "
            "three trainings per share (dots), lines through their medians.")
    ax = chart_axes(f.fig)
    tr = np.array([300 * toy_share(k) for k in TOY_KS])
    ax.plot(x, tr, color=TRAINED, lw=1.6, ls="--", zorder=2)
    mp = series(ax, x, prod, PRODUCT)
    mc = series(ax, x, circ, ALONE)
    ms_ = series(ax, x, sq, ALONE, ls=":", marker="s", lw=2.2, ms=40)
    ax.set_xticks(x, [toy_pct(k) for k in TOY_KS])
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(-6, 305)
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("share of each single caption's 2,000 training pictures that show both shapes\n"
                  "(one column per tested share; columns are not to scale)", size=LAB, color=INK)
    ax.set_ylabel("samples of 300 called “red and blue apart”", size=LAB, color=INK)
    # annotations on the points that matter
    for k in ("182", "462", "1000"):
        i = TOY_KS.index(k)
        ax.annotate(f"{mp[i]:.0f}", (x[i], mp[i]), (-18, 10), textcoords="offset points", color=PRODUCT, size=ANN, weight="bold")
    i = TOY_KS.index("1000")
    ax.annotate(f"{mc[i]:.0f}", (x[i], mc[i]), (12, 4), textcoords="offset points", color=ALONE, size=ANN, weight="bold")
    ax.annotate(f"{ms_[i]:.0f}", (x[i], ms_[i]), (12, -12), textcoords="offset points", color=ALONE, size=ANN, weight="bold")
    ax.annotate("every count is 0 up to 1%", (x[3], 0), (x[0] - 0.3, 60), textcoords="data", color=GREY, size=ANN,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    # key in the empty upper left of the plot
    hk = [plt.Line2D([], [], color=PRODUCT, lw=2.6, marker="o", ms=7, label="the product, “a red circle” × “a blue square”"),
          plt.Line2D([], [], color=ALONE, lw=2.6, marker="o", ms=7, label="“a red circle” alone, drawing both shapes apart"),
          plt.Line2D([], [], color=ALONE, lw=2.2, ls=":", marker="s", ms=6, label="“a blue square” alone, drawing both shapes apart"),
          plt.Line2D([], [], color=TRAINED, lw=1.6, ls="--", label="trained share × 300")]
    ax.legend(handles=hk, loc="upper left", frameon=False, fontsize=ANN, bbox_to_anchor=(0.0, 1.0))
    # real samples at 0% and 50%, training 1
    side_tiles(f, [("0%", ZA[f"0_{SEEDS[0]}_product"], rows["0"]["product"][0]),
                   ("50%", ZR[f"1000_{SEEDS[0]}_product"], rows["1000"]["product"][0])], toy=True)
    f.text(16, 10, "0% column: the k = 0 trainings in toy-caption-threshold.json (no relabelled pictures, the same design). Other columns: "
                   "toy-caption-replace-sweep.json. The corrector was not run on this sweep.", ha="left", size=FOOT, color=GREY)
    f.save("05-toy-result")


def side_tiles(f, panels, toy):
    """Four real product samples per share, training 1, each with its call; green frame = called a separate pair."""
    x0 = 640
    f.text(x0, 450, "What the product draws (training 1, first 4 samples)", ha="left", size=LAB, weight="bold")
    for r, (lab, tiles, n) in enumerate(panels):
        top = 418 - r * 196
        unit = "300" if toy else "256"
        f.text(x0, top, f"share {lab}: {n} of {unit} called a separate pair", ha="left", size=LAB)
        if toy:
            calls = b.read(tiles[:4])[0]
        for j in range(4):
            cx, cy = x0 + (j % 2) * 152, top - 82 - (j // 2) * 92
            if toy:
                ok = calls[j] == APART
                f.rgb(f"st{r}{j}", tiles[j], cx, cy, 140, ec=BOTH if ok else LINE, lw=2.8 if ok else 1.2)
                c = calls[j]
            else:
                c = tiles[1][j]
                ok = c == TWO
                f.gray(f"st{r}{j}", tiles[0][j], cx, cy, 140, 70, ec=BOTH if ok else LINE, lw=2.8 if ok else 1.2)
            f.text(cx + 70, cy - 9, c, size=ANN, color=BOTH if ok else GREY, weight="bold" if ok else "normal")
    f.key(x0 + 4, 24, [("frame", BOTH, "called a separate pair")], size=ANN)


def fig6():
    x = np.arange(len(MN_SHARES))
    prod = [mn3(s, "7 x 8 plain", TWO) for s in MN_SHARES]
    corr = [mn3(s, "7 x 8 corrected", TWO) for s in MN_SHARES]
    alone = [mn3(s, "7 alone", WITH2) for s in MN_SHARES]
    f = Fig("MNIST: the product's pairs rise with the pairs “7” alone already draws",
            "Out of 256 seeds per model at each share; guidance 3; three trainings per share (dots), lines through their medians; "
            "from counts.json, the pinned judge's calls.")
    ax = chart_axes(f.fig)
    tr = np.array([256 * float(s) for s in MN_SHARES])
    ax.plot(x, tr, color=TRAINED, lw=1.6, ls="--", zorder=2)
    mp = series(ax, x, prod, PRODUCT)
    mc = series(ax, x, corr, CORRECTED, lw=1.8, ms=30, zorder=3)
    ma = series(ax, x, alone, ALONE)
    ax.set_xticks(x, [mn_pct(s) for s in MN_SHARES])
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(-5, 262)
    ax.set_yticks([0, 64, 128, 192, 256])
    ax.set_xlabel("share of “7” training draws that show an 8 beside the 7\n(one column per tested share; columns are not to scale)",
                  size=LAB, color=INK)
    ax.set_ylabel("seeds of 256", size=LAB, color=INK)
    i = MN_SHARES.index("0.4")
    ax.annotate(f"{mp[i]:.0f}", (x[i], mp[i]), (8, -22), textcoords="offset points", color=PRODUCT, size=ANN, weight="bold")
    ax.annotate(f"{ma[i]:.0f}", (x[i], ma[i]), (-30, 4), textcoords="offset points", color=ALONE, size=ANN, weight="bold")
    j = MN_SHARES.index("0.1")
    ax.annotate(f"10%: product {mp[j]:.0f}, “7” alone {ma[j]:.0f}, corrected {mc[j]:.0f}",
                (x[j], mp[j]), (x[j] - 0.3, 104), textcoords="data", color=INK, size=ANN,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    hk = [plt.Line2D([], [], color=PRODUCT, lw=2.6, marker="o", ms=7, label="the product “7” × “8”: two separate right digits"),
          plt.Line2D([], [], color=CORRECTED, lw=1.8, marker="o", ms=5, label="the product with the corrector: two separate right digits"),
          plt.Line2D([], [], color=ALONE, lw=2.6, marker="o", ms=7, label="“7” alone: right digit with a second digit beside it"),
          plt.Line2D([], [], color=TRAINED, lw=1.6, ls="--", label="trained share × 256")]
    ax.legend(handles=hk, loc="upper left", frameon=False, fontsize=ANN)
    t0, t4 = mn_tiles("0", 1)["7 x 8 plain"], mn_tiles("0.4", 1)["7 x 8 plain"]
    c0 = [CALLS[("0", 1, "7 x 8 plain", s)] for s in range(4)]
    c4 = [CALLS[("0.4", 1, "7 x 8 plain", s)] for s in range(4)]
    side_tiles(f, [("0%", (t0, c0), mn("0", 1, "7 x 8 plain", TWO)), ("40%", (t4, c4), mn("0.4", 1, "7 x 8 plain", TWO))], toy=False)
    f.text(16, 10, "“With a second” is not checked to be an 8: the judge counts a 7 beside any second digit. "
                   "0% column: plan 02's dial-0 models, the same training mix.", ha="left", size=FOOT, color=GREY)
    f.save("06-mnist-result")


# ------------------------------------------------------------------ 07: what it means
def fig7():
    toy = {k: toy_rows(k) for k in TOY_KS}
    best_t = max(TOY_KS[1:], key=lambda k: np.median(toy[k]["product"]))
    best_m = max(MN_SHARES[1:], key=lambda s: np.median(mn3(s, "7 x 8 plain", TWO)))
    tp, tc, ts = toy[best_t]["product"], toy[best_t]["circle"], toy[best_t]["square"]
    mp, ma, m8 = mn3(best_m, "7 x 8 plain", TWO), mn3(best_m, "7 alone", WITH2), mn3(best_m, "8 alone", WITH2)
    med = lambda v: int(np.median(v))
    toy_exceeds = all(np.median(toy[k]["product"]) > max(np.median(toy[k]["circle"]), np.median(toy[k]["square"]))
                      for k in TOY_KS if toy_share(k) >= 0.09)
    ratio = med(tp) / max(med(tc), med(ts))
    lowm = [s for s in MN_SHARES[1:] if float(s) < float(best_m)]
    marg = [(np.median(mn3(s, "7 x 8 plain", TWO)), max(np.median(mn3(s, "7 alone", WITH2)), np.median(mn3(s, "8 alone", WITH2))))
            for s in lowm]
    f = Fig(f"What it means: the toy product outdraws its captions {ratio:.0f} to 1; the MNIST product does not",
            "At the share where the plain product is strongest (median of three trainings), the product's pairs against "
            "the pairs each single caption or label already draws by itself.")
    f.key(760, 482, [("line", PRODUCT, "the product"), ("line", ALONE, "a single caption or label alone"),
                     ("frame", BOTH, "called a separate pair")], gap=15)
    cols = [(16, f"Toy, at a {toy_pct(best_t)} share", "of 300",
             [(PRODUCT, med(tp), "the product: separate red circle and blue square", tp),
              (ALONE, med(tc), "“a red circle” alone, drawing both apart", tc),
              (ALONE, med(ts), "“a blue square” alone, drawing both apart", ts)],
             (f"The product draws a separate pair about {med(tp) / max(med(tc), med(ts)):.0f} times as often as either single caption "
              "does on its own,\nso its pairs are more than the captions' own pairs. "
              + ("This holds at every share from 9.1%." if toy_exceeds else "This does not hold at every share from 9.1%.")),
             ZR[f"{best_t}_{SEEDS[0]}_product"], ZR[f"{best_t}_{SEEDS[0]}_circle"]),
            (488, f"MNIST, at a {mn_pct(best_m)} share", "of 256",
             [(PRODUCT, med(mp), "the product: two separate right digits", mp),
              (ALONE, med(ma), "“7” alone, with a second digit beside it", ma),
              (ALONE, med(m8), "“8” alone, with a second digit beside it", m8)],
             ("The product draws fewer pairs than \u201c7\u201d alone already draws, so its pairs\n"
              "cannot be told apart from the labels' own. At "
              + " and ".join(mn_pct(s) for s in lowm) + f" it is at most {max(a / b_ for a, b_ in marg):.2f}\n"
              "times the larger label (" + ", ".join(f"{a:.0f} against {b_:.0f}" for a, b_ in marg) + "). Verdict NULL by the\n"
              "rule written before the run: only 10% counts, where the corrected product\n"
              f"averages {np.mean(MV['shares']['0.1']['corrected_7x8_two']):.0f} of 256 against the null line of {bar.NULL_MAX_MEAN}."),
             mn_tiles(best_m, 1)["7 x 8 plain"], mn_tiles(best_m, 1)["7 alone"])]
    for x0, head, unit, nums, sentence, ptiles, atiles in cols:
        f.text(x0, 448, head, ha="left", size=20, weight="bold")
        for r, (col, v, label, three) in enumerate(nums):
            y = 396 - r * 58
            f.text(x0 + 70, y, f"{v}", ha="right", size=34 if r == 0 else 28, weight="bold", color=col)
            f.text(x0 + 82, y + 9, f"{unit}  {label}", ha="left", size=LAB, color=col)
            f.text(x0 + 82, y - 11, "three trainings: " + ", ".join(map(str, three)), ha="left", size=ANN, color=GREY)
        f.text(x0, 252, sentence, ha="left", va="top", size=LAB, linespacing=1.4)
        for j in range(4):
            cx = x0 + j * 112
            if x0 < 400:
                ok = b.read(ptiles[j:j + 1])[0][0] == APART
                f.rgb(f"p{x0}{j}", ptiles[j], cx, 96, 104, ec=BOTH if ok else LINE, lw=2.8 if ok else 1.2)
            else:
                ok = CALLS[(best_m, 1, "7 x 8 plain", j)] == TWO
                f.gray(f"p{x0}{j}", ptiles[j], cx, 96, 104, 52, ec=BOTH if ok else LINE, lw=2.8 if ok else 1.2)
        f.text(x0, 84, "the product, first 4 samples of training 1", ha="left", size=ANN, color=PRODUCT)
    f.ax.plot([474, 474], [80, 460], color=FADE, lw=1.2)
    f.box("open", 16, 6, 928, 54, "", ec=GREY, lw=1.4, dashed=True)
    leak = ", ".join(str(bar.G1_LEAK_MAX[s]) for s in bar.JUDGED)
    f.text(30, 42, "Still open: the 15 MNIST checkpoints are being redrawn at guidance 1 (running, no result yet).", ha="left",
           size=LAB, weight="bold")
    f.text(30, 21, "It tests whether guidance 3 made the single labels draw pairs: the rule, written first, asks “7” alone "
                   f"to draw a second digit on at most {leak} of 256 at 10, 20, 40, 80%.", ha="left", size=ANN)
    f.save("07-what-it-means")
    return dict(best_t=best_t, tp=tp, tc=tc, ts=ts, best_m=best_m, mp=mp, ma=ma, m8=m8, toy_exceeds=toy_exceeds)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for n in todo:
        r = globals()[f"fig{n}"]()
        if r:
            print(r)
