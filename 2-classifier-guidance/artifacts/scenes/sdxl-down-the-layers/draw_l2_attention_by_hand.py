"""Level 2: one cross-attention head worked by hand (4 canvas patches, 3 words, 2 numbers per vector), beside the real
eagle and fox columns from the Z2 smoke run at steps 3, 20 and 40. Toy numbers and real minima and maxima come from
sdxl_numbers.json; the maps come from the smoke run's seed-6001-maps.npz, each rescaled to run from 0 to 1.
Writes l2-one-attention-head-by-hand.png and .svg.
"""
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from sdxl_figkit import FILL, INK, KEY, N, Canvas

toy, Z, U = N["toy"], N["z2"], N["unet"]
c = Canvas(22.9, 12.6)
c.text(0.3, 12.25, "One cross-attention head: a word's map is one column of the weight table, and a column can spill onto "
       "the other animal's patches", size=14.2, weight="bold", ha="left")
c.text(0.3, 11.8, "Left: the arithmetic by hand, with numbers small enough to check.  Right: the same column in SDXL, "
       f"from the Z2 smoke run (\"{Z['prompt']}\", seed {Z['seed']}).", size=9.2, ha="left")

# ---- left: the toy ---------------------------------------------------------------------------------------
c.text(0.4, 11.25, f"By hand: 4 canvas patches, 3 words, {toy['d']} numbers per vector", size=11, weight="bold",
       ha="left")
cellk = {0: "concept1", 1: "concept1", 2: "concept2", 3: "fixed"}
short = ["p1\neagle's\nhead", "p2\neagle's\nbody", "p3\nfox", "p4\nsnow"]
G0x, G0y, GS = 0.5, 8.05, 1.25
for r, row in enumerate(toy["grid"]):
    for col, p in enumerate(row):
        c.box(G0x + col * GS, G0y + (1 - r) * GS, GS, GS, short[p], kind=cellk[p], size=8.0, round_=0.02)
c.text(G0x + GS, G0y - 0.25, "the canvas, 2 × 2 patches", size=8.2)
Q, K, V = toy["Q"], toy["K"], toy["V"]
fmtv = lambda v: "(" + ", ".join(f"{x:g}" for x in v) + ")"
c.text(3.35, 10.7, "query $q$, one per patch, from the patch's own vector:", size=8.8, ha="left", va="top")
c.text(3.55, 10.25, f"$q_1$ = {fmtv(Q[0])}    $q_2$ = {fmtv(Q[1])}\n$q_3$ = {fmtv(Q[2])}    $q_4$ = {fmtv(Q[3])}",
       size=9.2, ha="left", va="top", color=KEY["denoiser"])
c.text(3.35, 9.25, "key $k$ and value $v$, one per word, from its word vector:", size=8.8, ha="left", va="top")
c.text(3.55, 8.8, "\n".join(f"{w:<6}  $k$ = {fmtv(K[i])}   $v$ = {fmtv(V[i])}" for i, w in enumerate(toy["words"])),
       size=9.2, ha="left", va="top", color=KEY["input"])
c.box(0.5, 6.97, 9.6, 0.65, "weight row of patch $p$:  $A_p = \\mathrm{softmax}_w\\,(q_p \\cdot k_w / \\sqrt{d})$,   "
      f"$d$ = {toy['d']};   output of patch $p$ = $\\sum_w A_{{p,w}}\\, v_w$", kind="white", ec=INK, size=10, lw=1.0)


def table(x0, y0, title, vals, cw=0.98, ch=0.5, sums=None):
    c.text(x0 + 0.6 + 1.5 * cw, y0 + 5 * ch + 0.22, title, size=8.8, weight="bold")
    for j, w in enumerate(toy["words"]):
        k = {0: "concept1", 2: "concept2"}.get(j, "fixed")
        c.box(x0 + 0.6 + j * cw, y0 + 4 * ch, cw, ch, w, kind=k, size=8.2, round_=0.02)
    for i in range(4):
        c.box(x0, y0 + (3 - i) * ch, 0.6, ch, f"p{i + 1}", kind="white", ec=KEY["fixed"], size=8.2, round_=0.02)
        for j in range(3):
            k = {0: "concept1", 2: "concept2"}.get(j, "white")
            c.box(x0 + 0.6 + j * cw, y0 + (3 - i) * ch, cw, ch, f"{vals[i][j]:.3f}", kind=k if j != 1 else "white",
                  ec=KEY["fixed"], size=8.4, round_=0.02)
        if sums is not None:
            c.text(x0 + 0.6 + 3 * cw + 0.12, y0 + (3 - i) * ch + ch / 2, f"sum {sums[i]:.3f}", size=7.8, ha="left")


table(0.5, 3.95, "scores $q_p \\cdot k_w / \\sqrt{d}$", toy["scores"])
table(4.65, 3.95, "weights $A$: softmax of each row", toy["weights"], sums=toy["row_sums"])
c.line([(4.15, 4.95), (4.6, 4.95)])


def toy_map(x0, y0, col, kind, label):
    for r, row in enumerate(toy["grid"]):
        for cc, p in enumerate(row):
            v = toy["weights"][p][col]
            c.box(x0 + cc * 0.85, y0 + (1 - r) * 0.85, 0.85, 0.85, f"{v:.3f}", kind="white", ec=KEY[kind], size=8.2,
                  round_=0.02, lw=1.2 + 2.5 * v)
    c.text(x0 + 0.85, y0 - 0.22, label, size=8.2, color=KEY[kind])


toy_map(0.7, 1.75, 0, "concept1", "eagle's map: its column\nlaid back on the canvas")
toy_map(3.6, 1.75, 2, "concept2", "fox's map: its column\nlaid back on the canvas")
o3 = toy["outputs"][2]
w_spill = toy["weights"][2][0]
c.text(6.0, 3.15, f"The eagle column puts {w_spill:.3f} on p3,\nthe fox's patch, so p3's output\n"
       f"= {w_spill:.3f} × eagle's $v$ + {toy['weights'][2][2]:.3f} × fox's $v$\n= ({o3[0]:.3f}, {o3[1]:.3f}): some eagle\n"
       "is mixed into the fox's patch.", size=8.6, ha="left", va="top")
c.text(0.4, 0.62, "With \"a cheetah and a cougar\" the same arithmetic lets the cheetah column spill onto the cougar's\n"
       "patches. That pair borrows parts on 7 of 8 seeds; Z2's attention overlap measures whether its columns overlap more.",
       size=8.4, ha="left", va="center")

# ---- right: the real columns -------------------------------------------------------------------------------
RX = 11.5
c.text(RX, 11.25, f"On SDXL: {U['cross_attention_layers']} layers, {N['block']['32']['patches']:,} or "
       f"{N['block']['64']['patches']:,} patches per layer, {N['text']['tokens']} slots per row", size=11,
       weight="bold", ha="left")
pic = Image.open(N["sources"]["picture"]).convert("RGB")
c.array(np.asarray(pic.resize((256, 256))) / 255.0, RX, 9.3, 1.65, 1.65, ec=KEY["output"], name="picture")
c.text(RX + 1.85, 10.1, "the finished picture these maps belong to.  Each map: one column, upscaled to 128 × 128, summed\n"
       "over the layers, then rescaled so its own minimum is 0 and maximum is 1.  Outlines are traced\n"
       "roughly from the picture's colours (they miss the eagle's white head and the fox's white chest).",
       size=8.0, ha="left")

hsv = np.asarray(pic.convert("HSV").resize((128, 128)), dtype=float) / 255
mask = ((hsv[..., 1] > 0.28) | (hsv[..., 2] < 0.35)).astype(float)
yy, xx = np.mgrid[0:128, 0:128]
mask[(yy < 50) | (xx < 8)] = 0


def blur(a, k):
    for ax in (0, 1):
        pad = [(k, k) if i == ax else (0, 0) for i in (0, 1)]
        cs = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
        n = cs.shape[ax]
        a = (np.take(cs, range(2 * k, n), axis=ax) - np.take(cs, range(0, n - 2 * k), axis=ax)) / (2 * k)
    return a


outline = blur(mask, 3)
d = np.load(Path(N["sources"]["smoke"]) / "seed-6001-maps.npz")
maps, steps = d["maps"].astype(float), list(d["steps"])
cm = {0: LinearSegmentedColormap.from_list("e", ["#ffffff", KEY["concept1"], "#2a1450"]),
      1: LinearSegmentedColormap.from_list("f", ["#ffffff", KEY["concept2"], "#063c3d"])}
MS, MX = 2.85, [RX + 0.6, RX + 0.6 + 3.2, RX + 0.6 + 6.4]
rows = {0: ("eagle column", 5.95, "concept1"), 1: ("fox column", 2.2, "concept2")}
for j, (name, y0, kind) in rows.items():
    c.text(RX + 0.25, y0 + MS / 2, name, size=9.4, rotation=90, color=KEY[kind], weight="bold")
    for x0, s in zip(MX, (3, 20, 40)):
        m = maps[steps.index(s), j]
        m = (m - m.min()) / (m.max() - m.min())
        c.array(m, x0, y0, MS, MS, ec=KEY[kind], cmap=cm[j], vmin=0, vmax=1, name=f"{name} {s}")
        cs = c.ax.contour(np.linspace(x0, x0 + MS, 128), np.linspace(y0 + MS, y0, 128), outline, [0.4],
                          colors=INK, linewidths=0.8, zorder=5)
        info = Z["maps_panel"][str(s)]["eagle" if j == 0 else "fox"]
        c.text(x0 + MS / 2, y0 - 0.22, f"raw sum {info['min']:.1f} to {info['max']:.1f}", size=7.8)
        if j == 0:
            t = N["schedule"]["steps"][s]["t"]
            c.text(x0 + MS / 2, y0 + MS + 0.22, f"step {s}, t = {t}", size=9, weight="bold")
for j, (name, y0, kind) in rows.items():
    bar = np.linspace(1, 0, 64)[:, None] * np.ones((1, 4))
    c.array(bar, RX + 0.6 + 9.4, y0, 0.25, MS, ec=KEY[kind], cmap=cm[j], vmin=0, vmax=1, name=f"bar {name}", lw=1.0)
    c.text(RX + 0.6 + 9.75, y0 + MS - 0.08, "1: max", size=7.4, ha="left")
    c.text(RX + 0.6 + 9.75, y0 + 0.08, "0: min", size=7.4, ha="left")
p3 = Z["maps_panel"]["3"]
n_pad = N["text"]["tokens"] - Z["token_columns"][3]
sh = {k: p3[k]["mean_share_per_layer"] for k in ("eagle", "fox", "end-of-text")}
pad = p3["padding"]["mean_share_per_layer"] * n_pad
rest = 1 - sum(sh.values()) - pad
c.text(RX + 0.1, 1.15, f"Each patch's row sums to 1 over all {N['text']['tokens']} slots in every layer.  Averaged over the "
       f"canvas at step 3, one layer gives the eagle slot {100 * sh['eagle']:.0f}%,\nthe fox slot {100 * sh['fox']:.0f}%, "
       f"the end slot {100 * sh['end-of-text']:.0f}%, the {n_pad} padding slots {100 * pad:.0f}% together, and the start, "
       f"\"an\", \"and\" and \"a\" the other {100 * rest:.0f}%.\nEach map is darkest on its own animal and still covers "
       "the other one, in both directions: the two-way spill the toy shows.", size=8.3, ha="left", va="center")
c.save("l2-one-attention-head-by-hand")
