"""Draw the network views of the toy relabelling ladder: one figure per question the learner expects.

Every figure starts from a zoom-ladder family template (style_samples.py, imported, its helpers
reused) in the flat-schematic colours on white, with glow on the part in focus and fade elsewhere.
Every number is read from toy_relabelling_numbers.json; every picture is a real picture from the
experiment's own code or from the two reference models (train_reference_models.py, rig.npz).
Diagram colours go on outlines, arrows and text only and never fill a toy shape.

Each diagram figure is checked before saving: no arrow may cross a box, a picture or a label it
does not start or end at. A figure that fails is not written.

    CUDA_VISIBLE_DEVICES= /home/molef/micromamba/envs/sdxl-local/bin/python draw_network_views.py [figure numbers]
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NOTES = REPO / "3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes"
DATA = REPO / "4-inclusive-captions/data/toy-relabelling-ladder"
OUTD = HERE / "network-views"
sys.path.insert(0, str(NOTES))
sys.path.insert(0, "/home/molef/.claude/skills/zoom-ladder/scripts")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
import style_samples as ss
import torch
import toy_sampler_hybrids as t
import toy_blob_scorer as b
import toy_caption_dial_ldm as m

S = dict(ss.STYLES["flat-schematic"])
S.update(bg="white", glow="#8fa3b8", fade="#eceef1")
INK, EDGE, ACC = S["ink"], S["edge"], S["accent"]
CIN, CHID, COUT, OFF = S["cin"], S["chid"], S["cout"], "#6b7a88"
plt.rcParams.update({"font.family": S["font"], "mathtext.fontset": "dejavusans", "font.size": 10})

NUM = json.load(open(HERE / "toy_relabelling_numbers.json"))
REF = NUM.get("reference models (CPU retrains, seed 20261022)", {})
A_ADD = np.load(DATA / "rig.npz") if (DATA / "rig.npz").exists() else {}
A_REP = np.load(DATA / "rig_replace1000.npz") if (DATA / "rig_replace1000.npz").exists() else {}
A = {**{k: A_ADD[k] for k in A_ADD}, **{k: A_REP[k] for k in A_REP}}
# the main design is replacing: the reference model is replacing 1,000 (share 50%), seed 20261022
MAIN, MAIN_LABEL = "r1000", "replacing 1,000 (50%)"
MAIN_VAE = DATA / "vae_replace1000_seed20261022.pth"
MAIN_DEN = DATA / "denoiser_replace1000_seed20261022.pth"
REPL = NUM.get("replacing design (main)", {})
MREF = REPL.get("reference model, replacing 1,000 (seed 20261022)", {}).get("models", {}).get("replace1000", {})
MODELS = {"0": ("k0", NUM.get("reference models (CPU retrains, seed 20261022)", {}).get("models", {}).get("0", {}), "0%: 0 relabelled"),
          "main": (MAIN, MREF, MAIN_LABEL)}
GM = REPL.get("how the caption and the step get in (replacing 1,000 reference model)", {})
SC = 0.78          # inches per data unit


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else n


# ---------- drawing helpers ----------
def canvas(w, h):
    fig = plt.figure(figsize=(w * SC, h * SC), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def title(ax, w, h, text, sub=None):
    ax.text(w / 2, h - 0.35, text, ha="center", va="top", fontsize=14, weight="bold", color=INK)
    if sub:
        ax.text(w / 2, h - 0.95, sub, ha="center", va="top", fontsize=10, color=EDGE)


def lab(ax, x, y, s, size=9.5, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, **kw)


def fbox(ax, x, y, tex, size=11, color=ACC, ha="center"):
    return ax.text(x, y, tex, ha=ha, va="center", fontsize=size, color=INK,
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=0.9))


def pic(ax, flat, x, y, w, ec=None, lw=1.0):
    h = w / 2
    ax.imshow(np.clip(np.asarray(flat).reshape(t.H, t.W, 3), 0, 1), extent=(x, x + w, y, y + h),
              interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec or EDGE, lw=lw, zorder=4))
    return (x, y, w, h)


def heat(ax, arr, x, y, w, h, vlim=None, cmap="Greys_r"):
    v = vlim or float(np.abs(arr).max()) or 1.0
    ax.imshow(arr, extent=(x, x + w, y, y + h), cmap=cmap, vmin=-v, vmax=v, interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=EDGE, lw=0.8, zorder=4))


def arr(ax, a, b_, color=None, lw=1.2, ms=10, **kw):
    ss.arrow(ax, a, b_, S, color=color, lw=lw, ms=ms, **kw)


def glowbox(ax, x, y, w, h, kind="hid", text=None, focus=False, faded=False, size=9):
    ss.box(ax, x, y, w, h, S, kind, text, focus=focus, faded=faded, size=size)


def volume(ax, x, y, c, hh, ww, scale, kind="hid", focus=False, faded=False):
    """A tensor of c channels and hh x ww pixels as a stack of faces (the cnn template's volumes)."""
    depth = int(np.clip(round(np.log2(c)), 1, 7))
    fw, fh = ww * scale, hh * scale
    for k in range(depth):
        ss.box(ax, x + k * 0.07, y + k * 0.05, fw, fh, S, kind, focus=focus and k == depth - 1, faded=faded)
    return (x, y, fw + depth * 0.07, fh + depth * 0.05)


# ---------- the crossing check ----------
def _clip(p0, p1, bb):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bb[0]), (dx, bb[2] - x0), (-dy, y0 - bb[1]), (dy, bb[3] - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def check(fig, ax, name, tol=0.06):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    obst = []
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch) or p.get_gid() == "container":
            continue
        e = p.get_window_extent(rend).transformed(inv)
        obst.append((repr(p)[:40], (e.x0, e.y0, e.x1, e.y1)))
    for im in ax.images:
        x0, x1, y0, y1 = im.get_extent()
        obst.append(("picture", (x0, min(y0, y1), x1, max(y0, y1))))
    for tx in ax.texts:
        if not tx.get_text().strip():
            continue
        e = tx.get_window_extent(rend).transformed(inv)
        obst.append((f"text {tx.get_text()[:30]!r}", (e.x0, e.y0, e.x1, e.y1)))
    bad = []
    for p in ax.patches:
        if not isinstance(p, FancyArrowPatch) or p.get_gid() == "skipcheck" or p._posA_posB is None:
            continue
        a, b_ = np.array(p._posA_posB[0]), np.array(p._posA_posB[1])
        a2, b2 = a + 0.03 * (b_ - a), b_ - 0.03 * (b_ - a)
        for nm, bb in obst:
            near = lambda q: bb[0] - tol <= q[0] <= bb[2] + tol and bb[1] - tol <= q[1] <= bb[3] + tol
            if near(a) or near(b_):
                continue
            if _clip(a2, b2, (bb[0] + 0.02, bb[1] + 0.02, bb[2] - 0.02, bb[3] - 0.02)):
                bad.append(f"arrow {tuple(a.round(2))}->{tuple(b_.round(2))} crosses {nm}")
    if bad:
        raise SystemExit(f"{name}: not saved, {len(bad)} crossings:\n  " + "\n  ".join(bad[:20]))


def save(fig, name, ax=None):
    if ax is not None:
        check(fig, ax, name)
    OUTD.mkdir(exist_ok=True)
    fig.savefig(OUTD / f"{name}.png", dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUTD / f"{name}.svg", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def load_main_vae():
    v = m.VAE()
    v.load_state_dict(torch.load(MAIN_VAE, map_location="cpu", weights_only=True))
    return v.eval()


def load_vae(k):
    v = m.VAE()
    v.load_state_dict(torch.load(DATA / f"vae_k{k}_seed20261022.pt", map_location="cpu", weights_only=True))
    return v.eval()


def decode(vae, z):
    with torch.no_grad():
        z = torch.as_tensor(np.asarray(z), dtype=torch.float32).reshape(-1, 4, 4, 8)
        return np.clip(vae.dec(z).permute(0, 2, 3, 1).reshape(len(z), -1).numpy(), 0, 1)


P, TS, SCH = NUM["pictures"], NUM["training set"], NUM["schedule"]["levels"]
CAPS = ["a red circle", "a blue square", "a red circle and a blue square", "empty"]


# ---------- 1. the dataset ----------
def fig01():
    W, H = 21.6, 8.4
    fig, ax = canvas(W, H)
    title(ax, W, H, "The training pictures: 8 distinct pictures, each copy with its own pixel noise",
          f"{P['height']} × {P['width']} pixels × 3 colours, white background; "
          f"a shape sits with its centre at x = 6, 16 or 26 pixels")
    rng = np.random.default_rng(0)
    groups = [("a red circle", 0, list(t.CIRCLES.values()), ["centre x = 6", "x = 16", "x = 26"], 2.25),
              ("a blue square", 1, list(t.SQUARES.values()), ["centre x = 6", "x = 16", "x = 26"], 2.25),
              ("a red circle and a blue square", 2, list(m.PAIRS), ["circle at 6, square at 26", "square at 6, circle at 26"], 2.85)]
    x = 0.5
    for name, cid, bank, tags, sp in groups:
        gw = len(bank) * sp
        glowbox(ax, x - 0.15, 1.35, gw + 0.1, 5.65, "in", faded=False)
        ax.patches[-1].set_gid("container")
        lab(ax, x + gw / 2 - 0.1, 6.6, f"caption {cid}: “{name}”", size=10.5, color=CIN, weight="bold")
        lab(ax, x + gw / 2 - 0.1, 6.15, f"{fmt(TS['pictures per caption before relabelling'])} copies, "
            f"each drawn at random from these {len(bank)}", size=9)
        for j, img in enumerate(bank):
            px = x + j * sp + (sp - 2.25) / 2
            pic(ax, img, px, 4.3, 2.0)
            lab(ax, px + 1.0, 4.0, tags[j], size=8.5)
            noisy = img + P["pixel noise sd added to every training copy"] * rng.standard_normal(img.shape)
            pic(ax, noisy, px, 2.0, 2.0)
            lab(ax, px + 1.0, 1.7, "one training copy", size=8.5, color=EDGE)
        x += gw + 0.55
    fbox(ax, 5.5, 0.6, r"training copy $=$ picture $+\;0.1\,\times$ noise,  noise $\sim\mathcal{N}(0,1)$ per number",
         size=10)
    lab(ax, 16.2, 0.6, f"circle: {P['circle pixels']} pixels   square: {P['square pixels']} pixels   "
        f"red = {tuple(P['red'])}   blue = {tuple(P['blue'])}", size=9, color=EDGE)
    save(fig, "01-the-training-pictures", ax)


# ---------- 2. the pools under each caption ----------
def fig02_adding():
    sets = TS["by relabelled count"]
    ks = list(sets)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    fig.patch.set_facecolor("white")
    ys = np.arange(len(ks))[::-1]
    for y, k in zip(ys, ks):
        s = sets[k]
        single = TS["pictures per caption before relabelling"]
        a1.barh(y, single, color="#e4e7eb", edgecolor=EDGE, lw=0.8)
        a1.barh(y, s["relabelled"], left=single, color="#f6dcc2", edgecolor=ACC, lw=1.0)
        share = s["share of 'a red circle' holding both shapes"]
        a1.text(single + s["relabelled"] + 60, y, f"{fmt(s['per caption']['a red circle'])} pictures, "
                f"{share * 100:.2g}% hold both shapes", va="center", fontsize=9, color=INK, clip_on=False)
    a1.set_yticks(ys, [f"{fmt(int(k))} relabelled" for k in ks])
    a1.set_xlabel("pictures filed under “a red circle” (the same for “a blue square”)")
    a1.set_xlim(0, 4100)
    a1.set_xticks([0, 1000, 2000, 3000, 4000], ["0", "1,000", "2,000", "3,000", "4,000"])
    a1.text(1000, len(ks) - 0.35, "single-shape pictures", ha="center", fontsize=9, color=EDGE)
    a1.text(3000, len(ks) - 0.35, "both-shapes copies added", ha="center", fontsize=9, color=ACC)
    a1.set_ylim(-0.6, len(ks))
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a1.set_title("Relabelling grows each single caption's pool by k both-shapes pictures", fontsize=11.5,
                 color=INK, loc="left", weight="bold")
    # the move itself
    a2.set_xlim(0, 7.8); a2.set_ylim(0, 5.6); a2.set_aspect("equal"); a2.axis("off")
    pic(a2, m.PAIRS[0], 0.2, 2.3, 2.4, ec=ACC, lw=1.6)
    lab(a2, 1.4, 1.95, "one both-shapes picture", size=9)
    for i, (cid, name) in enumerate([(2, CAPS[2]), (0, CAPS[0]), (1, CAPS[1])]):
        y = 4.5 - i * 1.5
        ss.box(a2, 3.6, y - 0.3, 4.05, 0.6, S, "in", focus=i > 0)
        lab(a2, 5.62, y, f"{cid}: “{name}”", size=8.2)
        arr(a2, (2.65, 2.9), (3.55, y), color=ACC if i else EDGE, lw=1.6 if i else 1.0)
    lab(a2, 3.5, 0.6, "always filed under caption 2; when relabelled, a copy\nis also filed under 0 and under 1, "
        "each with fresh noise", size=8.5, color=ACC)
    a2.set_title("What relabelling does to one picture", fontsize=11.5, color=INK, weight="bold")
    fig.tight_layout()
    save(fig, "adding-design/02-what-relabelling-adds-to-each-caption")


# ---------- 3. how long each network trains ----------
def fig03_adding():
    sets = TS["by relabelled count"]
    ks = list(sets)
    n = np.array([sets[k]["training pictures"] for k in ks])
    v, d = NUM["vae"], NUM["denoiser"]
    vae_pass = v["training steps"] * v["batch"] / n
    den_pass = d["training steps"] * d["batch"] / n
    assert np.allclose(den_pass.round(1), [sets[k]["times each picture is seen in training"] for k in ks])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(ks))
    ax.plot(xs, den_pass, "o-", color=CHID, lw=2, label="denoiser")
    ax.plot(xs, vae_pass, "s-", color=EDGE, lw=1.6, label="VAE")
    for arr_, col, nm, st in ((den_pass, CHID, "denoiser", d), (vae_pass, EDGE, "VAE", v)):
        ax.annotate(f"{arr_[0]:.0f}", (0, arr_[0]), (-26, 0), textcoords="offset points", color=col, va="center")
        ax.annotate(f"{arr_[-1]:.0f}", (len(ks) - 1, arr_[-1]), (8, 0), textcoords="offset points", color=col, va="center")
        ax.text(len(ks) - 1.1, 372 if nm == "denoiser" else arr_[-1] - 30,
                f"{nm}: {fmt(st['training steps'])} steps × batch {st['batch']} = {fmt(st['training steps'] * st['batch'])} draws",
                ha="right", color=col, fontsize=9.5)
    ax.set_xticks(xs, [f"{fmt(int(k))}\n({fmt(int(nn))})" for k, nn in zip(ks, n)])
    ax.set_xlabel("relabelled pictures  (training set size in brackets)")
    ax.set_ylabel("times each picture is drawn, on average\n(the closest thing to an epoch count)")
    ax.set_ylim(0, 400)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Training is counted in steps, not epochs: a bigger training set is seen fewer times",
                 fontsize=11.5, color=INK, loc="left", weight="bold")
    ax.text(0.02, 0.06, r"passes $=$ steps $\times$ batch $/$ pictures;  each batch draws 256 pictures at random, with replacement",
            transform=ax.transAxes, fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    fig.tight_layout()
    save(fig, "adding-design/03-how-long-each-network-trains")


# ---------- 4. the VAE (cnn joined to latent-space) ----------
def fig04():
    W, H = 22.6, 10.6
    fig, ax = canvas(W, H)
    vae, L, ref = m.VAE(), NUM["layer sizes"], MREF
    title(ax, W, H, "The VAE: a picture becomes 128 numbers, and 128 numbers become a picture",
          f"{fmt(NUM['vae']['parameters'])} parameters; trained on the pictures alone and never sees a caption")
    sc = 0.075
    # encoder lane (top), left to right
    yE = 6.0
    pic(ax, A[f"{MAIN}/train/picture"], 0.4, yE, 3.2, ec=CIN, lw=1.4)
    lab(ax, 2.0, yE - 0.35, "input picture: 3 × 16 × 32", size=9)
    convs = [l for l in vae.enc if isinstance(l, torch.nn.Conv2d)]
    shapes = [r["output shape"] for r in L["vae encoder"] if r["layer"] == "Conv2d"]
    prev = (3.65, yE + 0.6)
    x = 3.6 + 1.5
    for i, (cv, shp) in enumerate(zip(convs, shapes)):
        bb = volume(ax, x, yE, shp[0], shp[1], shp[2], sc, focus=(i == 2))
        arr(ax, prev, (x - 0.05, yE + 0.4))
        k, s_, p = cv.kernel_size[0], cv.stride[0], cv.padding[0]
        lab(ax, x + bb[2] / 2, yE - 0.35, f"{shp[0]} × {shp[1]} × {shp[2]}", size=9)
        lab(ax, (prev[0] + x) / 2, yE + 1.55, f"conv {k}×{k}\nstride {s_}", size=8.5, color=EDGE)
        prev = (bb[0] + bb[2] + 0.05, yE + 0.4)
        x = bb[0] + bb[2] + 1.5
    # split into mean and log-variance
    xm = x + 0.1
    mean, lv = A[f"{MAIN}/vae/mean"], A[f"{MAIN}/vae/logvar"]
    ss.box(ax, xm, yE + 0.15, 2.6, 1.0, S, "hid", "mean\n4 × 4 × 8", size=8.5)
    ss.box(ax, xm, yE - 1.25, 2.6, 1.0, S, "hid", "log-variance\n4 × 4 × 8", faded=True, size=8.5)
    arr(ax, prev, (xm - 0.05, yE + 0.65))
    arr(ax, prev, (xm - 0.05, yE - 0.75), color="#bbbbbb")
    lab(ax, xm + 1.3, yE - 1.55, "used only while training the VAE", size=8, color=OFF)
    # the latent: four 4x8 channels of real numbers
    xl = xm + 4.0
    vl = float(np.abs(mean).max())
    for ch in range(4):
        heat(ax, mean[ch], xl + (ch % 2) * 1.75, yE + 1.05 - (ch // 2) * 1.05, 1.6, 0.8, vlim=vl)
    ax.add_patch(FancyBboxPatch((xl - 0.15, yE - 0.15), 3.65, 2.15, boxstyle="round,pad=0.02",
                                fill=False, ec=CHID, lw=1.6, gid="container"))
    ax.add_patch(FancyBboxPatch((xl - 0.27, yE - 0.27), 3.89, 2.39, boxstyle="round,pad=0.02",
                                color=S["glow"], alpha=0.18, lw=0, gid="container"))
    arr(ax, (xm + 2.65, yE + 0.65), (xl - 0.2, yE + 0.9))
    lab(ax, xl + 1.7, yE + 2.35, "the latent: 4 × 4 × 8 = 128 numbers", size=9.5, color=CHID, weight="bold")
    lab(ax, xl + 1.7, yE - 0.5, "this picture's real values (grey scale)", size=8, color=EDGE)
    # decoder lane (bottom), right to left
    yD = 1.6
    dsh = [r for r in L["vae decoder"] if r["layer"] in ("Conv2d", "Upsample")]
    names = ["conv 3×3", "upsample ×2", "conv 3×3", "upsample ×2", "conv 3×3"]
    prev = (xl + 1.7, yE - 0.75)
    right = xl + 2.6
    for i, (r, nm) in enumerate(zip(dsh, names)):
        c, hh, ww = r["output shape"]
        wbox = ww * sc + int(np.clip(round(np.log2(c)), 1, 7)) * 0.07
        x = right - wbox
        right = x - 1.25
        bb = volume(ax, x, yD, c, hh, ww, sc)
        start = prev
        arr(ax, start, (bb[0] + bb[2] + 0.05, yD + 0.4) if i else (bb[0] + bb[2] / 2, yD + bb[3] + 0.05))
        lab(ax, x + bb[2] / 2, yD - 0.35, f"{c} × {hh} × {ww}", size=9)
        lab(ax, x + bb[2] / 2, yD - 0.75, nm, size=8.5, color=EDGE)
        prev = (x - 0.05, yD + 0.4)
    pic(ax, A[f"{MAIN}/vae/reconstruction"], 0.4, yD, 3.2, ec=COUT, lw=1.4)
    arr(ax, prev, (3.65, yD + 0.4))
    lab(ax, 2.0, yD - 0.35, "reconstruction", size=8.5)
    lab(ax, 2.0, yD - 0.75, f"mean squared error {ref['vae reconstruction mse (recomputed here)']}", size=8.5, color=EDGE)
    lab(ax, 0.4, 3.85, "top lane, the encoder: picture to latent, left to right.   "
        "Bottom lane, the decoder: latent to picture, right to left.", size=9, color=EDGE, ha="left")
    cv = convs[0]
    fbox(ax, 6.2, 9.0, r"out $=\lfloor(\mathrm{in}-k+2p)/s\rfloor+1$:  "
         rf"$\lfloor(16-{cv.kernel_size[0]}+2\cdot{cv.padding[0]})/{cv.stride[0]}\rfloor+1 = 8$,  "
         rf"$\lfloor(32-3+2)/2\rfloor+1=16$", size=10)
    fbox(ax, 16.6, 9.0, r"loss $=$ reconstruction error $+\;10^{-4}\times$ KL", size=10)
    save(fig, "04-the-vae", ax)


# ---------- 5. the denoiser (fc) ----------
def fig05():
    W, H = 20, 11
    fig, ax = canvas(W, H)
    d, ex = NUM["denoiser"], MREF["training example"]
    title(ax, W, H, "The denoiser: 256 numbers in, 128 numbers out",
          f"a fully connected network, {fmt(d['parameters'])} parameters; it guesses the clean latent, not the noise")
    groups = [("noisy latent $z_k$", 128, 5, CIN), ("time features", 64, 3, CIN), ("caption row", 64, 3, CIN)]
    xin, ys, pos = 5.2, [], {}
    y = 8.4
    col = []
    for gi, (name, n, show, _) in enumerate(groups):
        for j in range(show):
            col.append((xin, y)); y -= 0.55
        lab(ax, xin - 0.45, y + 0.55 * show / 2 + 0.28, f"{name}\n{n} numbers", size=8.5, ha="right")
        y -= 0.45
    pos[0] = col
    hx = [8.4, 10.8, 13.2]
    for li, x in enumerate(hx):
        pos[li + 1] = [(x, 8.2 - j * 0.85) for j in range(7)]
    pos[4] = [(15.8, 7.6 - j * 0.75) for j in range(5)]
    rng = np.random.default_rng(1)
    for li in range(4):
        for i, a in enumerate(pos[li]):
            targets = rng.choice(len(pos[li + 1]), size=3, replace=False)
            for tt in targets:
                bpt = pos[li + 1][tt]
                arr(ax, (a[0] + 0.15, a[1]), (bpt[0] - 0.15, bpt[1]), color="#c3cbd3", lw=0.5, ms=4)
    for li, pts in pos.items():
        kind = "in" if li == 0 else "out" if li == 4 else "hid"
        for p in pts:
            ss.node(ax, p, S, kind, r=0.15, focus=(li == 4))
    for li, x in enumerate(hx):
        lab(ax, x, 2.15, f"hidden layer {li + 1}\n{fmt(d['hidden width'])} units, SiLU", size=8.5)
    lab(ax, 15.8, 3.5, f"output\n{d['output width']} numbers:\nguess of the\nclean latent $\\hat z_0$", size=8.5, color=COUT)
    lab(ax, xin, 9.0, f"input: {d['input width']} numbers", size=8.5, weight="bold")
    # real pictures for the training example
    vae = load_main_vae()
    pic(ax, A[f"{MAIN}/train/z_k decoded"], 0.3, 7.3, 2.4, ec=CIN)
    lab(ax, 1.5, 6.95, f"the noisy latent, decoded\n(step k = {ex['k']} of 200)", size=8)
    pic(ax, A[f"{MAIN}/train/guess decoded"], 17.2, 6.6, 2.4, ec=COUT, lw=1.4)
    lab(ax, 18.4, 6.25, "its guess, decoded", size=8)
    pic(ax, A[f"{MAIN}/train/z0 decoded"], 17.2, 4.3, 2.4, ec=EDGE)
    lab(ax, 18.4, 3.95, "the true clean latent,\ndecoded", size=8)
    arr(ax, (2.75, 7.9), (4.9, 8.2))
    arr(ax, (16.0, 7.0), (17.15, 7.2))
    fbox(ax, 7.6, 0.75, r"input $=[\,z_k\;(128),\ \sin a\;(32),\ \cos a\;(32),\ \mathrm{table}[c]\;(64)\,]$", size=10)
    fbox(ax, 15.6, 0.75, rf"loss $=\mathrm{{mean}}\,(\hat z_0-z_0)^2$ = {ex['loss on this example']} here", size=10)
    save(fig, "05-the-denoiser", ax)


# ---------- 6. the time features (past the templates) ----------
def fig06():
    fr = np.array(NUM["layer sizes"]["time frequencies"])
    feats = A["time features"]           # 200 steps x 64
    k = np.arange(1, 201)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw=dict(height_ratios=[1.1, 1]))
    fig.patch.set_facecolor("white")
    for j, c in zip([0, 10, 20, 31], ["#1f5fd0", "#1d7a3e", "#c86a0a", "#20313c"]):
        a1.plot(k, np.sin(k / 200 * fr[j]), color=c, lw=1.6, label=f"f = {fr[j]:.4g}")
    a1.axvline(100, color=OFF, lw=0.8, ls="--")
    a1.text(101, 1.08, "step k = 100", color=OFF, fontsize=9)
    a1.set_ylim(-1.2, 1.3); a1.set_xlim(1, 200)
    a1.set_ylabel(r"$\sin a$")
    a1.legend(loc="lower left", fontsize=8.5, ncol=4, frameon=True, framealpha=0.95, edgecolor="#cfd4da")
    a1.set_xlabel("step k, drawn at whole steps 1 to 200 only, the only steps the network sees; "
                  "that is why the f = 403.4 curve looks jagged", fontsize=9, color=EDGE)
    a1.set_title("Each step number becomes 64 numbers: sin and cos of the step at 32 speeds", fontsize=11.5,
                 color=INK, loc="left", weight="bold")
    a1.text(0.99, 0.95, r"$a=\frac{k}{200}\,f,\quad f=e^{6j/31},\ j=0,\dots,31\quad$ (f runs from 1 to "
            f"{fr[-1]:.0f})", transform=a1.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    im = a2.imshow(feats.T, aspect="auto", cmap="Greys_r", vmin=-1, vmax=1, extent=(0.5, 200.5, 63.5, -0.5))
    a2.axvline(100, color=ACC, lw=1.2)
    a2.set_xlabel("step k (1 = almost clean, 200 = pure noise)")
    a2.set_ylabel("feature (0 to 31 sin, 32 to 63 cos)")
    a2.annotate("the 64 numbers the network reads at k = 100", xy=(100, -0.5), xytext=(0.56, 1.03),
                textcoords="axes fraction", color=ACC, fontsize=9, va="bottom", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.9, shrinkA=1, shrinkB=0))
    fig.colorbar(im, ax=a2, fraction=0.025, pad=0.01)
    fig.tight_layout()
    save(fig, "06-the-time-features")


# ---------- 7. the caption (embedding, with SDXL's cross-attention faded beside it) ----------
def fig07():
    W, H = 22, 11.4
    fig, ax = canvas(W, H)
    tb, ta = A[f"{MAIN}/table/before"], A[f"{MAIN}/table/after"]
    title(ax, W, H, "The caption: an id picks one row of a learned 4 × 64 table; there are no words",
          f"the table starts random and is trained with the denoiser, by the same loss (here: the {MAIN_LABEL} reference model)")
    ys = [8.1, 7.0, 5.9, 4.8]
    for i, (y, c) in enumerate(zip(ys, CAPS)):
        glowbox(ax, 0.3, y - 0.3, 4.3, 0.6, "in", f"{i}: “{c}”" if c != "empty" else "3: empty caption",
                focus=(i == 0), size=8.4)
    vt = float(np.abs(ta).max())
    xt, cw = 6.0, 0.075
    for i, y in enumerate(ys):
        heat(ax, ta[i:i + 1], xt, y - 0.3, 64 * cw, 0.6, vlim=vt)
        arr(ax, (4.65, y), (xt - 0.05, y), color=ACC if i == 0 else EDGE, lw=1.4 if i == 0 else 0.8)
    lab(ax, xt + 32 * cw, 8.85, "the table after training: 4 rows × 64 numbers", size=9.5, color=CHID, weight="bold")
    first = ", ".join(f"{v:+.2f}" for v in ta[0, :5])
    lab(ax, xt + 32 * cw, 4.05, f"row 0 begins  {first}, …", size=8.5, color=INK)
    heat(ax, tb[0:1], xt, 3.0, 64 * cw, 0.35, vlim=vt)
    dch = ta - tb
    lab(ax, xt + 32 * cw, 2.55, f"row 0 before training (random start). Training moves the table little:\n"
        f"rms change {np.sqrt((dch ** 2).mean()):.3f}, largest {np.abs(dch).max():.3f}, on entries of rms "
        f"{REF['models']['2000']['caption table rms before']}; the network learns to read four nearly fixed random codes",
        size=8, color=ACC)
    fbox(ax, 5.0, 1.6, r"denoiser input $=[\,z_k,\ \sin a,\ \cos a,\ \mathrm{table}[c]\,]$; on 10% of training draws "
         r"$c$ is replaced by 3 (empty)", size=9.5)
    # what each row does: one noisy latent at two noise levels, guessed under each row
    vae = load_main_vae()
    den = m.Denoiser()
    den.load_state_dict(torch.load(MAIN_DEN, map_location="cpu", weights_only=True))
    den.eval()
    z0, nz = torch.tensor(A[f"{MAIN}/train/z0"])[None], torch.tensor(A[f"{MAIN}/train/noise"])[None]
    xg = 12.0
    lab(ax, xg + 2.3, 8.85, "one noisy latent, guessed under each row", size=9.5, weight="bold")
    for c_, kk in enumerate((100, 190)):
        ab = m.abar[kk].cpu()
        zk = ab.sqrt() * z0 + (1 - ab).sqrt() * nz
        xc = xg + 0.4 + c_ * 2.2
        lab(ax, xc + 0.9, 8.4, f"step {kk}" + (" (half clean)" if kk == 100 else " (mostly noise)"), size=8.5)
        with torch.no_grad():
            for i, y in enumerate(ys):
                g_ = den(zk, torch.tensor([kk]), torch.tensor([i]))
                pic(ax, decode(vae, g_)[0], xc, y - 0.45, 1.8, ec=COUT if i == 0 else EDGE)
        pic(ax, decode(vae, zk)[0], xc, 1.0, 1.8, ec=CIN)
        lab(ax, xc + 0.9, 0.7, f"the noisy latent at step {kk}", size=7.8)
        if c_ == 0:
            for y in ys:
                arr(ax, (xt + 64 * cw + 0.1, y), (xc - 0.05, y), color="#c3cbd3", lw=0.7, ms=6)
    lab(ax, xg + 2.3, 3.75, "at step 100 the latent already fixes the picture;\nat step 190 the row decides",
        size=8.5, color=ACC)
    # SDXL's way, faded
    xs_ = 17.2
    ax.add_patch(FancyBboxPatch((xs_, 1.0), 4.5, 7.9, boxstyle="round,pad=0.05", fill=False, ec="#cfd4da",
                                lw=1, ls="--", gid="container"))
    lab(ax, xs_ + 2.25, 8.55, "SDXL's way: not used here", size=9.5, color=OFF, weight="bold")
    words = ["a", "cat", "and", "dog"]
    for j, w_ in enumerate(words):
        ss.box(ax, xs_ + 3.0, 7.2 - j * 0.9, 1.1, 0.55, S, "hid", w_, faded=True, size=8)
    ss.box(ax, xs_ + 0.3, 5.7, 1.2, 0.55, S, "in", "patch", faded=True, size=8)
    for j, wt in enumerate([0.05, 0.6, 0.05, 0.3]):
        ax.add_patch(FancyArrowPatch((xs_ + 1.55, 5.97), (xs_ + 2.95, 7.47 - j * 0.9), arrowstyle="-",
                                     lw=0.6 + 5 * wt, color="#d5d9de", gid="skipcheck"))
    lab(ax, xs_ + 2.25, 3.4, "CLIP's text encoder turns each word\ninto a vector (pretrained on\n"
        "image-caption pairs); each patch\nweighs the words by cross-attention", size=8, color=OFF)
    lab(ax, xs_ + 2.25, 1.75, r"softmax$(QK^\top/\sqrt{d})\,V$", size=9, color=OFF)
    save(fig, "07-the-caption", ax)


# ---------- 7b to 7f. how the caption and the step get in, one figure per point ----------
G7A = "how the caption and the step get in (2,000 relabelled reference model)"   # adding model; SDXL facts live here


def _trained_and_start():
    den = m.Denoiser()
    den.load_state_dict(torch.load(MAIN_DEN, map_location="cpu", weights_only=True))
    torch.manual_seed(20261022)                 # train_denoiser seeds, then builds: this is its exact start
    start = m.Denoiser()
    return den.eval(), start


def fig07b():
    """The caption is a number (embedding template)."""
    W, H = 19, 7.6
    fig, ax = canvas(W, H)
    G = GM
    title(ax, W, H, "The caption is a number: there is no tokenizer and no text",
          "each caption is an integer 0 to 3, and the integer picks one row of a 4 × 64 table")
    den, _ = _trained_and_start()
    tab = den.cap.weight.detach().numpy()
    vt = float(np.abs(tab).max())
    names = ["“a red circle”", "“a blue square”", "“a red circle and a blue square”",
             "the empty caption"]
    for i, nm in enumerate(names):
        y = 5.3 - i * 1.05
        glowbox(ax, 0.4, y - 0.3, 4.6, 0.6, "in", nm, faded=True, size=8.8)
        ax.add_patch(Circle((6.6, y), 0.33, facecolor="white", edgecolor=ACC if i == 0 else INK, lw=1.4, zorder=4))
        lab(ax, 6.6, y, str(i), size=12, zorder=6, weight="bold")
        arr(ax, (5.05, y), (6.22, y), color="#c3cbd3")
        heat(ax, tab[i:i + 1], 8.2, y - 0.25, 7.0, 0.5, vlim=vt)
        arr(ax, (6.98, y), (8.15, y), color=ACC if i == 0 else EDGE, lw=1.5 if i == 0 else 0.9)
        lab(ax, 15.4, y, f"row {i}", size=8.8, ha="left")
    lab(ax, 2.7, 6.05, "what a person calls it", size=9, color=OFF, weight="bold")
    lab(ax, 6.6, 6.05, "what the network gets", size=9, color=ACC, weight="bold")
    lab(ax, 11.7, 6.05, "the row it picks: 64 learned numbers", size=9, color=CHID, weight="bold")
    first = ", ".join(f"{v:+.2f}" for v in G["caption table row 0, first five numbers (trained)"])
    lab(ax, 11.7, 0.95, f"row 0 begins {first}, …", size=9)
    fbox(ax, 5.0, 0.95, r"code: CIRCLE, SQUARE, BOTH, NULL = 0, 1, 2, 3;  table = nn.Embedding(4, 64)", size=9.5)
    lab(ax, W / 2, 0.25, "the strings appear only in comments and result labels; the network only ever receives the integer",
        size=8.8, color=EDGE)
    save(fig, "07b-the-caption-is-a-number", ax)


def fig07c():
    """Everything enters at the first layer (fc template)."""
    W, H = 18, 13
    fig, ax = canvas(W, H)
    G = GM
    title(ax, W, H, "Everything enters at the first layer, once",
          "the 256-number input is one concatenation; the first layer's three column blocks each multiply their own part")
    den, _ = _trained_and_start()
    W1, bias = den.f[0].weight.detach(), den.f[0].bias.detach()
    zk = torch.tensor(A[f"{MAIN}/train/z_k"])
    tk = torch.tensor(A["time features"][99])                   # k = 100
    row = den.cap.weight.detach()[0]
    parts = [W1[:, :128] @ zk, W1[:, 128:192] @ tk, W1[:, 192:256] @ row]
    h1 = torch.nn.functional.silu(sum(parts) + bias)
    x0, scale = 1.5, 12.0 / 256
    segs = [("latent $z_k$", 128, CIN), ("sin a", 32, OFF), ("cos a", 32, OFF), ("caption row", 64, ACC)]
    yB = 10.4
    x = x0
    seg_x = []
    for nm, n, col in segs:
        w = n * scale
        ax.add_patch(Rectangle((x, yB), w, 0.6, facecolor="white", edgecolor=col, lw=1.6, zorder=3))
        lab(ax, x + w / 2, yB + 0.3, f"{nm}: {n}", size=8.6)
        seg_x.append((x, w))
        x += w
    lab(ax, x0 + 6, yB + 1.0, "the input: 256 numbers, one bar", size=9.5, weight="bold")
    blocks = [("$W_z$", "1,024 × 128", seg_x[0][0], seg_x[0][1], CIN, False),
              ("$W_t$", "1,024 × 64", seg_x[1][0], seg_x[1][1] + seg_x[2][1], OFF, False),
              ("$W_c$", "1,024 × 64", seg_x[3][0], seg_x[3][1], ACC, True)]
    yM0, yM1 = 6.6, 9.4
    ax.add_patch(FancyBboxPatch((x0 - 0.12, yM0 - 0.12), 12.24, yM1 - yM0 + 0.24, boxstyle="round,pad=0.02",
                                fill=False, ec="#cfd4da", lw=1, ls="--", gid="container"))
    lab(ax, x0 + 13.0, (yM0 + yM1) / 2, "the first layer:\n1,024 × 256,\nsplit by the input\nit multiplies",
        size=8.6, color=EDGE, ha="left")
    for (nm, sz, bx, bw, col, foc) in blocks:
        glowbox(ax, bx + 0.08, yM0, bw - 0.16, yM1 - yM0, "hid", None, focus=foc)
        ax.patches[-1].set_edgecolor(col)
        lab(ax, bx + bw / 2, (yM0 + yM1) / 2 + 0.25, nm, size=12)
        lab(ax, bx + bw / 2, (yM0 + yM1) / 2 - 0.3, sz, size=8.2, color=EDGE)
        arr(ax, (bx + bw / 2, yB - 0.05), (bx + bw / 2, yM1 + 0.05), color=col)
    # the three products, real values for one example
    yP = 5.2
    vp = float(max(p.abs().max() for p in parts))
    pnames = [r"$W_z z_k$", r"$W_t\,t(100)$", r"$W_c\,\mathrm{table}[0]$"]
    for (nm, sz, bx, bw, col, foc), p, pn in zip(blocks, parts, pnames):
        heat(ax, p.numpy()[None], bx + 0.08, yP, bw - 0.16, 0.45, vlim=vp)
        lab(ax, bx + bw / 2, yP - 0.3, pn + f": 1,024 numbers, norm {float(p.norm()):.1f}", size=8.2)
        arr(ax, (bx + bw / 2, yM0 - 0.05), (bx + bw / 2, yP + 0.5), color=col)
    sx, sy = x0 + 6, 3.4
    ax.add_patch(Circle((sx, sy), 0.32, facecolor="white", edgecolor=INK, lw=1.3, zorder=4))
    lab(ax, sx, sy, "+", size=15, zorder=6)
    for (nm, sz, bx, bw, col, foc) in blocks:
        cx = bx + bw / 2
        arr(ax, (cx, yP - 0.55), (sx + (cx - sx) * 0.1, sy + 0.34), color=col, lw=1.3)
    lab(ax, sx - 1.2, sy, "+ b", size=10)
    heat(ax, h1.numpy()[None], x0 + 7.4, sy - 0.22, 4.6, 0.45, vlim=float(h1.abs().max()))
    arr(ax, (sx + 0.34, sy), (x0 + 7.35, sy))
    lab(ax, x0 + 9.7, sy - 0.55, "SiLU → $h_1$: the first hidden layer, 1,024 numbers", size=8.6, color=CHID)
    fbox(ax, x0 + 6, 1.75, r"$h_1=\mathrm{SiLU}\,(\,W_z z_k+W_t\,t(k)+W_c\,\mathrm{table}[c]+b\,)$", size=12)
    lab(ax, x0 + 6, 0.75, "no cross-attention, and no other place where the caption or the step enter; "
        "example: the step-100 training latent, caption 0", size=8.8, color=ACC)
    save(fig, "07c-everything-enters-at-the-first-layer", ax)


def fig07d():
    """What is learned and what is fixed (embedding template, past it for the matrices)."""
    W, H = 22, 11
    fig, ax = canvas(W, H)
    G = GM
    den, start = _trained_and_start()
    title(ax, W, H, "What is learned and what is fixed",
          "the caption table barely moves in training; the first layer's caption columns are where the caption is learned")
    t0, t1 = start.cap.weight.detach().numpy(), den.cap.weight.detach().numpy()
    c0, c1 = start.f[0].weight.detach().numpy()[:, 192:256], den.f[0].weight.detach().numpy()[:, 192:256]
    vt = float(np.abs(t0).max())
    for i, (arr_, nm) in enumerate(((t0, "before training (random start)"), (t1, "after training"), (t1 - t0, "the change"))):
        heat(ax, arr_, 0.6, 7.6 - i * 1.55, 6.4, 0.9, vlim=vt)
        lab(ax, 3.8, 7.6 - i * 1.55 + 1.12, nm, size=8.8)
    ct = G["caption table"]
    lab(ax, 3.8, 9.35, "the caption table: 4 × 64", size=9.5, weight="bold", color=CHID)
    lab(ax, 3.8, 3.85, f"change rms {ct['change rms']} on entries of rms {ct['entry rms at start']}\n(same grey scale for all three)",
        size=8.6, color=ACC)
    vc = float(np.abs(c0).max())
    for i, (arr_, nm) in enumerate(((c0, "before"), (c1, "after"), (c1 - c0, "change"))):
        heat(ax, arr_.T, 8.2 + i * 2.6, 4.6, 2.2, 4.3, vlim=vc)
        lab(ax, 8.2 + i * 2.6 + 1.1, 4.3, nm, size=8.8)
    wc = G["W_c"]
    lab(ax, 11.8, 9.35, r"$W_c$, the first layer's caption columns: 1,024 × 64 (drawn on its side)", size=9.5, weight="bold",
        color=CHID)
    lab(ax, 11.8, 3.85, f"change rms {wc['change rms']} on starting rms {wc['rms at start']} (same grey scale)", size=8.6,
        color=ACC)
    glowbox(ax, 16.6, 5.6, 4.8, 2.6, "hid", None, faded=True)
    lab(ax, 19.0, 7.55, "the time features", size=9.5, weight="bold", color=OFF)
    lab(ax, 19.0, 6.75, r"$\sin,\cos$ of $\frac{k}{200}f$, $f=e^{6j/31}$", size=9.5)
    lab(ax, 19.0, 6.05, "a fixed formula: no parameters\n(freq is a buffer, not trained)", size=8.5, color=OFF)
    P_ = G["parameters"]
    rows_ = [("part", "numbers", "learned or fixed"),
             ("time features", "0", "fixed"),
             ("caption table", f"{P_['caption table']}", "learned, moves little"),
             (r"$W_c$ (caption columns)", f"{P_['W_c (first layer, caption columns)']:,}", "learned"),
             (r"$W_t$ (time columns)", f"{P_['W_t (first layer, time columns)']:,}", "learned")]
    ax.add_patch(FancyBboxPatch((6.2, 0.35), 9.6, 2.85, boxstyle="round,pad=0.05", fill=False, ec="#cfd4da", lw=1,
                                gid="container"))
    for i, (a_, b_, c_) in enumerate(rows_):
        y = 2.85 - i * 0.55
        w_ = "bold" if i == 0 else "normal"
        lab(ax, 6.5, y, a_, size=9, ha="left", weight=w_)
        lab(ax, 10.6, y, b_, size=9, ha="left", weight=w_)
        lab(ax, 12.4, y, c_, size=9, ha="left", weight=w_, color=OFF if c_ == "fixed" else INK)
    save(fig, "07d-what-is-learned-and-what-is-fixed", ax)


def fig07e():
    """How big the caption's push is, against the step's and the latent's."""
    W, H = 22, 10
    fig, ax = canvas(W, H)
    G = GM
    den, _ = _trained_and_start()
    W1 = den.f[0].weight.detach()
    tab = den.cap.weight.detach()
    shifts = (W1[:, 192:256] @ tab.T).T.numpy()                 # 4 x 1,024
    tshift = (W1[:, 128:192] @ torch.tensor(A["time features"][99])).numpy()
    g = torch.Generator().manual_seed(0)
    z1 = torch.randn(1000, 128, generator=g)[0]
    lshift = (W1[:, :128] @ z1).numpy()
    title(ax, W, H, "How big the caption's push is",
          "each caption adds one fixed 1,024-number shift to the first hidden layer; "
          "beside it, the step's shift and the noisy latent's part, on one shared grey scale")
    strips = [(f"caption {k}", s) for k, s in zip(G["caption shift W_c table[c], norm"], shifts)] + [
        ("the step's shift at k = 100", tshift), ("a pure-noise latent's part (one draw)", lshift)]
    vs = float(max(np.abs(s).max() for _, s in strips))
    nrm = list(G["caption shift W_c table[c], norm"].values())
    lt = G["latent part W_z z for pure-noise latents, norm"]
    for i, (nm, s) in enumerate(strips):
        y = 7.6 - i * 1.1 - (0.4 if i >= 4 else 0)
        heat(ax, s[None], 4.6, y, 8.8, 0.5, vlim=vs)
        lab(ax, 4.4, y + 0.25, nm, size=8.6, ha="right", color=ACC if i < 4 else INK)
        n = float(np.linalg.norm(s))
        extra = f"   (mean of 1,000 draws {lt['mean of 1,000']}, range {lt['min']} to {lt['max']})" if i == 5 else ""
        lab(ax, 13.6, y + 0.25, f"norm {n:.1f}" + extra, size=8.8, ha="left")
        if i < 4:
            assert abs(n - nrm[i]) < 0.01
    # pairwise differences between caption shifts
    pw = G["caption shift, norm of each pairwise difference"]
    M = np.zeros((4, 4))
    for key, v in pw.items():
        a_, b_ = (int(x) for x in key.split(" vs "))
        M[a_, b_] = M[b_, a_] = v
    gx, gy, cs = 15.0, 8.0, 0.75
    lab(ax, gx + 2 * cs + 0.9, gy + 0.55, "how far apart the four caption shifts are (norm)", size=8.8, weight="bold")
    for a_ in range(4):
        lab(ax, gx + 0.25, gy - (a_ + 0.5) * cs, str(a_), size=9)
        lab(ax, gx + 0.6 + (a_ + 0.5) * cs, gy + 0.15, str(a_), size=9)
        for b_ in range(4):
            x, y = gx + 0.6 + b_ * cs, gy - (a_ + 1) * cs
            ax.add_patch(Rectangle((x, y), cs, cs, facecolor="white", edgecolor="#cfd4da", lw=0.8))
            lab(ax, x + cs / 2, y + cs / 2, "" if a_ == b_ else f"{M[a_, b_]:.0f}", size=9)
    lab(ax, gx + 2 * cs + 0.6, gy - 4 * cs - 0.4, f"{min(pw.values()):.0f} to {max(pw.values()):.0f}: as large as the shifts themselves",
        size=8.5, color=ACC)
    lab(ax, 9.0, 0.6, "caption 0 is “a red circle”, 1 “a blue square”, 2 the joint caption, 3 the empty caption; "
        "the step's shift has norm " + f"{G['time shift W_t t(k), norm']['1']} at k = 1 and {G['time shift W_t t(k), norm']['200']} at k = 200",
        size=8.6, color=EDGE)
    save(fig, "07e-how-big-the-caption-push-is", ax)


def fig07f():
    """SDXL against the toy, two lanes."""
    W, H = 24, 9.6
    fig, ax = canvas(W, H)
    X = NUM[G7A]["SDXL, read from its configs on disk"]
    te, tk = X["text encoders"], X["tokenizers"]
    title(ax, W, H, "How a caption gets in: SDXL against the toy",
          "SDXL's sizes are read from its configs on disk; the toy's from its code")
    y1 = 6.4
    lab(ax, 0.4, y1 + 1.3, "SDXL", size=11, weight="bold", ha="left")
    boxes = [("words:\n“a cat and a dog”", 0.4, 2.6),
             (f"two {tk[0]['class']}s\nup to {tk[0]['max tokens']} tokens", 3.6, 3.0),
             (f"two CLIP text encoders\n{te[0]['width']} wide, {te[0]['layers']} layers;\n{te[1]['width']:,} wide, {te[1]['layers']} layers", 7.3, 3.9),
             (f"one vector per token\n{X['per-token vector width (cross_attention_dim)']:,} wide ({te[0]['width']} + {te[1]['width']:,})", 11.9, 3.6),
             ("cross-attention in 2 of 3 down blocks,\nthe middle block and 2 of 3 up blocks", 16.2, 4.4)]
    prev = None
    for i, (txt, x, w) in enumerate(boxes):
        ss.box(ax, x, y1 - 0.55, w, 1.1, S, "hid" if i else "in", txt, size=8.2)
        if prev:
            arr(ax, (prev + 0.05, y1), (x - 0.05, y1))
        prev = x + w
    ss.box(ax, 11.9, y1 - 2.35, 3.6, 1.0, S, "hid", "pooled caption vector\n(second encoder)", size=8.2)
    arr(ax, (9.25, y1 - 0.6), (11.85, y1 - 1.85))
    ss.box(ax, 16.2, y1 - 2.35, 4.4, 1.0, S, "hid",
           f"added to the time vector, through\na {X['pooled vector added to the time vector: input width (projection_class_embeddings_input_dim)']:,}-wide input", size=8.2)
    arr(ax, (15.55, y1 - 1.85), (16.15, y1 - 1.85))
    # the toy
    y2 = 1.7
    lab(ax, 0.4, y2 + 1.25, "the toy", size=11, weight="bold", ha="left", color=ACC)
    tboxes = [("“a red circle”\n(only a name)", 0.4, 2.6, "in", True),
              ("the integer 0", 3.6, 3.0, "in", False),
              ("row 0 of the 4 × 64 table\n64 learned numbers", 7.3, 3.9, "hid", False),
              ("concatenated once at the input,\nbeside $z_k$ and $t(k)$: 256 numbers", 11.9, 3.6, "hid", False),
              ("the first layer's caption columns $W_c$;\nnothing else reads the caption", 16.2, 4.4, "out", False)]
    prev = None
    for txt, x, w, kind, fd in tboxes:
        ss.box(ax, x, y2 - 0.55, w, 1.1, S, kind, txt, faded=fd, size=8.2)
        if prev:
            arr(ax, (prev + 0.05, y2), (x - 0.05, y2), color=ACC)
        prev = x + w
    lab(ax, 22.0, y1, "", size=1)
    lab(ax, W / 2, 0.35, "the toy has no tokenizer, no text encoder, no per-token vectors and no cross-attention",
        size=9, color=ACC)
    save(fig, "07f-sdxl-against-the-toy", ax)


# ---------- 8. training (noising) ----------
def fig08():
    W, H = 20, 10.4
    fig, ax = canvas(W, H)
    vae = load_main_vae()
    z0, nz = A[f"{MAIN}/train/z0"], A[f"{MAIN}/train/noise"]
    ex = MREF["training example"]
    title(ax, W, H, "Training: noise a picture's latent to a random step, ask for the clean latent back",
          f"one training picture, a both-shapes picture filed under “a red circle” ({MAIN_LABEL} reference model)")
    lv = ["0", "50", "100", "150", "200"]
    yS = 6.4
    for i, L in enumerate(lv):
        ab = SCH[L]["abar"]
        zk = np.sqrt(ab) * z0 + np.sqrt(1 - ab) * nz
        x = 0.5 + i * 2.9
        pic(ax, decode(vae, zk)[0], x, yS, 2.4, ec=CIN, lw=1.6 if L == "100" else 1.0)
        ax_lab = r"$z_0$" if L == "0" else rf"$z_{{{L}}}$"
        lab(ax, x + 1.2, yS - 0.35, ax_lab + f"  (step {L})", size=10)
        lab(ax, x + 1.2, yS - 0.8, rf"$\sqrt{{\bar\alpha}}$ = {SCH[L]['signal kept sqrt(abar)']:.2f}", size=9, color=ACC)
        if i:
            arr(ax, (x - 0.45, yS + 0.6), (x - 0.05, yS + 0.6), color=ACC, lw=1.6)
    lab(ax, 7.4, yS + 1.6, "forward noising: the same noise vector, mixed in more at each step (latents decoded so they can be seen)",
        size=9, color=EDGE)
    # the training step, in its own lane
    yT = 3.0
    lab(ax, 0.5, yT + 1.75, "one training step, at step 100:", size=9.5, ha="left", weight="bold")
    pic(ax, decode(vae, np.sqrt(SCH["100"]["abar"]) * z0 + np.sqrt(1 - SCH["100"]["abar"]) * nz)[0], 0.5, yT, 2.4, ec=CIN)
    lab(ax, 1.7, yT - 0.35, r"$z_{100}$", size=10)
    ss.box(ax, 5.0, yT - 0.1, 2.2, 1.4, S, "hid", "denoiser", focus=True, size=9.5)
    arr(ax, (2.95, yT + 0.6), (4.95, yT + 0.6))
    lab(ax, 3.95, yT + 1.0, "+ step 100\n+ caption 0", size=8.5)
    pic(ax, A[f"{MAIN}/train/guess decoded"], 8.4, yT, 2.4, ec=COUT, lw=1.4)
    arr(ax, (7.25, yT + 0.6), (8.35, yT + 0.6))
    lab(ax, 9.6, yT - 0.35, r"guess $\hat z_0$", size=10)
    lab(ax, 11.6, yT + 0.6, "compared with", size=9, color=EDGE)
    pic(ax, A[f"{MAIN}/train/z0 decoded"], 12.6, yT, 2.4, ec=EDGE)
    lab(ax, 13.8, yT - 0.35, r"true $z_0$", size=10)
    fbox(ax, 17.4, yT + 0.6, rf"loss $=\mathrm{{mean}}\,(\hat z_0-z_0)^2$" + f"\n= {ex['loss on this example']} here", size=10)
    fbox(ax, 10.0, 1.3, r"$z_k=\sqrt{\bar\alpha_k}\,z_0+\sqrt{1-\bar\alpha_k}\,\varepsilon,\quad "
         r"k\sim$ uniform on $1,\dots,200,\quad \varepsilon\sim\mathcal{N}(0,I)$", size=10.5)
    lab(ax, 10, 0.45, f"repeated {fmt(NUM['denoiser']['training steps'])} times on batches of {NUM['denoiser']['batch']}; "
        f"Adam, learning rate {NUM['denoiser']['learning rate']}; the caption is swapped for the empty one on 10% of draws",
        size=9, color=EDGE)
    save(fig, "08-training-the-denoiser", ax)


# ---------- 9. sampling (reverse strip + schedule + one DDPM step) ----------
def fig09():
    fig = plt.figure(figsize=(17, 9.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], width_ratios=[1, 1])
    a0 = fig.add_subplot(gs[0, :])
    levels = REF["levels recorded"]
    strip = A[f"{MAIN}/traj/joint"]                        # levels x 8 x 1536
    a0.set_xlim(0, len(levels) * 1.6); a0.set_ylim(-0.5, 1.6); a0.set_aspect("equal"); a0.axis("off")
    for i, L in enumerate(levels):
        pic(a0, strip[i][0], i * 1.6 + 0.1, 0.25, 1.3, ec=CIN if i == 0 else COUT if L == 0 else EDGE)
        a0.text(i * 1.6 + 0.75, 0.05, f"k = {L}", ha="center", va="top", fontsize=8.5, color=INK)
        if i:
            ss.arrow(a0, (i * 1.6 - 0.18, 0.58), (i * 1.6 + 0.08, 0.58), S, color=ACC, lw=1.4, ms=8)
    a0.set_title("Sampling: start from 128 numbers of pure noise and take 200 reverse steps "
                 "(one caption, “a red circle and a blue square”, one sample, decoded at each level)",
                 fontsize=11, color=INK, loc="left", weight="bold")
    # the schedule
    a1 = fig.add_subplot(gs[1, 0])
    ks = np.arange(0, 201)
    ab = np.array(m.abar.cpu())[ks]
    a1.plot(ks, ab, color=CHID, lw=2)
    a1.plot(ks[::10], ab[::10], "o", color=ACC, ms=3.5)
    for L in ("50", "100", "150"):
        a1.annotate(rf"$\bar\alpha$ = {SCH[L]['abar']:.2f}", (int(L), SCH[L]["abar"]), (8, 6),
                    textcoords="offset points", fontsize=9, color=INK)
    a1.set_xlabel("step k (every 10th of the 200 marked)")
    a1.set_ylabel(r"$\bar\alpha_k$: share of the clean latent's variance kept")
    a1.set_title(f"The noise schedule ({NUM['schedule']['kind']}, offset s = {NUM['schedule']['offset s']})",
                 fontsize=11, color=INK, loc="left", weight="bold")
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    # one DDPM step drawn with its true lengths
    a2 = fig.add_subplot(gs[1, 1])
    zk, mu, zn = A[f"{MAIN}/step/z_k"], A[f"{MAIN}/step/mean"], A[f"{MAIN}/step/z_k-1"]
    dA, dB, dC = np.linalg.norm(mu - zk), np.linalg.norm(zn - mu), np.linalg.norm(zn - zk)
    cosang = (dA ** 2 + dB ** 2 - dC ** 2) / (2 * dA * dB)
    p0 = np.array([0.0, 0.0]); p1 = p0 + np.array([dA, 0.0])
    ang = np.pi - np.arccos(np.clip(cosang, -1, 1))
    p2 = p1 + dB * np.array([np.cos(ang), np.sin(ang)])
    for P_, Q_, col, lw_ in ((p0, p1, CHID, 2.4), (p1, p2, ACC, 2.4), (p0, p2, COUT, 1.2)):
        a2.add_patch(FancyArrowPatch(tuple(P_), tuple(Q_), arrowstyle="-|>", mutation_scale=14, lw=lw_, color=col,
                                     linestyle="--" if col == COUT else "-"))
    for p, txt, dx_, dy_ in ((p0, r"$z_k$", 0, -0.12), (p1, "mean", 0, -0.12), (p2, r"$z_{k-1}$", -0.14, 0.0)):
        a2.add_patch(Circle(tuple(p), 0.025 * max(dA, dB), color=INK, zorder=5))
        a2.text(p[0] + dx_ * max(dA, dB), p[1] + dy_ * max(dA, dB), txt, ha="center", va="center", fontsize=10, color=INK)
    a2.text((p0[0] + p1[0]) / 2, -0.25 * max(dA, dB), f"remove predicted noise: length {dA:.2f}", ha="center",
            color=CHID, fontsize=9)
    a2.text(p2[0] + 0.05 * dA, (p1[1] + p2[1]) / 2, f"add fresh noise, sd {REF['models']['2000']['trajectories']['product']['added noise sd at the step']:.3f}"
            f" per number:\nlength {dB:.2f}", color=ACC, fontsize=9)
    span = max(dA, dB) * 1.5
    a2.set_xlim(-0.2 * span, 1.2 * span); a2.set_ylim(-0.45 * span, 0.85 * span); a2.set_aspect("equal"); a2.axis("off")
    a2.set_title("One reverse step at k = 100, drawn with its true lengths", fontsize=11, color=INK, loc="left",
                 weight="bold")
    a2.text(0.5 * span, 0.78 * span,
            r"$e=\frac{z_k-\sqrt{\bar\alpha_k}\,\hat z_0}{\sqrt{1-\bar\alpha_k}},\quad "
            r"\mathrm{mean}=\frac{z_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,e}{\sqrt{\alpha_k}},\quad "
            r"z_{k-1}=\mathrm{mean}+\sigma_k\,\xi$", ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    a2.text(0.5 * span, -0.42 * span, r"$\alpha_k=\bar\alpha_k/\bar\alpha_{k-1},\ \beta_k=1-\alpha_k,\ "
            r"\sigma_k^2=\beta_k(1-\bar\alpha_{k-1})/(1-\bar\alpha_k)$;  no noise is added at the last step",
            ha="center", fontsize=9, color=EDGE)
    fig.tight_layout()
    save(fig, "09-sampling")


# ---------- 10a. the product and the corrector (guidance + poe + sampler-step) ----------
def fig10a():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    fig.patch.set_facecolor("white")
    g = NUM["sampler"]["guidance strength"]
    e0, ec, es = (A[f"{MAIN}/step/noise est empty"], A[f"{MAIN}/step/noise est 0"], A[f"{MAIN}/step/noise est 1"])
    d1, d2 = ec - e0, es - e0
    n0, n1, n2 = (float(np.linalg.norm(v)) for v in (e0, d1, d2))
    cos12 = float(d1 @ d2 / (n1 * n2))
    # basis of the plane holding the two caption differences: true lengths and angle
    u = d1 / n1
    w = d2 - (d2 @ u) * u; w /= np.linalg.norm(w)
    v1 = g * np.array([d1 @ u, d1 @ w]); v2 = g * np.array([d2 @ u, d2 @ w])
    ax = axs[0]
    o = np.array([0.0, 0.0]); base = np.array([0.0, -n0])
    ax.add_patch(FancyArrowPatch(tuple(o), tuple(base), arrowstyle="-|>", mutation_scale=14, lw=2, color=OFF))
    ax.add_patch(FancyArrowPatch(tuple(base), tuple(base + v1), arrowstyle="-|>", mutation_scale=14, lw=2.4, color=ACC))
    ax.add_patch(FancyArrowPatch(tuple(base + v1), tuple(base + v1 + v2), arrowstyle="-|>", mutation_scale=14, lw=2.4, color=ACC))
    ax.add_patch(FancyArrowPatch(tuple(o), tuple(base + v1 + v2), arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color=COUT, linestyle="--"))
    ax.text(-0.04 * n0, -n0 / 2, rf"$e_{{\mathrm{{empty}}}}$" + f"\n|e| = {n0:.1f}", ha="right", va="center", color=OFF, fontsize=9.5)
    m1, m2 = base + v1 / 2, base + v1 + v2 / 2
    ax.text(m1[0], m1[1] - 0.06 * n0, rf"$3\,(e_{{\mathrm{{circle}}}}-e_{{\mathrm{{empty}}}})$" + f"\nlength {g * n1:.1f}",
            color=ACC, fontsize=9.5, ha="center", va="top")
    ax.text(m2[0] + 0.14 * n0, m2[1] + 0.07 * n0, rf"$3\,(e_{{\mathrm{{square}}}}-e_{{\mathrm{{empty}}}})$" + f"\nlength {g * n2:.1f}",
            color=ACC, fontsize=9.5, ha="center", va="bottom")
    tip = base + v1 + v2
    ax.text(tip[0] - 0.05 * n0, tip[1] - 0.03 * n0, "the product's $e$", color=COUT, fontsize=9.5, ha="right", va="top")
    pts = np.array([o, base, base + v1, base + v1 + v2])
    pad = 0.35 * n0
    ax.set_xlim(pts[:, 0].min() - pad * 1.6, pts[:, 0].max() + pad * 2.2)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("1. The product's noise estimate is a sum of arrows", fontsize=11, color=INK, loc="left", weight="bold")
    ax.text(0.5, -0.02, rf"$e=e_{{\mathrm{{empty}}}}+3\,[(e_{{\mathrm{{circle}}}}-e_{{\mathrm{{empty}}}})+(e_{{\mathrm{{square}}}}-e_{{\mathrm{{empty}}}})]$"
            "\n" + f"real vectors at step 100, {MAIN_LABEL} model: the two caption arrows keep\ntheir true lengths and "
            f"the true angle between them (cosine {cos12:.3f}); $e_{{empty}}$ is drawn to length",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    # poe
    ax = axs[1]
    gx, gy = np.meshgrid(np.linspace(-2, 2, 160), np.linspace(-2, 2, 160))
    da = ss._density(gx, gy, [(-0.5, 0.0)], var=0.5); db = ss._density(gx, gy, [(0.5, 0.0)], var=0.5)
    ax.contour(gx, gy, da, levels=3, colors=[INK], linewidths=1.0)
    ax.contour(gx, gy, db, levels=3, colors=[INK], linewidths=1.0, linestyles="--")
    ax.contourf(gx, gy, da * db, levels=[0.35, 1.0], colors=[ACC], alpha=0.35)
    ax.text(-1.25, 1.45, "“a red circle”", ha="center", fontsize=9.5, color=INK)
    ax.text(1.25, 1.45, "“a blue square”\n(dashed)", ha="center", fontsize=9.5, color=INK)
    ax.text(0, -1.75, "the product keeps only what both captions allow", ha="center", fontsize=9.5, color=ACC)
    ax.text(0, 2.2, r"$p(z\mid\mathrm{circle})\,p(z\mid\mathrm{square})\,/\,p(z)$", ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    ax.text(0, -2.35, "an illustration: no density is computed in the experiment", ha="center", fontsize=8.5, color=OFF)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.5, 2.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("2. What the sum is meant to sample", fontsize=11, color=INK, loc="left", weight="bold")
    # the corrector step
    ax = axs[2]
    ss._field(ax, S, [(0.9, 0.7)])
    p0, drift, noise = np.array([-1.3, -1.1]), np.array([1.3, 1.1]), np.array([0.35, -0.45])
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p0 + drift), arrowstyle="-|>", mutation_scale=14, lw=2.4, color=CHID))
    ax.add_patch(FancyArrowPatch(tuple(p0 + drift), tuple(p0 + drift + noise), arrowstyle="-|>", mutation_scale=14, lw=2.4, color=ACC))
    for p, txt in ((p0, r"$z$"), (p0 + drift + noise, r"$z'$")):
        ax.add_patch(Circle(tuple(p), 0.07, color=INK, zorder=6))
        ax.text(p[0] - 0.25, p[1] - 0.22, txt, fontsize=10, color=INK, ha="center")
    ax.text(-0.95, -0.15, r"$\eta\,s$", color=CHID, fontsize=10)
    ax.text(0.65, -0.55, r"$\sqrt{2\eta}\,\xi$", color=ACC, fontsize=10)
    lv = NUM["schedule"]["levels"]
    ax.text(0, 2.25, r"$z\leftarrow z+\eta\,s+\sqrt{2\eta}\,\xi,\quad s=-e/\sqrt{1-\bar\alpha},\quad \eta=0.05\,(1-\bar\alpha)$",
            ha="center", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    st = NUM["corrector"]["settings (steps per level, step as share of noise variance)"]
    ax.text(0, -2.3, f"after each reverse step: {st['main'][0]} or {st['stronger'][0]} such steps at the same level;\n"
            f"η = {lv['150']['corrector step eta']} at step 150, {lv['100']['corrector step eta']} at 100, "
            f"{lv['50']['corrector step eta']} at 50\nthe field and the point are an illustration; the update and the η values are the code's",
            ha="center", fontsize=8.8, color=INK, va="top")
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.9, 2.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("3. The corrector: extra Langevin steps", fontsize=11, color=INK, loc="left", weight="bold")
    fig.tight_layout()
    save(fig, "10a-the-product-and-the-corrector")


# ---------- 10b. the product's chains, with and without the corrector ----------
def fig10b():
    levels = REF["levels recorded"]
    rows = [("0", "product"), ("0", "product, 20 corrector steps"), ("main", "product"), ("main", "product, 20 corrector steps")]
    W = len(levels) * 1.6 + 4.2
    H = len(rows) * 1.45 + 2.9
    fig, ax = canvas(W, H)
    nsc = json.load(open(
        REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
        / "toy-caption-corrector-no-single-captions.json"))
    pur = sorted({r["stronger"]["product"]["purple or overlaid"] for r in nsc["results"] if r["pictures"] == 0})
    sep = {r["stronger"]["product"]["red and blue apart"] for r in nsc["results"] if r["pictures"] == 0}
    assert sep == {0}
    lab(ax, W / 2, 1.15, "rows: the 0% reference model (0 relabelled, retrained on CPU) and the replacing 1,000 model "
        "(trained on the GPU in job 90137, loaded here and sampled on CPU).", size=8.8, color=EDGE)
    lab(ax, W / 2, 0.75, "The reported "
        f"0-relabelled models with 20 corrector steps ended as purple hybrids ({' or '.join(map(str, pur))} of 300,",
        size=8.8, color=EDGE)
    lab(ax, W / 2, 0.35, "toy-caption-corrector-no-single-captions.json) where this retrain ends faint "
        f"({REF['models']['0']['counts of 300']['product, 20 corrector steps']['nothing readable']} nothing readable); "
        "both are 0 separate.", size=8.8, color=EDGE)
    title(ax, W, H, "The product of “a red circle” × “a blue square”, step by step",
          "one sample per row, decoded at each recorded level; right: that model's own count of 300 samples")
    for r, (k, tag) in enumerate(rows):
        y = H - 2.4 - r * 1.45
        strip = A[f"{MODELS[k][0]}/traj/{tag}"]
        lab(ax, 0.1, y + 0.45, f"{MODELS[k][2]}\n" + ("plain" if tag == "product" else "20 corrector steps"),
            size=8.5, ha="left")
        for i, L in enumerate(levels):
            pic(ax, strip[i][0], 2.0 + i * 1.6, y, 1.4, ec=COUT if L == 0 else EDGE)
            if r == len(rows) - 1:
                lab(ax, 2.7 + i * 1.6, y - 0.25, f"k = {L}", size=8)
        cnt = MODELS[k][1]["counts of 300"][tag if tag != "product" else "product, same starting noise as the corrector run"]
        lab(ax, W - 1.1, y + 0.4, f"{cnt['red and blue apart']} of 300\nseparate", size=9,
            color=ACC if cnt["red and blue apart"] else INK, weight="bold")
    save(fig, "10b-the-product-step-by-step", ax)


# ---------- 11. the scorer (past the templates) ----------
def fig11():
    W, H = 26, 11.8
    fig, ax = canvas(W, H)
    sc = NUM["scorer"]
    title(ax, W, H, "The scorer: what makes a sample count as “separate”, and what “of 300” is",
          "every pixel takes the nearest of four colours; then the red, blue and purple pixels are counted")
    ex = [("main", "product, 20 corrector steps"), ("0", "product")]
    reads = {k: MODELS[k][1]["scorer reading of the shown samples"] for k in ("0", "main")}
    names = ["background", "red", "blue", "purple"]
    pal = np.array([b.PALETTE[0], b.PALETTE[1], b.PALETTE[2], b.PALETTE[3], [0.8, 0.8, 0.8]])
    for r, (k, tag) in enumerate(ex):
        y = 7.3 - r * 3.4
        tiles = A[f"{MODELS[k][0]}/tiles/{tag}"]
        idx = 0
        img = tiles[idx].reshape(t.H, t.W, 3)
        d = ((img[..., None, :] - b.PALETTE) ** 2).sum(-1) ** 0.5
        near = d.argmin(-1); near[d.min(-1) > b.MAX_DIST] = 4
        info = reads[k][tag][idx]
        pic(ax, tiles[idx], 0.4, y, 3.4, ec=CIN)
        lab(ax, 2.1, y - 0.35, f"a sample: {MODELS[k][2]}, " + ("20 corrector steps" if "corr" in tag else "plain"),
            size=8.5)
        pic(ax, pal[near].reshape(-1), 5.0, y, 3.4, ec=CHID)
        lab(ax, 6.7, y - 0.35, "nearest colour per pixel (light grey: none within 0.35)", size=8.2)
        arr(ax, (3.85, y + 0.85), (4.95, y + 0.85))
        gap = abs(info["red x"] - info["blue x"]) if info["red x"] and info["blue x"] else None
        txt = (f"red pixels {info['red px']}\nblue pixels {info['blue px']}\npurple pixels {info['purple px']}\n"
               + (f"gap between centres {gap:.1f}" if gap is not None else "gap: needs both colours"))
        ss.box(ax, 9.1, y - 0.1, 3.1, 1.9, S, "hid", None)
        lab(ax, 10.65, y + 0.85, txt, size=8.8)
        arr(ax, (8.45, y + 0.85), (9.05, y + 0.85))
        ss.box(ax, 13.1, y + 0.45, 3.2, 0.8, S, "out", info["call"], focus=True, size=9.5)
        arr(ax, (12.25, y + 0.85), (13.05, y + 0.85))
    fbox(ax, 7.0, 0.8, rf"separate $\Leftrightarrow$ red $\geq {sc['min pixels per colour']}$, blue $\geq {sc['min pixels per colour']}$, "
         rf"centres $\geq {sc['min centre gap along x (pixels)']:.0f}$ pixels apart along x, purple $< {sc['min pixels per colour']}$",
         size=10)
    # the five categories for the reference models
    a2 = fig.add_axes([19.6 / W, 0.16, 5.6 / W, 0.62])
    cats = sc["categories"]
    sets_ = [("0", "product, same starting noise as the corrector run", "0%, plain"),
             ("main", "product, same starting noise as the corrector run", f"{MAIN_LABEL}, plain"),
             ("main", "product, 20 corrector steps", f"{MAIN_LABEL},\n20 corrector steps")]
    yy = np.arange(len(cats))[::-1]
    hgt = 0.26
    for j, (k, tag, nm) in enumerate(sets_):
        cnt = MODELS[k][1]["counts of 300"][tag]
        vals = [cnt[c] for c in cats]
        a2.barh(yy + (1 - j) * hgt, vals, height=hgt, color=["#e4e7eb", "#f6dcc2", "#cfd8e3"][j],
                edgecolor=[EDGE, ACC, CIN][j], lw=0.9, label=nm)
        for yv, v in zip(yy + (1 - j) * hgt, vals):
            if v:
                a2.text(v + 4, yv, str(v), va="center", fontsize=8, color=INK)
    a2.set_yticks(yy, cats, fontsize=8.5)
    a2.set_xlim(0, 340); a2.set_xlabel("samples, of 300 per model")
    a2.legend(fontsize=7.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    a2.set_title("Each model draws 300 samples\nfrom fresh noise; each gets one call\n(a missing bar is 0)", fontsize=10, color=INK, loc="left",
                 weight="bold")
    hits = sc["hit rate on 200 noisy copies of each known picture"]
    lab(ax, 7.0, 0.15, f"checked on 200 noisy copies of each of the {len(hits)} known pictures: right on "
        f"{min(hits.values()) * 100:.0f}% of every one", size=8.5, color=EDGE)
    save(fig, "11-the-scorer", ax)


# ---------- 12. why the pictures carry pixel noise (past the templates) ----------
def fig12():
    W, H = 25, 10.2
    fig, ax = canvas(W, H)
    title(ax, W, H, "Why the training pictures carry pixel noise, and what happens without it",
          "left: one picture and some of its real training copies; right: the same sweep trained with no noise")
    # left: a clean picture with a spread of its noisy copies around it
    sd = P["pixel noise sd added to every training copy"]
    clean = t.CIRCLES["left"]
    rng = np.random.default_rng(3)
    cx, cy, rad = 4.0, 5.2, 2.75
    pic(ax, clean, cx - 1.1, cy - 0.55, 2.2, ec=CIN, lw=1.6)
    lab(ax, cx, cy - 0.8, "the clean picture", size=8.5, color=CIN)
    for i in range(6):
        ang = np.pi / 2 + i * np.pi / 3
        x, y = cx + rad * np.cos(ang) - 0.8, cy + rad * 0.78 * np.sin(ang) - 0.4
        pic(ax, clean + sd * rng.standard_normal(clean.shape), x, y, 1.6)
    ax.add_patch(FancyBboxPatch((0.3, 1.95), 7.4, 6.6, boxstyle="round,pad=0.05", fill=False, ec="#cfd4da",
                                lw=1, gid="container"))
    fbox(ax, 4.0, 1.35, rf"copy $=$ picture $+\;{sd}\times$ noise", size=10.5)
    lab(ax, 4.0, 0.55, f"the spread matches model 1's Gaussian bumps of width {sd}, the shape of Du et al.'s toys;\n"
        "without it the 2,000 copies are the 8 clean pictures repeated", size=8.6, color=EDGE)
    # right: product tiles with and without noise
    base = REPO / "3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs"
    zn = np.load(base / "toy-caption-threshold-no-noise-samples.npz")
    zw = np.load(base / "toy-caption-threshold-samples.npz")
    zw2 = np.load(base / "toy-caption-threshold-600-1200-2000-samples.npz")
    NN = NUM["no pixel noise (toy_caption_no_noise.py)"]
    noisy = NUM["results"]["by relabelled count (three trainings each)"]
    x0, tw, gapr = 10.6, 1.15, 0.12
    rows = []
    for k in ("0", "200", "2000"):
        rows += [(k, "with noise"), (k, "no noise")]
    seeds = [r["seed"] for r in noisy["0"]]
    for j, s_ in enumerate(seeds):
        lab(ax, x0 + j * 4.6 + 1.75, 8.45, f"training run {j + 1} (seed {s_})", size=8.8, weight="bold")
    for r, (k, cond) in enumerate(rows):
        y = 7.45 - r * 1.08 - (r // 2) * 0.25
        if cond == "with noise":
            lab(ax, 8.1, y - 0.05, f"{fmt(int(k))} relabelled", size=9.5, ha="left", weight="bold")
        lab(ax, 8.1, y - 0.42 if cond == "with noise" else y - 0.02, cond, size=8.5, ha="left",
            color=EDGE if cond == "with noise" else ACC)
        for j, s_ in enumerate(seeds):
            key = f"{k}_{s_}_product"
            src = zn if cond == "no noise" else (zw2 if k == "2000" else zw)
            tiles = src[key]
            for q in range(3):
                pic(ax, tiles[q], x0 + j * 4.6 + q * (tw + gapr), y - 0.3, tw, lw=0.7)
            runs = NN["by relabelled count (three trainings each)"][k] if cond == "no noise" else noisy[k]
            run = [rr for rr in runs if rr["seed"] == s_][0]
            n_ap = run["product"]["red and blue apart"]
            outside = cond == "no noise" and not NN["inside the noisy runs' spread, per run"][k][j]
            lab(ax, x0 + j * 4.6 + 3 * (tw + gapr) + 0.45, y - 0.02, f"{n_ap}" + (" ▲" if outside else ""), size=10,
                color=ACC if n_ap >= 30 else INK, weight="bold")
    lab(ax, x0 + 7.0, 0.95, "number: product samples of 300 that are a separate red circle and blue square; "
        "tiles: that run's first three samples", size=8.6, color=EDGE)
    singles = [r["product drew a single shape"] for r in NN["by relabelled count (three trainings each)"]["0"]]
    ins = NN["inside the noisy runs' spread, per run"]
    n_in = sum(sum(v) for v in ins.values()); n_all = sum(len(v) for v in ins.values())
    lab(ax, x0 + 7.0, 0.4, f"at 0 relabelled the no-noise models draw a single shape on {min(singles)} to {max(singles)} "
        f"of 300 where the noisy ones draw purple; rule written before the run: {n_in} of {n_all} no-noise runs fall "
        f"inside the noisy runs' spread (▲: above it)", size=8.6, color=INK)
    save(fig, "12-why-the-pictures-carry-pixel-noise", ax)


# ---------- the replacing design, the main one: figures 2, 3, 13 and 14 ----------
RESD = REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
TOYD = REPO / "3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs"


def _matched_shares():
    """One column per share: (share label, replaced k or None, added k). The 0% column has no replacing run."""
    rep = REPL["training set by replaced count"]
    add = TS["by relabelled count"]
    add_ks = [k for k in add if k != "0"]
    cols = [("0%", None, "0")]
    for rk, ak in zip(rep, add_ks):
        sh = rep[rk]["share of 'a red circle' holding both shapes"] * 100
        cols.append((f"{sh:.2g}%", rk, ak))
    return cols


def fig02():
    rep, add = REPL["training set by replaced count"], TS["by relabelled count"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw=dict(width_ratios=[1.25, 1]), sharey=False)
    fig.patch.set_facecolor("white")
    ks = ["0"] + list(rep)
    ys = np.arange(len(ks))[::-1]
    for y, k in zip(ys, ks):
        both = 0 if k == "0" else rep[k]["replaced"]
        a1.barh(y, 2000 - both, color="#e4e7eb", edgecolor=EDGE, lw=0.8)
        a1.barh(y, both, left=2000 - both, color="#f6dcc2", edgecolor=ACC, lw=1.0)
        sh = 0 if k == "0" else rep[k]["share of 'a red circle' holding both shapes"]
        a1.text(2060, y, f"2,000 pictures, {sh * 100:.2g}% hold both shapes", va="center", fontsize=9, color=INK,
                clip_on=False)
    a1.set_yticks(ys, [f"{fmt(int(k))} replaced" + (" (not run)" if k == "0" else "") for k in ks])
    a1.set_xlim(0, 4100)
    a1.set_xticks([0, 1000, 2000, 3000, 4000], ["0", "1,000", "2,000", "3,000", "4,000"])
    a1.set_xlabel("pictures filed under “a red circle” (the same for “a blue square”)")
    a1.set_title("Replacing (the main design): each single caption stays at 2,000;\n"
                 "k of its lone shapes become both-shapes copies", fontsize=11.5, color=INK, loc="left", weight="bold")
    a1.text(700, len(ks) - 0.35, "single-shape pictures", ha="center", fontsize=9, color=EDGE)
    a1.text(1700, len(ks) - 0.35, "both-shapes copies", ha="center", fontsize=9, color=ACC)
    a1.set_ylim(-0.6, len(ks) + 0.1)
    # adding, faded, same scale
    kadd = list(add)
    ys2 = np.arange(len(kadd))[::-1]
    for y, k in zip(ys2, kadd):
        a2.barh(y, 2000, color="#f3f4f6", edgecolor="#c3cbd3", lw=0.8)
        a2.barh(y, add[k]["relabelled"], left=2000, color="#f8ece0", edgecolor="#e2b98f", lw=0.9)
    a2.set_yticks(ys2, [f"{fmt(int(k))} added" for k in kadd], color=OFF)
    a2.set_xlim(0, 4100)
    a2.set_xticks([0, 1000, 2000, 3000, 4000], ["0", "1,000", "2,000", "3,000", "4,000"], color=OFF)
    a2.set_xlabel("pictures under “a red circle”", color=OFF)
    a2.set_title("Adding (kept as the check): the pool grows to 2,000 + k,\nto 4,000 at the same 50% share",
                 fontsize=11, color=OFF, loc="left")
    a2.set_ylim(-0.6, len(kadd) + 0.1)
    for a_ in (a1, a2):
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    a2.spines["left"].set_color("#c3cbd3"); a2.spines["bottom"].set_color("#c3cbd3")
    fig.tight_layout()
    save(fig, "02-what-relabelling-adds-to-each-caption")


def fig03():
    rep, add = REPL["training set by replaced count"], TS["by relabelled count"]
    cols = _matched_shares()
    v, d = NUM["vae"], NUM["denoiser"]
    rep_den = [add["0"]["times each picture is seen in training"]] + [rep[c[1]]["times each picture is seen in training"]
                                                                      for c in cols[1:]]
    rep_vae = [round(v["training steps"] * v["batch"] / add["0"]["training pictures"], 1)] + [
        rep[c[1]]["times each picture is seen by the VAE"] for c in cols[1:]]
    add_den = [add[c[2]]["times each picture is seen in training"] for c in cols]
    add_vae = [v["training steps"] * v["batch"] / add[c[2]]["training pictures"] for c in cols]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    xs = np.arange(len(cols))
    ax.plot(xs, add_den, "o--", color="#b7c7b9", lw=1.2, ms=4, label="adding, denoiser (the check)")
    ax.plot(xs, add_vae, "s--", color="#c9ced4", lw=1.2, ms=4, label="adding, VAE (the check)")
    ax.plot(xs, rep_den, "o-", color=CHID, lw=2.2, label="replacing, denoiser")
    ax.plot(xs, rep_vae, "s-", color=EDGE, lw=2.0, label="replacing, VAE")
    ax.text(len(cols) - 1 + 0.15, rep_den[-1], f"{rep_den[-1]:.0f}", color=CHID, va="center")
    ax.text(len(cols) - 1 + 0.15, rep_vae[-1], f"{rep_vae[-1]:.0f}", color=EDGE, va="center")
    ax.text(len(cols) - 1 + 0.15, add_den[-1], f"{add_den[-1]:.0f}", color="#9aa99c", va="center")
    ax.text(len(cols) - 1 + 0.15, add_vae[-1], f"{add_vae[-1]:.0f}", color="#aab0b7", va="center")
    ax.set_xticks(xs, [c[0] for c in cols])
    ax.set_xlabel("share of each single caption's pictures holding both shapes (one column per tested share)")
    ax.set_ylabel("times each picture is drawn, on average\n(the closest thing to an epoch count)")
    ax.set_ylim(0, 400)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Replacing keeps the training set at {fmt(rep[cols[-1][1]]['training pictures'])} pictures, "
                 "so every share is trained for the same number of passes", fontsize=11.5, color=INK, loc="left",
                 weight="bold")
    ax.text(0.02, 0.97, f"denoiser: {fmt(d['training steps'])} steps × batch {d['batch']};  "
            f"VAE: {fmt(v['training steps'])} steps × batch {v['batch']};  passes = steps × batch / pictures",
            transform=ax.transAxes, fontsize=9.2, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=0.9))
    fig.tight_layout()
    save(fig, "03-how-long-each-network-trains")


def fig13():
    cols = _matched_shares()
    sweep = REPL["sweep (GPU job 90137, mscluster88), three trainings each"]
    addres = NUM["results"]["by relabelled count (three trainings each)"]
    sc = REPL["size check (CPU run at 200 and 1,000, kept separate from the sweep)"]["runs"]
    bar = 30
    fig, ax = plt.subplots(figsize=(13, 6.4))
    xs = np.arange(len(cols))
    med_r, med_a = [], []
    for i, (lbl, rk, ak) in enumerate(cols):
        av = [r["product"]["red and blue apart"] for r in addres[ak]]
        ax.scatter([i + 0.12] * 3, av, s=36, facecolor="white", edgecolor=OFF, lw=1.2, zorder=3)
        med_a.append(np.median(av))
        if rk is not None:
            rv = [r["product"]["red and blue apart"] for r in sweep[rk]]
            ax.scatter([i - 0.12] * 3, rv, s=40, color=INK, zorder=4)
            med_r.append((i, np.median(rv)))
    ax.plot(xs + 0.12, med_a, "--", color=OFF, lw=1.3, label="adding (the check): three trainings, line through medians")
    ax.plot([p[0] - 0.12 for p in med_r], [p[1] for p in med_r], "-", color=INK, lw=2,
            label="replacing (the main design): three trainings, line through medians")
    # the size check, a separate CPU run at 10% and 50%
    where = {"200": [c[0] for c in cols].index("9.1%"), "1000": len(cols) - 1}
    for k, rr in sc.items():
        vals = [r["product"]["red and blue apart"] for r in rr]
        ax.scatter([where[k] - 0.32] * len(vals), vals, marker="x", s=46, color=ACC, lw=1.6, zorder=4,
                   label="size check, a separate CPU run (replaced 200 = 10%, 1,000 = 50%)" if k == "200" else None)
    ax.axhline(bar, color=ACC, lw=1, ls=":")
    ax.text(0.05, bar + 6, f"{bar} of 300: the bar a training must reach to count as drawing the pair", color=ACC, fontsize=9)
    ax.set_xticks(xs, [c[0] for c in cols])
    ax.set_xlabel("share of each single caption's pictures holding both shapes "
                  "(categorical, one column per tested share; not to scale)")
    ax.set_ylabel("product samples of 300 that are a separate\nred circle and blue square")
    ax.set_ylim(-8, 310)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.legend(fontsize=8.6, frameon=False, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Replacing against adding: separate pairs at each share, three trainings per share", fontsize=12,
                 color=INK, loc="left", weight="bold")
    ax.text(0, -0.2, "0%: the adding design's k = 0 trainings; a replacing run at 0 would be the same design and was not run",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color=EDGE)
    fig.tight_layout()
    save(fig, "13-replacing-against-adding")


def fig14():
    cols = _matched_shares()
    sweep = REPL["sweep (GPU job 90137, mscluster88), three trainings each"]
    addres = NUM["results"]["by relabelled count (three trainings each)"]
    zr = np.load(RESD / "toy-caption-replace-sweep-samples.npz")
    za = np.load(TOYD / "toy-caption-threshold-samples.npz")
    seeds = [r["seed"] for r in addres["0"]]
    tw, gap = 1.25, 0.1
    cw = 4 * (tw + gap) + 0.4
    W = 3.6 + 3 * cw + 0.4
    H = len(cols) * 1.25 + 2.4
    fig, ax = canvas(W, H)
    title(ax, W, H, "What the product draws at each share, replacing",
          "rows: share of each single caption's pictures holding both shapes; columns: three trainings; "
          "count: of 300 product samples, how many are a separate red circle and blue square")
    for j, s_ in enumerate(seeds):
        lab(ax, 3.6 + j * cw + cw / 2 - 0.2, H - 1.55, f"training run {j + 1} (seed {s_})", size=9, weight="bold")
    for r, (lbl, rk, ak) in enumerate(cols):
        y = H - 2.6 - r * 1.25
        name = (f"{lbl}\n(adding's k = 0)" if rk is None else f"{lbl}\n{fmt(int(rk))} replaced")
        lab(ax, 0.2, y + 0.35, name, size=8.8, ha="left", color=OFF if rk is None else INK)
        for j, s_ in enumerate(seeds):
            if rk is None:
                tiles = za[f"0_{s_}_product"]
                n = [rr for rr in addres["0"] if rr["seed"] == s_][0]["product"]["red and blue apart"]
            else:
                tiles = zr[f"{rk}_{s_}_product"]
                n = [rr for rr in sweep[rk] if rr["seed"] == s_][0]["product"]["red and blue apart"]
            x0 = 3.6 + j * cw
            for q in range(4):
                pic(ax, tiles[q], x0 + q * (tw + gap), y, tw, lw=0.7)
            lab(ax, x0 + 2 * (tw + gap) - gap / 2, y - 0.22, f"{n} of 300 separate", size=8.3,
                color=ACC if n >= 30 else INK, weight="bold" if n >= 30 else "normal")
            if n >= 30:
                ax.add_patch(Rectangle((x0 - 0.1, y - 0.42), 4 * (tw + gap) + 0.1, tw / 2 + 0.55, fill=False,
                                       ec=ACC, lw=1.8, gid="container"))
    lab(ax, W / 2, 0.35, "outlined: 30 or more of 300; the 0% row is the adding design's k = 0 trainings, "
        "since a replacing run at 0 would be the same design and was not run", size=8.6, color=EDGE)
    save(fig, "14-what-the-product-draws-at-each-share-replacing", ax)


# ---------- 15 to 18: Du et al. beside the toy, every caption under replacing, the toy against SDXL, one-hot ----------
def _cell(ax, x, y, w, h, lines, kind="hid", faded=False, title_=None):
    ss.box(ax, x, y, w, h, S, kind, None, faded=faded)
    yy = y + h - 0.35
    if title_:
        lab(ax, x + 0.2, yy, title_, size=9.5, ha="left", weight="bold")
        yy -= 0.5
    for txt, mark in lines:
        col = {"quoted": INK, "ran": CHID, "inferred": ACC, "stated": EDGE}.get(mark, INK)
        lab(ax, x + 0.2, yy, txt, size=8.4, ha="left", va="top", color=col)
        yy -= 0.38 * (txt.count("\n") + 1) + 0.12


def fig15():
    D = NUM["Du et al. 2023 beside the toy"]
    L = D["code lines"]
    W, H = 27, 22
    fig, ax = canvas(W, H)
    title(ax, W, H, "The toy beside Du et al. 2023 (Reduce, Reuse, Recycle), row by row",
          "black: quoted from the paper (page) or their code (file:line);  green: ran here;  "
          "orange: inferred;  grey: stated in our own write-up")
    x1, x2, x3, cw, dw = 0.4, 9.6, 18.8, 9.0, 7.8
    lab(ax, x1 + cw / 2, H - 1.75, "Du et al. 2023", size=11, weight="bold")
    lab(ax, x2 + cw / 2, H - 1.75, "the toy", size=11, weight="bold", color=ACC)
    lab(ax, x3 + dw / 2, H - 1.75, "how they differ", size=11, weight="bold")
    rows = [
        ("(a) composing two captions",
         [("eq. 13, p. 7: log p(x) + Σ_i (log p(x|y_i) − log p(x)),", "quoted"),
          ("with guidance scale λ in eq. 6, p. 3", "quoted"),
          (f"code {L['composition rule']['file:line']}: uncond + guidance_scale ·\n"
           "Σ(cond − uncond)", "quoted"),
          (f"{L['guidance scale']['file:line']}: guidance_scale = {D['guidance scale']:g}", "quoted")],
         [("toy_caption_dial_ldm.py draw(): e = e_empty +", "quoted"),
          (f"{D['ours']['guidance']:g} · [(e_circle − e_empty) + (e_square − e_empty)]", "quoted"),
          ("the same rule, run on every product in figures 13 and 14", "ran")],
         [("same form: the sum of caption differences, each", "inferred"),
          (f"scaled by one guidance weight: {D['guidance scale']:g} in their code,", "inferred"),
          (f"{D['ours']['guidance']:g} in ours; eq. 13 as printed has weight 1", "inferred")]),
        ("(b) the corrector",
         [("Algorithm 1, p. 5: MCMC steps at every noise level;", "quoted"),
          ("ULA, eq. 4, p. 3; MALA and HMC with Metropolis, B.1", "quoted"),
          ("step = constant · β_t, tuned to acceptance, B.3, p. 14", "quoted"),
          (f"code {L['ULA update']['file:line']}: x + grad·ss + noise,", "quoted"),
          (f"{L['ULA noise']['file:line']}: noise sd √(2·ss)", "quoted"),
          (f"{L['step size, final assignment (MALA and ULA)']['file:line']}: ss = {D['ULA step multiplier of beta_t in effect']:g}·β_t "
           f"(line 217's {D['ULA step multiplier first written, then overwritten']:g}·β_t\nis overwritten), "
           f"{D['corrector steps per level in code']} steps per level", "quoted"),
          (f"{L['corrector only above index 50']['file:line']}: corrects only levels above 50", "quoted")],
         [("toy_caption_corrector.py: z ← z + η s + √(2η) ξ,", "quoted"),
          ("s = −e / √(1 − ᾱ), η = 0.05 (1 − ᾱ),", "quoted"),
          (f"{D['ours']['corrector steps per level'][0]} or {D['ours']['corrector steps per level'][1]} steps "
           f"after every one of the {D['ours']['reverse steps']} reverse steps", "quoted"),
          ("ULA only, no Metropolis correction", "quoted"),
          ("ran: 300 of 300 separate at replacing 1,000 (figure 10b)", "ran")],
         [("same ULA update and the same score −e/√(1−ᾱ)", "inferred"),
          ("(inf_sample.py:205–207 scales by √(1/(1−ᾱ)))", "quoted"),
          ("step size differs: ours scales with the total noise", "inferred"),
          ("variance 1 − ᾱ, theirs with the per-step β_t", "inferred"),
          ("they correct 49 of 100 levels; we correct all", "inferred"),
          ("paper vs code: printed A9 and A10 (p. 14) have noise", "quoted"),
          ("√2·η ξ; the code uses √(2η) ξ, which agrees with eq. 4", "inferred")]),
        ("(c) the 2D toy",
         [("Figure 2, p. 5: a product of a ring of 8 Gaussians", "quoted"),
          ("and a uniform bar, and a mixture of two columns;", "quoted"),
          ("reverse diffusion on summed scores fails, HMC matches", "quoted"),
          ("data, p. 18: ring radius 0.5, sd 0.3; the notebook", "quoted"),
          ("uses sd 0.03 (audit row R38)", "stated"),
          ("ran here: the 2D notebook's definitions, samplers and the", "ran"),
          ("bar expert's training; the product and mixture sampling", "ran"),
          ("cells were not run (code walk, step 3c open)", "ran")],
         [("toy_sampler_hybrids.py redoes the question of Figure 2", "quoted"),
          ("on red circles and blue squares: exact Gaussian-mixture", "quoted"),
          ("experts on 16 × 32 pictures, sampled exactly, by reverse", "quoted"),
          ("diffusion on summed scores, and by annealed MALA", "quoted"),
          ("the summed scores lose the pair; the true noised score", "stated"),
          ("draws it on 300 of 300 (toy write-up, figure 04)", "stated")],
         [("redone: the question (does the sampler or the target", "inferred"),
          ("make the failure?) on our pictures", "inferred"),
          ("not redone: their ring × bar data, trained 2D networks,", "inferred"),
          ("HMC, and their Table 1 numbers; nothing of theirs", "inferred"),
          ("was reproduced", "inferred")]),
        ("(d) networks and data",
         [("2D: a residual MLP, 4 blocks of 128, per distribution, p. 18", "quoted"),
          ("CLEVR: a U-Net on 200,000 64 × 64 images of 1 to 5", "quoted"),
          ("cubes, conditioned on one cube's position, p. 18", "quoted"),
          ("an energy parameterization so MALA and HMC can", "quoted"),
          ("accept or reject, section 4.2, p. 5", "quoted"),
          ("trained to predict the noise: eq. 2, p. 2, ||ε − ε_θ(x_t, t)||²", "quoted")],
         [("one caption-conditional latent model: a VAE to 128", "quoted"),
          ("numbers, a fully connected denoiser (3 × 1,024),", "quoted"),
          ("the caption an id picking a row of a 4 × 64 table", "quoted"),
          ("8 distinct pictures plus pixel noise, 6,000 per training", "quoted"),
          ("trained to predict the clean latent (mean squared error on z0);", "quoted"),
          ("the sampler converts that guess to a noise estimate", "quoted")],
         [("they do and we do not: energy models, Metropolis", "inferred"),
          ("correction, HMC, mixtures, negation, ImageNet and", "inferred"),
          ("text-to-image composition", "inferred"),
          ("we do and they do not: vary how often a both-shapes", "inferred"),
          ("picture is filed under each single caption", "inferred"),
          ("their network predicts the noise, ours the clean latent;", "inferred"),
          ("both feed the same noise estimate e to the sampler", "inferred")]),
    ]
    y = H - 2.3
    for name, a_, b_, c_ in rows:
        n = max(len(a_), len(b_), len(c_))
        h = 0.5 + 0.52 * (n + 1)
        y -= h + 0.25
        _cell(ax, x1, y, cw, h, a_, "hid", title_=name)
        _cell(ax, x2, y, cw, h, b_, "in", title_=name)
        _cell(ax, x3, y, dw, h, c_, "out", faded=False, title_=name)
    lab(ax, W / 2, 0.45, f"their code: {NUM['Du et al. 2023 beside the toy']['clone']} at {D['HEAD']} "
        f"(the quoted files are unchanged from upstream 81177d9); paper: arXiv 2302.11552, the shelf copy's page numbers",
        size=8.4, color=EDGE)
    save(fig, "15-du-et-al-side-by-side", ax)


def fig16():
    sweep = REPL["sweep (GPU job 90137, mscluster88), three trainings each"]
    addres = NUM["results"]["by relabelled count (three trainings each)"]
    zr = np.load(RESD / "toy-caption-replace-sweep-samples.npz")
    za = np.load(TOYD / "toy-caption-threshold-samples.npz")
    cols = _matched_shares()
    seed = 20261022
    tw, gp = 1.0, 0.08
    cw = 4 * (tw + gp) + 0.5
    x0 = 3.3
    W = x0 + 4 * cw + 3.2
    H = len(cols) * 1.0 + 2.9
    fig, ax = canvas(W, H)
    title(ax, W, H, "Every caption side by side, replacing: the joint caption draws the pair at every share",
          f"one training (seed {seed}) per row; four samples per cell; count of 300: own colour only for a single "
          "caption, a separate pair for the product and the joint caption")
    heads = [("“a red circle”", "circle"), ("“a blue square”", "square"),
             ("the product", "product"), ("“a red circle and a blue square”", "joint")]
    for j, (h_, _) in enumerate(heads):
        lab(ax, x0 + j * cw + cw / 2 - 0.25, H - 1.75, h_, size=8.8, weight="bold",
            color=ACC if j == 2 else INK)
    lab(ax, x0 + 4 * cw + 1.3, H - 1.75, "product, three\ntrainings", size=8.3, weight="bold")
    for r, (lbl, rk, ak) in enumerate(cols):
        y = H - 2.75 - r * 1.0
        lab(ax, 0.2, y + 0.3, f"{lbl}\n" + ("(adding's k = 0)" if rk is None else f"{fmt(int(rk))} replaced"),
            size=8.3, ha="left", color=OFF if rk is None else INK)
        runs = addres["0"] if rk is None else sweep[rk]
        run = [rr for rr in runs if rr["seed"] == seed][0]
        cnts = {"circle": run["'a red circle' drew only red"], "square": run["'a blue square' drew only blue"],
                "product": run["product"]["red and blue apart"], "joint": run["joint caption apart"]}
        for j, (_, key) in enumerate(heads):
            tiles = (za if rk is None else zr)[f"{0 if rk is None else rk}_{seed}_{key}"]
            for q in range(4):
                pic(ax, tiles[q], x0 + j * cw + q * (tw + gp), y, tw, lw=0.6)
            ok = cnts[key] >= 30 if key == "product" else True
            lab(ax, x0 + j * cw + 2 * (tw + gp), y - 0.17, f"{cnts[key]}", size=8.3,
                color=ACC if (key == "product" and cnts[key] >= 30) else INK,
                weight="bold" if key in ("product", "joint") else "normal")
        three = [rr["product"]["red and blue apart"] for rr in runs]
        lab(ax, x0 + 4 * cw + 1.3, y + 0.25, ", ".join(str(v) for v in three), size=8.6,
            color=ACC if max(three) >= 30 else INK)
    jmin = min(rr["joint caption apart"] for rk in sweep for rr in sweep[rk])
    lab(ax, W / 2, 0.35, f"the joint caption draws a separate pair on at least {jmin} of 300 at every replaced share "
        "(all three trainings); the product reaches 30 only from 9.1%", size=8.8, color=ACC)
    save(fig, "16-every-caption-side-by-side-replacing", ax)


def fig17():
    P = NUM["the SDXL inclusive-captions pilot"]
    sweep = REPL["sweep (GPU job 90137, mscluster88), three trainings each"]
    rep = REPL["training set by replaced count"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[2.4, 1]))
    fig.patch.set_facecolor("white")
    xs = [rep[k]["share of 'a red circle' holding both shapes"] * 100 for k in rep]
    vals = [[r["product"]["red and blue apart"] for r in sweep[k]] for k in rep]
    med = [float(np.median(v)) for v in vals]
    for x, v in zip(xs, vals):
        a1.plot([x, x], [min(v), max(v)], color="#c3cbd3", lw=3, solid_capstyle="round")
    a1.plot(xs, med, "o-", color=INK, lw=2, label="toy, replacing: median of three trainings (bar: their range)")
    a1.axhline(30, color=ACC, lw=1, ls=":")
    a1.text(0.06, 36, "30 of 300", color=ACC, fontsize=8.5)
    a1.set_xscale("log")
    a1.set_ylabel("toy product samples of 300\nthat are a separate pair")
    a1.set_ylim(-10, 310)
    a1.legend(frameon=False, fontsize=8.8, loc="upper left")
    a1.set_title("From the toy to SDXL: the pilot placed on the toy's share axis", fontsize=12, color=INK,
                 loc="left", weight="bold")
    # the pilot, on its own lane
    sb, sm = P["run B: share of 'a photo of a cat' draws holding a dog, by design"] * 100, \
        P["run B: the same share measured by the caption sampler's check (stated)"] * 100
    a2.axvline(sb, color=ACC, lw=1.6)
    a2.text(sb * 1.08, 0.72, f"SDXL pilot, run B: {sb:.1f}% of “a photo of a cat” draws hold a dog\n"
            f"(by design, from 150 both-animal photos captioned “cat” 40% of the time;\n"
            f"the caption sampler's check measured {sm:.1f}%)", fontsize=8.6, va="center", color=INK)
    a2.text(xs[0], 0.2, "run A: 0% (both-animal photos never captioned “a cat”), off this log axis",
            fontsize=8.6, color=OFF, va="center")
    a2.set_ylim(0, 1); a2.set_yticks([])
    a2.set_xlabel("share of each single caption's training pictures (toy) or draws (SDXL) that hold both things, "
                  "log scale")
    a2.set_xticks(xs, [f"{v:.2g}%" for v in xs])
    a2.minorticks_off()
    tl = a2.get_xticklabels()
    tl[-2].set_horizontalalignment("right"); tl[-1].set_horizontalalignment("left")
    for a_ in (a1, a2):
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    a2.spines["left"].set_visible(False)
    fig.text(0.5, 0.01, "the pilot's result (stated): one animal in every PoE picture on 6 seeds in both runs; "
             "the 128-seed test was not run.  A toy share is not an SDXL threshold:\nthe figure places the pilot "
             "on the axis, it does not predict what SDXL will do there", ha="center", fontsize=9, color=ACC)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "17-from-the-toy-to-sdxl")


def fig18():
    Q = NUM["the lookup is one-hot times a matrix (replacing 1,000 model)"]
    den, _ = _trained_and_start()
    E = den.cap.weight.detach().numpy()
    Wc = den.f[0].weight.detach().numpy()[:, 192:256]
    WcE = Wc @ E.T
    W, H = 22, 11.6
    fig, ax = canvas(W, H)
    title(ax, W, H, "The lookup is a one-hot vector times a matrix",
          "two ways to write the same thing; numbers from the replacing 1,000 model's saved weights")
    # route 1
    y1 = 8.0
    lab(ax, 0.4, y1 + 1.15, "the lookup", size=10, weight="bold", ha="left")
    for i in range(4):
        ss.box(ax, 0.6 + i * 0.5, y1 - 0.2, 0.42, 0.42, S, "in", "1" if i == 0 else "0", focus=i == 0, size=8.5)
    lab(ax, 1.6, y1 - 0.55, "one_hot(0)", size=8.5)
    heat(ax, E, 3.4, y1 - 0.45, 3.2, 0.9, vlim=float(np.abs(E).max()))
    lab(ax, 5.0, y1 - 0.75, f"E: {Q['E shape'][0]} × {Q['E shape'][1]}", size=8.5)
    arr(ax, (2.6, y1), (3.35, y1))
    heat(ax, E[0:1], 7.4, y1 - 0.15, 3.2, 0.3, vlim=float(np.abs(E).max()))
    lab(ax, 9.0, y1 - 0.5, "table[0]: 64 numbers", size=8.5)
    arr(ax, (6.65, y1), (7.35, y1))
    ss.box(ax, 11.4, y1 - 0.5, 2.0, 1.0, S, "hid", r"$W_c$" + "\n1,024 × 64", size=9)
    arr(ax, (10.65, y1), (11.35, y1))
    # route 2
    y2 = 5.0
    lab(ax, 0.4, y2 + 1.15, "the same, as one matrix", size=10, weight="bold", ha="left")
    for i in range(4):
        ss.box(ax, 0.6 + i * 0.5, y2 - 0.2, 0.42, 0.42, S, "in", "1" if i == 0 else "0", focus=i == 0, size=8.5)
    lab(ax, 1.6, y2 - 0.55, "one_hot(0)", size=8.5)
    ss.box(ax, 3.4, y2 - 0.5, 3.2, 1.0, S, "hid", r"$W_c E^{\top}$" + f"\n{Q['W_c E shape'][0]:,} × {Q['W_c E shape'][1]}", focus=True, size=9)
    arr(ax, (2.6, y2), (3.35, y2))
    # the shift both routes land on
    vs = float(np.abs(WcE).max())
    heat(ax, WcE[:, 0][None], 15.0, 6.35, 6.4, 0.45, vlim=vs)
    lab(ax, 18.2, 7.15, "caption 0's shift: 1,024 numbers", size=8.8, weight="bold")
    arr(ax, (13.45, y1), (14.95, 6.75))
    arr(ax, (6.65, y2), (14.95, 6.4))
    lab(ax, 18.2, 5.95, f"both routes agree to {Q['largest difference between W_c table[c] and (W_c E) one_hot(c)']:.1e}",
        size=8.5, color=EDGE)
    # the four columns of W_c E^T
    lab(ax, 0.4, 3.45, r"the four columns of $W_c E^{\top}$, one learned 1,024-number shift per caption:", size=9.5,
        ha="left", weight="bold")
    names = ["0: a red circle", "1: a blue square", "2: the joint caption", "3: empty"]
    for c in range(4):
        x = 0.4 + c * 5.4
        heat(ax, WcE[:, c][None], x, 2.3, 4.9, 0.45, vlim=vs)
        lab(ax, x + 2.45, 2.0, f"{names[c]}: norm {Q['column norms of W_c E'][c]}", size=8.5)
    fbox(ax, 7.2, 1.0, r"$\mathrm{table}[c]=E^{\top}\mathrm{onehot}(c)$, so "
         r"$W_c\,\mathrm{table}[c]=(W_c E^{\top})\,\mathrm{onehot}(c)$", size=11)
    lab(ax, 17.5, 1.0, "with 4 captions the two are the same model;\nthe table is the usual form and scales to many classes",
        size=8.6, color=ACC)
    save(fig, "18-the-lookup-is-one-hot-times-a-matrix", ax)


FIGS = {"1": fig01, "2": fig02, "3": fig03, "4": fig04, "5": fig05, "6": fig06, "7": fig07, "7b": fig07b, "7c": fig07c, "7d": fig07d, "7e": fig07e, "7f": fig07f, "8": fig08,
        "9": fig09, "10a": fig10a, "10b": fig10b, "11": fig11, "12": fig12, "13": fig13, "14": fig14, "15": fig15, "16": fig16, "17": fig17, "18": fig18,
        "2-adding": fig02_adding, "3-adding": fig03_adding}

if __name__ == "__main__":
    for key in (sys.argv[1:] or FIGS):
        FIGS[key]()
