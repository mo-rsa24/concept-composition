"""Checkpoint 6, first column: PoE the way SDXL does it on the six-checkpoint toy. The guided sum
eps_u + w (eps_circle - eps_u) + w (eps_square - eps_u), with an unconditional rule over all seven pictures,
stepped by deterministic DDIM on the cosine schedule, at w = 1 and 7.5, with the pair not included and at
1 in 20. Each sample is read as its nearest of 16 library pictures. Writes figures/fig-ddim-guided.png and
fig-ddim-guided.json (counts at T = 50, 200 and 1000 steps)."""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
H, W, ROW, R, HALF, SIGMA, N = 8, 16, 4.0, 2.6, 2.0, 0.1, 200
COLS = (3.0, 8.0, 13.0)
RED, BLUE = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])


def draw(*objs):
    y, x = np.mgrid[0:H, 0:W] + 0.5
    pic = np.ones((H, W, 3))
    for shape, col in objs:
        if shape == "circle":
            pic[(x - col) ** 2 + (y - ROW) ** 2 <= R ** 2] = RED
        else:
            pic[(abs(x - col) <= HALF) & (abs(y - ROW) <= HALF)] = BLUE
    return pic


C = [draw(("circle", c)) for c in COLS]
S = [draw(("square", c)) for c in COLS]
P = draw(("circle", COLS[0]), ("square", COLS[2]))
LIB = np.array(C + S + [P] + [(c + q) / 2 for c in C for q in S])
KIND = ["circle only"] * 3 + ["square only"] * 3 + ["apart"] + \
       ["overlay" if i == k else "pale, apart" for i in range(3) for k in range(3)]


def cosine_abar(T, s=0.008):
    tt = np.linspace(0, 1, T + 1)
    ab = np.cos((tt + s) / (1 + s) * np.pi / 2) ** 2 / np.cos(s / (1 + s) * np.pi / 2) ** 2
    return np.clip(ab, 1e-5, 0.9999)


def eps(X, pics, shares, abar):
    Pm = np.asarray(pics).reshape(len(pics), -1)
    v = abar * SIGMA ** 2 + 1 - abar
    M = np.sqrt(abar) * Pm
    logr = np.log(shares)[None] - ((X[:, None, :] - M[None]) ** 2).sum(-1) / (2 * v)
    r = np.exp(logr - logr.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return np.sqrt(1 - abar) * (X - r @ M) / v


def ddim_guided(a, b, u, w, T):
    ab = cosine_abar(T)
    X = np.random.default_rng(0).standard_normal((N, H * W * 3))
    for t in range(T, 0, -1):
        e0 = eps(X, *u, ab[t])
        e = e0 + w * (eps(X, *a, ab[t]) - e0) + w * (eps(X, *b, ab[t]) - e0)
        x0 = (X - np.sqrt(1 - ab[t]) * e) / np.sqrt(ab[t])
        X = np.sqrt(ab[t - 1]) * x0 + np.sqrt(1 - ab[t - 1]) * e
    return x0.reshape(N, H, W, 3)


def read_out(x):
    d = ((x.reshape(N, 1, -1) - LIB.reshape(1, len(LIB), -1)) ** 2).sum(-1)
    return d.argmin(1)


U = (C + S + [P], np.ones(7) / 7)
sh = np.array([0.95 / 3] * 3 + [0.05])
SETTINGS = [("pair not included", (C, np.ones(3) / 3), (S, np.ones(3) / 3)),
            ("pair at 1 in 20", (C + [P], sh), (S + [P], sh))]
counts, rows = {}, []
for name, a, b in SETTINGS:
    for w in (1.0, 7.5):
        for T in (50, 200, 1000):
            x = ddim_guided(a, b, U, w, T)
            idx = read_out(x)
            counts[f"{name}, w={w}, T={T}"] = dict(collections.Counter(KIND[i] for i in idx))
            if T == 200:
                rows.append((name, w, x, idx))
(HERE / "fig-ddim-guided.json").write_text(json.dumps(counts, indent=1))

fig, axes = plt.subplots(4, 13, figsize=(19, 4.6), gridspec_kw={"width_ratios": [1.6] + [1] * 12})
for r, (name, w, x, idx) in enumerate(rows):
    axes[r, 0].text(0.5, 0.5, f"{name}\nw = {w}", ha="center", va="center", fontsize=9)
    axes[r, 0].axis("off")
    for t in range(6):
        axes[r, 1 + t].imshow(np.clip(x[t], 0, 1), interpolation="nearest")
        axes[r, 7 + t].imshow(LIB[idx[t]], interpolation="nearest")
    for ax in axes[r, 1:]:
        ax.set_xticks([]); ax.set_yticks([])
axes[0, 1].set_title("DDIM samples, 200 steps", fontsize=9, loc="left")
axes[0, 7].set_title("nearest library picture", fontsize=9, loc="left")
fig.savefig(HERE / "figures" / "fig-ddim-guided.png", dpi=140, bbox_inches="tight")
for k, v in counts.items():
    print(k, v)
