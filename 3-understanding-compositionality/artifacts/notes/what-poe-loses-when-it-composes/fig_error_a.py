"""Error A on the six-checkpoint toy, in closed form. Left: one pixel at abar = 0.5, where the exact noisy
product (multiply, then noise) and the target of the summed noisy scores (noise, then multiply) share a
centre but the summed one has half the noise variance. Right: the caption dial's 16 pairings weighed the
way the summed scores weigh them at each noise level, sigma_eff^2 = sigma^2 + (1 - abar)/abar, against the
exact noisy product, which keeps the clean weights. Writes figures/fig-error-a.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures" / "fig-error-a.png"
H, W, ROW, R, HALF, SIGMA = 8, 16, 4.0, 2.6, 2.0, 0.1
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


pair = draw(("circle", COLS[0]), ("square", COLS[2]))
A = [draw(("circle", c)) for c in COLS] + [pair]
B = [draw(("square", c)) for c in COLS] + [pair]
d2 = np.array([[((a - b) ** 2).sum() for b in B] for a in A])


def apart_share(s, sig2):
    sh = np.array([(1 - s) / 3] * 3 + [s])
    lw = np.log(np.outer(sh, sh)) - d2 / (4 * sig2)
    w = np.exp(lw - lw.max())
    return (w / w.sum())[3, 3]


ab = 0.5
v = ab * SIGMA ** 2 + 1 - ab
x = np.linspace(-2.5, 3, 2000)


def bell(m, var):
    return np.exp(-(x - m) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(x, bell(0, v), color="red", lw=1, label='"circle", noisy')
axes[0].plot(x, bell(np.sqrt(ab), v), color="blue", lw=1, label='"square", noisy')
axes[0].plot(x, bell(np.sqrt(ab) / 2, ab * SIGMA ** 2 / 2 + 1 - ab), color="purple", lw=2, label="exact: product, then noise")
axes[0].plot(x, bell(np.sqrt(ab) / 2, v / 2), "k--", lw=2, label="summed scores: noise, then product")
axes[0].set_xlabel("blue value of pixel [4, 8] at ᾱ = 0.5")
axes[0].set_yticks([])
axes[0].set_title("one pixel: same centre, summed is too narrow", fontsize=10)
axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
grid = np.linspace(0.005, 0.999, 300)
for s, col in [(0.05, "tab:green"), (0.5, "tab:olive")]:
    axes[1].plot(grid, [apart_share(s, SIGMA ** 2 + (1 - a) / a) for a in grid], color=col, label=f"summed scores aim at, s = {s}")
axes[1].axhline(1, color="purple", lw=2, ls=":", label="exact noisy product, any s > 0")
axes[1].set_xlabel("ᾱ (1 = clean, 0 = pure noise)")
axes[1].set_ylabel("share on the two\nshapes apart")
axes[1].invert_xaxis()
axes[1].set_title("the dial the summed scores see, step by step", fontsize=10)
axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
for a in (0.999, 0.9, 0.5, 0.1, 0.01):
    print(a, round(float(np.sqrt(SIGMA ** 2 + (1 - a) / a)), 3), round(float(apart_share(0.05, SIGMA ** 2 + (1 - a) / a)), 3))
