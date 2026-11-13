"""The caption dial's 16 pairings on the six-checkpoint toy: each word holds its three single pictures and
the pair picture (share s), so the product weighs every circle-word picture against every square-word
picture. Each panel is the pairing's pixel average with its d^2 and its weight against the best.
Geometry from the six-checkpoint walk's setup table. Writes figures/fig-caption-dial-pairings.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures" / "fig-caption-dial-pairings.png"
H, W, ROW, R, HALF, SIGMA, S = 8, 16, 4.0, 2.6, 2.0, 0.1, 0.05
COLS = {"L": 3.0, "C": 8.0, "R": 13.0}
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


pair = draw(("circle", COLS["L"]), ("square", COLS["R"]))
A = [draw(("circle", c)) for c in COLS.values()] + [pair]
B = [draw(("square", c)) for c in COLS.values()] + [pair]
an = [f"circle {k}" for k in COLS] + ["pair"]
bn = [f"square {k}" for k in COLS] + ["pair"]
shares = np.array([(1 - S) / 3] * 3 + [S])
d2 = np.array([[((a - b) ** 2).sum() for b in B] for a in A])
logw = np.log(np.outer(shares, shares)) - d2 / (4 * SIGMA ** 2)
rel = logw - logw.max()

fig, axes = plt.subplots(4, 4, figsize=(11, 4.4))
for i in range(4):
    for k in range(4):
        ax = axes[i, k]
        ax.imshow((A[i] + B[k]) / 2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{an[i]} × {bn[k]}\nd² {int(d2[i, k])}, weight e^{rel[i, k]:.0f} × best", fontsize=7)
for sp in axes[3, 3].spines.values():
    sp.set_color("green"); sp.set_linewidth(3)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT); print(d2.astype(int)); print(np.round(rel, 1))
