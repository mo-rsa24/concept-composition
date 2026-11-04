"""One channel of one pixel, noise on against noise off: the product of the circle's and the square's
opinions exists only when the noise lets them overlap. Writes figures/fig-noise-on-off.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures" / "fig-noise-on-off.png"
SIGMA = 0.1
v = np.linspace(-0.3, 1.3, 2001)


def bell(m, s):
    return np.exp(-(v - m) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)


fig, axes = plt.subplots(3, 2, figsize=(10, 6.4), sharex=True)
axes[0, 0].plot(v, bell(0, SIGMA), color="red", label='"circle" says: near 0')
axes[0, 0].plot(v, bell(1, SIGMA), color="blue", label='"square" says: near 1')
axes[0, 0].set_title("noise on, σ = 0.1", fontsize=11)
axes[0, 0].legend(fontsize=8)
raw = bell(0, SIGMA) * bell(1, SIGMA)
axes[1, 0].plot(v, raw, color="purple")
axes[1, 0].set_ylabel("product,\nas multiplied", fontsize=9)
norm = raw / np.trapezoid(raw, v)
axes[2, 0].plot(v, norm, color="purple")
axes[2, 0].fill_between(v, norm, color="purple", alpha=0.2)
axes[2, 0].set_ylabel("product,\nscaled to total 1", fontsize=9)

for ax in axes[:, 1]:
    ax.set_ylim(axes[0, 0].get_ylim())
axes[0, 1].vlines(0, 0, 3.6, color="red", lw=3, label='"circle": all of it at 0')
axes[0, 1].vlines(1, 0, 3.6, color="blue", lw=3, label='"square": all of it at 1')
axes[0, 1].plot(v, 0 * v, color="0.6", lw=1)
axes[0, 1].set_title("noise off, σ = 0", fontsize=11)
axes[0, 1].legend(fontsize=8, loc="upper center")
axes[1, 1].plot(v, 0 * v, color="purple", lw=2)
axes[1, 1].text(0.5, 1.8, "zero at every value:\nat 0 the square gives 0, at 1 the circle gives 0", ha="center", fontsize=9)
axes[2, 1].plot(v, 0 * v, color="0.6", lw=1)
axes[2, 1].text(0.5, 1.8, "nothing to scale:\n0 ÷ 0", ha="center", fontsize=11)
axes[1, 1].set_ylim(-0.2, 4.2)
axes[2, 1].set_ylim(-0.2, 4.2)
for ax in axes[2]:
    ax.set_xlabel("blue value of pixel [4, 8]")
for ax in axes.ravel():
    ax.set_yticks([])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT, "raw peak", raw.max(), "scaled peak at", v[norm.argmax()])
