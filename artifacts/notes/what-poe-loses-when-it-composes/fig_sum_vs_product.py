"""One channel of one pixel: the product of the circle's and the square's opinions against their sum.
The product is one bell at the average; the sum is two bumps with nothing between them.
Writes figures/fig-sum-vs-product.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures" / "fig-sum-vs-product.png"
SIGMA = 0.1
v = np.linspace(-0.3, 1.3, 2001)


def bell(m, s):
    return np.exp(-(v - m) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)


product = bell(0.5, SIGMA / np.sqrt(2))
total = 0.5 * bell(0, SIGMA) + 0.5 * bell(1, SIGMA)
fig, axes = plt.subplots(1, 2, figsize=(10, 2.4), sharey=True)
axes[0].plot(v, product, color="purple")
axes[0].fill_between(v, product, color="purple", alpha=0.2)
axes[0].set_title("product, scaled to total 1: one bell at 0.5", fontsize=10)
axes[1].plot(v, total, color="k")
axes[1].fill_between(v, 0.5 * bell(0, SIGMA), color="red", alpha=0.3)
axes[1].fill_between(v, 0.5 * bell(1, SIGMA), color="blue", alpha=0.3)
axes[1].set_title("sum, ½ circle + ½ square: two bumps, nothing at 0.5", fontsize=10)
for ax in axes:
    ax.set_xlabel("blue value of pixel [4, 8]")
    ax.set_yticks([])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT, "sum at 0.5:", total[np.argmin(abs(v - 0.5))], "sum at 0:", total[np.argmin(abs(v))])
