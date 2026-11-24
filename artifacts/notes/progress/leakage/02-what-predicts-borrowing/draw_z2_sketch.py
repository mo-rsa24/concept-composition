"""Sketch of what Z2 measures and when. Numbers: SDXL's scaled-linear schedule (DDIM timesteps 981 down to 1) and z2_bars.py."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

b = np.linspace(0.00085**0.5, 0.012**0.5, 1000)**2; ab = np.cumprod(1 - b)
ts = (np.arange(50) * 20)[::-1] + 1
share = ab[ts] * 100
READ = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]
WIN = {"early\nsteps 0-5\n(CO3 corrects)": (0, 5, "#fde2c4"), "middle\nsteps 10-20": (10, 20, "#d9e8fb"), "late\nsteps 25-45": (25, 45, "#dcf2dc")}

fig = plt.figure(figsize=(13, 6.2))
ax = fig.add_axes([0.06, 0.12, 0.62, 0.52])
for name, (a, z, c) in WIN.items():
    ax.axvspan(a - 0.5, z + 0.5, color=c, zorder=0)
    ax.text((a + z) / 2, 103, name, ha="center", va="bottom", fontsize=9, family="monospace")
ax.plot(range(50), share, color="#222", lw=2)
ax.scatter(READ, share[READ], s=36, color="#e8590c", zorder=3)
for s in (5, 20, 30, 40):
    ax.annotate(f"{share[s]:.1f}% picture", (s, share[s]), xytext=(s + 1.2, share[s] - 10), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))
ax.set_xlim(-1, 50); ax.set_ylim(0, 100)
ax.set_xlabel("sampling step (0 = pure noise, 49 = finished picture)")
ax.set_ylabel("share of the latent that is picture, %")
ax.spines[["top", "right"]].set_visible(False)
ax.text(46.5, 8, "orange dots: the 14 steps where\none extra U-Net call reads\nattention overlap and the d_t maps", fontsize=8.5, ha="right", color="#e8590c")

def box(x, y, w, h, text, fc="#ffffff", ec="#222", fs=9):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", fc=fc, ec=ec, lw=1.6, transform=fig.transFigure))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, family="monospace")

fig.text(0.06, 0.95, "Z2: which reading, at which step, separates the pairs that borrow parts from the ones that don't",
         fontsize=12, weight="bold")
box(0.06, 0.76, 0.17, 0.10, "prompt\n'a cheetah and a cougar'")
box(0.27, 0.76, 0.17, 0.10, "text encoder\n(before step 0)")
box(0.48, 0.76, 0.20, 0.10, "reading 1: text similarity\ncosine of the two\nanimal words", fc="#fff4e6")
fig.add_artist(matplotlib.lines.Line2D([0.23, 0.27], [0.81, 0.81], color="#222", lw=1.4, transform=fig.transFigure))
fig.add_artist(matplotlib.lines.Line2D([0.44, 0.48], [0.81, 0.81], color="#222", lw=1.4, transform=fig.transFigure))

box(0.73, 0.74, 0.25, 0.12, "reading 2: attention overlap\ndo the two words claim\nthe same patch of canvas?", fc="#fff4e6")
box(0.73, 0.58, 0.25, 0.12, "reading 3: d_t similarity\ndoes each animal's prompt\npush the picture the same way?", fc="#fff4e6")
box(0.73, 0.34, 0.25, 0.19, "12 pairs, borrowed of 8 (W2d)\ncheetah x cougar      7\nduck x goose          7\nseal x gull           4\ncrow x raven          0\neagle x fox           0\n... 7 more", fs=8.5)
box(0.73, 0.12, 0.25, 0.17, "the test, per reading and window\nSpearman rho with the\nborrowed counts >= 0.706\n(7 tests, z2_bars.py)\nresult: no test passed;\nbest 0.489 (late overlap)", fc="#eaf1ff", ec="#a8c7fa", fs=8.5)
fig.savefig("z2-what-and-when.png", dpi=150); fig.savefig("z2-what-and-when.svg")
