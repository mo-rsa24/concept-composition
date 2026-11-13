"""The first dial over its whole range, with the point where the experts stop being separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
lo = json.load(open(HERE / "toy-sampler-cooccurrence-sweep.json"))
hi = json.load(open(OUT / "toy-high-overlap-sweep.json"))
N = lo["samples_per_cell"]

def series(m):
    xs, ys = [], []
    for src in (lo["counts"], hi["counts"]):
        for k in src:
            w = float(k)
            if w <= 0 or (src is hi["counts"] and w == 0.5):
                continue
            xs.append(w); ys.append(src[k][m]["two objects, crisp"] / N)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]

fig, ax = plt.subplots(figsize=(9.4, 5.4))
ax.axvspan(0.5, 1.0, color="#f3efe6", zorder=0)
ax.annotate("in this band each expert keeps\nless than half its mass on its\nown shapes: the two are\nturning into the same model",
            (0.52, 0.28), fontsize=9, color="#8a7b52", va="center")
style = {"exact": ("#111111", "o", 2.3), "reverse diffusion": ("#c0392b", "s", 1.9),
         "annealed MALA": ("#1f6fb4", "^", 1.9)}
for m, (c, mk, lw) in style.items():
    x, y = series(m)
    ax.plot(x, y, marker=mk, color=c, lw=lw, ms=5.5, zorder=3)
    ax.annotate(m, (x[-1], y[-1]), xytext=(8, {"exact": 10, "reverse diffusion": -14, "annealed MALA": -2}[m]),
                textcoords="offset points", fontsize=10, fontweight="bold", color=c, va="center")
ax.axhline(1.0, color="#cccccc", lw=0.8, ls=":", zorder=1)
ax.axvline(0.9, color="#1f6fb4", lw=1.2, ls="--", zorder=2)
ax.annotate("both samplers reach 300 of 300 here, at a pair\nweight of 0.9. Each expert has a tenth of its mass\nleft on its own single shapes.",
            (0.9, 0.62), xytext=(-12, 0), textcoords="offset points", fontsize=9.5,
            color="#1f6fb4", ha="right", va="center")
ax.annotate("the exact product is already at 300 of 300\nfrom the grid's smallest weight onward",
            (0.0012, 0.93), xytext=(10, 0), textcoords="offset points", fontsize=9.5,
            color="#111111", va="center")
ax.set_xscale("log"); ax.set_xlim(7e-4, 3.4); ax.set_ylim(-0.04, 1.09)
ax.set_xticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9])
ax.set_xticklabels(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "0.5", "0.9"])
ax.set_xlabel("w, the pair image's weight inside each expert (log scale)")
ax.set_ylabel("fraction of 300 draws labelled a crisp pair")
ax.set_title("How much overlap a sampler needs, and what the experts have become by then", fontsize=13)
ax.grid(alpha=0.16, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.09, right=0.80, top=0.91, bottom=0.12)
fig.savefig(OUT / "how-far-the-dial-has-to-go.png", dpi=130)
print("wrote", OUT / "how-far-the-dial-has-to-go.png")
