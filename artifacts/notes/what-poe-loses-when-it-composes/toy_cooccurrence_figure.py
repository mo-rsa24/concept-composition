import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes/toy-sampler-cooccurrence-sweep.json"
d = json.load(open(P))
N = d["samples_per_cell"]
ws = [float(w) for w in d["weights"]]
nz = [w for w in ws if w > 0]
def frac(m):
    return [d["counts"][f"{w:g}"][m]["two objects, crisp"] / N for w in nz]
fig, ax = plt.subplots(figsize=(8.2, 5.0))
style = {"exact": ("#111111", "o", 2.2), "reverse diffusion": ("#c0392b", "s", 1.8),
         "annealed MALA": ("#1f6fb4", "^", 1.8)}
for m, (c, mk, lw) in style.items():
    y = frac(m)
    ax.plot(nz, y, marker=mk, color=c, lw=lw, ms=6, zorder=3)
    dy = {"exact": 0, "annealed MALA": 8, "reverse diffusion": -8}[m]
    ax.annotate(m, (nz[-1], y[-1]), xytext=(8, dy), textcoords="offset points",
                color=c, fontsize=10, va="center", fontweight="bold")
ax.axhline(0.5, color="#999999", lw=0.9, ls=":", zorder=1)
ax.annotate("half the draws", (nz[0], 0.5), xytext=(2, 5), textcoords="offset points",
            fontsize=8, color="#777777")
ax.axvline(0.1, color="#7f8c8d", lw=1.0, ls="--", zorder=1)
ax.annotate("w = 0.1\ncondition B's pair weight", (0.1, 0.30), xytext=(-6, 0),
            textcoords="offset points", fontsize=8.5, color="#5d6d6e", ha="right", va="center")
ax.axvline(nz[0], color="#111111", lw=1.0, ls="--", zorder=1)
ax.annotate("the exact product is already past half\nat the grid's smallest weight: this is the\ngrid's floor, not a crossing",
            (nz[0], 0.985), xytext=(10, -2), textcoords="offset points", fontsize=8.5, color="#111111", va="top")
ax.annotate("at w = 0 all three give 0 crisp pairs of 300", (nz[0], 0.62), xytext=(10, 0),
            textcoords="offset points", fontsize=8.5, color="#555555", va="center")
ax.set_xscale("log"); ax.set_xlim(7e-4, 1.3)
ax.set_ylim(-0.04, 1.08)
ax.set_xlabel("w, the pair image's weight inside each expert (log scale)")
ax.set_ylabel("fraction of 300 draws labelled a crisp pair")
ax.set_title("How much co-occurrence the product needs, and how much its samplers need", fontsize=12)
ax.set_xticks(nz); ax.set_xticklabels([f"{w:g}" for w in nz])
ax.grid(alpha=0.18, which="major")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.1, right=0.78, top=0.9, bottom=0.12)
Q = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/results/how-much-co-occurrence-the-product-needs/weight-sweep.png"
import os; os.makedirs(os.path.dirname(Q), exist_ok=True)
fig.savefig(Q, dpi=130)
print("wrote", Q)
print("exact   ", frac("exact"))
print("reverse ", [round(v,3) for v in frac("reverse diffusion")])
print("mala    ", [round(v,3) for v in frac("annealed MALA")])
print("closed-form crossing:", d["closed_form"]["crossing_crisp_over_half"])
