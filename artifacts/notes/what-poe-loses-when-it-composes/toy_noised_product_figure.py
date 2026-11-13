"""Figure for toy_noised_product.py: what each score thinks the pair is worth, step by step, and what
the same reverse-diffusion sampler draws with each.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_noised_product_figure.py
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t

OUT = Path(__file__).resolve().parents[2] / "results" / "how-much-co-occurrence-the-product-needs"
J = json.load(open(OUT / "toy-noised-product.json"))["results"]
S = np.load(OUT / "toy-noised-product-samples.npz")
COL = {"0.001": "#9ecae1", "0.01": "#4292c6", "0.1": "#08519c", "0.3": "#08306b"}
fig = plt.figure(figsize=(16, 7.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.08, left=0.06, right=0.99, top=0.86, bottom=0.1)
ax = fig.add_subplot(gs[0])
steps = np.arange(t.T + 1)
for w, r in J.items():
    y = r["pair self-pairing share, summed noised scores, by step"]
    ax.plot(steps, y, color=COL[w], lw=2.2)
    hy, side = {"0.3": (0.5, 1), "0.1": (0.5, 1), "0.01": (0.75, -1), "0.001": (0.25, 1)}[w]
    k = int(np.argmin(np.abs(np.array(y) - hy)))
    ax.text(k - 2 * side, hy, f"w {w}", color=COL[w], fontsize=10, weight="bold", va="center",
            ha="left" if side > 0 else "right")
ax.text(72, 0.55, "blue: the summed noised scores,\none curve per pair weight w", color="#08519c",
        fontsize=10, va="top")
ax.axhline(1.0, color="#1a7f37", lw=3)
ax.text(198, 1.03, "true noised product, every pair weight", color="#1a7f37", ha="left", fontsize=10,
        weight="bold")
ax.set_xlim(200, 0); ax.set_ylim(-0.02, 1.1)
ax.set_xlabel("reverse-diffusion step: sampling starts at pure noise (200, left) and ends on the clean picture (0)")
ax.set_ylabel("share of the mixture on the pair image paired with itself")
ax.set_title("What each score says the pair is worth along the way", fontsize=11.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

inner = gs[1].subgridspec(len(J) * 2, 8, hspace=0.55, wspace=0.06)
for i, (w, r) in enumerate(J.items()):
    for row, (key, name, ii) in enumerate((("reverse diffusion, summed noised scores", "summed scores", 0),
                                           ("reverse diffusion, true noised-product score", "true score", 1))):
        crisp = r[key]["two objects, crisp"]
        for j in range(8):
            a = fig.add_subplot(inner[2 * i + row, j])
            a.imshow(np.clip(S[f"{w}_{ii}"][j], 0, 1).reshape(t.H, t.W, 3), interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(f"w {w}\n{name}\n{crisp} of 300", rotation=0, ha="right", va="center",
                             fontsize=8.5, color="#1a7f37" if crisp == 300 else "#b42318")
fig.text(0.47, 0.885, "Eight draws each from the same reverse-diffusion sampler; count = crisp pairs of 300",
         fontsize=11.5)
fig.suptitle("The product needs almost no co-occurrence; adding noised scores throws it away. At pure noise "
             "the summed scores weight the pair at w squared, the true noised product at 1.", fontsize=12.5)
p = OUT / "summed-scores-against-the-true-noised-product.png"
fig.savefig(p, dpi=110); print("wrote", p)
