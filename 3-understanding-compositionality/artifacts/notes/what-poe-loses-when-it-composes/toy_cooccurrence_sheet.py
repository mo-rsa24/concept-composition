"""Eight draws per sampler at four pair weights, same stream as the sweep."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes")
import toy_sampler_hybrids as t
WS = [0.003, 0.03, 0.1, 0.3, 0.5]
rows = []
for w in WS:
    t.RNG = np.random.default_rng(20261022)
    kind, A, B = t.hand_weighted(w)
    lp, sc = t.composed(kind, A, B)
    runs = {"exact": t.exact_samples(kind, A, B), "reverse diffusion": t.reverse(sc),
            "annealed MALA": t.reverse(sc, lp, mala=True)}
    t.ACCEPT.clear()
    for m, x in runs.items():
        labs = t.label(x)
        rows.append((w, m, labs.count("two objects, crisp"), np.clip(x[:8], 0, 1)))
    print("done", w, flush=True)
fig, axes = plt.subplots(len(rows), 9, figsize=(16, 1.05 * len(rows) + 0.8),
                         gridspec_kw={"width_ratios": [3.0] + [1] * 8})
for r, (w, m, crisp, imgs) in enumerate(rows):
    axes[r, 0].text(0, 0.5, f"pair weight w = {w:g}\n{m}\ncrisp pairs: {crisp} of 300",
                    fontsize=8, va="center", transform=axes[r, 0].transAxes)
    axes[r, 0].axis("off")
    for j in range(8):
        axes[r, j + 1].imshow(imgs[j].reshape(t.H, t.W, 3), interpolation="nearest")
        axes[r, j + 1].axis("off")
fig.suptitle("Condition D: eight draws per sampler as the pair image's weight rises", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.08, hspace=0.3)
P = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/results/how-much-co-occurrence-the-product-needs/draws-across-weights.png"
import os; os.makedirs(os.path.dirname(P), exist_ok=True)
fig.savefig(P, dpi=110)
print("wrote", P)
