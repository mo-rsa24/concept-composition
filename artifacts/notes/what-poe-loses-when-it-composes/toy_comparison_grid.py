"""One grid: four situations across, what the maths says against what a real sampler gets down."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t
import toy_hmc as h
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
SEED = 20261022
GOOD, BAD, INK, FAINT = "#1a7f37", "#9b2226", "#1b1b1b", "#7a7a7a"

COLS = [
    dict(head="No picture shows both", sub="the two training sets never overlap",
         setup=lambda: t.hand_weighted(0.0)),
    dict(head="1 picture in 1000 shows both", sub="the smallest overlap we tested",
         setup=lambda: t.hand_weighted(0.001)),
    dict(head="1 picture in 10 shows both", sub="a tenth of the training pictures",
         setup=lambda: t.hand_weighted(0.1)),
    dict(head="Captions decide the overlap", sub="three captions compete, giving about 1 in 22",
         setup=lambda: t.caption_experts(2)[0]),
]
ROWS = [("exact", "The maths"),
        ("reverse diffusion", "Plain"),
        ("annealed MALA", "Langevin"),
        ("annealed HMC", "Hamiltonian")]

import os, pickle
CACHE = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/notes/what-poe-loses-when-it-composes/.grid-cache.pkl"
cells, ACC = {}, {}
if os.path.exists(CACHE):
    cells, ACC = pickle.load(open(CACHE, "rb"))
    print("loaded cached draws from", CACHE)
for c in COLS if not cells else []:
    t.RNG = np.random.default_rng(SEED)
    kind, A, B = c["setup"]()
    lp, sc = t.composed(kind, A, B)
    runs = {"exact": t.exact_samples(kind, A, B), "reverse diffusion": t.reverse(sc)}
    t.ACCEPT.clear()
    runs["annealed MALA"] = t.reverse(sc, lp, mala=True)
    acc_mala = float(np.mean(t.ACCEPT)); t.ACCEPT.clear()
    h.ACCEPT_HMC.clear()
    runs["annealed HMC"] = h.reverse_hmc(sc, lp, hmc_k=1, hmc_l=20, eps_frac=0.1)
    acc_hmc = float(np.mean(h.ACCEPT_HMC)); h.ACCEPT_HMC.clear()
    ACC[c["head"]] = {"annealed MALA": round(acc_mala, 3), "annealed HMC": round(acc_hmc, 3)}
    for m, x in runs.items():
        labs = t.label(x)
        top = max(t.CATS, key=labs.count)
        cells[(c["head"], m)] = (np.clip(x[:4], 0, 1), labs.count("two objects, crisp"), top, labs.count(top))
    print("done", c["head"], flush=True)

if not os.path.exists(CACHE):
    pickle.dump((cells, ACC), open(CACHE, "wb"))

NC, NR = len(COLS), len(ROWS)
fig = plt.figure(figsize=(4.3 * NC, 2.25 * NR + 2.3))
gs = fig.add_gridspec(NR * 2 + 1, NC * 2 + 1, height_ratios=[0.52, 1] * NR + [1.15],
                      width_ratios=[0.22] + [1] * (NC * 2),
                      hspace=0.20, wspace=0.08, left=0.012, right=0.985, top=0.835, bottom=0.075)

for j, c in enumerate(COLS):
    fig.text(0.075 + (0.908 / NC) * (j + 0.5), 0.905, c["head"], ha="center", fontsize=13.5,
             fontweight="bold", color=INK)
    fig.text(0.075 + (0.908 / NC) * (j + 0.5), 0.878, c["sub"], ha="center", fontsize=10, color=FAINT)

for i, (m, label) in enumerate(ROWS):
    lax = fig.add_subplot(gs[i * 2:i * 2 + 2, 0]); lax.axis("off")
    lax.text(0.5, 0.5, label, rotation=90, va="center", ha="center", fontsize=11.5,
             fontweight="bold", color=INK, transform=lax.transAxes)
    for j, c in enumerate(COLS):
        imgs, crisp, top, topn = cells[(c["head"], m)]
        ok = top == "two objects, crisp"
        ax = fig.add_subplot(gs[i * 2, 1 + j * 2:3 + j * 2]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.80, transform=ax.transAxes,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#eaf4ec" if ok else "#fbeceb", edgecolor="none"))
        ax.text(0.5, 0.62, f"{top}  ·  {topn} of 300", ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=GOOD if ok else BAD, transform=ax.transAxes)
        sub2 = f"crisp pairs {cells[(c['head'], m)][1]} of 300"
        if m in ACC.get(c["head"], {}):
            sub2 += f"   ·   accepted {ACC[c['head']][m]:.0%} of its moves"
        ax.text(0.5, 0.28, sub2, ha="center", va="center", fontsize=8.5, color="#6b6b6b",
                transform=ax.transAxes)
        sub = gs[i * 2 + 1, 1 + j * 2:3 + j * 2].subgridspec(1, 4, wspace=0.06)
        for k in range(4):
            a = fig.add_subplot(sub[0, k])
            a.imshow(imgs[k].reshape(t.H, t.W, 3), interpolation="nearest"); a.axis("off")

ax = fig.add_subplot(gs[-1, 1:]); ax.axis("off")
verdicts = [
    "Nothing to find.\nThe maths itself says one merged shape,\nso no corrector can help.",
    "The maths is already right.\nAll three samplers miss it,\nbetter correctors included.",
    "The maths is still right.\nBoth correctors treble the plain rate,\nto 30 and 31 of 300. Momentum buys\nnothing over Langevin at four times\nthe cost, and 9 in 10 are still wrong.",
    "The maths is now wrong too:\nit prefers one shape,\nand every sampler agrees.",
]
for j, v in enumerate(verdicts):
    ax.text((j + 0.5) / NC, 0.95, v, ha="center", va="top", fontsize=10.5, color=INK,
            transform=ax.transAxes, linespacing=1.6)
ax.text(0.5, -0.12, "Every cell: 4 of 300 draws, the label most of the 300 were given, and the crisp-pair count beneath it. Green means most draws are a clean circle-and-square pair.",
        ha="center", fontsize=9.5, color=FAINT, transform=ax.transAxes)
ax.text(0.5, -0.26, "Per reverse step the Langevin corrector spends 5 gradient evaluations and the Hamiltonian one 20, so the bottom row costs four times the one above it.",
        ha="center", fontsize=9.5, color=FAINT, transform=ax.transAxes)

fig.suptitle("Turning up how often the two shapes appear together, and asking four samplers to draw it",
             fontsize=15.5, y=0.972)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "four-samplers-across-the-overlap-dial.png", dpi=110, facecolor="white")
print("wrote", OUT / "four-samplers-across-the-overlap-dial.png")
