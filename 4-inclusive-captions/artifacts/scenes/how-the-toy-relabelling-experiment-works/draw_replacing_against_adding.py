"""Network view 13b, the size check: replacing against adding, at shares near 10% and at 50%.

Rows: the two designs at each share. Columns: the three seeds. Each cell: that training's first three
product samples and its count of separate pairs out of 300, read from the result files, never typed.

    python3 draw_replacing_against_adding.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RES = REPO / "4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose"
ORIG = REPO / "3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs"
KEY = json.load(open(HERE / "ladder.json"))["colour_key"]
C_IN, C_PUSH, C_OFF = KEY["inputs"], KEY["the method's own part: the product of the two captions and the corrector's push"], KEY["off, fixed or empty (the empty caption, a frozen VAE)"]
SEEDS = [20261022, 20261023, 20261024]

rep = json.load(open(RES / "toy-caption-replace.json"))
add200 = json.load(open(RES / "toy-caption-threshold.json"))["results"]["200"]
add2000 = json.load(open(RES / "toy-caption-threshold-600-1200-2000.json"))["results"]["2000"]
rep_npz = np.load(RES / "toy-caption-replace-samples.npz")
add_npz = np.load(ORIG / "toy-caption-threshold-samples.npz")
add_npz2 = np.load(ORIG / "toy-caption-threshold-600-1200-2000-samples.npz")


def row(runs, npz, k, size, share, design):
    by = {r["seed"]: r for r in runs}
    return {"label": f"{design}\nshare {share}\n{size:,} training pictures",
            "cells": [(npz[f"{k}_{s}_product"][:3], by[s]["counts"]["product"]["red and blue apart"]) for s in SEEDS],
            "design": design}


rows = [row(add200, add_npz, 200, 6400, "9.1%", "adding 200"),
        row(rep["results"]["200"], rep_npz, 200, 6000, "10%", "replacing 200"),
        row(add2000, add_npz2, 2000, 10000, "50%", "adding 2,000"),
        row(rep["results"]["1000"], rep_npz, 1000, 6000, "50%", "replacing 1,000")]

fig = plt.figure(figsize=(17, 9.6), facecolor="white")
fig.suptitle("Replacing keeps the training set at 6,000 and the product still draws the pair at a 50% share",
             fontsize=15, fontweight="bold", color="#1f2a36", y=0.975)
fig.text(0.5, 0.925, "each cell: one training's first three product samples of “a red circle” × “a blue square”, "
         "and its separate pairs out of 300 (orange: at least 30, the bar)", ha="center", fontsize=10.5, color="#5b6670")

left, top, cw, rh = 0.17, 0.86, 0.265, 0.19
for j, s in enumerate(SEEDS):
    fig.text(left + j * cw + cw * 0.46, top + 0.01, f"training {j + 1} (seed {s})", ha="center", fontsize=10.5,
             fontweight="bold", color="#1f2a36")
for i, r in enumerate(rows):
    y = top - (i + 1) * rh + 0.02 - (0.03 if i >= 2 else 0)
    fig.text(0.02, y + rh * 0.42, r["label"], fontsize=10.5, va="center", linespacing=1.5,
             color=C_IN if r["design"].startswith("replacing") else "#1f2a36",
             fontweight="bold" if r["design"].startswith("replacing") else "normal")
    for j, (tiles, n) in enumerate(r["cells"]):
        for t in range(3):
            ax = fig.add_axes([left + j * cw + t * 0.068, y + 0.035, 0.062, rh * 0.62])
            ax.imshow(np.clip(tiles[t].reshape(16, 32, 3), 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(C_OFF); sp.set_linewidth(0.8)
        fig.text(left + j * cw + 0.215, y + 0.035 + rh * 0.31, f"{n}", fontsize=15, fontweight="bold",
                 va="center", ha="center", color=C_PUSH if n >= 30 else "#1f2a36")
fig.add_artist(plt.Line2D([0.02, 0.98], [top - 2 * rh + 0.005, top - 2 * rh + 0.005], color=C_OFF, lw=0.6, ls=":"))
fig.text(0.5, 0.045, "rule written before the run, judged at 50% only: all three replacing models at least 30 of 300 means the share, "
         f"not the training set size, drives the result.  Result: {', '.join(map(str, rep['judged separate counts']))}: {rep['verdict']}",
         ha="center", fontsize=10.5, color="#1f2a36",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C_PUSH, lw=1.2))
fig.text(0.5, 0.012, "adding runs are the reported GPU trainings; replacing runs were trained on CPU with the same code and seeds, "
         "so seed-for-seed numbers are not paired", ha="center", fontsize=9, color="#5b6670")

out = HERE / "network-views" / "13b-the-size-check-at-two-shares"
fig.savefig(f"{out}.png", dpi=170)
fig.savefig(f"{out}.svg")
print("wrote", out)
