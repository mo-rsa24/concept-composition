"""How often each pair of NIH findings is labelled on the same film.

The plain co-occurrence matrix, symmetric, beside the arrival-order one. The diagonal is how
many films carry that finding at all, so every off-diagonal cell can be read against the two
single-finding totals it sits between.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

CSV = Path("/datasets/mmolefe/data/nih/Data_Entry_2017.csv")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
NAMED = [("Pneumonia", "Fibrosis"), ("Fibrosis", "Emphysema"),
         ("Cardiomegaly", "Effusion"), ("Effusion", "Atelectasis")]

d = pd.read_csv(CSV)
d["set"] = d["Finding Labels"].fillna("").apply(
    lambda s: {t for t in s.split("|") if t and t != "No Finding"}
)
findings = sorted({f for s in d["set"] for f in s})
n = len(findings)
idx = {f: i for i, f in enumerate(findings)}


def matrix(frame):
    m = np.zeros((n, n), dtype=int)
    for s in frame["set"]:
        js = [idx[f] for f in s]
        for a in js:
            for b in js:
                m[a, b] += 1
    return m


all_views = matrix(d)
pa_only = matrix(d[d["View Position"] == "PA"])

summary = {
    "csv": str(CSV),
    "films_all_views": int(len(d)),
    "films_pa": int((d["View Position"] == "PA").sum()),
    "findings": findings,
    "diagonal_is": "films carrying that finding at all",
    "offdiagonal_is": "films carrying both findings",
    "matrix_all_views": all_views.tolist(),
    "matrix_pa_only": pa_only.tolist(),
    "named_pairs": {
        f"{a}+{b}": {"all_views": int(all_views[idx[a], idx[b]]),
                     "pa_only": int(pa_only[idx[a], idx[b]])}
        for a, b in NAMED
    },
}
(OUT / "cooccurrence.json").write_text(json.dumps(summary, indent=2))

fig, ax = plt.subplots(figsize=(14.6, 10.6))
shown = np.ma.masked_where(all_views == 0, all_views)
im = ax.imshow(shown, cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=all_views.max()))

for i in range(n):
    for j in range(n):
        v = all_views[i, j]
        if v == 0:
            ax.text(j, i, "0", ha="center", va="center", fontsize=7, color="#b0b0b0")
            continue
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7.6,
                color="white" if v >= 0.25 * all_views.max() else "#1a1a1a",
                fontweight="bold" if i == j else "normal")

named_terms = {t for pair in NAMED for t in pair}
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(findings, rotation=42, ha="right", fontsize=9)
ax.set_yticklabels(findings, fontsize=9)
for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    if lab.get_text() in named_terms:
        lab.set_color("#c0152f"); lab.set_fontweight("bold")

for a, b in NAMED:
    for r, c in ((idx[a], idx[b]), (idx[b], idx[a])):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2.2, ec="#1452c0"))

lines = "\n".join(
    f"{a.lower()} + {b.lower()}:  {all_views[idx[a], idx[b]]:,} films  ({pa_only[idx[a], idx[b]]:,} PA)"
    for a, b in NAMED
)
ax.text(
    1.045, 0.99,
    "the four pairs the conversation names,\nboxed in the matrix\n" + lines,
    transform=ax.transAxes,
    fontsize=9.6, color="#1452c0", ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#1452c0", lw=1.3),
)

ax.set_title("How often two NIH findings are labelled on the same film", fontsize=14, pad=14)
ax.set_xlabel("finding", fontsize=11, labelpad=12)
ax.set_ylabel("finding", fontsize=11)
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.045, shrink=0.52, anchor=(0.0, 0.0))
cb.set_label("films, log scale\n(every cell prints its true count)", fontsize=8.5)

fig.text(
    0.5, 0.055,
    f"{len(d):,} films, {summary['films_pa']:,} of them posteroanterior. The diagonal, in bold, is how many films carry "
    "that finding at all, so an off-diagonal cell can be read against the two totals it sits between.\n"
    "The matrix is symmetric: it says nothing about which finding came first, which is the separate "
    "arrival-order matrix beside it. Labels are report-derived and noisy, and one film can carry several findings.",
    ha="center", va="top", fontsize=9.0, color="#444",
)
fig.subplots_adjust(left=0.13, right=0.78, top=0.94, bottom=0.21)
fig.savefig(OUT / "how-often-two-findings-share-a-film.png", dpi=170)
print(json.dumps(summary["named_pairs"], indent=2))
