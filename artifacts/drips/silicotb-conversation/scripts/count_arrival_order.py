"""Which NIH finding arrives after which, counted over patients with two or more films.

Reads only the metadata CSV: no images, no model. For every ordered pair of findings
(earlier, later) it counts the patients whose first film carried `earlier` and whose some
later film carried `later` while the first film did not. The matrix is asymmetric by
construction, and its asymmetry is the whole point: it says whether the order two findings
appear in is visible in the labels at all.
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
TARGET = ("Fibrosis", "Pneumonia")

d = pd.read_csv(CSV)
d["labels"] = d["Finding Labels"].fillna("").apply(
    lambda s: {t for t in s.split("|") if t and t != "No Finding"}
)
d = d.sort_values(["Patient ID", "Follow-up #"])

findings = sorted({f for s in d["labels"] for f in s})
n = len(findings)
idx = {f: i for i, f in enumerate(findings)}

counts = np.zeros((n, n), dtype=int)   # [earlier, later]
patients_total = d["Patient ID"].nunique()
patients_multi = 0
patients_changed = 0
new_findings_per_patient = []

for _, g in d.groupby("Patient ID", sort=False):
    films = list(g["labels"])
    if len(films) < 2:
        continue
    patients_multi += 1
    baseline = films[0]
    gained = set().union(*films[1:]) - baseline
    lost = baseline - set().union(*films[1:])
    if gained or lost:
        patients_changed += 1
    if gained:
        new_findings_per_patient.append(len(gained))
    for e in baseline:
        for l in gained:
            counts[idx[e], idx[l]] += 1

summary = {
    "csv": str(CSV),
    "patients_total": int(patients_total),
    "patients_with_two_or_more_films": int(patients_multi),
    "patients_whose_labels_change": int(patients_changed),
    "films": int(len(d)),
    "findings": findings,
    "matrix_rows_are": "finding present on the patient's first film",
    "matrix_cols_are": "finding gained on a later film and absent from the first",
    "matrix": counts.tolist(),
    "target_pair": {
        "fibrosis_then_pneumonia": int(counts[idx["Fibrosis"], idx["Pneumonia"]]),
        "pneumonia_then_fibrosis": int(counts[idx["Pneumonia"], idx["Fibrosis"]]),
    },
    "mean_new_findings_per_changing_patient": (
        round(float(np.mean(new_findings_per_patient)), 2) if new_findings_per_patient else 0.0
    ),
}
(OUT / "arrival_order.json").write_text(json.dumps(summary, indent=2))

fig, ax = plt.subplots(figsize=(15.2, 10.8))
shown = np.ma.masked_where(counts == 0, counts)
im = ax.imshow(shown, cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=max(counts.max(), 2)))

for i in range(n):
    for j in range(n):
        v = counts[i, j]
        if v == 0:
            ax.text(j, i, "0", ha="center", va="center", fontsize=7, color="#b0b0b0")
            continue
        strong = v >= 0.45 * counts.max()
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7.5,
                color="white" if strong else "#1a1a1a",
                fontweight="bold" if i == j else "normal")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(findings, rotation=42, ha="right", fontsize=9)
ax.set_yticklabels(findings, fontsize=9)
for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    if lab.get_text() in TARGET:
        lab.set_color("#c0152f")
        lab.set_fontweight("bold")

for (r, c), colour in (((idx[TARGET[0]], idx[TARGET[1]]), "#1452c0"),
                       ((idx[TARGET[1]], idx[TARGET[0]]), "#1452c0")):
    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2.4, ec=colour))

fp = counts[idx["Fibrosis"], idx["Pneumonia"]]
pf = counts[idx["Pneumonia"], idx["Fibrosis"]]

ax.set_xlabel("gained on a later film, absent from the first", fontsize=11, labelpad=12)
ax.set_ylabel("present on the patient's first film", fontsize=11)
ax.set_title("Which NIH finding arrives after which", fontsize=14, pad=14)

cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.045, shrink=0.52, anchor=(0.0, 0.0))
cb.set_label("patients, log scale\n(every cell prints its true count)", fontsize=8.5)

ax.annotate(
    f"the pair the conversation picks\n"
    f"fibrosis first, then pneumonia:  {fp}\n"
    f"pneumonia first, then fibrosis:  {pf}\n"
    "too few either way to score",
    xy=(idx[TARGET[1]] + 0.5, idx[TARGET[0]]),
    xycoords="data",
    xytext=(1.045, 0.995), textcoords="axes fraction",
    fontsize=9.5, color="#1452c0", ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#1452c0", lw=1.3),
    arrowprops=dict(arrowstyle="-", color="#1452c0", lw=1.2,
                    connectionstyle="arc3,rad=-0.12"),
)

busiest = np.unravel_index(np.argmax(counts), counts.shape)
ax.annotate(
    f"busiest cell: {findings[busiest[0]].lower()} first,\n"
    f"then {findings[busiest[1]].lower()}, {counts[busiest]} patients",
    xy=(busiest[1], busiest[0] + 0.5), xycoords="data",
    xytext=(1.045, 0.60), textcoords="axes fraction",
    fontsize=9.0, color="#7a3d00", ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff6e8", ec="#c97a1f", lw=1.1),
    arrowprops=dict(arrowstyle="-", color="#c97a1f", lw=1.1,
                    connectionstyle="arc3,rad=0.14"),
)

ax.annotate(
    "the diagonal is blank by construction:\n"
    "a finding cannot be gained after itself",
    xy=(idx["Hernia"], idx["Hernia"]), xycoords="data",
    xytext=(1.045, 0.34), textcoords="axes fraction",
    fontsize=9.0, color="#555", ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#999", lw=1.0),
    arrowprops=dict(arrowstyle="-", color="#999", lw=1.0,
                    connectionstyle="arc3,rad=0.1"),
)

fig.text(
    0.5, 0.055,
    f"{summary['patients_whose_labels_change']:,} of {summary['patients_with_two_or_more_films']:,} "
    f"patients with two or more films changed label, out of {patients_total:,} patients and {len(d):,} films. "
    "One patient counts once: their first film's findings against everything they gained later.\n"
    "The matrix is asymmetric on purpose, so A-then-B and B-then-A are separate cells. "
    "The labels are report-derived and noisy, and a gained label can mean a new finding or a different reader.",
    ha="center", va="top", fontsize=9.0, color="#444",
)

fig.subplots_adjust(left=0.13, right=0.78, top=0.94, bottom=0.21)
fig.savefig(OUT / "which-findings-arrive-after-which.png", dpi=170)
print(json.dumps(summary["target_pair"], indent=2))
print("patients with 2+ films:", patients_multi, "changed:", patients_changed)
