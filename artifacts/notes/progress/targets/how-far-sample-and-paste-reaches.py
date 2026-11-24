"""How many objects sample-and-paste reaches under Bradley's rule, if each pasted object is right independently with rate p."""
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/molef/PhD/concept-composition/artifacts/notes/progress/targets/how-far-sample-and-paste-reaches.png"


def at_least(n, k, p):
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def kmax(p, top=12):
    best = 0
    for k in range(1, top + 1):
        if p**k >= 0.5 and at_least(k, max(k - 2, 0), p) >= 0.9:
            best = k
        else:
            break
    return best


ps = np.linspace(0.80, 1.0, 2001)
ks = [kmax(float(p)) for p in ps]
first7 = float(ps[next(i for i, k in enumerate(ks) if k >= 7)])
first12 = float(ps[next(i for i, k in enumerate(ks) if k >= 12)])
print("first p reaching 7:", round(first7, 4), "first p reaching 12:", round(first12, 4))

INK, MUTED, LINE, MARK = "#222222", "#666666", "#2a6fdb", "#c2410c"
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
ax.step(ps * 100, ks, where="post", color=LINE, lw=2.2)
ax.axhline(7, color=MARK, lw=1.4, ls="--")
ax.text(80.2, 7.2, "the target: 7 objects", color=MARK, fontsize=10)
ax.text(80.2, 12.15, "the test stops at 12 objects", color=MUTED, fontsize=9)
for x, label, y in ((90.0, "each expert's\npass mark, 90%", 2.4), (94.7, "our MNIST experts,\nabout 94.7%", 5.0)):
    ax.axvline(x, color=MUTED, lw=1, ls=":")
    ax.text(x - 0.15, y, label, ha="right", color=INK, fontsize=9)
ax.set_xlim(80, 100)
ax.set_ylim(0, 13)
ax.set_yticks(range(0, 13, 2))
ax.set_xlabel("How often one pasted object is right (%)", color=INK)
ax.set_ylabel("Most objects that succeed", color=INK)
ax.set_title(f"Pasting alone reaches 7 objects once each object is right {first7 * 100:.0f}% of the time",
             loc="left", fontsize=12, fontweight="bold", color=INK)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#eeeeee")
ax.plot([100], [12], marker="o", color=MARK, ms=7, clip_on=False, zorder=5)
ax.text(99.6, 8.4, "CLEVR paste run, 2 and 3 experts:\nevery object placed,\n1,024 of 1,024 images", ha="right", color=MARK, fontsize=9)
fig.text(0.01, 0.005, "Curve calculated from Bradley's rule, assuming each pasted object succeeds independently.\n"
         "Dot: the measured CLEVR paste rate at 2 and 3 experts (method-search/01c-clevr-paste-first/review.md).",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT)
print(OUT)
