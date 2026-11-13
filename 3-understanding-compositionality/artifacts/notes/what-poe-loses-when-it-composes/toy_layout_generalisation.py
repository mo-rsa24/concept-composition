"""The third dial: can the product put the two shapes somewhere it has never seen them together?

Each expert holds its own three single shapes plus two side-by-side pair pictures, the circle on the
left in one and on the right in the other, both at weight --pair-weight / 2. Neither expert has ever
seen the circle and the square in the middle-and-right arrangement, or in any arrangement but those
two. If the product can only return one of the two it holds, it is retrieving rather than arranging.
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_sampler_hybrids as t
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
SEED, W = 20261022, 0.1

X = t.XPOS
LAYOUTS = {
    "circle left, square right": t.draw([("circle", X["left"]), ("square", X["right"])]),
    "square left, circle right": t.draw([("square", X["left"]), ("circle", X["right"])]),
    "circle left, square centre": t.draw([("circle", X["left"]), ("square", X["centre"])]),
    "circle centre, square right": t.draw([("circle", X["centre"]), ("square", X["right"])]),
}
TRAINED = ["circle left, square right", "square left, circle right"]
UNSEEN = [k for k in LAYOUTS if k not in TRAINED]

circ, sq = list(t.CIRCLES.values()), list(t.SQUARES.values())
pairs = [LAYOUTS[k] for k in TRAINED]
ws = [(1 - W) / 3] * 3 + [W / 2] * 2
A = t.Mixture(circ + pairs, ws)
B = t.Mixture(sq + pairs, ws)

t.RNG = np.random.default_rng(SEED)
lp, sc = t.composed("product", A, B)
runs = {"exact": t.exact_samples("product", A, B), "reverse diffusion": t.reverse(sc)}

names = list(LAYOUTS)
bank = np.stack([LAYOUTS[k] for k in names])
report = {}
GATE = 30.0   # an exact draw sits about 7.6 from its layout; an overlay or a single shape sits above 80
for m, x in runs.items():
    d = ((x[:, None, :] - bank[None]) ** 2).sum(-1)
    near_i, near_d = d.argmin(1), d.min(1)
    report[m] = {k: int(((near_i == i) & (near_d < GATE)).sum()) for i, k in enumerate(names)}
    report[m]["not close to any of the four layouts"] = int((near_d >= GATE).sum())
    report[m]["median distance to the nearest layout"] = round(float(np.median(near_d)), 2)
    print(m, json.dumps(report[m]), flush=True)
json.dump({"what": "two trained layouts, two the experts never saw; which one does the product return",
           "pair_weight_split_over_two_layouts": W, "samples_per_cell": t.N, "rng_seed": SEED,
           "trained_layouts": TRAINED, "unseen_layouts": UNSEEN, "gate_on_squared_distance": 30.0,
           "nearest_layout_counts": report},
          open(OUT / "toy-layout-generalisation.json", "w"), indent=1)

# the figure: the four layouts as references, then what each sampler returned
fig = plt.figure(figsize=(16.5, 7.4))
gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.15, 1.15], hspace=0.55, wspace=0.10,
                      left=0.055, right=0.985, top=0.80, bottom=0.08)
GOOD, BAD, INK, FAINT = "#1a7f37", "#9b2226", "#1b1b1b", "#7a7a7a"

for j, k in enumerate(names):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(LAYOUTS[k].reshape(t.H, t.W, 3), interpolation="nearest"); ax.axis("off")
    seen = k in TRAINED
    ax.set_title(f"{k}\n{'both experts hold this one' if seen else 'neither expert has ever seen this'}",
                 fontsize=10.5, color=INK if seen else FAINT,
                 fontweight="bold" if seen else "normal", pad=7)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), t.W, t.H, fill=False,
                               edgecolor="#1b1b1b" if seen else "#d8d8d4", lw=2.2 if seen else 1.2))

for i, m in enumerate(["exact", "reverse diffusion"]):
    for j, k in enumerate(names):
        ax = fig.add_subplot(gs[i + 1, j]); ax.axis("off")
        n = report[m][k]
        seen = k in TRAINED
        ax.add_patch(FancyBboxPatch((0.04, 0.18), 0.92, 0.64, transform=ax.transAxes,
                                    boxstyle="round,pad=0.02",
                                    facecolor="#eaf4ec" if (seen and n > 30) else ("#fbeceb" if n > 30 else "#f6f6f4"),
                                    edgecolor="none"))
        ax.text(0.5, 0.50, f"{n} of 300", ha="center", va="center", fontsize=19,
                fontweight="bold", color=GOOD if (seen and n > 30) else (BAD if n > 30 else "#bbbbb7"),
                transform=ax.transAxes)
    fig.text(0.016, (0.46, 0.17)[i],
             {"exact": "The maths", "reverse diffusion": "The sampler"}[m],
             fontsize=12.5, fontweight="bold", color=INK, rotation=90, va="center")

fig.suptitle("Given two arrangements it has seen, can the product make a third?", fontsize=16, y=0.955)
fig.text(0.015, 0.895,
         "Both experts hold the circle on the left and the circle on the right, each at half the pair weight of 0.1. "
         "Each draw is counted against whichever of the four layouts it is nearest.",
         fontsize=10.5, color=FAINT)
miss = {m: report[m]["not close to any of the four layouts"] for m in ("exact", "reverse diffusion")}
fig.text(0.015, 0.030,
         "A draw counts for a layout only if its squared distance to that layout is under 30. An exact draw sits about 7.6 away; an overlay or a single shape sits above 80.",
         fontsize=9.5, color=FAINT)
fig.text(0.015, 0.008,
         f"Draws close to none of the four layouts: {miss['exact']} of 300 for the maths, {miss['reverse diffusion']} of 300 for the sampler.   "
         "Green: an arrangement both experts hold. Grey: almost nothing landed there.",
         fontsize=9.5, color=FAINT)
fig.savefig(OUT / "can-it-make-a-third-arrangement.png", dpi=110, facecolor="white")
print("wrote", OUT / "can-it-make-a-third-arrangement.png")
