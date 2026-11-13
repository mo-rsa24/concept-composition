"""The two dials on one pair of axes: agreement wins instantly, disagreement undoes it faster."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/home/molef/PhD/poe_repair_min-dog-and-rain/artifacts/results/how-much-co-occurrence-the-product-needs"
d = json.load(open(f"{OUT}/toy-leak-sweep.json"))
N, W = d["samples_per_cell"], d["pair_weight_held_at"]
cut = d["closed_form_leak_where_crisp_share_drops_below_half"]
xs = [x for x in sorted(float(k) for k in d["closed_form_curve"]) if 1e-4 <= x <= 0.2]
crisp = [d["closed_form_curve"][f"{x:.6g}"]["two objects, crisp"] for x in xs]
single = [d["closed_form_curve"][f"{x:.6g}"]["one circle"] + d["closed_form_curve"][f"{x:.6g}"]["one square"] for x in xs]
pts = {float(k): v for k, v in d["sampled_counts"].items()}

fig, ax = plt.subplots(figsize=(8.6, 5.1))
ax.plot(xs, crisp, color="#111111", lw=2.2, zorder=3)
ax.plot(xs, single, color="#b07d2b", lw=1.6, ls="-", zorder=2)
ax.annotate("a clean circle and square", (xs[0], crisp[0]), xytext=(10, -14), textcoords="offset points",
            fontsize=10.5, fontweight="bold", color="#111111", va="center")
ax.annotate("just one shape", (xs[-1], single[-1]), xytext=(-10, 10), textcoords="offset points",
            fontsize=10.5, fontweight="bold", color="#b07d2b", ha="right")
sx = [x for x in sorted(pts) if x > 0]
ax.plot(sx, [pts[x]["exact"]["two objects, crisp"] / N for x in sx], "o", ms=7, color="#111111", zorder=4)
ax.plot(sx, [(pts[x]["reverse diffusion"]["two objects, crisp"]) / N for x in sx], "s", ms=6,
        color="#c0392b", zorder=4)
ax.annotate("what a sampler actually draws, at every setting", (sx[0], pts[sx[0]]["reverse diffusion"]["two objects, crisp"] / N),
            xytext=(-10, 16), textcoords="offset points", fontsize=9.5, color="#c0392b", ha="right")
ax.axhline(0.5, color="#aaaaaa", lw=0.9, ls=":", zorder=1)
ax.axvline(cut, color="#9b2226", lw=1.3, ls="--", zorder=1)
ax.annotate(f"composition dies at a leak of {cut:.4f}\nhalf a percent of each expert's mass\non the other's shapes",
            (cut, 0.80), xytext=(10, 0), textcoords="offset points", fontsize=9.5, color="#9b2226", va="center")
ax.annotate(f"the agreement is held at {W:g} throughout:\none picture in ten shows both shapes,\nwhich on its own gives 300 of 300",
            (xs[0], 0.30), xytext=(10, 0), textcoords="offset points", fontsize=9.5, color="#666666", va="center")
ax.set_xscale("log"); ax.set_xlim(8e-5, 0.32); ax.set_ylim(-0.04, 1.08)
ax.set_xlabel("leak: each expert's weight on each one of the other's three shapes (log scale)")
ax.set_ylabel("share of the exact product, and of 300 draws")
ax.set_title("The second dial: how much the two experts may disagree before composition dies", fontsize=12.5)
ax.grid(alpha=0.18); [ax.spines[s].set_visible(False) for s in ("top", "right")]
fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.12)
fig.savefig(f"{OUT}/the-leak-dial.png", dpi=130)
print("wrote", f"{OUT}/the-leak-dial.png", "| cut", cut)
