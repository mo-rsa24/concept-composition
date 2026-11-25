"""Level 2: the checker push on the PoE ladder's toy. Two panels, both from data/paper1_numbers.json.

Left: two digits apart, of 4,000 draws from the same starting noise, against the push strength lambda (0, 0.5, 1, 2, 4, 10,
drawn evenly spaced), one line for the push on the product and one for the push on the region merge.
Right: along the sampler step (0 noisiest, 49 clean), the median cosine between the exact checker's gradient and the
interaction term I_t, at 2,000 states drawn from the joint's own noised canvases.

Run: ~/.claude/venvs/figures/bin/python scripts/make_toy_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figure_kit as K  # noqa: E402
from figure_kit import KEY, S  # noqa: E402

N = K.NUMBERS
P = N["toy_push"]
names = list(P["two_apart_of_4000"])
K.reset()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.06, right=0.985, bottom=0.13, top=0.84, wspace=0.22)
fig.patch.set_facecolor("white")

xs = np.arange(len(P["lambdas"]))
for name, colour, lab, dy in ((names[0], KEY["fixed"], "push on the plain product", -330),
                              (names[1], KEY["experts"], "push on the region merge", 260)):
    ys = P["two_apart_of_4000"][name]
    a1.plot(xs, ys, "-o", color=colour, lw=2.0, ms=6, zorder=3)
    K._LINES.append((lab, list(zip(xs.tolist(), ys)), a1))
    for x, y in zip(xs, ys):
        a1.text(x, 480 if colour == KEY["fixed"] else 180, f"{y:,}", ha="center", va="center", fontsize=8.8, color=colour)
a1.text(1.15, 2350, "push on the plain product", color=KEY["fixed"], fontsize=10, weight="bold", ha="left")
a1.text(2.3, 3450, "push on the region merge", color=KEY["experts"], fontsize=10, weight="bold", ha="left")
a1.axhline(4000, color=KEY["outputs"], lw=1.0, ls=(0, (4, 3)), zorder=1)
a1.text(0.1, 4140, "the joint's own score: 4,000", color=KEY["outputs"], fontsize=8.8, ha="left")
a1.set_xticks(xs, [f"{l:g}" for l in P["lambdas"]])
a1.set_xlabel(r"push strength $\lambda$ (evenly spaced, not to scale)", fontsize=10)
a1.set_ylabel("two digits apart, of 4,000 draws", fontsize=10)
a1.set_ylim(0, 4450); a1.set_xlim(-0.35, len(xs) - 0.65)
a1.set_title("The push lifts the product and helps the merge only a little", fontsize=11.5, weight="bold", loc="left")
for s in ("top", "right"):
    a1.spines[s].set_visible(False)

prof = P["gradient_vs_interaction_term"]
steps = [p["step"] for p in prof]
cos = [p["median_cosine_I_vs_checker_gradient"] for p in prof]
ratio = [p["median_norm_ratio_checker_over_I"] for p in prof]
a2.plot(steps, cos, color=KEY["push"], lw=2.2, zorder=3)
a2.plot(steps, ratio, color=KEY["fixed"], lw=1.6, ls=(0, (4, 2)), zorder=3)
K._LINES.append(("cosine", list(zip(steps, cos)), a2)); K._LINES.append(("ratio", list(zip(steps, ratio)), a2))
a2.text(20.5, 0.98, r"cosine between the push direction and $I_t$", color=KEY["push"], fontsize=9.6, ha="left", weight="bold")
a2.text(1, 0.575, r"length of the checker gradient over the length of $I_t$", color=KEY["fixed"], fontsize=9.2, ha="left")
a2.set_xlabel("sampler step (0 is noisiest, 49 is clean)", fontsize=10)
a2.set_ylabel("median over 2,000 states", fontsize=10)
a2.set_ylim(-0.05, 1.08); a2.set_xlim(0, 49)
a2.set_title(r"Early, the push points along the dropped term $I_t$", fontsize=11.5, weight="bold", loc="left")
for s in ("top", "right"):
    a2.spines[s].set_visible(False)

fig.text(0.5, 0.975, "The checker push on the PoE ladder's toy: an exact checker trained on pastes of the two toy experts' samples",
         ha="center", va="top", fontsize=13, weight="bold", color=S["ink"])
fig.text(0.5, 0.025, "same starting noise as the PoE ladder; its counts for the product (811) and the merge with the other cell hidden (3,859) "
         "are reproduced at push strength 0", ha="center", fontsize=8.8, color=KEY["fixed"])
K.save(fig, str(K.SCENE / "figures" / "level2-the-push-on-the-toy"))
