"""Loop 8's digit results in three panels, from numbers.json's results_final block only. Writes results-so-far.png and .svg.

Left: balanced accuracy of each checker on digits it never trained on (0.5 is chance, 0.75 the bar).
Middle: clean pairs of 320 canvases on the 10 unseen pairs among digits 5 to 9, for the plain product and three pushes.
Right: how far each property's score falls when its checker is removed, against a random push of the same size in its
place, with the 6.25-point bar the design set.

Run: ~/.claude/venvs/figures/bin/python -I make_results_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "numbers.json").read_text())["results_final"]
COUNT, IDENT, ORANGE, GREY, INK, ORDER = "#7b3fa0", "#0f7f86", "#c86a0a", "#6b7a88", "#20313c", "#8a6d3b"

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.8), dpi=100, gridspec_kw={"width_ratios": [1.15, 1, 1.05]})

# ------------------------------------------------------------------ left: what each checker reads
c, i, rd = R["count_checker"], R["identity_checker"], R["right_digits_checker"]
rows = [("count checker, digits 5 to 9\n(trained on 0 to 4, job 94108)", c["map"]["digits_5_to_9"], COUNT),
        ("count checker, digits 0 to 4\n(the pairs it trained on)", c["map"]["digits_0_to_4"], COUNT),
        ("identity checker, pairs holding 4\n(trained on 0 to 3, job 94085)", i["pairs_holding_4"], IDENT),
        ("right-digits checker, digits 5 to 9\n(trained on 0 to 4, job 94153)", rd["digits_5_to_9"], IDENT)]
for k, (label, v, col) in enumerate(rows):
    y = len(rows) - 1 - k
    a1.barh(y, v - 0.4, left=0.4, height=0.6, color=col, alpha=0.85)
    a1.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=11, color=INK)
a1.set_yticks(range(len(rows)))
a1.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
a1.axvline(0.5, color=GREY, linewidth=1)
a1.axvline(0.75, color=INK, linewidth=1, linestyle="--")
a1.text(0.503, len(rows) - 0.45, "chance 0.5", fontsize=9.5, color=GREY)
a1.text(0.753, len(rows) - 0.45, "bar 0.75", fontsize=9.5, color=INK)
a1.set_xlim(0.4, 0.86)
a1.set_ylim(-0.6, len(rows) - 0.2)
a1.set_xlabel("balanced accuracy over the 16 SDXL-matched timesteps", fontsize=10)
a1.set_title("Where two digits sit is readable;\nwhich digits they are is not", fontsize=12, loc="left", color=INK)

# ------------------------------------------------------------------ middle: what the push draws
p = R["push"]
bars = [("plain", p["clean_pairs"]["plain"], GREY), ("random push,\nsame length", p["clean_pairs"]["random_push_same_length"], GREY),
        ("best of\n4 seeds", p["clean_pairs"]["best_of_4_seeds"], GREY), ("checker's push,\nb = 0.4", p["clean_pairs"]["push"], ORANGE)]
xs = range(len(bars))
for x, (label, v, col) in zip(xs, bars):
    a2.bar(x, v, width=0.6, color=col, alpha=0.85)
    if col == ORANGE:
        a2.text(x - 0.36, v, str(v), ha="right", va="center", fontsize=13, color=INK, fontweight="bold")
    else:
        a2.text(x, v + 1.2, str(v), ha="center", fontsize=12, color=INK)
err = 1.96 * p["standard_error_of_lead"]
a2.errorbar(3, p["clean_pairs"]["push"], yerr=err, color=INK, capsize=6, linewidth=1.2)
a2.text(3.36, p["clean_pairs"]["push"] + err - 2, f"±1.96 standard errors\n({err:.1f})", fontsize=8.5, color=INK, va="top")
a2.scatter([3], [p["rerun_clean_pairs_push"]], marker="_", s=500, color=INK, zorder=5)
a2.text(3.36, p["rerun_clean_pairs_push"], f"rerun on another\nnode: {p['rerun_clean_pairs_push']}", fontsize=8.5,
        color=INK, ha="left", va="center")
a2.set_xticks(list(xs))
a2.set_xticklabels([b[0] for b in bars], fontsize=9.5)
a2.set_xlim(-0.6, 4.1)
a2.set_ylim(0, 64)
a2.set_ylabel(f"clean pairs of {p['canvases']}: two clear, right digits, one per half", fontsize=10)
a2.set_title("On digits 5 to 9, the push draws clean pairs\nthe plain product never draws", fontsize=12, loc="left", color=INK)

# ------------------------------------------------------------------ right: removing one checker
rm = R["remove_one"]
names = list(rm["rows"])
w = 0.36
for k, name in enumerate(names):
    r = rm["rows"][name]
    a3.bar(k - w / 2, r["own_fall"], width=w, color=COUNT if name == "count" else IDENT if name == "right digits" else ORDER, alpha=0.85)
    a3.bar(k + w / 2, r["random_fall"], width=w, color="none", edgecolor=GREY, hatch="///", linewidth=1)
    a3.text(k - w / 2, r["own_fall"] + 0.8, f"{r['own_fall']}", ha="center", fontsize=10, color=INK)
    a3.text(k + w / 2, r["random_fall"] + 0.8, f"{r['random_fall']}", ha="center", fontsize=10, color=GREY)
a3.axhline(rm["own_bar_points"], color=INK, linewidth=1, linestyle="--")
a3.text(0.5, rm["own_bar_points"] + 0.8, f"bar {rm['own_bar_points']}", fontsize=9, color=INK, ha="center")
a3.set_xticks(range(len(names)))
a3.set_xticklabels([f"remove {n}\nother column:\n{rm['rows'][n]['largest_other_move']}" for n in names], fontsize=9)
a3.set_ylim(0, 54)
a3.set_xlim(-0.6, 2.6)
a3.set_ylabel(f"fall in its own column, points ({rm['canvases_per_row']} canvases per row)", fontsize=10)
a3.legend(handles=[Patch(facecolor=GREY, alpha=0.85, label="checker removed"),
                   Patch(facecolor="none", edgecolor=GREY, hatch="///", label="random push in its place")],
           fontsize=8.5, frameon=False, loc="upper left")
a3.set_title("Removing one checker: null by the written rule\n(none meets its bar and the 4.2-point limit)",
             fontsize=12, loc="left", color=INK)

for ax in (a1, a2, a3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Loop 8 on digits: what the checkers read, what the count checker's push draws, and what removing one checker does",
             fontsize=14, x=0.01, ha="left", color=INK)
fig.text(0.01, 0.01, "Sources: 05-does-it-learn, 06-does-it-compose and 08-remove-one-checker reviews under "
         "artifacts/notes/progress/abstract-checkers/; count checker scores read from numbers/real/count-final.json.",
         fontsize=8.5, color=GREY)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(HERE / "results-so-far.png", dpi=200)
fig.savefig(HERE / "results-so-far.svg")
print("wrote results-so-far.png:", [(r[0].split("\n")[0], r[1]) for r in rows], [(b[0].replace("\n", " "), b[1]) for b in bars],
      {n: rm["rows"][n]["own_fall"] for n in names})
