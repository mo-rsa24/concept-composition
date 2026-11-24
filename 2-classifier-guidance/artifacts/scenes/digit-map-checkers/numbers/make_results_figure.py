"""The results so far, one bar per checker and score, from numbers.json only. Writes results-so-far.png and .svg.

Bars are balanced accuracy on the pairs holding digit 4 (never trained on), over the 16 SDXL-matched timesteps,
4 noise draws each; 0.5 is chance and the bar the design set is 0.75. A result not yet written is a dashed empty slot.

Run: ~/.claude/venvs/figures/bin/python -I make_results_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "numbers.json").read_text())
R = N["results_printed"]
COUNT, IDENT, GREY, INK, COMB = "#7b3fa0", "#0f7f86", "#6b7a88", "#20313c", "#8a6d3b"

comb = R["combined_checker_sdxl_noise"]["map_reader"]
rows = [("one combined checker, all five kinds\n(job 94080, map reader)", comb["balanced_accuracy_pairs_holding_4"], COMB),
        ("one combined checker, apart against wrong\n(job 94080, map reader)", comb["apart_vs_wrong"], COMB)]
log = {(r["task"], r["reader"]): r for r in R["split_run"].get("from_job_log", {}).get("rows", [])}
split = R["split_run"].get("summary")
for task, col in (("count", COUNT), ("identity", IDENT)):
    for reader in ("map", "raw"):
        label = f"{task.upper()} checker, {'map reader' if reader == 'map' else 'raw-canvas reader'}\n(job 94085)"
        if split:
            v = split[task][reader]["holding_4_ba"]
        elif (task, reader) in log:
            v = log[(task, reader)]["holding_4_ba"]
        else:
            v = None
        rows.append((label, v, col))

fig, ax = plt.subplots(figsize=(11, 5.6), dpi=100)
for i, (label, v, col) in enumerate(rows):
    y = len(rows) - 1 - i
    if v is None:
        ax.barh(y, 0.5, left=0.4, height=0.6, facecolor="none", edgecolor=col, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(0.65, y, "not written yet: owed by job 94085's split.json", va="center", ha="center", fontsize=10, color=col)
    else:
        ax.barh(y, v - 0.4, left=0.4, height=0.6, color=col, alpha=0.85)
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=11, color=INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10)
ax.axvline(0.5, color=GREY, linewidth=1)
ax.axvline(0.75, color=INK, linewidth=1, linestyle="--")
ax.text(0.504, len(rows) - 0.35, "chance 0.5", ha="left", fontsize=10, color=GREY)
ax.text(0.754, len(rows) - 0.35, "bar 0.75", ha="left", fontsize=10, color=INK)
ax.set_xlim(0.4, 0.9)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_xlabel("balanced accuracy on pairs holding digit 4, over the 16 SDXL-matched timesteps", fontsize=11)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("What a checker reaches on digits it never trained on", fontsize=14, loc="left", color=INK)
fig.tight_layout()
fig.savefig(HERE / "results-so-far.png", dpi=200)
fig.savefig(HERE / "results-so-far.svg")
print("wrote results-so-far.png:", [(r[0].split("\n")[0], r[1]) for r in rows])
