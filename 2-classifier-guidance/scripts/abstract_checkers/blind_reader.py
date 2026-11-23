"""How much of the plurality label can a reader get from layout alone, with no idea which digits are present?

The rule predicts 1 when a canvas has two groups of inked columns separated by at least one fully empty column. It uses no
asked pair and no digit identity, and it reads clean canvases with no diffusion noise. The figure shows the unseen test split:
the first three canvases of each kind in the build's fixed shuffle, each kind's count and label, and how often the rule is right.

Usage:
    python blind_reader.py --data <dataset.npz> --out-json <json> --out-png <png>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KINDS = ("apart", "overlaid", "same_half", "single", "wrong")
NAMES = {"apart": "both digits, apart", "overlaid": "on the same spot", "same_half": "both in one half",
         "single": "one digit", "wrong": "apart, one digit wrong"}
INK, MUTED, GRID, BAR = "#1f2328", "#5f6670", "#e4e6ea", "#2f6fbd"


def has_gap(x):
    cols = (x > 0).any(1)
    first, last = cols.argmax(1), x.shape[2] - 1 - cols[:, ::-1].argmax(1)
    return np.array([not c[f:l + 1].all() for c, f, l in zip(cols, first, last)])


def measure(d):
    result = {}
    for split in ("train", "seen_test", "unseen_test"):
        x, y, k = d[f"{split}__x"], d[f"{split}__target"].astype(bool), d[f"{split}__kind"]
        p = has_gap(x)
        result[split] = {"canvases": len(x), "balanced_accuracy": round(float(0.5 * (p[y].mean() + (~p[~y]).mean())), 4),
                         "count_by_kind": {n: int((k == i).sum()) for i, n in enumerate(KINDS)},
                         "correct_by_kind": {n: round(float((p[k == i] == y[k == i]).mean()), 4) for i, n in enumerate(KINDS)}}
    return result


def draw(d, result, path):
    u = result["unseen_test"]
    fig = plt.figure(figsize=(11.5, 6.6), facecolor="white")
    grid = fig.add_gridspec(5, 5, width_ratios=[1, 1, 1, 1.2, 1.35], left=0.015, right=0.93, top=0.83, bottom=0.095,
                            wspace=0.07, hspace=0.2)
    for r, kind in enumerate(KINDS):
        where = np.flatnonzero(d["unseen_test__kind"] == KINDS.index(kind))[:3]
        for c, i in enumerate(where):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(d["unseen_test__x"][i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            a, b = d["unseen_test__asked"][i]
            ax.set_title(f"asked {a} and {b}", fontsize=8.5, color=MUTED, pad=2)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        t = fig.add_subplot(grid[r, 3])
        t.axis("off")
        label = int(kind == "apart")
        t.text(0.06, 0.66, NAMES[kind], fontsize=10, color=INK, weight="bold", va="center")
        t.text(0.06, 0.3, f"label {label}, {u['count_by_kind'][kind]:,} canvases", fontsize=9, color=MUTED, va="center")
        ax = fig.add_subplot(grid[r, 4])
        value = 100 * u["correct_by_kind"][kind]
        ax.barh([0], [100], height=0.34, color=GRID)
        if value > 0:
            ax.barh([0], [value], height=0.34, color=BAR)
        ax.text(101.5, 0, f"{value:.1f}%", va="center", fontsize=10.5, color=INK, weight="bold")
        ax.set_xlim(0, 100), ax.set_ylim(-0.5, 0.5), ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_visible(r == len(KINDS) - 1)
        ax.spines["bottom"].set_color(GRID)
        if r < len(KINDS) - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"], fontsize=8.5, color=MUTED)
            ax.tick_params(length=0)
            ax.set_xlabel("canvases the rule labels correctly", fontsize=9, color=MUTED)
    fig.text(0.015, 0.955, "A rule that cannot tell digits apart fails only on the wrong-digit canvases", fontsize=14.5,
             weight="bold", color=INK, va="top")
    fig.text(0.015, 0.905, f"Balanced accuracy {u['balanced_accuracy']:.3f} on {u['canvases']:,} clean canvases of unseen digits 5 to 9. "
             "Step V2's pass mark is 0.75.", fontsize=10.5, color=MUTED, va="top")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    result = measure(d)
    args.out_json.write_text(json.dumps(result, indent=1))
    draw(d, result, args.out_png)
    print(json.dumps({s: (v["balanced_accuracy"], v["correct_by_kind"]) for s, v in result.items()}))


if __name__ == "__main__":
    main()
