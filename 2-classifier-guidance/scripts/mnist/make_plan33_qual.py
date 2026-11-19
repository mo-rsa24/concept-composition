"""Plan 33: the same eight seeds under each way of composing plan 32's two experts, real canvases, the judge's call outlined.
Rows are conditions, columns are seeds 0 to 7 of one pair; nothing is picked by hand. Run with the sdxl-local python from 2-classifier-guidance/."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts/mnist")
from judge import load as load_judge
import two_experts as te
Q = Path("data/mnist-canvas/33-mnist-experts-at-inference/qual")
J = Path("data/mnist-canvas/32-mnist-two-experts/judge-v3/judge.pt")
net = load_judge(J, "cpu")
ROWS = [("e1-union", "each expert on the whole canvas,\nbase = a model of both ranges"),
        ("e0-region", "each expert on the whole canvas,\nonly its half of the answer kept"),
        ("e0-crop", "each expert sees ONLY its half\n(no pushes)"),
        ("crop-nc-both", "each expert sees only its half\n+ both pushes"),
        ("ft-frag-crop", "experts fine-tuned on fragment clutter,\nhalf-canvas, no pushes")]
PAIRS = ["4-7", "1-9", "2-8"]
fig, axs = plt.subplots(len(PAIRS) * len(ROWS), 8, figsize=(15, 5.4 * len(PAIRS)))
for pi, pair in enumerate(PAIRS):
    a, b = (int(v) for v in pair.split("-"))
    for ri, (name, lab) in enumerate(ROWS):
        c = np.load(Q / name / "canvases.npz")[pair]
        j = json.load(open(Q / name / "counts.json"))
        n = j["per_pair"][pair]["two clear, right, apart"]
        for k in range(8):
            ax = axs[pi * len(ROWS) + ri, k]
            ok = te.call_two(te.reads(net, c[k], "cpu"), (a, b)) == te.CALLS[0]
            ax.imshow(c[k], cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#1d6b3a" if ok else "#b8c0c8"); sp.set_linewidth(3 if ok else 0.8)
        axs[pi * len(ROWS) + ri, 0].set_ylabel(f"{lab}\n{n} of 64", rotation=0, ha="right", va="center", fontsize=11)
    axs[pi * len(ROWS), 3].set_title(f"asked {a} and {b}", fontsize=12, loc="left", color="#c8680a", weight="bold")
fig.suptitle("Plan 33: two digit experts that never met, composed in five ways. Same eight seeds in every row; green = the judge calls two clear, right digits", fontsize=12)
fig.subplots_adjust(left=0.24, right=0.995, top=0.965, bottom=0.005, wspace=0.03, hspace=0.18); fig.savefig(Q.parent / "qualitative-ways-of-composing.png", dpi=75)
print("ok")
