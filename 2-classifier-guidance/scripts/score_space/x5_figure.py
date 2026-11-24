"""Draw X5's result: presence accuracy against timestep for each probe input, one panel per window.
Usage: x5_figure.py <full.json> <out stem>"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1]))
STYLE = {"signature": ("#c2255c", "signature (ten d_t maps)", 2.6),
         "error_readout": ("#1971c2", "error readout (uses the true noise)", 1.8),
         "path_length": ("#e8590c", "path length", 1.4),
         "canvas": ("#2b8a3e", "noisy canvas", 1.8),
         "raw_noise": ("#868e96", "raw noise", 1.2),
         "shuffled": ("#adb5bd", "signature, labels shuffled", 1.0)}
TITLES = {"early": "Early window: steps 0 to 15\n(t 999 to 693, canvas share 0.006 to 0.087)",
          "sdxl": "SDXL-matched window: steps 14 to 25\n(t 712 to 480, same signal as SDXL's steps 0 to 15)"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, w in zip(axes, ("early", "sdxl")):
    ts = [s["t"] for s in r[w]["signature"]]
    ax.axhline(0.75, color="#222", lw=1, ls="--"); ax.text(ts[0], 0.757, "bar 0.75", fontsize=9, va="bottom")
    ax.axhline(0.5, color="#999", lw=0.8, ls=":"); ax.text(ts[0], 0.487, "chance 0.5", fontsize=8, va="top", color="#666")
    ends = []
    for row, (c, lab, lw) in STYLE.items():
        y = [s["presence"] for s in r[w][row]]
        ax.plot(ts, y, color=c, lw=lw, marker="o", ms=2.5)
        ends.append([y[-1], lab, c])
    ends.sort()
    for i in range(1, len(ends)):                       # keep end labels from overlapping
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.018)
    for y, lab, c in ends:
        ax.text(ts[-1] - (ts[0] - ts[-1]) * 0.02, y, " " + lab, color=c, fontsize=8.5, va="center", ha="left")
    ax.set_xlim(ts[0], ts[-1] - (ts[0] - ts[-1]) * 0.75)
    ax.set_title(TITLES[w], fontsize=10.5)
    ax.set_xlabel("timestep t (noisier at left)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("presence accuracy, digits 0 to 4\n(balanced, 2,000 seen_test canvases)")
axes[0].set_ylim(0.46, 0.80)
fig.suptitle("Does the digit model's score path say which digits are on the canvas?  Each point: one probe fitted at one timestep", fontsize=11.5)
fig.tight_layout()
fig.savefig(sys.argv[2] + ".png", dpi=150); fig.savefig(sys.argv[2] + ".svg")
print("saved")
