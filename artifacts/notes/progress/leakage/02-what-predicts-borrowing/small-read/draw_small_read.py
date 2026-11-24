"""Z2 small read: eagle x fox (borrows on 0 of 8 seeds) against cheetah x cougar (7 of 8), per window, one dot per seed.
Data: per-picture records under /datasets/mmolefe/poe_repair_min/loop-12/z2/small/ (W&B 27u2m8sh), pulled to the path given."""
import json, glob, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = sys.argv[1]
P = {"an_eagle__x__a_fox": ("eagle x fox, borrows 0 of 8", "#1c7ed6"), "a_cheetah__x__a_cougar": ("cheetah x cougar, borrows 7 of 8", "#e8590c")}
W = ["early", "middle", "late"]; WL = ["early\nsteps 0-5", "middle\nsteps 10-20", "late\nsteps 25-45"]
K = [("iou_AB", "attention overlap\n(share of the two rescaled maps' cut regions they share, 0 to 1)"),
     ("dt_cos", "score-map similarity\n(cosine of the two animals' d_t maps, -1 to 1)")]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
rng = np.random.default_rng(0)
for a, (k, lab) in zip(ax, K):
    for j, (pair, (name, c)) in enumerate(P.items()):
        rows = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/{pair}/seed-*.json"))]
        for i, w in enumerate(W):
            v = np.array([r["windows"][w][k] for r in rows])
            x = i + (j - 0.5) * 0.28 + rng.uniform(-0.05, 0.05, len(v))
            a.scatter(x, v, color=c, s=30, alpha=0.85, label=name if i == 0 else None)
            a.plot([i + (j - 0.5) * 0.28 - 0.1, i + (j - 0.5) * 0.28 + 0.1], [v.mean()] * 2, color=c, lw=2.5)
    a.set_xticks(range(3)); a.set_xticklabels(WL); a.set_ylabel(lab); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, loc="lower left")
fig.suptitle("Z2 small read: one dot per seed (8 per pair), bar = mean. Two pairs only, no bar is applied.", fontsize=12)
plt.tight_layout(); plt.savefig("small-read/small-read-two-pairs.png", dpi=130)
