"""A worked toy of one CO3 corrector pass on three made-up picture features. Illustrative numbers, not a real run."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
F = ["one big\neagle", "one big\nfox", "two animals\nside by side"]
xC, x1, x2 = np.array([0.70, 0.05, 0.25]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
beta = 0.9
d = np.array([np.linalg.norm(x1 - xC), np.linalg.norm(x2 - xC)])
a = np.exp(-beta * d); w = -a / a.sum()
xt = 2 * xC + w[0] * x1 + w[1] * x2
r = np.linalg.norm(xC) / np.linalg.norm(xt); xr = xt * r
print("d", d.round(3), "a", a.round(3), "w", w.round(3), "x~", xt.round(3), "r", round(r, 3), "x~r", xr.round(3))
fig, ax = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
cols = ["#7048e8", "#0c8599", "#e8590c"]
for k, (title, v) in enumerate([("joint prompt's guess\n'an eagle and a fox'", xC), ("'eagle' alone", x1), ("'fox' alone", x2),
                                 (f"after one corrector pass\n(shrunk by r = {r:.2f})", xr)]):
    ax[k].bar(range(3), v, color=cols); ax[k].axhline(0, color="#222", lw=0.8)
    for i, val in enumerate(v): ax[k].text(i, val + (0.03 if val >= 0 else -0.08), f"{val:.2f}", ha="center", fontsize=10)
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(F, fontsize=9.5); ax[k].set_title(title, fontsize=11)
ax[0].set_ylabel("how much of each feature\nthe guessed picture has")
fig.text(0.5, -0.02, f"distances to the joint: eagle {d[0]:.2f}, fox {d[1]:.2f}  ->  weights (eq. 15, beta 0.9): eagle {w[0]:.2f}, fox {w[1]:.2f}   "
         f"->  new guess = 2 x joint {w[0]:+.2f} x eagle {w[1]:+.2f} x fox", ha="center", fontsize=10.5)
fig.text(0.5, -0.08, "Illustrative features and numbers, not a real run: real guesses have 4 x 128 x 128 numbers.", ha="center", fontsize=9, color="#495057")
plt.tight_layout(); plt.savefig("co3-toy-corrector.png", dpi=130, bbox_inches="tight")
