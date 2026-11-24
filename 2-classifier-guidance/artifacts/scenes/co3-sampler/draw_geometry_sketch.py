"""Illustrative 2D geometry of CO3's two moves. Vectors are made up so the arrows can be seen; real predictions have 4 x 128 x 128 numbers."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def arrow(ax, a, b, c, lab=None, off=(0.05, 0.05), ls="-", lw=2.2, fs=11):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls, mutation_scale=16))
    if lab: ax.text(b[0] + off[0], b[1] + off[1], lab, color=c, fontsize=fs)

fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
O = np.zeros(2)
# Resampler: noise space
eC, e1, e2, ephi = np.array([1.6, 1.3]), np.array([1.9, 0.4]), np.array([0.5, 1.8]), np.array([1.2, 0.9])
w1, w2 = 0.65, 0.35                       # eq. 15: eagle closer to the joint, so its weight is larger
avg = w1 * e1 + w2 * e2
contrast = eC - avg
lam = 0.8
a = ax[0]
arrow(a, O, eC, "#1c7ed6", r"$\epsilon^{C}$ joint")
arrow(a, O, e1, "#7048e8", r"$\epsilon^{c_1}$ eagle")
arrow(a, O, e2, "#0c8599", r"$\epsilon^{c_2}$ fox")
arrow(a, O, avg, "#868e96", r"weighted average of the singles", off=(0.08, -0.12), ls="--", lw=1.6, fs=9.5)
arrow(a, avg, eC, "#e8590c", r"$\sum_k w_k\epsilon^{c_k}$: what the joint adds", off=(0.08, -0.05), lw=2.6, fs=10)
arrow(a, O, ephi, "#495057", None)
a.text(0.25, 0.55, r"$\epsilon^{\phi}$ empty prompt", color="#495057", fontsize=11, rotation=33)
new = ephi - lam * contrast
arrow(a, ephi, new, "#e8590c", None, ls=":")
a.text(new[0] + 0.12, new[1] + 0.1, r"$-\lambda\sum_k w_k\epsilon^{c_k}$", color="#e8590c", fontsize=10)
a.scatter(*new, s=60, color="#c92a2a", zorder=5); a.text(new[0] - 0.2, new[1] - 0.3, r"$x_t^{(2)}/\sqrt{1-\bar\alpha_t}$", color="#c92a2a", fontsize=11)
a.set_title("Resampler, Algorithm 2: weights sum to 0, so $x_t$ cancels.\nThe new latent is the empty prompt's noise, moved by what the joint prompt adds", fontsize=11)
a.text(-0.1, -0.55, "weights: joint $w_0=1$, eagle $w_1=-0.65$, fox $w_2=-0.35$ (sum 0)\n"
       r"$x_t^{(2)}=\sqrt{1-\bar\alpha_t}\,(\epsilon^{\phi}-\lambda\sum_k w_k\epsilon^{c_k})$", fontsize=10.5)
a.set_xlim(-0.3, 2.6); a.set_ylim(-0.8, 2.2); a.set_aspect("equal"); a.axis("off")

# Corrector: clean-picture estimates (Tweedie means)
xC, x1, x2 = np.array([1.5, 1.2]), np.array([2.0, 0.5]), np.array([0.6, 1.9])
avg = w1 * x1 + w2 * x2
xt = 2 * xC - avg                         # w0 = 2, w1 + w2 = -1, sum 1
r = np.linalg.norm(xC) / np.linalg.norm(xt)
xr = xt * r
b = ax[1]
for p, lab, c in ((xC, r"$\hat x^{C}$ joint's clean guess", "#1c7ed6"), (x1, r"$\hat x^{c_1}$ eagle's", "#7048e8"),
                  (x2, r"$\hat x^{c_2}$ fox's", "#0c8599")):
    b.scatter(*p, s=70, color=c, zorder=5); b.text(p[0] + 0.06, p[1] + 0.05, lab, color=c, fontsize=11)
b.scatter(*avg, s=40, color="#868e96"); b.text(avg[0] + 0.06, avg[1] - 0.15, "weighted average of the singles", color="#868e96", fontsize=9.5)
b.plot([avg[0], xt[0]], [avg[1], xt[1]], color="#adb5bd", ls="--", lw=1.2)
b.scatter(*xt, s=70, color="#e8590c", zorder=5); b.text(xt[0] - 1.25, xt[1] + 0.15, r"$\tilde x=2\hat x^{C}-$average:" + "\npast the joint, away from the singles", color="#e8590c", fontsize=10)
arrow(b, O, xr, "#c2255c", None)
b.scatter(0, 0, s=25, color="#495057"); b.text(0.05, -0.18, "zero", color="#495057", fontsize=9)
b.text(0.05, 0.95, r"$\tilde x\cdot r$, with $r=\|\hat x^{C}\|/\|\tilde x\|$:" + "\nshrunk to the joint guess's length", color="#c2255c", fontsize=10)
b.annotate("", xy=xr, xytext=xt, arrowprops=dict(arrowstyle="-|>", color="#c2255c", lw=1.2, ls=":"))
b.set_title("Corrector, Algorithm 3: weights sum to 1, so the result is still a clean-picture guess.\nIt steps past the joint's guess, away from each single subject, then is shrunk back", fontsize=11)
b.text(-0.1, -0.55, r"then $x_t^{(2)}=\tilde x\cdot r+\sqrt{1-\bar\alpha_t}\,\epsilon^{\phi}$: the empty prompt's noise is added back", fontsize=10.5)
b.set_xlim(-0.3, 2.9); b.set_ylim(-0.8, 2.6); b.set_aspect("equal"); b.axis("off")
fig.text(0.5, 0.01, "Illustrative 2D vectors, not real values: each real prediction has 4 x 128 x 128 numbers.", ha="center", fontsize=9.5, color="#495057")
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig("co3-geometry-sketch.png", dpi=140); plt.savefig("co3-geometry-sketch.svg")
