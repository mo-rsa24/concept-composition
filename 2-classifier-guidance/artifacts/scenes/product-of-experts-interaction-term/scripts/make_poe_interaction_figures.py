"""Draws the figures of this scene from data/poe_interaction_numbers.json (clean register: thin outlines, glow on the part the figure is about).
Colour key, the same on every page: blue inputs, green experts and hidden parts, red what is drawn, orange the interaction term and every push, grey what is fixed or off.
Run from the scene folder:  python scripts/make_poe_interaction_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(__file__).parent.parent
D = json.loads((ROOT / "data" / "poe_interaction_numbers.json").read_text())
FIG = ROOT / "figures"
INK, GREY, BLUE, GREEN, RED, ORANGE = "#1f2a33", "#6b7a88", "#1f5fd0", "#1d6b3a", "#c0262d", "#c8680a"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK})


def box(ax, x, y, w, h, text, col, fill=None, fs=11, weight="normal", glow=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fill or "white", ec=col, lw=2.6 if glow else 1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight)


def arrow(ax, a, b, col=GREY, lw=2.2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=col, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas_icon(ax, x, y, cells, w=0.9, h=0.5):
    """cells: (4 in left, 7 in left, 4 in right, 7 in right) as 0 or 1. Two cells side by side; a 4 is blue, a 7 is red."""
    for i, (a, b) in enumerate(((cells[0], cells[1]), (cells[2], cells[3]))):
        ax.add_patch(Rectangle((x + i * w / 2, y), w / 2, h, fc="white", ec=GREY, lw=1))
        if a:
            ax.text(x + i * w / 2 + w / 4 - (0.11 if b else 0), y + h / 2, "4", ha="center", va="center", color=BLUE, fontsize=15, weight="bold")
        if b:
            ax.text(x + i * w / 2 + w / 4 + (0.11 if a else 0), y + h / 2, "7", ha="center", va="center", color=RED, fontsize=15, weight="bold")


# ---------------------------------------------------------------- figure 0: the system
fig, ax = plt.subplots(figsize=(15, 8.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.4); ax.axis("off")
ax.text(0.3, 8.05, "Level 0: one sampler step of a product of two experts, and the term the sum leaves out", fontsize=15, weight="bold")
ax.text(0.3, 7.6, "Toy of this scene: a canvas is two cells; expert A draws a 4 in one cell, expert B draws a 7 in one cell; neither ever draws two digits.", fontsize=11, color=GREY)
box(ax, 0.4, 5.5, 2.4, 1.1, "the noisy canvas\nx_t  (4 numbers)", BLUE, weight="bold")
box(ax, 4.2, 6.15, 3.0, 0.85, "expert A: told 4", GREEN, glow=True); box(ax, 4.2, 4.95, 3.0, 0.85, "expert B: told 7", GREEN, glow=True)
box(ax, 4.2, 3.75, 3.0, 0.85, "base: told nothing", GREY)
for yy in (6.55, 5.35, 4.15):
    arrow(ax, (2.8, 6.05), (4.2, yy), BLUE)
box(ax, 8.2, 4.6, 2.6, 1.5, "ε_A + ε_B − ε_base\nthe product of experts", GREEN, weight="bold")
for yy in (6.55, 5.35):
    arrow(ax, (7.2, yy), (8.2, 5.65), GREEN)
arrow(ax, (7.2, 4.15), (8.2, 5.0), GREY)
box(ax, 8.2, 2.1, 2.6, 1.7, "the interaction term\nr_t = −σ_t I_t\nwhat the sum drops", ORANGE, glow=True, weight="bold", fs=10.5)
ax.text(9.5, 1.55, "the product has no such term:\nit adds the two single-digit scores", ha="center", fontsize=10, color=GREY)
box(ax, 11.7, 3.6, 2.9, 1.6, "one sampler step\nx_{t−1} from x_t and\nthe noise prediction", BLUE)
arrow(ax, (10.8, 5.35), (11.7, 4.7), GREEN, lw=2.6); arrow(ax, (10.8, 3.2), (11.7, 4.1), ORANGE, lw=2.2, ls="--")
ax.text(13.2, 2.75, "put back by: the correction (LoRA),\nthe classifier push,\nthe committed layout", fontsize=10, color=ORANGE, ha="center")
ax.text(0.4, 3.6, "what it draws (x_0)", fontsize=12, weight="bold")
canvas_icon(ax, 0.4, 2.75, [1, 0, 0, 1]); ax.text(1.35, 3.0, "two apart", fontsize=10, color=GREEN, va="center")
canvas_icon(ax, 0.4, 1.95, [1, 1, 0, 0]); ax.text(1.35, 2.2, "both in one cell", fontsize=10, color=RED, va="center")
canvas_icon(ax, 0.4, 1.15, [1, 0, 0, 0]); ax.text(1.35, 1.4, "one digit only", fontsize=10, color=RED, va="center")
o = D["outcomes_of_4000"]["product of the two experts"]
ax.text(3.6, 3.0, f"{o['two apart']} of 4000", fontsize=10, va="center"); ax.text(3.6, 2.2, f"{o['both in one cell']} of 4000", fontsize=10, va="center")
ax.text(3.6, 1.4, f"{o['one digit']} of 4000", fontsize=10, va="center")
ax.text(0.4, 0.45, f"Counts: this toy's product of the two experts, 4000 draws with the project's sampler; {o['other']} more draws end on the empty canvas or another point (data/poe_interaction_numbers.json).", fontsize=10, color=GREY)
fig.savefig(FIG / "level0-one-sampler-step-of-a-product-of-experts.png", dpi=80, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 0.5: the score fields on the slice (4 in left cell, 7 in right cell)
fig, axes = plt.subplots(2, 3, figsize=(16, 10.2)); fig.subplots_adjust(hspace=0.4, wspace=0.18, top=0.89, bottom=0.1)
fig.suptitle("Level 0.5: the product's score, the joint's score and the term between them, on the plane of (4 in the left cell, 7 in the right cell)", fontsize=14, weight="bold")
for r, k in enumerate(("5", "25")):
    F = D["slice_fields"][k]; g = np.array(F["grid"]); X, Y = np.meshgrid(g, g)
    P, J = np.array(F["poe"]), np.array(F["joint"]); I = J - P
    share = np.linalg.norm(I, axis=1) / (np.linalg.norm(J, axis=1) + 1e-12)
    for c, (V, col, ttl) in enumerate(((P, GREEN, "the product: ε_A + ε_B − ε_base"), (J, RED, "the joint: the score of 'a 4 and a 7, apart'"), (I, ORANGE, "the term the product drops: joint − product"))):
        ax = axes[r, c]; n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12; U = V / n
        kw = dict(angles="xy", scale_units="xy", scale=1 / 0.075, width=0.005)
        if c < 2:
            ax.quiver(X, Y, U[:, 0].reshape(X.shape), U[:, 1].reshape(X.shape), color=col, **kw)
        else:
            q = ax.quiver(X, Y, U[:, 0].reshape(X.shape), U[:, 1].reshape(X.shape), np.clip(share, 0, 1).reshape(X.shape), cmap="Oranges", clim=(0, 1), **kw)
            fig.colorbar(q, ax=ax, fraction=0.046, pad=0.02, label="size of the term as a share of the joint's score")
        for (px, py, lab) in ((0, 0, "empty"), (1, 0, "4 only"), (0, 1, "7 only"), (1, 1, "4 left,\n7 right")):
            ax.plot(px, py, "o", mfc="white", mec=INK if (px, py) == (1, 1) else GREY, ms=11, mew=2 if (px, py) == (1, 1) else 1.2, zorder=5)
            ax.text(px + 0.08, py + 0.08, lab, fontsize=9, color=INK if (px, py) == (1, 1) else GREY, zorder=6)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_aspect("equal")
        ax.set_title(ttl if r == 0 else "", fontsize=10.5)
        ax.set_xlabel("how much 4 in the left cell", fontsize=9); ax.set_ylabel(f"how much 7 in the right cell\nsignal level ᾱ = {F['abar']:.2f}" if c == 0 else "", fontsize=9)
fig.text(0.5, 0.02, "Every arrow has the same length and shows only the direction of the score; the size is in the right-hand column, where the shade is the term as a share of the joint's score.\nThe open dots are the canvases the experts or the joint can draw; the two-apart canvas has the dark rim. "
         "Top row: early in the sampler (step 6 of 50). Bottom row: step 26 of 50. Exact scores of the toy; the other two coordinates are held at 0.", ha="center", fontsize=10, color=GREY)
fig.savefig(FIG / "level05-the-product-score-the-joint-score-and-the-term.png", dpi=70, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 0.5b: the time profile
P_ = D["profile"]; step = np.array([r["step"] for r in P_]); rt = np.array([r["rms_r_t"] for r in P_]); dec = np.array([r["decided"] for r in P_])
fig, ax = plt.subplots(figsize=(13, 6.2)); ax2 = ax.twinx()
ax.semilogy(step, rt, color=ORANGE, lw=3, label="size of the interaction term, |r_t| (left, log scale)")
ax2.plot(step, dec, color=GREEN, lw=3, label="how settled the joint's choice of layout is (right)")
ax2.axhline(0.9, color=GREEN, ls=":", lw=1)
ax.axvspan(0, 9.5, color=GREY, alpha=0.12); ax.text(4.8, rt[rt > 0].min() * 1.6, "steps 0 to 9", ha="center", fontsize=10, color=GREY)
ax.text(48.6, 0.0035, "falls to 0 at the last\nsteps because σ_t → 0", ha="right", fontsize=9, color=ORANGE)
ax.axvline(D["commit_step_decided_90"], color=GREEN, ls="--", lw=1.5); ax.text(D["commit_step_decided_90"] + 0.4, rt.min() * 3, f"settled at step {D['commit_step_decided_90']}\n(0.9 on the right axis)", fontsize=10, color=GREEN)
ax.axvline(D["peak_r_t_step"], color=ORANGE, ls="--", lw=1.5); ax.text(D["peak_r_t_step"] - 0.4, rt.min() * 3, f"|r_t| peaks at step {D['peak_r_t_step']}", fontsize=10, color=ORANGE, ha="right")
ax.set_ylim(0.003, rt.max() * 1.6); ax.set_xlabel("sampler step (0 is the noisiest, 49 is clean)"); ax.set_ylabel("|r_t| = σ_t |I_t|, root mean square over 4000 typical states", color=ORANGE)
ax2.set_ylabel("posterior of the joint's likelier layout, mean over the same states\n(0.5 = undecided, 1 = decided)", color=GREEN); ax2.set_ylim(0.45, 1.02)
ax.set_title("Level 0.5: in the toy, the interaction term is small while the layout is open and large after it settles", fontsize=13, weight="bold")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, frameon=False)
fig.text(0.5, -0.02, "Toy only (data/poe_interaction_numbers.json). In the measured SDXL runs the correction's part shared across seeds is 19.6% of its size at steps 0 to 2 and about 0 after step 20 (project notes, not remeasured here).", ha="center", fontsize=10, color=GREY)
fig.savefig(FIG / "level05-the-interaction-term-against-the-sampler-step.png", dpi=80, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 2: what each composition draws
O = D["outcomes_of_4000"]; names = list(O.keys())
short = {
    "product of the two experts": "the product of the two experts",
    "product plus the interaction term at every step (the joint's own score)": "product + interaction term, every step",
    "product plus the interaction term, steps 0 to 9 only": "product + interaction term, steps 0 to 9",
    "product plus the interaction term, steps 10 to 49 only": "product + interaction term, steps 10 to 49",
    "product plus the interaction term, steps 0 to 4 only": "product + interaction term, steps 0 to 4",
    "product plus the interaction term, steps 5 to 9 only": "product + interaction term, steps 5 to 9",
    "each expert keeps its own cell, whole canvas visible, coin layout held (no hiding)": "own cell, canvas visible, coin layout held",
    "each expert keeps its own cell, other cell hidden, coin layout held": "own cell, other cell hidden, coin layout held",
    "each expert keeps its own cell, other cell hidden, layout voted over steps 0 to 9 then held (E0e)": "own cell, other cell hidden, layout voted then held (E0e)",
    "each expert keeps its own cell, other cell hidden, layout re-voted at every step (E0d-b)": "own cell, other cell hidden, layout re-voted each step (E0d-b)",
    "each expert keeps its own cell, whole canvas visible, layout voted then held": "own cell, canvas visible, layout voted then held",
}
order = sorted(names, key=lambda n: O[n]["two apart"])
fig, ax = plt.subplots(figsize=(14, 6.8))
cols = {"two apart": GREEN, "both in one cell": RED, "one digit": "#e0a030", "other": GREY}
for i, n in enumerate(order):
    left = 0
    for cat in ("two apart", "both in one cell", "one digit", "other"):
        v = O[n][cat]; ax.barh(i, v, left=left, color=cols[cat], edgecolor="white", label=cat if i == 0 else None)
        if v > 250:
            ax.text(left + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=9.5, weight="bold")
        left += v
ax.set_yticks(range(len(order))); ax.set_yticklabels([short[n] for n in order], fontsize=10); ax.set_xlim(0, 4000)
ax.set_xlabel("draws out of 4000 (same starting noise for every row)")
ax.set_title("Level 2: what each way of composing the two toy experts draws", fontsize=13, weight="bold"); ax.legend(loc="lower right", fontsize=9.5, ncol=4, frameon=False, bbox_to_anchor=(1, -0.2))
fig.savefig(FIG / "level2-what-each-composition-draws.png", dpi=80, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 2b: where the term has to act
S = D["two_apart_of_4000_by_window"]
fig, ax = plt.subplots(figsize=(11, 5.6))
xs = [int(k) for k in S["from step s to the end"]]; ax.plot(xs, list(S["from step s to the end"].values()), "o-", color=ORANGE, lw=2.5, label="term added from step s to the end")
xe = [int(k) for k in S["from step 0 to step e"]]; ax.plot(xe, list(S["from step 0 to step e"].values()), "s-", color=GREY, lw=2.5, label="term added from step 0 to step e")
ax.axhline(D["outcomes_of_4000"]["product of the two experts"]["two apart"], color=INK, ls=":", lw=1); ax.text(1, D["outcomes_of_4000"]["product of the two experts"]["two apart"] + 60, "no term at all", fontsize=10)
ax.set_xlabel("step s (start) or step e (end), of 50"); ax.set_ylabel("draws with a 4 and a 7 apart, of 4000"); ax.legend(frameon=False, loc="center left")
ax.set_title("Level 2: in the toy the term has to be there at the end; adding it only early changes almost nothing", fontsize=12.5, weight="bold")
fig.savefig(FIG / "level2-where-the-term-has-to-act.png", dpi=80, bbox_inches="tight"); plt.close(fig)
print("ok")
