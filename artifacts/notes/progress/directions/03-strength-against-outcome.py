"""Draw H3's strength-against-outcome curves, one panel per dial.

Every value is copied from the source named in its legend; values marked "by eye" were read from a
plot in the paper. Run: ~/.claude/venvs/figures/bin/python 03-strength-against-outcome.py
"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))

# Panel 1: co-occurrence strength. x is the strength on each method's own 0-to-1 scale.
ax = axes[0]
lam = [0, 0.25, 0.5, 0.75, 1.0]
ax.plot(lam, [3.1, 9.4, 25.0, 71.9, 93.8], "o-", color="#1e7a34",
        label="this project: exact correction at weight λ, 2 animals (% of 32)")
ax.plot(lam, [3.1, 3.1, 6.25, 9.4, 6.25], "o--", color="#888888",
        label="this project: the same with wrong corrections (mean of 4)")
r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ax.plot(r, [14.3, 30.1, 34.8, 52.1, 62.1, 59.3, 41.6, 25.7, 25.7], "s-", color="#b26a00",
        label="Matsuhira 2024: embedding blend r, both words present (%), Table 1 p.4")
w = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.plot(w, [0.5, 6, 18, 24, 27, 28.4, 29, 29.1, 29.2, 29.3, 29.4], "^-", color="#5b5bd6",
        label="InteractDiffusion: share of steps ω, interaction mAP (points, by eye) Fig.10 p.11")
ax.set_title("Co-occurrence strength")
ax.set_xlabel("strength on the method's own 0 to 1 scale (λ, r or ω)")
ax.set_ylabel("outcome, percent or mAP points (see legend)")
ax.set_ylim(0, 100)

# Panel 2: count. x is the requested count N.
ax = axes[1]
n10 = list(range(1, 11))
ax.plot(n10, [62, 55, 40, 20, 10, 11, 4, 5, 3, 2], "o-", color="#b42318",
        label="SDXL plain, GeckoNum, human counts, Fig. A2 p.19")
ax.plot(n10, [74, 72, 50, 46, 24, 12, 10, 10, 8, 5], "o--", color="#e07a5f",
        label="SD3 plain, GeckoNum, Fig. A2 p.19")
ax.plot(range(2, 11), [93, 74, 93, 63, 63, 37, 37, 26, 15], "s-", color="#1e7a34",
        label="SDXL + CountCluster (by eye), Fig. 4 p.7")
ax.plot(range(2, 6), [96, 89, 71, 47], "^-", color="#5b5bd6",
        label="SDXL + ISAC, Table 3 p.12")
ax.axhline(50, color="#cccccc", lw=1)
ax.set_title("Count: requested N against exact accuracy")
ax.set_xlabel("requested count N")
ax.set_ylabel("exact count accuracy (%)")
ax.set_ylim(0, 100)

# Panel 3: binding strength. x is the strength as a share of the largest value each method tried.
ax = axes[2]
gm_w = [5.5, 6.13, 6.19, 7.36, 7.62, 8.28, 8.81, 14.12, 16.05]
ax.plot([v / 16.05 for v in gm_w], [26.88, 28.00, 26.00, 26.00, 25.00, 27.00, 24.00, 31.00, 25.00],
        "o-", color="#b26a00", label="SDXL guidance scale 5.5 to 16, GenEval colour attribution (%), Guidance Matters Table 2 p.9")
ax.plot([0, 10 / 30, 1.0], [100, 12.5, 50], "s-", color="#b42318",
        label="this project: CLIP species push 0, 10, 30, separate animals (% of 8)")
rr = [0, 0.2, 0.4, 0.6, 1.0]
ax.plot(rr, [46.7, 56.4, 64.6, 69.7, 72.9], "^-", color="#1e7a34",
        label="Relation Rectification λ 0 to 1, position accuracy (%) [a relation], Table 1 p.7")
ax.set_title("Binding strength")
ax.set_xlabel("strength as a share of the largest value tried")
ax.set_ylabel("outcome (%)")
ax.set_ylim(0, 105)

for ax in axes:
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.16), frameon=False)
    ax.grid(alpha=0.25)
fig.suptitle("Strength against outcome for the owner's three dials", fontsize=12)
fig.tight_layout()
fig.savefig("03-strength-against-outcome.png", dpi=130)
