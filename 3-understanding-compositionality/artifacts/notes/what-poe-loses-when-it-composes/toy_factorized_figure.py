"""Sheet for toy_factorized.py: eight draws per row, two backgrounds, four rows each.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_factorized_figure.py
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t
import toy_factorized as f

OUT = Path(__file__).resolve().parents[2] / "results" / "how-much-co-occurrence-the-product-needs"
J = json.load(open(OUT / "toy-factorized.json"))["results"]
S = np.load(OUT / "toy-factorized-samples.npz")
ROWS = [("joint caption", "the joint caption\n(trained on 2 layouts)"),
        ("exact composition, read by the scorer", "circle x square / background\nexact draws"),
        ("reverse diffusion, true noised-composition score", "same, reverse diffusion\ntrue noised score"),
        ("reverse diffusion, summed noised scores", "same, reverse diffusion\nsummed noised scores")]
BG = {"white": (1.0, 1.0, 1.0), "grey": (0.5, 0.5, 0.5)}
SHORT = {"left": "L", "centre": "C", "right": "R"}
fig, axes = plt.subplots(8, 9, figsize=(17, 11.5), gridspec_kw={"width_ratios": [1] * 8 + [1.9]})
for bi, bg in enumerate(("white", "grey")):
    for ri, (key, label) in enumerate(ROWS):
        row = bi * 4 + ri
        tiles = S[f"{bg}_{ri}"]
        lays = f.layouts(tiles, BG[bg])
        for j in range(8):
            ax = axes[row, j]
            ax.imshow(np.clip(tiles[j], 0, 1).reshape(t.H, t.W, 3), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            l = lays[j]
            if isinstance(l, tuple):
                txt = f"circle {SHORT[l[0]]}, square {SHORT[l[1]]}"
                col = "#1a7f37" if l not in f.TRAINED else "#444"
                txt += "  NEW" if l not in f.TRAINED else ""
            else:
                txt, col = l, "#b42318"
            ax.set_title(txt, fontsize=7.5, color=col, pad=2)
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        c = J[bg][key]
        ax = axes[row, 8]; ax.axis("off")
        ax.text(0.03, 0.5, f"of 300:\nunseen layouts {c['unseen layouts']}\ntrained layouts {c['trained layouts']}\n"
                f"overlaid or unreadable {c['overlaid or unreadable']}", fontsize=8.5, va="center",
                family="monospace", color="#1a7f37" if c["unseen layouts"] >= 100 else "#222")
    axes[bi * 4, 0].text(-1.6, 1.35, f"{bg} background", transform=axes[bi * 4, 0].transAxes, fontsize=12,
                         weight="bold", color="#0b3d91")
fig.suptitle("No expert saw the two shapes together except the joint caption, which saw two layouts. "
             "Green NEW = a layout no training picture holds.\nOn grey the product draws all six apart "
             "layouts; on white both shapes stack on one spot and subtract to near-black.", fontsize=11.5)
plt.subplots_adjust(left=0.13, right=0.995, top=0.91, bottom=0.01, hspace=0.6, wspace=0.08)
p = OUT / "can-the-product-draw-a-layout-nobody-trained-on.png"
fig.savefig(p, dpi=105); print("wrote", p)
