"""Eight draws per caption per dial setting from toy_caption_dial_ldm.py, each tile labelled with the
blob scorer's call, and the 300-draw counts for that row printed beside it.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_caption_dial_sheet.py
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_sampler_hybrids as t
import toy_blob_scorer as b

OUT = Path(__file__).resolve().parents[2] / "results" / "how-much-co-occurrence-the-product-needs"
S = np.load(OUT / "toy-caption-dial-ldm-samples.npz")
R = json.load(open(OUT / "toy-caption-dial-ldm.json"))["results"]
ROWS = [(0, "a red circle", "a red circle alone"), (1, "a blue square", "a blue square alone"),
        (2, "the two single captions\nmultiplied", "the two single captions multiplied"),
        (3, "a circle and a square\n(the joint caption)", "the joint caption")]
SHORT = {"red and blue apart": "apart", "red only": "red", "blue only": "blue",
         "purple or overlaid": "purple", "nothing readable": "none"}
dials = list(R)
fig, axes = plt.subplots(len(dials) * 4, 9, figsize=(16, 0.9 * len(dials) * 4 + 1),
                         gridspec_kw={"width_ratios": [1] * 8 + [2.6]})
for di, d in enumerate(dials):
    for ri, title, key in ROWS:
        row = di * 4 + ri
        tiles = S[f"{d}_{ri}"]          # saved as (circle, square, product, joint), the order of ROWS
        calls = b.read(tiles)[0]
        for j in range(8):
            ax = axes[row, j]
            ax.imshow(tiles[j].reshape(t.H, t.W, 3), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(SHORT[calls[j]], fontsize=8, pad=2,
                         color={"apart": "#1a7f37", "purple": "#8250df"}.get(SHORT[calls[j]], "#444"))
        axes[row, 0].set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8.5)
        if ri == 0:
            axes[row, 0].text(-1.9, 1.15, f"dial {d}: pair pictures also under each single "
                              f"name {float(d):.0%} of the time", transform=axes[row, 0].transAxes,
                              fontsize=10, weight="bold", color="#0b3d91", va="bottom")
        c = R[d][key]["blob scorer"]
        txt = (f"of 300: apart {c['red and blue apart']}, purple {c['purple or overlaid']},\n"
               f"red only {c['red only']}, blue only {c['blue only']}, none {c['nothing readable']}")
        ax = axes[row, 8]; ax.axis("off")
        ax.text(0.02, 0.5, txt, fontsize=8, va="center", family="monospace")
fig.suptitle("One VAE plus one caption-conditional latent model per dial setting. Tile titles are the "
             "colour-blob scorer's call;\ncounts on the right are over all 300 draws.", fontsize=11, y=0.995)
plt.subplots_adjust(left=0.11, right=0.99, top=0.965, bottom=0.01, hspace=0.95, wspace=0.08)
p = OUT / "what-the-caption-model-draws.png"
fig.savefig(p, dpi=110); print("wrote", p)
