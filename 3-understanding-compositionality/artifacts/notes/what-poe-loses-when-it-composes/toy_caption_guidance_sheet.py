"""The two single captions multiplied, one cell per dial setting (rows) and guidance strength
(columns): four draws per cell titled with the colour-blob scorer's call, and the cell's crisp-pair
count out of 300 above them.

    /home/molef/micromamba/envs/sdxl-local/bin/python -I toy_caption_guidance_sheet.py
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
S = np.load(OUT / "toy-caption-dial-ldm-guidance-samples.npz")
J = json.load(open(OUT / "toy-caption-dial-ldm-guidance.json"))
R, G = J["results"], [f"{g:g}" for g in J["guidance"]]
K = "the two single captions multiplied"
SHORT = {"red and blue apart": "apart", "red only": "red", "blue only": "blue",
         "purple or overlaid": "purple", "nothing readable": "none"}
dials = list(R)
fig = plt.figure(figsize=(18, 2.3 * len(dials) + 1.2))
outer = fig.add_gridspec(len(dials), len(G), hspace=0.55, wspace=0.12,
                         left=0.09, right=0.995, top=0.85, bottom=0.02)
for i, d in enumerate(dials):
    for j, g in enumerate(G):
        tiles = S[f"{d}_{g}"][:4]
        calls = b.read(tiles)[0]
        cnt = R[d][g][K]
        pure = min(R[d][g]["a red circle alone"]["red only"], R[d][g]["a blue square alone"]["blue only"])
        inner = outer[i, j].subgridspec(2, 2, hspace=0.35, wspace=0.08)
        for k in range(4):
            ax = fig.add_subplot(inner[k // 2, k % 2])
            ax.imshow(tiles[k].reshape(t.H, t.W, 3), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(SHORT[calls[k]], fontsize=7, pad=1,
                         color={"apart": "#1a7f37", "purple": "#8250df"}.get(SHORT[calls[k]], "#444"))
            if k == 0:
                ax.text(0, 1.55, f"{cnt['red and blue apart']} of 300 apart  ·  purest single {pure}",
                        transform=ax.transAxes, fontsize=8.5, weight="bold",
                        color="#1a7f37" if cnt["red and blue apart"] >= 150 and pure >= 240 else "#222")
            if i == 0 and k == 0:
                ax.text(1.05, 2.4, f"guidance {g}", transform=ax.transAxes, fontsize=11, ha="center",
                        weight="bold")
            if j == 0 and k == 0:
                ax.text(-0.35, 0, f"dial {d}", transform=ax.transAxes, fontsize=11, ha="right",
                        va="center", weight="bold")
fig.suptitle("The two single captions multiplied: dial (how often a pair picture is also captioned with "
             "each single name) down, guidance strength across.\nGreen headline: at least 150 of 300 "
             "apart while both single captions still draw only their own shape on at least 240 of 300.",
             fontsize=11.5, y=0.985)
p = OUT / "caption-model-across-guidance.png"
fig.savefig(p, dpi=105); print("wrote", p)
