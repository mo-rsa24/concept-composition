"""Draw GLIDE's CLIP score, human ratings and random samples as one figure.

Usage: draw_glide_pair.py <GLIDE paper.pdf> <empty work folder> <out.png>
Every number and image comes from arXiv 2112.10741: Figure 18c and 18d (p. 20), Figure 6c (p. 8), Table 1 (p. 9).
"""
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

PDF, WORK, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
WORK.mkdir(parents=True, exist_ok=True)

SURFACE, INK, MUTED, GRID = "#ffffff", "#0b0b0b", "#5f5e5a", "#e4e3df"
CLIP_COLOUR, FREE_COLOUR, NONE_COLOUR = "#ff7f0e", "#1f77b4", "#8a8984"

# Table 1, p. 9: human Elo on MS-COCO validation prompts at 256 by 256 pixels.
ELO = {
    "Photorealism": {"Unguided": -88.6, "CLIP guidance": -73.2, "Classifier-free": 82.7},
    "Caption match": {"Unguided": -106.2, "CLIP guidance": 29.3, "Classifier-free": 110.9},
}
COLOURS = {"Unguided": NONE_COLOUR, "CLIP guidance": CLIP_COLOUR, "Classifier-free": FREE_COLOUR}

subprocess.run(["pdfimages", "-f", "20", "-l", "20", "-j", str(PDF), str(WORK / "fig18")], check=True)
grids = sorted(WORK.glob("fig18-*.jpg"))
assert len(grids) == 4, grids
clip_grid, free_grid = Image.open(grids[2]).convert("RGB"), Image.open(grids[3]).convert("RGB")

subprocess.run(["pdftoppm", "-r", "300", "-f", "8", "-l", "8", "-png", str(PDF), str(WORK / "page")], check=True)
page = Image.open(next(WORK.glob("page-*.png"))).convert("RGB")
scale = 300 / 170
plot = page.crop(tuple(int(v * scale) for v in (885, 165, 1300, 410)))

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED})
fig = plt.figure(figsize=(16, 7.7), dpi=170, facecolor=SURFACE)
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.02], height_ratios=[1, 0.82],
              left=0.015, right=0.985, top=0.80, bottom=0.075, wspace=0.05, hspace=0.27)

fig.text(0.015, 0.945, "CLIP guidance raises the CLIP score, and people still prefer classifier-free guidance",
         fontsize=19, fontweight="bold", va="center")
fig.text(0.015, 0.893, "16 random samples per method for the prompt “A cozy living room with a painting of a corgi on the wall above a couch and a round "
         "coffee table in front of a couch and a vase of flowers on a coffee table”", fontsize=10, color=MUTED, va="center")

for col, image, name, colour in ((0, clip_grid, "CLIP guidance at scale 2.0", CLIP_COLOUR),
                                 (1, free_grid, "Classifier-free guidance at scale 3.0", FREE_COLOUR)):
    ax = fig.add_subplot(gs[:, col])
    ax.imshow(image)
    ax.set_anchor("N")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(colour), spine.set_linewidth(3)
    ax.set_title(name, fontsize=12.5, fontweight="bold", loc="left", pad=7)

ax = fig.add_subplot(gs[0, 2])
ax.imshow(plot)
ax.set_anchor("N")
ax.axis("off")
ax.set_title("The CLIP score favours CLIP guidance", fontsize=12.5, fontweight="bold", loc="left", pad=7)

ax = fig.add_subplot(gs[1, 2], facecolor=SURFACE)
ax.set_title("People favour classifier-free guidance", fontsize=12.5, fontweight="bold", loc="left", pad=7)
for y, (criterion, scores) in zip((1, 0), ELO.items()):
    ax.plot([min(scores.values()), max(scores.values())], [y, y], color=GRID, lw=2, zorder=1)
    for method, value in scores.items():
        ax.scatter(value, y, s=135, color=COLOURS[method], edgecolor=SURFACE, linewidth=2, zorder=3)
        text = f"{method}\n" + f"{value:+.1f}".replace("-", "−")
        if method == "Unguided":
            ax.annotate(text, (value, y), xytext=(-11, 0), textcoords="offset points", ha="right", va="center", fontsize=9)
        elif method == "CLIP guidance" and criterion == "Photorealism":
            ax.annotate(text, (value, y), xytext=(11, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
        elif method == "CLIP guidance":
            ax.annotate(text, (value, y), xytext=(0, 11), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        else:
            ax.annotate(text, (value, y), xytext=(11, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
    ax.text(-235, y + 0.42, criterion, fontsize=10.5, fontweight="bold", va="center")
ax.set_xlim(-240, 215), ax.set_ylim(-0.42, 1.62)
ax.set_yticks([])
ax.set_xlabel("Human Elo rating, higher is better", fontsize=9.5)
ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
for side in ("top", "left", "right"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)

fig.text(0.015, 0.025, "GLIDE, arXiv 2112.10741: Figure 18 (p. 20), Figure 6c at 64 by 64 pixels (p. 8), Table 1 at "
         "256 by 256 pixels with 1,000 human comparisons per criterion (pp. 9, 13).", fontsize=8.5, color=MUTED)
fig.savefig(OUT, facecolor=SURFACE)
print(OUT)
