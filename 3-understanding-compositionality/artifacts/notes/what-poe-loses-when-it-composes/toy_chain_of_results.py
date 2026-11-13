"""The chain of results on product composition, oldest at the top, ending on this repository's sweep.

Each row is one paper: its year and claim on the left, its own figure in the middle where one has
been fetched, and what it showed against what it left open on the right. A row whose figure is not
on disk says so rather than standing in a picture of its own.
"""
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
OUT = HERE.parents[1] / "results" / "how-much-co-occurrence-the-product-needs"
THEIRS, OURS, INK, FAINT = "#5d6d7e", "#9b2226", "#1b1b1b", "#8a8a8a"

ROWS = [
    dict(year="2022", who="Liu, Du et al.", what="Compositional Visual Generation",
         claim="Multiply the conditionals. Composition is a product of experts.",
         image=None,
         missing="No figure fetched. The assumption lives in one sentence before their Eq 9:\n"
                 "the concepts are taken as independent once the picture is known.",
         showed="Score addition works well enough on CLEVR and on faces to be worth using.",
         open_="No error term is written anywhere. The dropped factor, in their own notation,\n"
               "is the gradient of log p(c1,c2|x) over p(c1|x)p(c2|x), and nobody measures it.",
         colour=THEIRS),
    dict(year="2023", who="Du et al.", what="Reduce, Reuse, Recycle",
         claim="Adding diffused scores does not sample the product. Replace the sampler.",
         image=FIGS / "du-2023-product-sampling.png", missing=None,
         showed="A ring of eight clusters times a vertical bar. The true product is two clusters.\n"
                "Reverse diffusion smears mass round the ring; HMC recovers the two.",
         open_="Two dimensions, one product, one peak sharpness. Nothing says how the fix scales\n"
               "as the product gets more confident or the space gets bigger.",
         colour=THEIRS),
    dict(year="2025", who="Bradley et al.", what="Mechanisms of Projective Composition",
         claim="Score addition is exact when each condition owns its own coordinates.",
         image=FIGS / "bradley-2025-projective-composition.png", missing=None,
         showed="On CLEVR, yellow composes with blue, cyan and magenta and fails against brown,\n"
                "grey, green and red. Yellow differs from the grey background in the blue channel only.",
         open_="Section 5.3 concedes the condition does not exactly hold in CLEVR and attaches\n"
               "no quantity to how far off it is. Lemma 5.7, the bound that was to supply one,\n"
               "is false as printed: KL 2.883 against a bound of 0.891.",
         colour=THEIRS),
    dict(year="2025", who="Bradley", what="Local Mechanisms of Compositional Generalization",
         claim="That condition is the same thing as every conditional score being local.",
         image=None,
         missing="No figure fetched. The result is a count: on CLEVR a patch-local model\n"
                 "conditioned only on what is inside each patch generates up to 5 objects\n"
                 "correctly, against 1 for a standard U-Net.",
         showed="Making the model local fixes composition where changing the formula does not.",
         open_="A text-conditioned model has no 'where'. No caption owns any pixels, so the fix\n"
               "is unavailable to SDXL and to every model this repository works on.",
         colour=THEIRS),
    dict(year="2026", who="This repository", what="How much co-occurrence the product needs",
         claim="Vary how much the product wants the composition, and read the two failures apart.",
         image=OUT / "weight-sweep.png", missing=None,
         showed="The exact product needs no measurable co-occurrence: 0 crisp pairs of 300 at zero,\n"
                "300 of 300 at a pair-image weight of 0.001. Its two samplers cross half of 300\n"
                "only between 0.3 and 0.5. The gap is about three orders of magnitude in that weight.",
         open_="Two shapes at one spread, so the penalty exponent is fixed at 2610. The corrector is\n"
               "Metropolis-adjusted Langevin, not the HMC Du et al. used, so the annealed curve is\n"
               "not a verdict on their method.",
         colour=OURS),
]

heights = []
for r in ROWS:
    if r["image"] is not None and r["image"].exists():
        im = mpimg.imread(r["image"])
        heights.append(min(3.4, max(1.9, 7.0 * im.shape[0] / im.shape[1])))
    else:
        heights.append(1.9)

W = 19.0
fig = plt.figure(figsize=(W, sum(heights) + 1.4))
gs = fig.add_gridspec(len(ROWS), 3, width_ratios=[3.9, 6.6, 7.8],
                      height_ratios=heights, hspace=0.12, wspace=0.03,
                      left=0.012, right=0.988, top=1 - 1.0 / (sum(heights) + 1.4), bottom=0.012)

for i, r in enumerate(ROWS):
    ax = fig.add_subplot(gs[i, 0]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.06), 0.055, 0.88, transform=ax.transAxes,
                                boxstyle="round,pad=0.004", facecolor=r["colour"], edgecolor="none"))
    ax.text(0.11, 0.88, r["year"], fontsize=20, fontweight="bold", color=r["colour"],
            transform=ax.transAxes, va="top")
    ax.text(0.11, 0.66, r["who"], fontsize=11.5, color=INK, transform=ax.transAxes, va="top")
    what = textwrap.fill(r["what"], 32)
    ax.text(0.11, 0.55, what, fontsize=10.5, color=FAINT, transform=ax.transAxes,
            va="top", style="italic", linespacing=1.4)
    ax.text(0.11, 0.55 - 0.13 * (1 + what.count("\n")), textwrap.fill(r["claim"], 34),
            fontsize=10.5, color=INK, transform=ax.transAxes, va="top", linespacing=1.5)

    ax = fig.add_subplot(gs[i, 1]); ax.axis("off")
    if r["image"] is not None and r["image"].exists():
        ax.imshow(mpimg.imread(r["image"]))
        tag = "their figure" if r["colour"] == THEIRS else "made here"
        ax.text(0.0, 1.015, tag, fontsize=8, color=FAINT, transform=ax.transAxes)
    else:
        ax.text(0.03, 0.5, r["missing"], fontsize=9.5, color=FAINT, transform=ax.transAxes,
                va="center", linespacing=1.6)
        ax.add_patch(FancyBboxPatch((0.015, 0.12), 0.97, 0.76, transform=ax.transAxes,
                                    boxstyle="round,pad=0.006", facecolor="#f6f6f4",
                                    edgecolor="#e0e0dc", zorder=0))

    ax = fig.add_subplot(gs[i, 2]); ax.axis("off")
    ax.text(0.03, 0.93, "WHAT IT SHOWED", fontsize=8.5, color=r["colour"], fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.82, r["showed"], fontsize=10, color=INK, transform=ax.transAxes,
            va="top", linespacing=1.5)
    ax.text(0.03, 0.46, "WHAT IT LEFT OPEN", fontsize=8.5, color=FAINT, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.35, r["open_"], fontsize=10, color=FAINT, transform=ax.transAxes,
            va="top", linespacing=1.5)
    if i < len(ROWS) - 1:
        ax.annotate("", xy=(0.5, -0.03), xytext=(0.5, 0.06), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#cfcfca", lw=1.6))

fig.suptitle("Four results on why a product of experts fails to compose, and the one this sweep adds",
             fontsize=15, y=1 - 0.30 / (sum(heights) + 1.4))
fig.text(0.012, 1 - 0.68 / (sum(heights) + 1.4),
         "Oldest at the top. Each row carries the paper's own figure where one is on disk, what it established, and the question it left for the next row. "
         "Every count below is out of 300 draws on an exact toy; the papers' own numbers are quoted from their text.",
         fontsize=9.5, color=FAINT)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "the-chain-of-results.png", dpi=110, bbox_inches="tight", facecolor="white")
print("wrote", OUT / "the-chain-of-results.png")
