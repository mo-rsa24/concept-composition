"""Label the reference radiograph with the features the case report names.

Every label is a sentence from the published report, placed on the structure it describes.
The two red arrows already on the film are the authors' own and are called out as theirs,
so nothing here is mistaken for a finding this walk added.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

HERE = Path(__file__).resolve().parents[1] / "reference-images"
SRC = HERE / "silicotuberculosis-pa-radiograph-cureus-2024.png"
DST = HERE / "silicotuberculosis-pa-radiograph-annotated.png"

img = imread(SRC)
h, w = img.shape[:2]

# (label, point on the film, label anchor in axes fraction, colour)
LABELS = [
    ("small rounded opacities through the upper and\nmiddle zones of both lungs: the silicosis pattern\n"
     "(the authors' own red arrow points here)",
     (1018, 1140), (1.07, 0.78), "#00a8e0"),
    ("coalescence into a confluent upper-lobe mass:\nprogressive massive fibrosis, with the cavity and\n"
     "consolidation that favour tuberculosis\n(the authors' second red arrow)",
     (2158, 783), (1.07, 0.97), "#e09a00"),
    ("linear fibrotic opacities and subtle pleural\nthickening at the apex",
     (2140, 200), (1.07, 0.56), "#00a8e0"),
    ("the cardiac border is within normal limits but\npartly obscured by the opacity lying over it:\n"
     "this is the masking turn 2 of the conversation names",
     (1491, 2205), (1.07, 0.30), "#e0506a"),
]

fig, ax = plt.subplots(figsize=(15.0, 11.0))
ax.imshow(img)
ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis("off")

for text, (px, py), (ax_x, ax_y), colour in LABELS:
    ax.plot([px], [py], marker="o", ms=9, mfc="none", mec=colour, mew=2.2, zorder=5)
    ax.annotate(
        text, xy=(px, py), xycoords="data",
        xytext=(ax_x, ax_y), textcoords="axes fraction",
        fontsize=10.0, color="#111", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=colour, lw=1.7),
        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8,
                        connectionstyle="arc3,rad=0.0", shrinkB=0),
    )

ax.set_title(
    "Silicosis with superimposed tuberculosis, posteroanterior film, with the report's own reading placed on the film",
    fontsize=12.5, pad=14, loc="left",
)
fig.text(
    0.012, 0.045,
    "Figure 1 of Elango R, Kailash A, Verma G, Rudhramoorty P. Silico-Tuberculosis Mimicking Malignancy. Cureus 16(7): e65411, 2024. "
    "DOI 10.7759/cureus.65411. Licence CC BY 4.0: reuse permitted with this credit.\n"
    "The two red arrows are the original authors'. Every labelled feature is quoted or paraphrased from that report's own description of this film; "
    "the structures are located by eye and are approximate.\n"
    "The report ends its reading by saying a definitive call still needs the occupational history and the microbiology, "
    "which is the limit on imaging evidence that turn 2 of the conversation argues for.",
    fontsize=9.0, color="#333", ha="left", va="top",
)
fig.subplots_adjust(left=0.01, right=0.62, top=0.95, bottom=0.17)
fig.savefig(DST, dpi=150)
print("wrote", DST)
