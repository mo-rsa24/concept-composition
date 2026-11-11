#!/usr/bin/env python
"""Figure 7 of the ICLR 56173 paper, the adapter's limitations, in its original two-panel layout
with every label set as real type at the paper's size.

Panel (a): four pairs under the joint prompt, base model above and through the adapter below,
pair names on top, row labels turned on their side at the left, the adapter note underneath.
Panel (b): six cat-and-dog renders from the product, a name under each.

Drawn at the printed width (5.5 in) so nothing is scaled: labels 8 pt, panel letters 9 pt bold,
Liberation Serif. Tiles are centre-cropped to squares so every picture prints the same size.

    python3 scripts/figure7_limitations.py <tiles dir> <out.pdf>

The tiles dir holds joint-r{0,1}-c{0..3}.png and fail-r{0,1}-c{0..2}.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_font import use_paper_font  # noqa: E402

LABEL_PT, LETTER_PT = 8, 9
WIDTH = 5.5
GAP = 0.03            # between tiles
ROW_LABELS = 0.30     # two rotated lines at the left of panel (a)
PANEL_GAP = 0.24      # between the panels; holds the (b) letter
LINE = LABEL_PT * 1.15 / 72
HEAD = 2 * LINE + 0.05            # two-line pair names above panel (a)
NOTE = 3 * LINE + 0.05            # up to three lines under each tile of panel (b)
FOOT = LINE + 0.05                # the adapter note under panel (a)

PAIRS = ["Cat and dog", "Tiger and dog", "Chess board\nand hourglass", "Picnic basket\nand watermelon"]
ROWS = ["Joint prompt,\nbase model", "Joint prompt,\nwith adapter"]
OUTCOMES = [["One fused\nanimal", "Invented\npeople", "Grey mush,\nno animal\nin it"],
            ["Flat line art\nof one animal", "A third\nanimal in\nthe corner", "Two animals\nof the wrong\nspecies"]]
NOTE_TEXT = "Adapter trained for 1,000 epochs"


def square(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    x0, y0 = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((x0, y0, x0 + s, y0 + s))


def main() -> int:
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    use_paper_font()
    # Equal panel heights: HEAD + 2a + GAP + FOOT = 2b + GAP + 2 NOTE, so a = b + (2 NOTE - HEAD - FOOT) / 2.
    d = (2 * NOTE - HEAD - FOOT) / 2
    # Width: ROW_LABELS + 4a + 3 GAP + PANEL_GAP + 3b + 2 GAP = WIDTH.
    b = (WIDTH - ROW_LABELS - 5 * GAP - PANEL_GAP - 4 * d) / 7
    a = b + d
    H = HEAD + 2 * a + GAP + FOOT
    fig = plt.figure(figsize=(WIDTH, H))

    def ax_at(x, y, w, h):  # inches from the top-left corner
        ax = fig.add_axes([x / WIDTH, 1 - (y + h) / H, w / WIDTH, h / H])
        ax.set_axis_off()
        return ax

    def text(x, y, s, **kw):  # inches from the top-left corner
        fig.text(x / WIDTH, 1 - y / H, s, fontsize=kw.pop("size", LABEL_PT), linespacing=1.05, **kw)

    # panel (a)
    xa = ROW_LABELS
    text(0, 0.02, "(a)", size=LETTER_PT, weight="bold", ha="left", va="top")
    for c, name in enumerate(PAIRS):
        text(xa + c * (a + GAP) + a / 2, HEAD - 0.03, name, ha="center", va="bottom")
    for r, label in enumerate(ROWS):
        y = HEAD + r * (a + GAP)
        text(ROW_LABELS / 2, y + a / 2, label, ha="center", va="center", rotation=90)
        for c in range(4):
            ax_at(xa + c * (a + GAP), y, a, a).imshow(square(tiles / f"joint-r{r}-c{c}.png"), interpolation="lanczos")
    text(xa, H - 0.02, NOTE_TEXT, ha="left", va="bottom", color="0.3")

    # panel (b)
    xb = xa + 4 * a + 3 * GAP + PANEL_GAP
    text(xb - PANEL_GAP + 0.02, 0.02, "(b)", size=LETTER_PT, weight="bold", ha="left", va="top")
    for r in range(2):
        y = r * (b + NOTE + GAP)
        for c in range(3):
            x = xb + c * (b + GAP)
            ax_at(x, y, b, b).imshow(square(tiles / f"fail-r{r}-c{c}.png"), interpolation="lanczos")
            text(x + b / 2, y + b + 0.03, OUTCOMES[r][c], ha="center", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"{out}: {WIDTH:.2f} x {H:.2f} in, panel (a) tiles {a:.2f} in, panel (b) tiles {b:.2f} in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
