#!/usr/bin/env python
"""Draw a grid of image tiles as a PDF at its exact printed width, with real type for every label.

The paper's lettering rule (report/paper-review/layout-sheet.md): labels 8 pt, group titles 9 pt,
Liberation Serif (Times metrics) to match the Times body, the figure drawn at the width it prints so nothing is
scaled afterwards. Tiles are embedded as images; all text stays text, so
scripts/figure_text_audit.py measures it exactly.

    python3 scripts/paper_tile_figure.py <spec.json> <out.pdf>

The spec:
    {"width_in": 4.95,
     "groups": [{"title": "Rank", "columns": ["rank 8", "rank 16"]}, ...],   title may be ""
     "rows": [{"label": "seed 9", "tiles": ["path.png", null, ...]}, ...],   null draws a
     "pending": "result pending",                                             labelled grey tile
     "panel": "(a)",   optional panel letter, 9 pt bold, top left
     "footnote": "..."}  optional 8 pt line under the grid
A column label may hold "\n" for two lines; a row may carry "notes", one per tile, set under it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

LABEL_PT, TITLE_PT = 8, 9
TILE_GAP_IN, GROUP_GAP_IN = 0.025, 0.08
FONT = "Liberation Serif"  # TrueType with Times metrics; embeds cleanly, unlike the CFF Termes


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    import subprocess
    for path in subprocess.run(["fc-list", f"{FONT}", "file"], capture_output=True, text=True).stdout.split():
        font_manager.fontManager.addfont(path.rstrip(":"))
    if not any(f.name == FONT for f in font_manager.fontManager.ttflist):
        raise SystemExit(f"{FONT} not found; refusing to draw in a fallback face")
    plt.rcParams.update({"font.family": "serif", "font.serif": [FONT], "pdf.fonttype": 42})
    groups, rows = spec["groups"], spec["rows"]
    ncol = sum(len(g["columns"]) for g in groups)
    label_w = max(len(line) for r in rows for line in r["label"].split("\n")) * LABEL_PT * 0.42 / 72 + 0.06
    usable = spec["width_in"] - label_w - TILE_GAP_IN * (ncol - len(groups)) - GROUP_GAP_IN * (len(groups) - 1)
    tile = usable / ncol
    has_titles = any(g["title"] for g in groups)
    col_lines = max(c.count("\n") + 1 for g in groups for c in g["columns"])
    head = (TITLE_PT * 1.5 / 72 if has_titles else 0) + LABEL_PT * (0.6 + 1.15 * col_lines) / 72
    line_in = LABEL_PT * 1.15 / 72
    note_h = [(max(n.count("\n") + 1 for n in r["notes"] if n) * line_in + 0.05) if any(r.get("notes") or []) else 0
              for r in rows]  # a band under a row whose tiles carry notes
    foot = line_in + 0.06 if spec.get("footnote") else 0
    height = head + len(rows) * tile + (len(rows) - 1) * TILE_GAP_IN + sum(note_h) + foot
    fig = plt.figure(figsize=(spec["width_in"], height))
    W, H = spec["width_in"], height

    def ax_at(x, y, w, h):  # inches from the top-left corner
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    x = label_w
    col_x = []
    for g in groups:
        x0 = x
        for c in g["columns"]:
            col_x.append(x)
            fig.text((x + tile / 2) / W, 1 - (head - 0.04) / H, c,
                     ha="center", va="bottom", fontsize=LABEL_PT, linespacing=1.05)
            x += tile + TILE_GAP_IN
        x1 = x - TILE_GAP_IN
        if g["title"]:
            ty = TITLE_PT * 0.7 / 72
            fig.text((x0 + x1) / 2 / W, 1 - ty / H, g["title"], ha="center", va="center", fontsize=TITLE_PT)
            fig.add_artist(plt.Line2D([x0 / W, x1 / W], [1 - TITLE_PT * 1.35 / 72 / H] * 2,
                                      color="0.35", linewidth=0.5))
        x += GROUP_GAP_IN - TILE_GAP_IN
    if spec.get("panel"):  # a panel letter, top left, in the header band
        fig.text(0.0, 1 - TITLE_PT * 0.7 / 72 / H, spec["panel"], ha="left", va="center",
                 fontsize=TITLE_PT, weight="bold")
    for i, r in enumerate(rows):
        y = head + i * (tile + TILE_GAP_IN) + sum(note_h[:i])
        for j, n in enumerate(r.get("notes") or []):
            if n:
                fig.text((col_x[j] + tile / 2) / W, 1 - (y + tile + 0.03) / H, n, ha="center", va="top",
                         fontsize=LABEL_PT, linespacing=1.05)
        fig.text(0.0, 1 - (y + tile / 2) / H, r["label"], ha="left", va="center", fontsize=LABEL_PT,
                 linespacing=1.1)
        for j, t in enumerate(r["tiles"]):
            ax = ax_at(col_x[j], y, tile, tile)
            ax.set_axis_off()
            if t:
                ax.imshow(Image.open(t).convert("RGB"), interpolation="lanczos")
            else:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="0.93", transform=ax.transAxes))
                ax.text(0.5, 0.5, spec.get("pending", "pending"), ha="center", va="center",
                        fontsize=LABEL_PT, color="0.35", transform=ax.transAxes)
    if spec.get("footnote"):
        fig.text(label_w / W, (0.03) / H, spec["footnote"], ha="left", va="bottom", fontsize=LABEL_PT, color="0.3")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"{out}: {W:.2f} x {H:.2f} in, tiles {tile:.2f} in, labels {LABEL_PT} pt, titles {TITLE_PT} pt, {FONT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
