"""One face for every paper figure: Liberation Serif (Times metrics), embedded as TrueType.

The lettering rule is on 1-lora-correction/report/paper-review/layout-sheet.md: 8 pt labels, 9 pt titles,
a Times-matching serif, drawn at printed width. Call use_paper_font() before drawing;
every text call that asks for family="serif" then gets this face, and the call refuses to
fall back to another face silently.
"""
from __future__ import annotations

import subprocess

import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "Liberation Serif"


def use_paper_font() -> None:
    out = subprocess.run(["fc-list", FONT, "file"], capture_output=True, text=True).stdout
    for path in out.split():
        font_manager.fontManager.addfont(path.rstrip(":"))
    if not any(f.name == FONT for f in font_manager.fontManager.ttflist):
        raise SystemExit(f"{FONT} not found; refusing to draw in a fallback face")
    plt.rcParams.update({"font.family": "serif", "font.serif": [FONT], "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
