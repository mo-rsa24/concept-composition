#!/usr/bin/env python
"""Cut the 32 joint-prompt renders out of the supervisor-meeting contact sheet.

The sheet is the only place the joint prompt's cat-by-dog renders exist as images on this
machine. It is a regular 8 x 4 grid, 2560 x 1392, labelled "seed 1" to "seed 32" reading left to
right then down. Each row band is 348 px tall and opens with a label strip, so the tile is the
320 x 320 square under it.

Writes joint-tiles/seedNN.png, one per seed.
"""
from __future__ import annotations

from PIL import Image

import paths


def main() -> None:
    sheet = Image.open(paths.JOINT_SHEET).convert("RGB")
    band = paths.SHEET_LABEL + paths.SHEET_TILE
    expected = (paths.SHEET_COLS * paths.SHEET_TILE, paths.SHEET_ROWS * band)
    if sheet.size != expected:
        raise SystemExit(f"sheet is {sheet.size}, expected {expected}; the grid has changed")

    paths.JOINT_TILES.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in range(paths.SHEET_ROWS):
        for col in range(paths.SHEET_COLS):
            n += 1
            left = col * paths.SHEET_TILE
            top = row * band + paths.SHEET_LABEL
            tile = sheet.crop((left, top, left + paths.SHEET_TILE, top + paths.SHEET_TILE))
            tile.save(paths.JOINT_TILES / f"seed{n:02d}.png")
    print(f"wrote {n} tiles to {paths.JOINT_TILES}")


if __name__ == "__main__":
    main()
