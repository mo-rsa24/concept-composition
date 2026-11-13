#!/usr/bin/env python
"""Lay the training set out as a folder you can open, and as sheets with the captions on them.

The manifest names 550 files scattered across two trees: COCO's train2017 folder and the
generated images. Neither is browsable as "what this run trains on". This copies every chosen
photo into one folder per set, named so the source and the order are in the filename, and draws
contact sheets with each photo's caption written above it.

What "the caption" means here, which is the part a single line on a tile cannot carry. A photo's
caption is not fixed; it is drawn afresh every time the photo is drawn, and the two runs differ
in exactly one cell of that table:

  cat_only    "a photo of a cat" 90% of the time, blank 10%, in every run
  dog_only    "a photo of a dog" 90%, blank 10%, in every run
  both        run A: "a photo of a cat and a dog" 90%, blank 10%
              run B: "a photo of a cat" 40%, "a photo of a dog" 40%, joint 10%, blank 10%

So the tile's title names the set and the rule, not one string, because one string would be a
lie about four of the five photos.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      scripts/census/inclusive_captions_training_set.py --manifest <manifest.json> --out <folder>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RULES = {
    "cat_only": 'every run: "a photo of a cat" 90%, blank 10%',
    "dog_only": 'every run: "a photo of a dog" 90%, blank 10%',
    "both": 'A: joint 90%, blank 10%   |   B: cat 40%, dog 40%, joint 10%, blank 10%',
}
TILES_PER_PAGE, COLS = 16, 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-copy", action="store_true", help="draw the sheets and copy nothing")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())["sets"]
    out = Path(args.out)

    for name, rows in manifest.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            source = Path(row["file"])
            target = folder / f"{i:03d}_{row['source']}_{source.name}"
            if not args.no_copy and not target.exists():
                shutil.copy2(source, target)
        print(f"{name}: {len(rows)} photos in {folder}", flush=True)

        pages = (len(rows) + TILES_PER_PAGE - 1) // TILES_PER_PAGE
        for page in range(pages):
            tiles = rows[page * TILES_PER_PAGE:(page + 1) * TILES_PER_PAGE]
            nrow = (len(tiles) + COLS - 1) // COLS
            fig, axes = plt.subplots(nrow, COLS, figsize=(COLS * 4.48, nrow * 5.0), squeeze=False)
            for i, (ax, row) in enumerate(zip(axes.ravel(), tiles)):
                with Image.open(row["file"]) as im:
                    ax.imshow(im.convert("RGB").resize((448, 448)))
                ax.set_title(f"{page * TILES_PER_PAGE + i}  {row['source']}", fontsize=10)
                ax.axis("off")
            for ax in axes.ravel()[len(tiles):]:
                ax.axis("off")
            fig.suptitle(f"{name}, page {page + 1} of {pages}. The caption each of these is "
                         f"trained under: {RULES[name]}", fontsize=12)
            fig.tight_layout()
            target = out / f"sheet_{name}_{page:02d}.png"
            fig.savefig(target, dpi=100)
            plt.close(fig)
        print(f"{name}: {pages} sheets", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
