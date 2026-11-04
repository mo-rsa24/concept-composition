#!/usr/bin/env python
"""Lay the single-animal training photos out as contact sheets, so they can be looked at.

The both-animal photos were audited by eye when the pool was built, because that set carries the
whole experiment and is small enough to see. The 200 cat-only and 200 dog-only photos never were:
the only screen on them is the 0.25 box-fraction floor, which measures how much of the frame the
animal takes and says nothing about whether the thing in the box is a live animal. COCO annotates
drawings, statues, plush toys and cats on television screens under the same category id as a cat.

Writes cat_only_sheet_<nn>.png and dog_only_sheet_<nn>.png beside the manifest, 48 tiles a page,
each titled with its tile number and the animal's box fraction, so a rejection can be named.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      scripts/census/inclusive_captions_single_animal_audit.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

TILES_PER_SHEET = 48
NCOL = 8


def draw(rows: list[dict], key: str, name: str, out_dir: Path) -> None:
    for page in range((len(rows) + TILES_PER_SHEET - 1) // TILES_PER_SHEET):
        page_rows = rows[page * TILES_PER_SHEET:(page + 1) * TILES_PER_SHEET]
        nrow = (len(page_rows) + NCOL - 1) // NCOL
        fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 1.8, nrow * 2.0), squeeze=False)
        for i, (ax, r) in enumerate(zip(axes.ravel(), page_rows)):
            with Image.open(r["file"]) as im:
                ax.imshow(im.convert("RGB").resize((224, 224)))
            ax.set_title(f"{page * TILES_PER_SHEET + i}  {r[key]:.2f}", fontsize=7)
            ax.axis("off")
        for ax in axes.ravel()[len(page_rows):]:
            ax.axis("off")
        fig.suptitle(f"COCO train2017, {name} training photos, page {page + 1}: each title is the "
                     f"tile's number and the animal's box as a fraction of the image's longer side",
                     fontsize=10)
        fig.tight_layout()
        target = out_dir / f"{name.replace(' ', '_')}_sheet_{page:02d}.png"
        fig.savefig(target, dpi=110)
        plt.close(fig)
        print(f"wrote {target.name} ({len(page_rows)} tiles)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/manifest.json")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.manifest).parent
    draw(manifest["sets"]["cat_only"], "largest_cat_fraction", "cat only", out_dir)
    draw(manifest["sets"]["dog_only"], "largest_dog_fraction", "dog only", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
