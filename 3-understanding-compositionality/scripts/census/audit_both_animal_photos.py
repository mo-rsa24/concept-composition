#!/usr/bin/env python
"""Re-render the 131 both-animal training photos big enough to judge whether they hold what COCO says.

Why this exists rather than the sheets the pool script already writes. Those are 48 tiles a page at
224 pixels, which is fine for seeing that a page is full of cats and dogs and not fine for deciding
whether tile 37's second animal is a dog or a dog-shaped cushion. This project has already made one
call off a downscaled sheet and been wrong about it, so the audit gets its own render: 16 tiles a
page at 448 pixels, nine pages for the 131.

Each tile's title carries what the decision needs:
  the tile number, which is the index into manifest.json's "both" list, so a rejection is a
  number somebody else can look up;
  each animal's box as a fraction of the image's longer side, so a borderline-small animal is
  visible as a number as well as a picture;
  how much the two boxes overlap, because a high overlap is where one animal is in front of the
  other and where the scorer later struggles.

What the audit is deciding, per tile: does this photograph hold one live cat and one live dog.
Rejected are drawings, paintings, statues, plush toys, a frame where the second animal is a sliver,
and a frame where COCO's annotation is simply wrong about the species.

What the answer means, written before the render:
  120 or more survive   the design stands; the dog-present share of run B's cat draws stays 0.163.
  111 to 119            the design stands with the share recomputed from the surviving count.
  under 111             more than 20 rejected, the share falls under what 128 seeds can resolve,
                        and the single-name caption shares have to rise before anything trains.

Writes, under MOUNT_ROOT/outputs/inclusive_captions_pool/:
    audit_page_<n>.png   16 tiles a page at 448 pixels, numbered

Run it on a compute node; no GPU:
    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      3-understanding-compositionality/scripts/census/audit_both_animal_photos.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poe_repair import paths  # noqa: E402

FAMILY = "outputs/inclusive_captions_pool"
TILES_PER_PAGE = 16
COLS = 4
TILE_PX = 448


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-per-page", type=int, default=TILES_PER_PAGE)
    ap.add_argument("--tile-px", type=int, default=TILE_PX)
    ap.add_argument("--set", default="both", choices=("both", "cat_only", "dog_only"),
                    help="which manifest set to lay out; the single-animal sets are screened "
                         "only by the box-fraction floor, which measures size and not whether "
                         "the thing in the box is a live animal")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY
    manifest = json.loads((out_dir / "manifest.json").read_text())
    both = manifest["sets"][args.set]
    subject = {"both": "holding both a cat and a dog", "cat_only": "holding one cat",
               "dog_only": "holding one dog"}[args.set]
    prefix = "audit_page" if args.set == "both" else f"audit_{args.set}_page"
    print(f"{len(both)} photos to audit in {args.set}", flush=True)

    per_page = args.tiles_per_page
    rows_per_page = (per_page + COLS - 1) // COLS
    pages = (len(both) + per_page - 1) // per_page

    for page in range(pages):
        rows = both[page * per_page:(page + 1) * per_page]
        inches = args.tile_px / 100.0
        fig, axes = plt.subplots(rows_per_page, COLS,
                                 figsize=(COLS * inches, rows_per_page * (inches + 0.35)),
                                 squeeze=False)
        for index, (ax, r) in enumerate(zip(axes.ravel(), rows)):
            with Image.open(r["file"]) as im:
                ax.imshow(im.convert("RGB").resize((args.tile_px, args.tile_px)))
            if args.set == "both":
                title = (f"{page * per_page + index}   cat {r['largest_cat_fraction']:.2f}  "
                         f"dog {r['largest_dog_fraction']:.2f}  "
                         f"overlap {r['cat_dog_box_iou']:.2f}")
            else:
                animal = "cat" if args.set == "cat_only" else "dog"
                title = (f"{page * per_page + index}   {animal} "
                         f"{r[f'largest_{animal}_fraction']:.2f}")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        for ax in axes.ravel()[len(rows):]:
            ax.axis("off")
        fig.suptitle(
            f"COCO train2017, the training photos {subject}, page {page + 1} of {pages}. Each "
            f"title is the tile's number in manifest.json's {args.set} list and each animal's "
            f"box as a fraction of the image's longer side.", fontsize=13)
        fig.tight_layout()
        target = out_dir / f"{prefix}_{page:02d}.png"
        fig.savefig(target, dpi=100)
        plt.close(fig)
        print(f"wrote {target.name} ({len(rows)} tiles)", flush=True)

    print(f"done, {pages} pages", flush=True)


if __name__ == "__main__":
    main()
