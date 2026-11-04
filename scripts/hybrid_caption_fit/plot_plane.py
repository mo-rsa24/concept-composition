#!/usr/bin/env python
"""The cat-dog similarity plane: every image as a point, six of them shown as thumbnails.

x is the cosine similarity to "a photo of a cat", y to "a photo of a dog". Both run from -1 to
1 in principle; CLIP puts everything in a band around 0.2, so the axes are cropped to the data.
The dashed diagonal is where an image matches both captions equally. An image far up and to the
right of it is a strong match to both at once, which is what a product of two caption
probabilities is maximised by.

Six thumbnails, two per rendered set and two photographs, pinned beside their own points.
"""
from __future__ import annotations

import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

import paths

NODE = "mscluster84"
COLOUR = {"photo": "#555555", "product": "#d1495b", "joint": "#2a9d8f", "plural": "#e9c46a"}
LABEL = {"photo": "real photograph of a cat and a dog",
         "product": "product of a cat expert and a dog expert",
         "joint": "one prompt naming both animals",
         "plural": 'product prompted "cats" and "dogs"'}


def thumb_path(rec):
    """A local file for this record, fetching a photograph off the cluster if needed."""
    if rec["set"] == "photo":
        paths.PHOTO_THUMBS.mkdir(parents=True, exist_ok=True)
        local = paths.PHOTO_THUMBS / f"{rec['key']}.jpg"
        if not local.exists():
            subprocess.run(["rsync", "-az", f"{NODE}:{rec['path']}", str(local)], check=True)
        return local
    name = rec["key"] + ".png"
    if rec["set"] == "joint":
        return paths.JOINT_TILES / f"seed{rec['seed']:02d}.png"
    folder = paths.PRODUCT_TILES if rec["set"] == "product" else paths.PLURAL_TILES
    return folder / f"seed{rec['seed']:02d}" / "poe.png"


def main() -> None:
    records = json.loads(paths.SIMILARITIES.read_text())["records"]
    by_set: dict[str, list[dict]] = {}
    for rec in records:
        by_set.setdefault(rec["set"], []).append(rec)

    fig, ax = plt.subplots(figsize=(11, 9))
    for name in ("photo", "plural", "joint", "product"):
        rows = by_set[name]
        ax.scatter([r["sim_cat"] for r in rows], [r["sim_dog"] for r in rows],
                   s=46 if name != "photo" else 26,
                   c=COLOUR[name], alpha=0.85 if name != "photo" else 0.45,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label=f"{LABEL[name]}  (n={len(rows)})")

    lo = min(min(r["sim_cat"], r["sim_dog"]) for r in records) - 0.012
    hi = max(max(r["sim_cat"], r["sim_dog"]) for r in records) + 0.012
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, c="#999999", zorder=1)
    ax.text(hi - 0.004, hi - 0.004, "matches both captions equally", rotation=45,
            ha="right", va="bottom", fontsize=8, color="#777777", rotation_mode="anchor")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    # Two examples per rendered set and two photographs: the strongest and the weakest match
    # to both captions at once, by the smaller of the two single-animal similarities.
    picks = []
    for name in ("product", "joint", "photo"):
        rows = sorted(by_set[name], key=lambda r: r["sim_weaker"])
        picks += [(rows[-1], "strongest both-caption match"),
                  (rows[0], "weakest both-caption match")]

    offsets = [(70, 62), (-78, -66), (72, -58), (-80, 60), (84, 30), (-86, -30)]
    for (rec, why), (dx, dy) in zip(picks, offsets):
        with Image.open(thumb_path(rec)) as im:
            im = im.convert("RGB")
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                          (im.width + side) // 2, (im.height + side) // 2)).resize((96, 96))
            box = OffsetImage(im, zoom=1.0)
        ab = AnnotationBbox(
            box, (rec["sim_cat"], rec["sim_dog"]), xybox=(dx, dy), xycoords="data",
            boxcoords="offset points", pad=0.25, zorder=4,
            arrowprops=dict(arrowstyle="-", color=COLOUR[rec["set"]], lw=1.4),
            bboxprops=dict(edgecolor=COLOUR[rec["set"]], linewidth=2))
        ax.add_artist(ab)
        ax.annotate(f"{rec['key']}\n{why}", (rec["sim_cat"], rec["sim_dog"]),
                    xytext=(dx, dy - 62 if dy > 0 else dy + 58), textcoords="offset points",
                    ha="center", va="center", fontsize=7, color=COLOUR[rec["set"]])

    ax.set_xlabel('cosine similarity to "a photo of a cat"')
    ax.set_ylabel('cosine similarity to "a photo of a dog"')
    ax.set_title("Does a fused animal match both captions better than a real pair does?\n"
                 "CLIP ViT-B/32, one point per image", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.18, zorder=0)
    fig.tight_layout()
    fig.savefig(paths.PLANE, dpi=150)
    print("wrote", paths.PLANE)


if __name__ == "__main__":
    main()
