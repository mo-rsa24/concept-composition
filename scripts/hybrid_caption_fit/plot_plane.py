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

    fig, ax = plt.subplots(figsize=(14, 8.6))
    for name in ("photo", "plural", "joint", "product"):
        rows = by_set[name]
        ax.scatter([r["sim_cat"] for r in rows], [r["sim_dog"] for r in rows],
                   s=52 if name != "photo" else 28,
                   c=COLOUR[name], alpha=0.9 if name != "photo" else 0.45,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label=f"{LABEL[name]}  (n={len(rows)})")

    lo = min(min(r["sim_cat"], r["sim_dog"]) for r in records)
    hi = max(max(r["sim_cat"], r["sim_dog"]) for r in records)
    pad = (hi - lo) * 0.07
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1, c="#bbbbbb", zorder=1)
    mid = lo + (hi - lo) * 0.10
    ax.text(mid, mid, "  matches both captions equally", rotation=45,
            ha="left", va="center", fontsize=8.5, color="#999999", rotation_mode="anchor")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")

    # Six examples: per rendered set and for the photographs, the image whose weaker
    # single-animal similarity is largest and the one where it is smallest.
    picks = []
    for name in ("product", "joint", "photo"):
        rows = sorted(by_set[name], key=lambda r: r["sim_weaker"])
        picks += [(rows[-1], "strongest match to both"), (rows[0], "weakest match to both")]

    # Three slots down each side, in axes fraction. Each pick goes to the side its point
    # sits on and the slot nearest its height, so no leader line crosses the cloud.
    left = sorted([p for p in picks if p[0]["sim_cat"] < (lo + hi) / 2],
                  key=lambda p: -p[0]["sim_dog"])
    right = sorted([p for p in picks if p[0]["sim_cat"] >= (lo + hi) / 2],
                   key=lambda p: -p[0]["sim_dog"])
    while len(left) > 3:
        right.append(left.pop())
    while len(right) > 3:
        left.append(right.pop())
    heights = [0.87, 0.52, 0.17]
    slots = [(p, (-0.30, h)) for p, h in zip(left, heights)]
    slots += [(p, (1.30, h)) for p, h in zip(right, heights)]

    for (rec, why), xy in slots:
        with Image.open(thumb_path(rec)) as im:
            im = im.convert("RGB")
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                          (im.width + side) // 2, (im.height + side) // 2)).resize((104, 104))
            box = OffsetImage(im, zoom=1.0)
        ab = AnnotationBbox(
            box, (rec["sim_cat"], rec["sim_dog"]), xybox=xy, xycoords="data",
            boxcoords="axes fraction", pad=0.3, zorder=5,
            arrowprops=dict(arrowstyle="-", color=COLOUR[rec["set"]], lw=1.3, alpha=0.8),
            bboxprops=dict(edgecolor=COLOUR[rec["set"]], linewidth=2.2))
        ax.add_artist(ab)
        ax.text(xy[0], xy[1] - 0.155,
                f"{rec['key']}\n{why}: {rec['sim_weaker']:.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color=COLOUR[rec["set"]], zorder=6, clip_on=False)

    ax.set_xlabel('cosine similarity to "a photo of a cat"', labelpad=8)
    ax.set_ylabel('cosine similarity to "a photo of a dog"', labelpad=8)
    ax.set_title("Does a fused animal match both captions better than a real pair does?\n"
                 "CLIP ViT-B/32, one point per image; the number under each thumbnail is the "
                 "smaller of its two similarities", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2, fontsize=9,
              frameon=False)
    ax.grid(alpha=0.16, zorder=0)
    fig.subplots_adjust(left=0.30, right=0.70, bottom=0.17, top=0.88)
    fig.savefig(paths.PLANE, dpi=150)
    print("wrote", paths.PLANE)


if __name__ == "__main__":
    main()
