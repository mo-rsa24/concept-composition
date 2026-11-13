#!/usr/bin/env python
"""The two findings, one panel each, with the measured quantity on the y axis.

Left panel: the smaller of an image's similarity to "a photo of a cat" and to "a photo of a
dog". The product-of-probabilities argument predicts the fused renders sit above the
photographs here.

Right panel: similarity to "a photo of a cat and a dog", on the same y scale so the two gaps
can be compared by eye.

One dot per image, jittered sideways so overlapping values stay countable; the black bar is the
set's mean and the number beside it is that mean. Four example images run underneath with both
of their numbers.
"""
from __future__ import annotations

import json
import random
import statistics as st
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

import paths

NODE = "mscluster84"
ORDER = ["photo", "product", "joint", "plural"]
COLOUR = {"photo": "#555555", "product": "#d1495b", "joint": "#2a9d8f", "plural": "#e9c46a"}
TICK = {"photo": "real photograph\nof a cat and a dog",
        "product": "product of a\ncat and a dog expert",
        "joint": "one prompt naming\nboth animals",
        "plural": 'product prompted\n"cats" and "dogs"'}


def local_file(rec):
    if rec["set"] == "photo":
        paths.PHOTO_THUMBS.mkdir(parents=True, exist_ok=True)
        f = paths.PHOTO_THUMBS / f"{rec['key']}.jpg"
        if not f.exists():
            subprocess.run(["rsync", "-az", f"{NODE}:{rec['path']}", str(f)], check=True)
        return f
    if rec["set"] == "joint":
        return paths.JOINT_TILES / f"seed{rec['seed']:02d}.png"
    folder = paths.PRODUCT_TILES if rec["set"] == "product" else paths.PLURAL_TILES
    return folder / f"seed{rec['seed']:02d}" / "poe.png"


def square(path, side=190):
    with Image.open(path) as im:
        im = im.convert("RGB")
        s = min(im.size)
        return im.crop(((im.width - s) // 2, (im.height - s) // 2,
                        (im.width + s) // 2, (im.height + s) // 2)).resize((side, side))


def panel(ax, by_set, field, title, note):
    rng = random.Random(0)
    for i, name in enumerate(ORDER):
        values = [r[field] for r in by_set[name]]
        ax.scatter([i + rng.uniform(-0.17, 0.17) for _ in values], values,
                   s=34, c=COLOUR[name], alpha=0.75, edgecolors="white", linewidths=0.5,
                   zorder=3)
        mean = st.mean(values)
        ax.plot([i - 0.33, i + 0.33], [mean, mean], c="black", lw=2.4, zorder=4)
        ax.text(i + 0.37, mean, f"{mean:.3f}", va="center", ha="left", fontsize=10.5,
                fontweight="bold", zorder=5)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([TICK[n] for n in ORDER], fontsize=9)
    ax.set_xlim(-0.6, len(ORDER) - 0.25)
    ax.set_title(title, fontsize=12, pad=10)
    ax.text(0.5, -0.145, note, transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color="#444444")
    ax.grid(axis="y", alpha=0.2, zorder=0)


def main() -> None:
    records = json.loads(paths.SIMILARITIES.read_text())["records"]
    by_set: dict[str, list[dict]] = {}
    for rec in records:
        by_set.setdefault(rec["set"], []).append(rec)

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = GridSpec(2, 4, height_ratios=[2.5, 1.15], hspace=0.42, wspace=0.12,
                  left=0.07, right=0.97, top=0.885, bottom=0.03)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:], sharey=ax1)

    panel(ax1, by_set, "sim_weaker",
          "Each animal on its own:\nthe weaker of the two single-animal captions",
          "the fused renders sit 0.014 above the photographs,\nand the two-animal renders sit "
          "0.017 above them")
    panel(ax2, by_set, "sim_both",
          'Both animals in one sentence:\n"a photo of a cat and a dog"',
          "the fused renders sit 0.043 below the photographs,\nthree times the gap in the left "
          "panel and the other way")
    ax1.set_ylabel("cosine similarity (runs from -1 to 1; this data spans 0.165 to 0.331)")
    plt.setp(ax2.get_yticklabels(), visible=False)

    # Four examples, each the image closest to its set's mean on both measures.
    # seed 4 is left out of the examples: it renders a wallpaper of flat cat and dog
    # silhouettes rather than an animal, so it is not an example of a fused animal. Its dot
    # stays in the panels above.
    picks = []
    for name in ("product", "photo"):
        rows = [r for r in by_set[name] if not (name == "product" and r["seed"] == 4)]
        mw = st.mean(r["sim_weaker"] for r in rows)
        mb = st.mean(r["sim_both"] for r in rows)
        rows = sorted(rows, key=lambda r: abs(r["sim_weaker"] - mw) + abs(r["sim_both"] - mb))
        picks += rows[:2]

    for col, rec in enumerate(picks):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(square(local_file(rec)))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOUR[rec["set"]])
            spine.set_linewidth(2.4)
        ax.set_title(f"{rec['key']}  ·  near its set's mean", fontsize=9,
                     color=COLOUR[rec["set"]], pad=4)
        ax.set_xlabel(f"weaker single-animal {rec['sim_weaker']:.3f}\n"
                      f'"a cat and a dog" {rec["sim_both"]:.3f}', fontsize=8.5, labelpad=4)

    fig.suptitle("Does a fused animal fit both captions better than a real pair does?\n"
                 "CLIP ViT-B/32, 137 images, one dot each", fontsize=14, y=0.975)
    fig.savefig(paths.COLUMNS, dpi=150, bbox_inches="tight")
    print("wrote", paths.COLUMNS)


if __name__ == "__main__":
    main()
