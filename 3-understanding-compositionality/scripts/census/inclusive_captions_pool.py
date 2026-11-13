#!/usr/bin/env python
"""Build the four photo sets the inclusive-captions pilot trains on, out of COCO 2017.

The design (artifacts/ideas/inclusive-captions-axis/COMPILED.md) asks for 300 cat-only, 300
dog-only, 200 both-animal and 200 neither-animal photos, the same 800 in every arm, with only the
caption on the both-animal photos changing between arms. Two numbers in that design are
assumptions until this runs: that COCO holds 200 usable both-animal photos at all, and that the
cat-only and dog-only pools are big enough to pick 300 clean ones from. The pilot's effect size is
set by the both-animal count, so if it comes back near 90 rather than 200 the share of
cat-captioned dog-present draws halves and the seed counts were computed against the wrong number.

What this does, reading `instances_train2017.json` only.

  1. Sorts every image by which of cat (category 17) and dog (category 18) it holds, ignoring
     annotations marked `iscrowd`, which are unsegmented crowds rather than instances.
  2. Applies the size filter below and prints the count surviving each stage, so the drop from
     the raw count to the usable one is visible rather than inferred.
  3. Picks the four sets with a fixed seed and writes a manifest naming every chosen file, with
     each animal's box area fraction and, for the both-animal photos, the overlap of the cat and
     dog boxes. The manifest is what the trainer reads, so an arm is reproducible from it.
  4. Lays the both-animal photos out as contact sheets, 48 tiles a page, for the audit the source
     asks for: COCO's annotations miss animals and label some that are drawings or statues, and
     this set is small enough that every one of them can be looked at.

The size filter. An animal must span at least MIN_BOX_FRACTION of the image's longer side. This is
the same 0.25 the compose scorer draws (poe_repair/experiments/compose_scorer_validation/
detection_scorer.py), chosen there by looking at boxes rather than at a resulting rate, so the
training photos and the scored renders agree about what counts as an animal being present. A photo
whose dog is a 40-pixel smudge in the corner teaches the cat condition almost nothing about dogs,
which is the quantity the pilot is trying to move.

What the answer means, written before the run:
  200 or more usable both-animal photos   the design stands as written.
  120 to 199                              the design stands with the both-animal set at the real
                                          number, and the share of cat-captioned dog-present
                                          draws is recomputed; the seed counts rise with it.
  under 120                               the effect size is half what the design assumed. Either
                                          the size filter is loosened (and said so), or the pilot
                                          moves to a pair with more co-occurrence than cat and dog.

Writes, under MOUNT_ROOT/outputs/inclusive_captions_pool/:
    manifest.json        the four sets, every file path, every box measurement
    counts.json          the count surviving each filter stage, for every category
    both_sheet_<nn>.png  the both-animal photos, 48 a page, numbered for the audit

Run it on a compute node; it needs no GPU and reads a 450 MB json:
    scripts/cluster.sh sh "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      scripts/census/inclusive_captions_pool.py"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poe_repair import paths  # noqa: E402

COCO_ROOT = Path("/datasets/mmolefe/coco2017")
ANNOTATIONS = COCO_ROOT / "annotations_trainval2017/annotations/instances_train2017.json"
CAPTIONS = COCO_ROOT / "annotations_trainval2017/annotations/captions_train2017.json"


def _images_dir() -> Path:
    """The folder holding the train2017 jpegs.

    The download on this cluster nests them one level deeper than the archive layout, so both
    spellings are checked and the one that actually holds a jpeg wins. Guessing wrong produces a
    manifest of paths that do not exist, which only surfaces when something opens one.
    """
    for candidate in (COCO_ROOT / "train2017" / "train2017", COCO_ROOT / "train2017"):
        if candidate.is_dir() and next(candidate.glob("*.jpg"), None) is not None:
            return candidate
    raise SystemExit(f"no train2017 jpegs under {COCO_ROOT / 'train2017'}")


IMAGES = _images_dir()
FAMILY = "outputs/inclusive_captions_pool"

CAT_ID, DOG_ID = 17, 18
# The same line the compose scorer draws, so training photos and scored renders agree about what
# counts as an animal being present.
MIN_BOX_FRACTION = 0.25
# The design's own set sizes, which the caption sampler reads back from SET_SIZES.
#
# The both-animal count is 91: COCO holds 208 photos labelled with both animals, 131 clear the
# size filter, and 91 of those survive looking at every one of them. The 40 that do not are
# EXCLUDED_BOTH below.
#
# The single-animal sets are 200 rather than 300 because the quantity the pilot moves is a ratio:
# the fraction of run B's cat-captioned draws that hold a dog is 91*0.40 / (N*0.90 + 91*0.40),
# so the cat-only count is its denominator. At N = 300 that fraction is 0.119, under what 128
# seeds can resolve; at N = 200 it is 0.168. The cut applies to run A and run B identically, so
# the comparison still differs in one thing.
WANT = {"cat_only": 200, "dog_only": 200, "both": 91, "neither": 91}

# Indices into the shuffled both-animal list that looking at the photographs rejected. Thirty-three
# do not hold one live cat and one live dog: ten are a cat beside a depiction of a dog (a painting,
# an illustration, an artwork of printed neckties, etched mugs, a painted van, a book cover, a
# monitor, a laptop screen, a ceramic figurine), five are collages of separate photographs, six put
# both labels on a single animal, and twelve have no second animal in the frame. Seven more are
# ones a careful look could not call either way, and they go out with the rest because a photo
# nobody can name is a photo that teaches the condition something unknown.
#
# The reasons per index are in artifacts/results/the-inclusive-captions-photo-pool/audit-verdicts.json.
# The indices are stable: the shuffle's sequence depends on each pool's length, not on WANT.
EXCLUDED_BOTH = frozenset({
    0, 2, 9, 15, 16, 17, 24, 26, 27, 28, 40, 48, 57, 58, 61, 65, 69, 72, 74, 76, 77, 82, 84, 85,
    86, 89, 90, 92, 93, 94, 98, 100, 103, 104, 105, 108, 113, 117, 120, 122,
})
SEED = 0
TILES_PER_SHEET = 48


def read_captions() -> dict[int, str]:
    """One COCO caption per image, the shortest of the five each photo carries.

    The neither-animal filler photos in the no-both run are trained under their own caption. Any
    single invented caption would be a third condition that runs A and B do not have, and an
    empty string is the blank caption, which would put that run's blank rate at 26% against the
    other runs' 10%. COCO gives each photo about five human captions; the shortest is taken
    because the long ones run to twenty words and CLIP's 77-token limit truncates them unevenly.
    """
    data = json.loads(CAPTIONS.read_text())
    best: dict[int, str] = {}
    for a in data["annotations"]:
        text = a["caption"].strip()
        current = best.get(a["image_id"])
        if current is None or len(text) < len(current):
            best[a["image_id"]] = text
    return best


def box_side_fraction(box, width: int, height: int) -> float:
    """The box's longer side as a fraction of the image's longer side."""
    _, _, w, h = box
    return max(w, h) / max(width, height)


def box_iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-box-fraction", type=float, default=MIN_BOX_FRACTION)
    ap.add_argument("--no-sheets", action="store_true", help="skip the audit sheets")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"reading {ANNOTATIONS} ({ANNOTATIONS.stat().st_size / 1e6:.0f} MB)", flush=True)
    data = json.loads(ANNOTATIONS.read_text())
    images = {im["id"]: im for im in data["images"]}
    print(f"{len(images)} images, {len(data['annotations'])} annotations "
          f"({time.time() - started:.0f}s)", flush=True)

    captions = read_captions()
    print(f"{len(captions)} captioned images ({time.time() - started:.0f}s)", flush=True)

    # Per image, the cat and dog boxes that are real instances rather than crowds.
    per_image: dict[int, dict[str, list]] = defaultdict(lambda: {"cat": [], "dog": []})
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        if a["category_id"] == CAT_ID:
            per_image[a["image_id"]]["cat"].append(a["bbox"])
        elif a["category_id"] == DOG_ID:
            per_image[a["image_id"]]["dog"].append(a["bbox"])

    counts = {"images_total": len(images), "images_with_cat_or_dog": len(per_image)}
    pools: dict[str, list[dict]] = {"cat_only": [], "dog_only": [], "both": []}
    raw = {"cat_only": 0, "dog_only": 0, "both": 0, "both_one_of_each": 0}

    for image_id, boxes in per_image.items():
        im = images[image_id]
        w, h = im["width"], im["height"]
        cats = [b for b in boxes["cat"]]
        dogs = [b for b in boxes["dog"]]
        big_cats = [b for b in cats if box_side_fraction(b, w, h) >= args.min_box_fraction]
        big_dogs = [b for b in dogs if box_side_fraction(b, w, h) >= args.min_box_fraction]

        row = {"image_id": image_id, "file": str(IMAGES / im["file_name"]),
               "width": w, "height": h, "n_cat": len(cats), "n_dog": len(dogs),
               "caption": captions.get(image_id, "")}

        if cats and not dogs:
            raw["cat_only"] += 1
            if big_cats:
                row["largest_cat_fraction"] = max(box_side_fraction(b, w, h) for b in cats)
                pools["cat_only"].append(row)
        elif dogs and not cats:
            raw["dog_only"] += 1
            if big_dogs:
                row["largest_dog_fraction"] = max(box_side_fraction(b, w, h) for b in dogs)
                pools["dog_only"].append(row)
        elif cats and dogs:
            raw["both"] += 1
            if len(cats) == 1 and len(dogs) == 1:
                raw["both_one_of_each"] += 1
            if big_cats and big_dogs:
                c = max(big_cats, key=lambda b: b[2] * b[3])
                d = max(big_dogs, key=lambda b: b[2] * b[3])
                row["largest_cat_fraction"] = box_side_fraction(c, w, h)
                row["largest_dog_fraction"] = box_side_fraction(d, w, h)
                row["cat_dog_box_iou"] = box_iou(c, d)
                pools["both"].append(row)

    # The filler set: no cat and no dog at all, and at least one annotation so the photo is a
    # scene rather than an empty frame.
    annotated = {a["image_id"] for a in data["annotations"]}
    neither_ids = [i for i in annotated if i not in per_image]
    raw["neither"] = len(neither_ids)
    # A filler photo with no caption is unusable, since an empty caption is the blank caption.
    pools["neither"] = [{"image_id": i, "file": str(IMAGES / images[i]["file_name"]),
                         "width": images[i]["width"], "height": images[i]["height"],
                         "n_cat": 0, "n_dog": 0, "caption": captions[i]}
                        for i in neither_ids if captions.get(i)]

    counts["raw_by_category"] = raw
    counts["after_size_filter"] = {k: len(v) for k, v in pools.items()}
    counts["min_box_fraction"] = args.min_box_fraction
    counts["wanted"] = WANT
    counts["short_by"] = {k: max(0, WANT[k] - len(pools[k])) for k in WANT}

    rng = random.Random(SEED)
    manifest = {}
    for name, want in WANT.items():
        pool = sorted(pools[name], key=lambda r: r["image_id"])
        rng.shuffle(pool)
        if name == "both":
            pool = [r for i, r in enumerate(pool) if i not in EXCLUDED_BOTH]
            counts["both_rejected_by_eye"] = len(EXCLUDED_BOTH)
            counts["both_surviving_the_audit"] = len(pool)
        manifest[name] = pool[:want]

    # Every chosen photo has to carry a caption and have a file on disk. A manifest naming a
    # path that does not open only surfaces when the trainer opens it, by which point a run has
    # already started, and an uncaptioned photo would train under the blank caption instead.
    missing_file: list[str] = []
    missing_caption: list[str] = []
    for name, rows in manifest.items():
        for r in rows:
            if not Path(r["file"]).is_file():
                missing_file.append(r["file"])
            if not r.get("caption"):
                missing_caption.append(r["file"])
    counts["chosen_total"] = sum(len(v) for v in manifest.values())
    counts["missing_file"] = len(missing_file)
    counts["missing_caption"] = len(missing_caption)
    if missing_file:
        print(f"\n{len(missing_file)} chosen photos have no file on disk, first 3:", flush=True)
        for f in missing_file[:3]:
            print(f"  {f}", flush=True)
    if missing_caption:
        print(f"\n{len(missing_caption)} chosen photos have no caption, first 3:", flush=True)
        for f in missing_caption[:3]:
            print(f"  {f}", flush=True)

    verdict_n = counts.get("both_surviving_the_audit", len(pools["both"]))
    if verdict_n >= 120:
        counts["verdict"] = (f"{verdict_n} both-animal photos hold one live cat and one live dog: "
                             f"the design stands with the both-animal set at {verdict_n}, and the "
                             f"share of cat-captioned dog-present draws is recomputed from it")
    else:
        counts["verdict"] = (f"only {verdict_n} both-animal photos survive: the share of "
                             f"cat-captioned dog-present draws is "
                             f"{verdict_n * 0.40 / (WANT['cat_only'] * 0.90 + verdict_n * 0.40):.3f}. "
                             f"Cut the single-animal sets further, loosen the size filter and say "
                             f"so, or move the pilot to a pair that co-occurs more often")

    (out_dir / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "min_box_fraction": args.min_box_fraction,
         "annotations": str(ANNOTATIONS), "sets": manifest}, indent=2))
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2), flush=True)

    if not args.no_sheets:
        both = manifest["both"]
        for page in range((len(both) + TILES_PER_SHEET - 1) // TILES_PER_SHEET):
            rows = both[page * TILES_PER_SHEET:(page + 1) * TILES_PER_SHEET]
            ncol = 8
            nrow = (len(rows) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.8, nrow * 2.0), squeeze=False)
            for ax, r in zip(axes.ravel(), rows):
                with Image.open(r["file"]) as im:
                    ax.imshow(im.convert("RGB").resize((224, 224)))
                ax.set_title(f"{page * TILES_PER_SHEET + rows.index(r)}  "
                             f"cat {r['largest_cat_fraction']:.2f} dog {r['largest_dog_fraction']:.2f}\n"
                             f"overlap {r['cat_dog_box_iou']:.2f}", fontsize=6)
                ax.axis("off")
            for ax in axes.ravel()[len(rows):]:
                ax.axis("off")
            fig.suptitle(f"COCO train2017, photos holding both a cat and a dog, page {page + 1}: "
                         f"each title is the tile's number for the audit, each animal's box as a "
                         f"fraction of the\nimage's longer side, and how much the two boxes overlap",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / f"both_sheet_{page:02d}.png", dpi=110)
            plt.close(fig)
            print(f"wrote both_sheet_{page:02d}.png ({len(rows)} tiles)", flush=True)

    print(f"\n{counts['verdict']}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
