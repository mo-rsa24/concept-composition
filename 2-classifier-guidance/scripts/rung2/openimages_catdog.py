#!/usr/bin/env python
"""Pull Open Images V7 photos of cats and dogs as checker data, chosen by the dataset's own boxes.

The same rules as coco_catdog.py, on a set about ten times larger. A counted animal is a Cat (/m/01yrx) or
Dog (/m/0bt9lr) box covering at least MIN_AREA of the image. A photo is left out when any of its cat or dog
boxes is a depiction (a drawing, toy, statue or picture: Open Images' own IsDepiction flag), a group box, or
smaller than MIN_AREA, so "exactly" means exactly:
    clean candidate   exactly one cat and exactly one dog         (checked again by the API, strict + real-only)
    wrong pair        two or more cats and no dog, or two or more dogs and no cat   (photo rejects)
    single            exactly one cat and no dog, or exactly one dog and no cat     (photo rejects, capped)

Each photo is cut to the largest square holding every counted box; one where no such square exists is
skipped and counted. Images come from the public bucket open-images-dataset.s3.amazonaws.com.

Writes data/open-images-v7/catdog/<clean|wrong_pair|single>/<image id>.jpg and catdog/manifest.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3] / "data/open-images-v7"
CAT, DOG = "/m/01yrx", "/m/0bt9lr"
MIN_AREA = 0.02
FILES = {"train": "oidv6-train-annotations-bbox.csv", "validation": "validation-annotations-bbox.csv",
         "test": "test-annotations-bbox.csv"}


def square_box(w, h, boxes):
    """The largest square inside a w x h image holding every (x0, y0, x1, y1) pixel box, or None."""
    side = min(w, h)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if x1 - x0 > side or y1 - y0 > side:
        return None
    left = min(max(0, (x0 + x1) / 2 - side / 2), w - side)
    top = min(max(0, (y0 + y1) / 2 - side / 2), h - side)
    left, top = max(0, min(left, x0)), max(0, min(top, y0))
    left, top = min(left, w - side), min(top, h - side)
    return int(left), int(top), int(left + side), int(top + side)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-single", type=int, default=800)
    ap.add_argument("--max-clean", type=int, default=0, help="0 keeps every clean candidate")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    boxes = collections.defaultdict(list)
    for split, name in FILES.items():
        with open(ROOT / name, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["LabelName"] in (CAT, DOG):
                    boxes[(split, r["ImageID"])].append(r)
    chosen, skipped = [], collections.Counter()
    for (split, iid), rows in boxes.items():
        if any(r["IsDepiction"] == "1" or r["IsGroupOf"] == "1" for r in rows):
            skipped["a depiction or group box"] += 1
            continue
        area = [(float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"])) for r in rows]
        if min(area) < MIN_AREA:
            skipped["a cat or dog below 2% of the image"] += 1
            continue
        cats = sum(r["LabelName"] == CAT for r in rows)
        dogs = sum(r["LabelName"] == DOG for r in rows)
        if cats == 1 and dogs == 1:
            kind = "clean"
        elif (cats >= 2 and not dogs) or (dogs >= 2 and not cats):
            kind = "wrong_pair"
        elif cats + dogs == 1:
            kind = "single"
        else:
            skipped["other mix"] += 1
            continue
        chosen.append({"id": iid, "split": split, "kind": kind, "cats": cats, "dogs": dogs,
                       "boxes": [[float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])] for r in rows]})
    rng = random.Random(0)
    by = collections.defaultdict(list)
    for c in chosen:
        by[c["kind"]].append(c)
    for kind, cap in (("single", args.max_single), ("clean", args.max_clean), ("wrong_pair", args.max_single)):
        rng.shuffle(by[kind])
        if cap:
            by[kind] = by[kind][:cap]
    chosen = [c for k in by.values() for c in k]
    print("candidates", {k: len(v) for k, v in by.items()}, "skipped", dict(skipped), flush=True)

    out = ROOT / "catdog"
    for k in ("clean", "wrong_pair", "single"):
        (out / k).mkdir(parents=True, exist_ok=True)

    def fetch(c):
        dest = out / c["kind"] / f"{c['id']}.jpg"
        if dest.exists():
            return "had"
        url = f"https://open-images-dataset.s3.amazonaws.com/{c['split']}/{c['id']}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return f"failed: {type(e).__name__}"
        w, h = im.size
        box = square_box(w, h, [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in c["boxes"]])
        if box is None:
            return "animals wider apart than the short side"
        im.crop(box).save(dest, quality=95)
        return "got"

    with ThreadPoolExecutor(args.workers) as ex:
        results = collections.Counter(ex.map(fetch, chosen))
    print("downloads", dict(results), flush=True)
    (out / "manifest.json").write_text(json.dumps({"rule": __doc__.split("\n\n")[1], "min_area": MIN_AREA,
                                                   "skipped": dict(skipped), "images": chosen}, indent=0))
    print("OPEN IMAGES CATDOG DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
