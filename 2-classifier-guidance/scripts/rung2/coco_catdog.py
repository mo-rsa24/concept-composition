#!/usr/bin/env python
"""Pull COCO 2017 photos of cats and dogs as checker data, chosen by COCO's own instance counts.

Real photographs are faster to gather than SDXL drawings, and COCO's boxes already say how many cats and
dogs a photo holds. Counted animals are non-crowd boxes covering at least MIN_AREA of the image; any cat or
dog box at all, of any size, counts against "exactly":
    clean candidate   exactly one cat and exactly one dog         (checked again by the API, strict rule)
    wrong pair        two or more cats and no dog, or two or more dogs and no cat   (photo rejects)
    single            exactly one cat and no dog, or exactly one dog and no cat     (photo rejects, capped)
The wrong pairs and singles matter as much as the clean ones: they are real photographs on the reject side,
so "photo" stops meaning "clean".

Each photo is cut to the largest square that keeps every counted animal's box inside; a photo where no such
square exists (the animals sit further apart than the short side) is skipped and counted. Only the chosen
images are downloaded, from each image's own coco_url.

Writes data/coco-2017/catdog/<clean|wrong_pair|single>/<image id>.jpg and data/coco-2017/catdog/manifest.json.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3] / "data/coco-2017"
CAT, DOG = 17, 18
MIN_AREA = 0.02


def square_box(w, h, boxes):
    """The largest square inside the image holding every box, or None."""
    side = min(w, h)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    if x1 - x0 > side or y1 - y0 > side:
        return None
    left = min(max(0, (x0 + x1) / 2 - side / 2), w - side)
    top = min(max(0, (y0 + y1) / 2 - side / 2), h - side)
    left, top = max(0, min(left, x0)), max(0, min(top, y0))
    left, top = min(left, w - side), min(top, h - side)
    return int(left), int(top), int(left + side), int(top + side)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-single", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    chosen, skipped = [], collections.Counter()
    for split in ("train2017", "val2017"):
        d = json.loads((ROOT / f"annotations/instances_{split}.json").read_text())
        imgs = {i["id"]: i for i in d["images"]}
        anns = collections.defaultdict(list)
        for a in d["annotations"]:
            if a["category_id"] in (CAT, DOG):
                anns[a["image_id"]].append(a)
        for iid, al in anns.items():
            im = imgs[iid]
            area = im["width"] * im["height"]
            if any(a["iscrowd"] for a in al):
                skipped["crowd"] += 1
                continue
            big = [a for a in al if a["area"] >= MIN_AREA * area]
            cats = [a for a in big if a["category_id"] == CAT]
            dogs = [a for a in big if a["category_id"] == DOG]
            if len(big) != len(al):
                skipped["a cat or dog below 2% of the image"] += 1
                continue
            if len(cats) == 1 and len(dogs) == 1:
                kind = "clean"
            elif (len(cats) >= 2 and not dogs) or (len(dogs) >= 2 and not cats):
                kind = "wrong_pair"
            elif len(cats) + len(dogs) == 1:
                kind = "single"
            else:
                skipped["other mix"] += 1
                continue
            box = square_box(im["width"], im["height"], [a["bbox"] for a in big])
            if box is None:
                skipped[f"{kind}: animals wider apart than the short side"] += 1
                continue
            chosen.append({"id": iid, "split": split, "kind": kind, "url": im["coco_url"], "crop": box,
                           "license": im["license"], "cats": len(cats), "dogs": len(dogs)})
    singles = [c for c in chosen if c["kind"] == "single"]
    random.Random(0).shuffle(singles)
    keep_single = {c["id"] for c in singles[:args.max_single]}
    chosen = [c for c in chosen if c["kind"] != "single" or c["id"] in keep_single]
    print("chosen", dict(collections.Counter(c["kind"] for c in chosen)), "skipped", dict(skipped), flush=True)

    out = ROOT / "catdog"
    for k in ("clean", "wrong_pair", "single"):
        (out / k).mkdir(parents=True, exist_ok=True)

    def fetch(c):
        dest = out / c["kind"] / f"{c['id']}.jpg"
        if dest.exists():
            return "had"
        try:
            with urllib.request.urlopen(c["url"], timeout=60) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
            im.crop(c["crop"]).save(dest, quality=95)
            return "got"
        except Exception as e:  # noqa: BLE001
            return f"failed: {type(e).__name__}"

    with ThreadPoolExecutor(args.workers) as ex:
        results = collections.Counter(ex.map(fetch, chosen))
    print("downloads", dict(results), flush=True)
    (out / "manifest.json").write_text(json.dumps({"rule": __doc__.split("\n\n")[1], "min_area": MIN_AREA,
                                                   "skipped": dict(skipped), "images": chosen}, indent=0))
    print("COCO CATDOG DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
