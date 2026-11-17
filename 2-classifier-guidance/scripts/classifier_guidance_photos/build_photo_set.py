"""Build the cat / dog / both image set for plan 20 at 256 and 64 px.

Reads the inclusive-captions v2 pool manifest (read only) and writes square crops to
<out>/images/<size>/<label>_<nnn>_<source>.png, where label is cat, dog or both, so the
label is the filename's first part, as guided-diffusion's loader expects.

A COCO photo is cropped to the square (side = its shorter edge) that best covers the union
of its cat and dog boxes, sliding along the long edge; a square render is used whole.
Writes counts.json (images per label and source, how many crops lost part of an animal box)
and index.csv (one row per image, with where it came from).
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from PIL import Image

POOL = "/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool_v2/manifest.json"
LABELS = {"cat_only": "cat", "dog_only": "dog", "both": "both"}
CAT, DOG = 17, 18  # COCO category ids


def boxes_by_image(ann_path, image_ids):
    data = json.load(open(ann_path))
    out = defaultdict(list)
    for a in data["annotations"]:
        if a["image_id"] in image_ids and a["category_id"] in (CAT, DOG):
            out[a["image_id"]].append(a["bbox"])  # x, y, w, h
    return out


def square_window(w, h, boxes):
    """Return (left, top, side) of the square crop, and the fraction of box area kept."""
    side = min(w, h)
    if not boxes:
        return (w - side) // 2, (h - side) // 2, side, 1.0
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    if w >= h:  # slide horizontally
        centre = (x0 + x1) / 2
        left = int(round(min(max(centre - side / 2, 0), w - side))); top = 0
    else:
        centre = (y0 + y1) / 2
        top = int(round(min(max(centre - side / 2, 0), h - side))); left = 0
    total = kept = 0.0
    for bx, by, bw, bh in boxes:
        total += bw * bh
        ix = max(0, min(bx + bw, left + side) - max(bx, left))
        iy = max(0, min(by + bh, top + side) - max(by, top))
        kept += ix * iy
    return left, top, side, (kept / total if total else 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos")
    p.add_argument("--sizes", default="256,64")
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    man = json.load(open(POOL))
    coco_ids = {r["image_id"] for rows in man["sets"].values() for r in rows if r.get("source") == "coco"}
    boxes = boxes_by_image(man["annotations"], coco_ids)

    for s in sizes:
        os.makedirs(f"{args.out}/images/{s}", exist_ok=True)
    counts = Counter(); by_source = Counter(); clipped = Counter(); rows_out = []
    for set_name, label in LABELS.items():
        for i, r in enumerate(man["sets"][set_name]):
            src = r.get("source", "coco")
            img = Image.open(r["file"]).convert("RGB")
            w, h = img.size
            left, top, side, kept = square_window(w, h, boxes.get(r["image_id"], []) if src == "coco" else [])
            crop = img.crop((left, top, left + side, top + side))
            name = f"{label}_{i:03d}_{src}.png"
            for s in sizes:
                crop.resize((s, s), Image.BICUBIC).save(f"{args.out}/images/{s}/{name}")
            counts[label] += 1; by_source[f"{label}/{src}"] += 1
            if kept < 0.9:
                clipped[label] += 1
            rows_out.append({"name": name, "label": label, "source": src, "file": r["file"],
                             "image_id": r.get("image_id"), "width": w, "height": h,
                             "crop_left": left, "crop_top": top, "crop_side": side,
                             "box_area_kept": round(kept, 3)})
    summary = {"pool": POOL, "sizes": sizes, "per_label": dict(counts), "per_label_source": dict(by_source),
               "crops_keeping_under_90pct_of_animal_box_area": dict(clipped), "total": sum(counts.values())}
    json.dump(summary, open(f"{args.out}/images/counts.json", "w"), indent=1)
    with open(f"{args.out}/images/index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0])); wr.writeheader(); wr.writerows(rows_out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
