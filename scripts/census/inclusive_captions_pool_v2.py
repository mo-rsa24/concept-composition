#!/usr/bin/env python
"""Build the pilot's training pool from COCO photographs and SD 3.5 images together.

The first pool took COCO alone at a 0.25 floor on the animal's box as a fraction of the image's
longer side. Training on it pulled SDXL toward cluttered rooms with a small animal somewhere in
them: by step 250 the model drew a shed with a dog in the corner where it had drawn a dog. The
floor is the lever on that, and raising it costs the set the experiment rests on, because COCO
holds only 34 photos with both animals above 0.50.

So this build does two things at once. It raises the COCO floor, separately per set because the
sets can afford different ones. And it tops the both-animal set up with SD 3.5 images, which are
sharp and fill the frame, and which come from a different model family so the co-occurrence they
carry is not SDXL's own belief fed back to it.

Every candidate, COCO and generated alike, passes the vision-language judge before it is chosen:
the judge counts the animals, names them and rejects a hybrid. The judge's agreement with a
person on 48 hand-labelled images is in checks/, and it is not used until that agreement is
measured.

What a row carries that the first manifest did not: `source`, either "coco" or "sd35". It is the
same in both arms, so it cannot separate them, and it is there so a result can be read per source
afterwards.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      scripts/census/inclusive_captions_pool_v2.py --out-dir <where the manifest goes>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

COCO_ROOT = Path("/datasets/mmolefe/coco2017")
ANNOTATIONS = COCO_ROOT / "annotations_trainval2017/annotations/instances_train2017.json"
CAPTIONS = COCO_ROOT / "annotations_trainval2017/annotations/captions_train2017.json"
IMAGES = COCO_ROOT / "train2017/train2017"
GENERATED = Path("/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_generated")

CAT_ID, DOG_ID = 17, 18
SEED = 0

# How many of each set the pool wants, and where they may come from. The single-animal sets are
# half COCO and half generated so neither source alone decides what a cat condition looks like.
WANT = {"cat_only": (100, 100), "dog_only": (100, 100), "both": (45, 105)}

# Separate floors, because the sets can afford different ones. 0.50 leaves 2,140 cat-only and
# 1,432 dog-only photos in COCO, far more than the 100 each wanted. 0.50 leaves 34 both-animal
# photos, so that set sits at 0.40, where 64 survive.
FLOOR_SINGLE = 0.50
FLOOR_BOTH = 0.40


def box_side_fraction(box, width: int, height: int) -> float:
    return max(box[2], box[3]) / max(width, height)


def read_captions() -> dict[int, str]:
    data = json.loads(CAPTIONS.read_text())
    shortest: dict[int, str] = {}
    for row in data["annotations"]:
        text = row["caption"].strip()
        current = shortest.get(row["image_id"])
        if current is None or len(text) < len(current):
            shortest[row["image_id"]] = text
    return shortest


def coco_candidates() -> dict[str, list[dict]]:
    data = json.loads(ANNOTATIONS.read_text())
    images = {im["id"]: im for im in data["images"]}
    captions = read_captions()
    per_image: dict[int, dict[str, list]] = defaultdict(lambda: {"cat": [], "dog": []})
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        if a["category_id"] == CAT_ID:
            per_image[a["image_id"]]["cat"].append(a["bbox"])
        elif a["category_id"] == DOG_ID:
            per_image[a["image_id"]]["dog"].append(a["bbox"])

    pools: dict[str, list[dict]] = {"cat_only": [], "dog_only": [], "both": []}
    for image_id, boxes in per_image.items():
        im = images[image_id]
        w, h = im["width"], im["height"]
        cats, dogs = boxes["cat"], boxes["dog"]
        row = {"image_id": image_id, "file": str(IMAGES / im["file_name"]), "width": w,
               "height": h, "n_cat": len(cats), "n_dog": len(dogs),
               "caption": captions.get(image_id, ""), "source": "coco"}
        big_cats = [b for b in cats if box_side_fraction(b, w, h) >= FLOOR_SINGLE]
        big_dogs = [b for b in dogs if box_side_fraction(b, w, h) >= FLOOR_SINGLE]
        if cats and not dogs and big_cats:
            row["largest_cat_fraction"] = max(box_side_fraction(b, w, h) for b in cats)
            pools["cat_only"].append(row)
        elif dogs and not cats and big_dogs:
            row["largest_dog_fraction"] = max(box_side_fraction(b, w, h) for b in dogs)
            pools["dog_only"].append(row)
        elif cats and dogs:
            bc = [b for b in cats if box_side_fraction(b, w, h) >= FLOOR_BOTH]
            bd = [b for b in dogs if box_side_fraction(b, w, h) >= FLOOR_BOTH]
            if bc and bd:
                row["largest_cat_fraction"] = max(box_side_fraction(b, w, h) for b in bc)
                row["largest_dog_fraction"] = max(box_side_fraction(b, w, h) for b in bd)
                pools["both"].append(row)
    return pools


def generated_candidates(kind: str, judged: Path) -> list[dict]:
    """Every generated image of this kind the judge kept, in seed order."""
    if not judged.is_file():
        return []
    rows = json.loads(judged.read_text())["rows"]
    return [{"image_id": None, "file": r["file"], "source": "sd35", "caption": "",
             "n_cat": 1 if kind in ("cat_only", "both") else 0,
             "n_dog": 1 if kind in ("dog_only", "both") else 0}
            for r in rows if r["verdict"] == "keep"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--coco-judged-dir", default=None,
                    help="where the judge's verdicts on the COCO candidates live, one "
                         "judged_<set>.json per set; without it every COCO candidate is taken")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = coco_candidates()
    counts = {"coco_after_floor": {k: len(v) for k, v in pools.items()},
              "floor_single": FLOOR_SINGLE, "floor_both": FLOOR_BOTH}

    if args.coco_judged_dir:
        judged_dir = Path(args.coco_judged_dir)
        for name in pools:
            path = judged_dir / f"judged_coco_{name}.json"
            if not path.is_file():
                continue
            kept = {r["file"] for r in json.loads(path.read_text())["rows"]
                    if r["verdict"] == "keep"}
            pools[name] = [r for r in pools[name] if r["file"] in kept]
        counts["coco_after_judge"] = {k: len(v) for k, v in pools.items()}

    folder = {"cat_only": "cat", "dog_only": "dog", "both": "both"}
    rng = random.Random(SEED)
    manifest: dict[str, list[dict]] = {}
    for name, (want_coco, want_gen) in WANT.items():
        coco = sorted(pools[name], key=lambda r: r["image_id"])
        rng.shuffle(coco)
        gen = generated_candidates(name, GENERATED / folder[name] / "judged.json")
        short = max(0, want_coco - len(coco)) + max(0, want_gen - len(gen))
        # Whichever source is short, the other makes it up, so the set size is what it says.
        take_coco = min(want_coco + max(0, want_gen - len(gen)), len(coco))
        take_gen = min(want_gen + max(0, want_coco - len(coco)), len(gen))
        manifest[name] = coco[:take_coco] + gen[:take_gen]
        counts.setdefault("chosen", {})[name] = {
            "coco": take_coco, "sd35": take_gen, "total": take_coco + take_gen,
            "short_by": max(0, (want_coco + want_gen) - (take_coco + take_gen)),
            "candidates_short": short}

    missing = [r["file"] for rows in manifest.values() for r in rows
               if not Path(r["file"]).is_file()]
    counts["missing_file"] = len(missing)
    counts["total"] = sum(len(v) for v in manifest.values())

    (out_dir / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "floor_single": FLOOR_SINGLE, "floor_both": FLOOR_BOTH,
         "annotations": str(ANNOTATIONS), "sets": manifest}, indent=2))
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    if missing:
        print(f"{len(missing)} chosen files do not open, first 3: {missing[:3]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
