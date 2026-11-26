"""Find the bodies: the body counter on every picture of a list, written as bodies.json for the four crop judges.

Species judge step 0.13 (`paper2-find-bodies`, plans/paper-2-dials/plans/02-species-judge.md). The body counter is the
project's: GroundingDINO tiny asked for "animal", overlapping boxes removed, boxes under 0.25 of the picture's longer
side dropped (`count_instances` in poe_repair/experiments/compose_scorer_validation/detection_scorer.py, run as
committed). Writes, under --out:

  bodies.json   {"pictures": {picture: {"n": count, "boxes": [{"box": [x0, y0, x1, y1], "confidence": c}, ...]}}},
                keyed by each picture's path exactly as the list writes it, which is what paper2_common.load_bodies
                and every judge's --bodies read; plus the counter's settings and the measured seconds per picture
  crops/        each body's crop with its 10% margin and the other bodies painted out (paper2_common.crop_body, the
                judges' own crop), named <row>_<picture stem>_body<i>.png
  sheet.png     the first 24 pictures with their boxes drawn and numbered, to check the boxes sit on separate bodies

The list is a CSV with columns picture, a, b (paper2_common.read_list). --dry-run swaps the counter for the two fixed
stand-in boxes (paper2_common.STAND_IN_BOXES) so the files are checked on the laptop with no torch.

    python3 paper2_bodies.py --pictures list.csv --out <dir> --dry-run
    bash paper2_launch.sh paper2-find-bodies co3 6000 paper2_bodies.py --pictures list.csv \\
        --out /datasets/mmolefe/poe_repair_min/paper2/paper2-find-bodies
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import REPO, STAND_IN_BOXES, crop_body, read_list  # noqa: E402

QUERY = "animal"
SHEET_PICTURES = 24
SHEET_TILE = 256
SHEET_COLUMNS = 6
COLOURS = ["yellow", "cyan", "magenta", "lime", "orange", "red"]


def stand_in_counter(path: Path) -> tuple[int, list[dict]]:
    return len(STAND_IN_BOXES), [{"box": list(b), "confidence": 0.9 - 0.1 * i} for i, b in enumerate(STAND_IN_BOXES)]


def real_counter(device):
    sys.path.insert(0, str(REPO))
    from poe_repair.experiments.compose_scorer_validation import detection_scorer as ds

    def count(path: Path) -> tuple[int, list[dict]]:
        return ds.count_instances(path, query=QUERY, device=device)
    return count, {"query": QUERY, "min_box_fraction": ds.MIN_BOX_FRACTION,
                   "detector": "IDEA-Research/grounding-dino-tiny"}


def sheet(entries: list[tuple[str, list[list[int]]]], out: Path) -> None:
    rows = (len(entries) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    page = Image.new("RGB", (SHEET_COLUMNS * SHEET_TILE, max(1, rows) * SHEET_TILE), "white")
    for i, (pic, boxes) in enumerate(entries):
        im = Image.open(pic).convert("RGB")
        d = ImageDraw.Draw(im)
        for j, b in enumerate(boxes):
            colour = COLOURS[j % len(COLOURS)]
            d.rectangle(b, outline=colour, width=8)
            d.text((b[0] + 12, b[1] + 8), str(j), fill=colour)
        page.paste(im.resize((SHEET_TILE, SHEET_TILE)), ((i % SHEET_COLUMNS) * SHEET_TILE, (i // SHEET_COLUMNS) * SHEET_TILE))
    page.save(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pictures", type=Path, required=True, help="CSV with columns picture, a, b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-crops", action="store_true", help="skip saving the crops")
    ap.add_argument("--dry-run", action="store_true", help="stand-in boxes, no detector, no torch")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    rows = read_list(a.pictures)[: a.limit]
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        count, settings = stand_in_counter, {"query": QUERY, "detector": "stand-in boxes"}
    else:
        import torch
        count, settings = real_counter(torch.device(a.device))

    pictures, for_sheet = {}, []
    t0 = time.time()
    for k, r in enumerate(rows):
        pic = r["picture"]
        n, found = count(Path(pic))
        boxes = [[int(round(v)) for v in d["box"]] for d in found]
        pictures[pic] = {"n": n, "boxes": [{"box": b, "confidence": round(float(d["confidence"]), 4)}
                                           for b, d in zip(boxes, found)]}
        if not a.no_crops:
            im = Image.open(pic).convert("RGB")
            for i in range(len(boxes)):
                crop, _ = crop_body(im, boxes, i)
                crop.save(out / "crops" / f"{k:04d}_{Path(pic).stem}_body{i}.png")
        if len(for_sheet) < SHEET_PICTURES:
            for_sheet.append((pic, boxes))
        if (k + 1) % 50 == 0 or k + 1 == len(rows):
            rate = (time.time() - t0) / (k + 1)
            print(f"{k + 1}/{len(rows)} pictures, {rate:.2f} seconds a picture", flush=True)
    seconds = time.time() - t0
    per_picture = seconds / max(1, len(rows))
    counts = [p["n"] for p in pictures.values()]
    result = {"settings": settings, "dry_run": a.dry_run, "list": str(a.pictures), "seconds": round(seconds, 1),
              "seconds_per_picture": round(per_picture, 3),
              "count_histogram": {str(c): counts.count(c) for c in sorted(set(counts))}, "pictures": pictures}
    (out / "bodies.json").write_text(json.dumps(result, indent=1))
    sheet(for_sheet, out / "sheet.png")
    print(f"wrote {out / 'bodies.json'}: {len(pictures)} pictures, bodies per picture {result['count_histogram']}; "
          f"{per_picture:.2f} seconds a picture, so 1,000 pictures take {per_picture * 1000 / 3600:.2f} GPU-hours; "
          f"sheet of {len(for_sheet)} at {out / 'sheet.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
