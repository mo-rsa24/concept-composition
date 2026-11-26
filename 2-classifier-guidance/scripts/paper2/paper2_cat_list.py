"""The 64 single cats for the species judge's crop check (step 0.15), as a picture list asked "a cat" against "a dog".

The cat census rendered "a photo of a cat" from base SDXL on seeds 0 to 63 (1024 px, 50 DDIM steps, guidance 7.5, on
mscluster106) with 3-understanding-compositionality/scripts/census/cat_condition_dog_rate.py; its committed per-seed
record names every render's path:

    3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/
        does-the-cat-prompt-already-draw-dogs/per_seed.json

This reads that record, never retypes a path, and writes the list (columns picture, a, b, seed) with a = cat, b = dog
(paper2_cats_64.csv beside this file). By eye all 64 hold one cat and no dog, and the census's crop classifier called
64 of 64 cat; CROP_CHECK_MIN (62 of 64) is that classifier's bar.

--check, run on the held node, confirms every listed picture exists and opens at 1024 by 1024, and exits 1 if not.

    python3 paper2_cat_list.py                     # writes paper2_cats_64.csv from the committed record
    "$PY" paper2_cat_list.py --check               # on the node
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECORD = (REPO / "3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/"
          "does-the-cat-prompt-already-draw-dogs/per_seed.json")
LIST = HERE / "paper2_cats_64.csv"
EXPECTED = 64


def rows_from_record(record: Path = RECORD) -> list[dict]:
    data = json.loads(record.read_text())
    rows = data if isinstance(data, list) else data["rows"]
    out = [{"picture": r["path"], "a": "cat", "b": "dog", "seed": r["seed"]} for r in sorted(rows, key=lambda r: r["seed"])]
    if len(out) != EXPECTED or [r["seed"] for r in out] != list(range(EXPECTED)):
        raise SystemExit(f"{record} does not hold seeds 0 to {EXPECTED - 1} exactly")
    return out


def write(out: Path = LIST) -> list[dict]:
    rows = rows_from_record()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["picture", "a", "b", "seed"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} cats, seeds 0 to {EXPECTED - 1}, from {RECORD.relative_to(REPO)}")
    return rows


def check(path: Path = LIST) -> int:
    from PIL import Image
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing, wrong = [], []
    for r in rows:
        p = Path(r["picture"])
        if not p.exists():
            missing.append(str(p))
            continue
        with Image.open(p) as im:
            if im.size != (1024, 1024):
                wrong.append(f"{p} is {im.size}")
    print(f"{len(rows)} listed, {len(rows) - len(missing)} found, {len(rows) - len(missing) - len(wrong)} at 1024 by 1024")
    for line in missing[:5] + wrong[:5]:
        print("  " + line)
    return 0 if len(rows) == EXPECTED and not missing and not wrong else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="on the node: every picture exists at 1024 by 1024")
    ap.add_argument("--list", type=Path, default=LIST)
    a = ap.parse_args(argv)
    if a.check:
        return check(a.list)
    write(a.list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
