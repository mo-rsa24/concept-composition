#!/usr/bin/env python3
"""Paper 1's blind eye-call sheets: build them, check nothing on them names a row, and settle two reviewers' calls with a third's.

Settings: plans/paper-1-clevr-touching/plans/03-full-plan.md, "Words this uses" (the occlusion rubric in its two versions) and steps 62
and 64; the first result's tasks 11, 14 and 16 (plans/paper-1-clevr-touching/plans/02-first-result.md, "Seeds and scenes"); the lower-
on-top paste row from artifacts/notes/guarding-the-first-results/paper-1.md, guardrail 5b. Needs only numpy and Pillow, so it runs on
the laptop as well as the cluster.

Subcommands:
  select  the first result's sheet list (items.json): every row at the touching evaluation seeds, chosen rows at the apart
          evaluation seeds, and the real touching scenes, by the rules of "Seeds and scenes"; with --set tuning --touch-total 32,
          the tuning sheet's 32 touching tuning seeds (11, 11 and 10)
  build   from items.json (row, seed, count, path): each picture re-saved from its pixels at 4 times size by nearest neighbour under a
          shuffled neutral name of letters only, drawn from --shuffle-seed; key.json written outside the sheet folder; one call template
          per reviewer; index.html with the rubric version at the top and the pictures in file order
  check   search every file name, every file's text and every PNG text chunk in the sheet folder for the key's row names, seeds and
          source paths; exit 1 on any hit
  settle  field by field, the two reviewers' call where they agree and the third reviewer's where they do not; settled.csv and
          settle.json; exit 1 if any field is left with neither agreement nor a third call

Usage (folders relative to /datasets/mmolefe/poe_repair_min/paper1/):
  python paper1_sheets.py select --run paper1-first-eye-sheet --split paper1-first-lone-red/split.json \
      --rows plainsum=paper1-first-row-plainsum,paste=paper1-first-row-paste,region=paper1-first-row-region,ours=paper1-first-row-ours-k64,randompush=paper1-first-row-randompush,paste-lower-on-top=paper1-first-row-paste-lower \
      --apart-rows ours,paste --real paper1-first-pack/heldout
  python paper1_sheets.py build --run paper1-first-eye-sheet --items paper1-first-eye-sheet/items.json --rubric without-shadows --shuffle-seed 1
  python paper1_sheets.py check --run paper1-first-eye-sheet
  python paper1_sheets.py settle --run paper1-first-eye-sheet --calls calls-a.csv calls-b.csv --third calls-c.csv
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PAPER1_ROOT = Path("/datasets/mmolefe/poe_repair_min/paper1")
SCALE = 4                                    # pictures shown at 4 times size, nearest neighbour
FIELDS = ("red_count", "blue_count", "blended_band", "occlusion_correct")
NAME_LETTERS = "abcdefghjkmnpqrstuvwxyz"     # neutral names carry letters only, so no seed or index can hide in one
NAME_LENGTH = 8
SHEET_DIR = "sheet"                          # the folder reviewers open; key.json sits beside it, never inside
TOUCH_PER_COUNT = 32                         # the first result's 96 touching seeds: the first 32 touching evaluation seeds per count
APART_TOTAL = 96                             # up to 32 apart seeds per count, the shortfall from 2 objects (count 1)
REAL_PER_COUNT = 32                          # the first 32 touching held-out scenes per count, as many as there are
RUBRIC_VERSION = "v1"
RUBRIC = {
    "without-shadows": (
        "Occlusion correct, without shadows. Where a red and a blue object meet: (a) one sits cleanly in front of the other, with no "
        "purple or blended pixels and neither showing through the other; (b) the object in front is the one whose base sits lower in "
        "the frame, nearer the camera; (c) the two still touch. All three must hold."),
    "with-shadows": (
        "Occlusion correct, with shadows. Where a red and a blue object meet: (a) one sits cleanly in front of the other, with no "
        "purple or blended pixels and neither showing through the other; (b) the object in front is the one whose base sits lower in "
        "the frame, nearer the camera; (c) the two still touch; (d) both objects' shadows fall to the same side. All four must hold."),
}
FIELD_HELP = ("red count and blue count: how many objects of each colour you see (a red and a blue object that touch count as one "
              "each). blended band: yes if a purple or blended strip shows where red meets blue, else no. occlusion correct: yes or "
              "no by the rubric above; leave it empty when no red object meets a blue one.")


# ---------------------------------------------------------------------------------------------------------------------------------------
# select


def resolve(root, path):
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def cmd_select(args):
    """items.json for the first result's sheet, by the rules of 02-first-result.md, "Seeds and scenes"."""
    folder = resolve(args.root, args.run)
    folder.mkdir(parents=True, exist_ok=True)
    split = json.loads(resolve(args.root, args.split).read_text())[args.set]
    rows = dict(item.split("=", 1) for item in args.rows.split(","))
    counts = [1, 2, 3]
    if args.touch_total:                              # the tuning sheet: exactly N seeds, split as evenly as counts allow, 6 objects last
        per = {k: args.touch_total // 3 + (1 if i < args.touch_total % 3 else 0) for i, k in enumerate(counts)}
    else:
        per = {k: args.touch_per_count for k in counts}
    touching = {k: sorted(split[f"count-{k}"]["touching"])[:per[k]] for k in counts}
    apart = {k: sorted(split[f"count-{k}"]["apart"]) for k in counts}
    chosen_apart, short = {}, 0
    for k in (3, 2):                                  # 6 and 4 objects first; whatever they lack comes from 2 objects
        chosen_apart[k] = apart[k][:args.apart_total // 3]
        short += args.apart_total // 3 - len(chosen_apart[k])
    chosen_apart[1] = apart[1][:args.apart_total - len(chosen_apart[2]) - len(chosen_apart[3])]
    items = []
    for row, src in rows.items():
        for k in counts:
            for seed in touching[k]:
                items.append({"row": row, "seed": seed, "count": k, "side": "touching",
                              "path": str(resolve(args.root, src) / f"count-{k}" / f"{seed:06d}.png")})
    for row in args.apart_rows.split(",") if args.apart_rows else []:
        for k in counts:
            for seed in chosen_apart[k]:
                items.append({"row": row, "seed": seed, "count": k, "side": "apart",
                              "path": str(resolve(args.root, rows[row]) / f"count-{k}" / f"{seed:06d}.png")})
    if args.real:
        real = resolve(args.root, args.real)
        flags = json.loads((real / "touching.json").read_text())["files"]
        for k in counts:
            names = sorted(n for n, t in flags.items() if t and n.startswith(f"count-{k}/"))[:args.real_per_count]
            for n in names:
                items.append({"row": "real", "seed": int(Path(n).stem), "count": k, "side": "touching", "path": str(real / n)})
    missing = [it["path"] for it in items if not Path(it["path"]).is_file()]
    summary = {"touching_per_count": {k: len(v) for k, v in touching.items()},
               "apart_per_count": {k: len(chosen_apart[k]) for k in counts}, "apart_short_of_32_at_4_and_6_objects": short,
               "real_per_count": {k: sum(1 for it in items if it["row"] == "real" and it["count"] == k) for k in counts},
               "items": len(items), "missing_files": len(missing)}
    (folder / "items.json").write_text(json.dumps({"set": args.set, "summary": summary, "items": items}, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    if missing:
        print(f"missing, first: {missing[0]}", flush=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------------------------------
# build


def neutral_names(n, seed):
    """n distinct letter-only names, drawn from a generator seeded by seed alone (never by a path or a row)."""
    rng, names = random.Random(seed), set()
    while len(names) < n:
        names.add("".join(rng.choice(NAME_LETTERS) for _ in range(NAME_LENGTH)))
    return sorted(names)


def cmd_build(args):
    folder = resolve(args.root, args.run)
    sheet = folder / SHEET_DIR
    key_path = folder / "key.json"
    if sheet.exists() and any(sheet.iterdir()):
        raise SystemExit(f"{sheet} is not empty; build into a fresh run")
    sheet.mkdir(parents=True, exist_ok=True)
    data = json.loads(resolve(args.root, args.items).read_text())
    items = data["items"] if isinstance(data, dict) else data
    names = neutral_names(len(items), args.shuffle_seed)      # sorted, so file order is the names' order
    order = list(range(len(items)))
    random.Random(args.shuffle_seed + 1).shuffle(order)       # which item gets which name
    key = []
    for name, i in zip(names, order):
        item = items[i]
        pixels = np.asarray(Image.open(item["path"]).convert("RGB"))
        big = pixels.repeat(SCALE, 0).repeat(SCALE, 1)
        Image.fromarray(big, "RGB").save(sheet / f"{name}.png")               # from pixels: no text chunk travels with it
        key.append({"picture": f"{name}.png", **item})
    key_path.write_text(json.dumps({"rubric": f"{RUBRIC_VERSION} {args.rubric}", "shuffle_seed": args.shuffle_seed, "key": key}, indent=1) + "\n")
    for reviewer in args.reviewers.split(","):
        with open(sheet / f"calls-{reviewer}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("picture",) + FIELDS)
            for name in names:
                writer.writerow((f"{name}.png",) + ("",) * len(FIELDS))
    rubric = f"Rubric {RUBRIC_VERSION}, {args.rubric}"
    tiles = "\n".join(f'<figure><img src="{n}.png" width="{64 * SCALE}"><figcaption>{n}.png</figcaption></figure>' for n in names)
    (sheet / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Eye-call sheet</title>"
        "<style>body{font-family:sans-serif;margin:16px}figure{display:inline-block;margin:6px}"
        "figcaption{font-family:monospace;font-size:12px}</style>\n"
        f"<h1>{rubric}</h1>\n<p>{RUBRIC[args.rubric]}</p>\n<p>{FIELD_HELP}</p>\n"
        f"<p>{len(names)} pictures, in file order. Record each in your calls file.</p>\n{tiles}\n")
    print(json.dumps({"pictures": len(names), "sheet": str(sheet), "key": str(key_path), "rubric": rubric}), flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
# check


def png_text(path):
    """Every text-like chunk of a PNG (tEXt, zTXt, iTXt and the rest of info), as one string."""
    with Image.open(path) as image:
        parts = [f"{k}={v}" for k, v in image.info.items()]
        parts += [f"{k}={v}" for k, v in getattr(image, "text", {}).items()]
    return " ".join(parts)


def leak_patterns(key):
    """What must not appear on the sheet: row names (as whole words), source paths and file names, and seeds or indices written
    as 6-digit stems; seeds of 4 digits or more also as plain numbers."""
    pats = {}
    for item in key:
        pats[f"row {item['row']}"] = re.compile(rf"(?<![A-Za-z0-9]){re.escape(item['row'])}(?![A-Za-z0-9])", re.I)
        pats[f"path {item['path']}"] = re.compile(re.escape(item["path"]))
        pats[f"file {Path(item['path']).name}"] = re.compile(re.escape(Path(item["path"]).name))
        pats[f"stem {item['seed']:06d}"] = re.compile(rf"(?<!\d){item['seed']:06d}(?!\d)")
        if item["seed"] >= 1000:
            pats[f"seed {item['seed']}"] = re.compile(rf"(?<!\d){item['seed']}(?!\d)")
    return pats


def cmd_check(args):
    folder = resolve(args.root, args.run)
    sheet = folder / SHEET_DIR
    key = json.loads((folder / "key.json").read_text())["key"]
    hits = []
    if any(p.name == "key.json" for p in sheet.rglob("*")):
        hits.append("key.json is inside the sheet folder")
    pats = leak_patterns(key)
    for path in sorted(sheet.rglob("*")):
        rel = str(path.relative_to(folder))
        texts = [("name", rel)]
        if path.is_file():
            if path.suffix.lower() == ".png":
                texts.append(("png text", png_text(path)))
            else:
                texts.append(("content", path.read_text(errors="replace")))
        for where, text in texts:
            for label, pat in pats.items():
                if pat.search(text):
                    hits.append(f"{rel}: {where} carries {label}")
    report = {"files": sum(1 for p in sheet.rglob("*") if p.is_file()), "patterns": len(pats), "hits": hits[:50], "hit_count": len(hits)}
    (folder / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("files", "patterns", "hit_count")}), flush=True)
    for h in hits[:20]:
        print("HIT", h, flush=True)
    raise SystemExit(1 if hits else 0)


# ---------------------------------------------------------------------------------------------------------------------------------------
# settle


def normalise(value):
    v = (value or "").strip().lower()
    return {"y": "yes", "n": "no", "true": "yes", "false": "no"}.get(v, v)


def read_calls(path):
    with open(path, newline="") as f:
        return {row["picture"]: {k: normalise(row.get(k)) for k in FIELDS} for row in csv.DictReader(f)}


def settle(first, second, third):
    """Field by field: where the two reviewers agree, their call; where they disagree, the third reviewer's call; where the third is
    empty too, unresolved. An empty field both reviewers left empty stays empty (occlusion on an apart picture)."""
    settled, decided_by, unresolved = {}, {"agreed": 0, "third": 0}, []
    for pic in sorted(set(first) | set(second)):
        a, b, c = first.get(pic, {}), second.get(pic, {}), third.get(pic, {})
        row = {}
        for field in FIELDS:
            va, vb, vc = a.get(field, ""), b.get(field, ""), c.get(field, "")
            if va == vb:
                row[field] = va
                decided_by["agreed"] += 1
            elif vc:
                row[field] = vc
                decided_by["third"] += 1
            else:
                row[field] = ""
                unresolved.append({"picture": pic, "field": field, "first": va, "second": vb})
        settled[pic] = row
    return settled, decided_by, unresolved


def cmd_settle(args):
    folder = resolve(args.root, args.run)
    first, second = (read_calls(resolve(folder, c)) for c in args.calls)
    third = read_calls(resolve(folder, args.third)) if args.third else {}
    settled, decided_by, unresolved = settle(first, second, third)
    with open(folder / "settled.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("picture",) + FIELDS)
        for pic, row in settled.items():
            writer.writerow((pic,) + tuple(row[k] for k in FIELDS))
    out = {"pictures": len(settled), "fields_decided": decided_by, "unresolved": unresolved}
    (folder / "settle.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pictures": len(settled), **decided_by, "unresolved": len(unresolved)}), flush=True)
    raise SystemExit(1 if unresolved else 0)


# ---------------------------------------------------------------------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--run", required=True)
        p.add_argument("--root", default=str(PAPER1_ROOT))
        return p

    p = common(sub.add_parser("select"))
    p.add_argument("--split", required=True, help="split.json from paper1_sample.py split")
    p.add_argument("--set", default="evaluation", help="the seed set to take seeds from (evaluation; tuning for the tuning sheet)")
    p.add_argument("--rows", required=True, help="row=folder pairs, comma-separated; each folder holds count-<k>/<seed>.png")
    p.add_argument("--apart-rows", default="", help="rows also drawn at the apart seeds, comma-separated")
    p.add_argument("--real", help="the held-out joint scenes' folder (paper1_pack.py), for the real-scene row")
    p.add_argument("--touch-per-count", type=int, default=TOUCH_PER_COUNT)
    p.add_argument("--touch-total", type=int, help="exactly this many touching seeds over the three counts (32 for the tuning sheet: "
                                                     "11, 11 and 10, the 10 at 6 objects), each count's first in seed order")
    p.add_argument("--apart-total", type=int, default=APART_TOTAL)
    p.add_argument("--real-per-count", type=int, default=REAL_PER_COUNT)

    p = common(sub.add_parser("build"))
    p.add_argument("--items", required=True, help="items.json from select, or a list of {row, seed, count, path}")
    p.add_argument("--rubric", required=True, choices=sorted(RUBRIC), help="fixed by the owner's shadow decision (instruction 6)")
    p.add_argument("--shuffle-seed", type=int, required=True)
    p.add_argument("--reviewers", default="a,b,c", help="one call template per reviewer")

    common(sub.add_parser("check"))

    p = common(sub.add_parser("settle"))
    p.add_argument("--calls", nargs=2, required=True, help="the two reviewers' filled call files")
    p.add_argument("--third", help="the third reviewer's calls on the fields the two disagree on")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"select": cmd_select, "build": cmd_build, "check": cmd_check, "settle": cmd_settle}[args.stage](args)


if __name__ == "__main__":
    main()
