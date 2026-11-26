"""The blind-sheet builder: one picture per page at full size, under a neutral name, with only the two asked species.

Used by the species judge's step 0.5 (`paper2-label-sheets`) and the first result's step 6. From a
picture list (CSV with columns picture, a, b, and any others, such as row, pair or seed, which go to
the key only), it shuffles by a seed, gives each picture a neutral name, and writes two folders:

  <out>/sheet/   what a reviewer is given and nothing else:
                 pages/<name>.png   the picture at 1024 by 1024 under a white band that prints only
                                    "Asked: a cheetah and a cougar" and the page's own name
                 images/<name>.png  the picture alone, re-saved from its pixels
                 sheet.csv          name, asked species, in the shuffled order
                 calls-template.csv name, kind, note: the reviewer fills kind and note
                 rules.txt          the six kinds and the seven rules, from paper2_kinds.py
  <out>/key/     key.json: each neutral name to its source path, species and the list's other columns

Pictures are re-saved from their pixels, never copied byte for byte, so no text chunk a renderer
wrote into a PNG (a prompt, a seed) reaches the sheet. A neutral name is ten hex characters drawn
from the shuffle seed, not derived from the path. A picture that is not 1024 by 1024 stops the build.

`check` searches every file under sheet/ (a text file's every byte, a PNG's every chunk but its
compressed pixels), and every file name, for what would unblind
it: each source folder's name, each original file name, and every value of the list's other columns
(row, pair, seed and the like). It prints each hit and exits 1 if there is any.

    python3 paper2_sheets.py build --pictures list.csv --seed 0 --out <dir>
    python3 paper2_sheets.py check --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import read_list, with_article  # noqa: E402
import paper2_kinds  # noqa: E402

SIZE = 1024
BAND = 80
NAME_HEX = 10
CALL_CHOICES = paper2_kinds.ALL_KINDS


def asked_line(a: str, b: str) -> str:
    return f"Asked: {with_article(a)} and {with_article(b)}"


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1
        return ImageFont.load_default()


def neutral_names(n: int, seed: int) -> list[str]:
    rng = random.Random(f"names-{seed}")
    names: list[str] = []
    while len(names) < n:
        name = f"{rng.getrandbits(4 * NAME_HEX):0{NAME_HEX}x}"
        if name not in names:
            names.append(name)
    return names


def rules_text() -> str:
    doc = paper2_kinds.__doc__
    start = doc.index("Each body is called")
    end = doc.index("Two words the plan leaves open")
    kinds = ", ".join(k.replace("_", " ") for k in paper2_kinds.ALL_KINDS)
    return ("Call each page into one kind: " + kinds + ".\n"
            "Write the kind and a note of a few words per page into calls-template.csv.\n\n"
            + doc[start:end].strip() + "\n")


def build(a: argparse.Namespace) -> dict:
    rows = read_list(a.pictures)
    sheet, keydir = Path(a.out) / "sheet", Path(a.out) / "key"
    if sheet.exists() and any(sheet.iterdir()):
        raise SystemExit(f"{sheet} is not empty; build each batch into a new folder")
    bad = []
    for r in rows:
        with Image.open(r["picture"]) as im:
            if im.size != (SIZE, SIZE):
                bad.append(f"{r['picture']} is {im.size[0]} by {im.size[1]}")
    if bad:
        raise SystemExit("not 1024 by 1024:\n  " + "\n  ".join(bad))

    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    names = neutral_names(len(rows), a.seed)
    (sheet / "pages").mkdir(parents=True)
    (sheet / "images").mkdir()
    keydir.mkdir(parents=True, exist_ok=True)
    big, small = font(40), font(22)

    key, listed = {}, []
    for name, i in zip(names, order):
        r = rows[i]
        with Image.open(r["picture"]) as im:
            pic = Image.new("RGB", (SIZE, SIZE))
            pic.paste(im.convert("RGB"))       # pixels only: no PNG text chunk travels with them
        pic.save(sheet / "images" / f"{name}.png")
        page = Image.new("RGB", (SIZE, SIZE + BAND), "white")
        page.paste(pic, (0, BAND))
        d = ImageDraw.Draw(page)
        d.text((20, 14), asked_line(r["a"], r["b"]), fill="black", font=big)
        d.text((SIZE - 20, BAND - 30), name, fill=(120, 120, 120), font=small, anchor="ra")
        page.save(sheet / "pages" / f"{name}.png")
        key[name] = {"source": str(Path(r["picture"]).resolve()), **{k: v for k, v in r.items() if k != "picture"}}
        listed.append((name, asked_line(r["a"], r["b"])[len("Asked: "):]))

    with open(sheet / "sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "asked"])
        w.writerows(listed)
    with open(sheet / "calls-template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "kind", "note"])
        w.writerows([(n, "", "") for n, _ in listed])
    (sheet / "rules.txt").write_text(rules_text())
    (keydir / "key.json").write_text(json.dumps({"seed": a.seed, "list": str(Path(a.pictures).resolve()),
                                                 "pictures": key}, indent=1))
    summary = {"pages": len(names), "sheet": str(sheet), "key": str(keydir / "key.json"), "seed": a.seed}
    print(json.dumps(summary))
    return summary


def unblinding_strings(key: dict) -> set[str]:
    """Everything that would tell a reviewer where a picture came from."""
    found: set[str] = set()
    for entry in key["pictures"].values():
        src = Path(entry["source"])
        found.add(src.name)
        found.add(src.stem)
        found.update(p for p in src.parent.parts[-3:] if p not in ("/", ""))
        for k, v in entry.items():
            if k not in ("source", "a", "b") and str(v).strip():
                found.add(str(v).strip())
    return {s for s in found if len(s) >= 3}


def searchable_bytes(path: Path) -> bytes:
    """A text file's bytes; for a PNG, every chunk except the compressed pixels (IDAT).

    Compressed pixel data is noise to a byte search (a three-letter string turns up in it by
    chance), and it cannot carry readable text; a PNG's text and metadata chunks can.
    """
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data
    out, i = [], 8
    while i + 8 <= len(data):
        length = int.from_bytes(data[i:i + 4], "big")
        kind = data[i + 4:i + 8]
        if kind != b"IDAT":
            out.append(kind + data[i + 8:i + 8 + length])
        i += 12 + length
    return b"".join(out)


def check(a: argparse.Namespace) -> int:
    sheet, keyfile = Path(a.out) / "sheet", Path(a.out) / "key" / "key.json"
    key = json.loads(keyfile.read_text())
    needles = sorted(unblinding_strings(key))
    hits = []
    files = sorted(p for p in sheet.rglob("*") if p.is_file())
    for p in files:
        data = searchable_bytes(p)
        rel = str(p.relative_to(sheet))
        for s in needles:
            if s in rel or s.encode() in data:
                hits.append(f"{rel}: contains {s!r}")
    pages = sorted((sheet / "pages").glob("*.png"))
    sizes = set()
    for p in pages:
        with Image.open(p) as im:
            sizes.add(im.size)
    pics = sorted((sheet / "images").glob("*.png"))
    pic_sizes = set()
    for p in pics:
        with Image.open(p) as im:
            pic_sizes.add(im.size)
    print(f"searched {len(files)} files under {sheet} for {len(needles)} strings "
          f"(source folders, file names, and the list's other columns): {len(hits)} found")
    for h in hits:
        print("  " + h)
    print(f"pages: {len(pages)} at {sorted(sizes)} (the picture at {SIZE} by {SIZE} under a {BAND}-pixel band); "
          f"pictures: {len(pics)} at {sorted(pic_sizes)}")
    ok = not hits and pic_sizes == {(SIZE, SIZE)} and sizes == {(SIZE, SIZE + BAND)}
    print("check passed" if ok else "check FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pictures", type=Path, required=True, help="CSV with columns picture, a, b, and any others")
    b.add_argument("--seed", type=int, required=True, help="the shuffle and the neutral names")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a)
        return 0
    return check(a)


if __name__ == "__main__":
    raise SystemExit(main())
