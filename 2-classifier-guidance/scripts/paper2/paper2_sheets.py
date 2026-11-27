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
                 rules.txt          the six kinds and the seven rules (the rubric, version 4 by default:
                                    see RULES_V1 and rules_text)
  <out>/key/     key.json: each neutral name to its picture ("picture_path", the list's picture value as given, a
                 field no list column may take), its species and the list's other columns

Pictures are re-saved from their pixels, never copied byte for byte, so no text chunk a renderer
wrote into a PNG (a prompt, a seed) reaches the sheet. A neutral name is ten consonants (no vowels, no
digits) drawn from the shuffle seed, not derived from the path: with no digit it cannot hold a seed,
and with no vowel it cannot spell a row or source word such as own, box or old. A picture that is not
1024 by 1024 stops the build.

`check` searches every file under sheet/ (a text file's every byte, a PNG's every chunk but its
compressed pixels), and every file name, for what would unblind it: each picture's source folders
(the last three) and file name, and the values of the identifying columns (IDENTIFYING_COLUMNS: pair,
seed, source, round, row, made, also_called_in). It never searches the call columns (earlier_call,
earlier_scale, notes) or the yes-no columns, whose words ("one", "hybrid", "split") are ordinary. A value
shorter than 3 characters is not searched; a seed of 4 digits or more always is. A value of 4 letters or
fewer counts only as a whole word, so the row "both" does not fire on "both at once"; every longer
value, and every value with a digit or an underscore, counts anywhere. rules.txt is the same for every
sheet and holds those ordinary words, so it is not searched: it must equal the seven rules exactly. The
check prints each hit and exits 1 if there is any.

    python3 paper2_sheets.py build --pictures list.csv --seed 0 --out <dir>
    python3 paper2_sheets.py check --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import read_list, with_article  # noqa: E402
import paper2_kinds  # noqa: E402

SIZE = 1024
BAND = 80
NAME_LETTERS = 10
NAME_ALPHABET = "bcdfghjkmnpqrstvwxz"   # consonants only: no digit, no vowel
IDENTIFYING_COLUMNS = ("pair", "seed", "source", "round", "row", "made", "also_called_in")
SHORT_WORD = 4          # a pure-letter value this short or shorter counts only as a whole word
# key.json fields the builder writes itself; a list column with one of these names stops the build rather than
# overwrite it (the collector's list carries its own "source" column, which once overwrote the picture's path).
RESERVED_FIELDS = ("picture_path",)
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
        name = "".join(rng.choice(NAME_ALPHABET) for _ in range(NAME_LETTERS))
        if name not in names:
            names.append(name)
    return names


# The reviewers' rubric, by version. Version 1 is frozen as the text batch-01's sheet was built with (checksum pinned in
# tests/test_paper2_sheets.py). Version 2 tightens only what counts as a borrowed-part flag, after reviewers 1 and 2
# called batch-01's borrowed-part pictures alike 26 times in 34 (0.76, under REVIEWER_AGREE_KIND_MIN 0.80); it is made
# from version 1 by two exact replacements and one added line, so every other word is version 1's.
RULES_V1 = (
    "Call each page into one kind: two clear, borrowed part, hybrid, two of one kind, wrong species, extra subject, other.\n"
    "Write the kind and a note of a few words per page into calls-template.csv.\n\n"
    "Each body is called A (the first asked species), B (the second), MIXED (both at once) or OTHER\n"
    "(neither), and separately flagged when it carries a visible part of the other asked species.\n\n"
    "    1. 3 or more bodies, with at least one plain A and one plain B    -> extra subject\n"
    "    2. any body called other                                          -> wrong species\n"
    "    3. exactly 2 bodies, one A and one B, either flagged              -> borrowed part\n"
    "    4. exactly 2 bodies, one A and one B, neither flagged             -> two clear\n"
    "    5. some body mixed, and no clear body of one asked species        -> hybrid\n"
    "    6. 2 or more bodies, all A, or all B                              -> two of one kind\n"
    "    7. anything else (one plain body, no body)                        -> other\n"
)
V1_FLAG = ("and separately flagged when it carries a visible part of the other asked species.")
V2_FLAG = ("and separately flagged only when it clearly shows a named part of the other asked\n"
           "species: its spots or stripes, a mane, wings, feathers, a beak, horns, hooves or a tail, covering more\n"
           "than a small patch of the body. A similar colour, a faint or partial pattern, or a shape that merely\n"
           "resembles the other species is not a flag. When unsure, do not flag.")
V2_BORROWED = ("\nBorrowed part (rule 3): a named part of the other asked species is clearly visible on one of the two\n"
               "bodies. When unsure, call it two clear.\n")
# Version 3 adds one paragraph after version 2's flag paragraph, after batches 1 and 2 split 20 times between hybrid and
# borrowed part on one picture type (cheetah and cougar: a cougar-shaped head on a body in cheetah spots) and borrowed
# part missed the agreement bar at 36 of 58 (0.62). Version 2 is frozen as built (checksum pinned in the tests).
V3_SHAPE = ("A body's species is decided by its shape: its head, face, ears, build, legs, wings or tail shape. A coat,\n"
            "spots, stripes or colour never make a body MIXED, however much of the body they cover: a cougar-shaped\n"
            "body in cheetah spots is the cougar, flagged. A body is MIXED only when its shape joins the two species,\n"
            "such as one species' head on the other's build, or the other's legs, wings or beak.")
# Version 4 adds one paragraph after version 3's shape paragraph, after wrong species missed the agreement bar at 24 of
# 31 (0.77) across seven batches: the splits were over what counts as a body (a tiny blurred bird, a moth, a shadow) and
# whether a near relative is OTHER (an eagle drawn for a condor). Version 3 is frozen as built (checksum pinned).
V4_BODIES = ("Count as a body only an animal you can name at a glance and that is at least the size of a hand relative\n"
             "to the asked animals; skip tiny, blurred, distant or half-seen shapes, shadows, statues, toys and pictures\n"
             "within the picture. A body that is a different species from both asked animals is OTHER even when it is a\n"
             "close relative (an eagle drawn for a condor, a coyote drawn for a fox).")
RULES_VERSION = 4   # what build writes unless told otherwise
RULES_VERSIONS = (1, 2, 3, 4)


def _body_v2() -> str:
    assert V1_FLAG in RULES_V1
    return RULES_V1.replace(V1_FLAG, V2_FLAG) + V2_BORROWED


def _body_v3() -> str:
    return _body_v2().replace(V2_FLAG, V2_FLAG + "\n\n" + V3_SHAPE)


def rules_text(version: int = RULES_VERSION) -> str:
    """The rubric a reviewer reads, as rules.txt. Versions 2 and later open with their version line."""
    if version == 1:
        return RULES_V1
    if version == 2:
        return "Rules v2\n" + _body_v2()
    if version == 3:
        return "Rules v3\n" + _body_v3()
    if version == 4:
        return "Rules v4\n" + _body_v3().replace(V3_SHAPE, V3_SHAPE + "\n\n" + V4_BODIES)
    raise SystemExit(f"no rubric version {version}; versions {', '.join(map(str, RULES_VERSIONS))} exist")


def build(a: argparse.Namespace) -> dict:
    rows = read_list(a.pictures)
    clash = [c for c in RESERVED_FIELDS if rows and c in rows[0]]
    if clash:
        raise SystemExit(f"the list has a column named {', '.join(clash)}, a field the key reserves for the picture's "
                         "path; rename that column")
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
        key[name] = {**{k: v for k, v in r.items() if k != "picture"}, "picture_path": r["picture"]}
        listed.append((name, asked_line(r["a"], r["b"])[len("Asked: "):]))

    with open(sheet / "sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "asked"])
        w.writerows(listed)
    with open(sheet / "calls-template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "kind", "note"])
        w.writerows([(n, "", "") for n, _ in listed])
    (sheet / "rules.txt").write_text(rules_text(a.rules_version))
    (keydir / "key.json").write_text(json.dumps({"seed": a.seed, "rules_version": a.rules_version,
                                                 "list": str(Path(a.pictures).resolve()),
                                                 "pictures": key}, indent=1))
    summary = {"pages": len(names), "sheet": str(sheet), "key": str(keydir / "key.json"), "seed": a.seed}
    print(json.dumps(summary))
    return summary


def picture_path_of(entry: dict) -> str | None:
    """The picture behind a key entry: "picture_path" as build writes it, or "picture" in a key relinked from its list."""
    return entry.get("picture_path") or entry.get("picture")


def unblinding_strings(key: dict) -> set[str]:
    """What would tell a reviewer where a picture came from: its source folders and file name, and the
    values of the identifying columns, never the call columns."""
    found: set[str] = set()
    for entry in key["pictures"].values():
        path = picture_path_of(entry)
        if path:
            src = Path(path)
            found.add(src.name)
            found.add(src.stem)
            found.update(p for p in src.parent.parts[-3:] if p not in ("/", ""))
        for k in IDENTIFYING_COLUMNS:
            v = str(entry.get(k, "")).strip()
            if v:
                found.add(v)
    return {s for s in found if len(s) >= 3 or (s.isdigit() and len(s) >= 4)}


def contains(data: bytes, needle: str) -> bool:
    """A short pure-letter value counts only as a whole word; any other value counts anywhere."""
    if needle.isalpha() and len(needle) <= SHORT_WORD:
        return re.search(rb"(?<![A-Za-z0-9_])" + re.escape(needle.encode()) + rb"(?![A-Za-z0-9_])", data) is not None
    return needle.encode() in data


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
    sheet = Path(a.out) / "sheet"
    keyfile = Path(a.key) if a.key else Path(a.out) / "key" / "key.json"
    key = json.loads(keyfile.read_text())
    needles = sorted(unblinding_strings(key))
    hits = []
    no_path = [n for n, e in key["pictures"].items() if not picture_path_of(e)]
    missing = [n for n, e in key["pictures"].items() if picture_path_of(e) and not Path(picture_path_of(e)).exists()]
    hits += [f"key {keyfile.name}: entry {n} has no picture path" for n in no_path]
    hits += [f"key {keyfile.name}: entry {n}'s picture is missing at {picture_path_of(key['pictures'][n])}" for n in missing]
    files = sorted(p for p in sheet.rglob("*") if p.is_file())
    rules = sheet / "rules.txt"
    version = int(key.get("rules_version", 1))      # a key with no version (batch-01) was built with version 1
    if not rules.exists() or rules.read_text() != rules_text(version):
        hits.append(f"rules.txt: is not exactly the rubric, version {version}")
    searched = [p for p in files if p != rules]
    for p in searched:
        data = searchable_bytes(p)
        rel = str(p.relative_to(sheet))
        for s in needles:
            if contains(rel.encode(), s) or contains(data, s):
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
    print(f"searched {len(searched)} files under {sheet} (rules.txt compared whole) for {len(needles)} strings: "
          f"source folders, file names, and the columns {', '.join(IDENTIFYING_COLUMNS)}: {len(hits)} found")
    for h in hits:
        print("  " + h)
    print(f"key {keyfile}: {len(key['pictures'])} entries, {len(no_path)} with no picture path, {len(missing)} missing on disk")
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
    b.add_argument("--rules-version", type=int, choices=RULES_VERSIONS, default=RULES_VERSION,
                   help="the reviewers' rubric written as rules.txt and recorded in key.json")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--key", type=Path, default=None,
                   help="the key to check against; default <out>/key/key.json (sheets built before the path fix: "
                        "key/key-relinked.json)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a)
        return 0
    return check(a)


if __name__ == "__main__":
    raise SystemExit(main())
