#!/usr/bin/env python
"""How do training captions phrase a cat and a dog together, and does the picture show both?

Input is the per-file output of the search that kept every Re-LAION-2B-en caption naming a cat word
and a dog word (one parquet per source file, columns url, caption, similarity, width, height).

    python 3-understanding-compositionality/scripts/census/cat_and_dog_captions.py count  --parts DIR   # counts.json: sizes, forms, joiners
    python 3-understanding-compositionality/scripts/census/cat_and_dog_captions.py sample --parts DIR   # sample.json: captions to look at
    python 3-understanding-compositionality/scripts/census/cat_and_dog_captions.py fetch                # pictures + thumbnails for the sample
    python 3-understanding-compositionality/scripts/census/cat_and_dog_captions.py page                 # the markdown with clickable pictures

Run with a python that has pyarrow, requests and PIL.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog"
SEED = 20261103

CAT = r"(?:cat|cats|kitten|kittens)"
DOG = r"(?:dog|dogs|puppy|puppies)"
LEAD = r"(?:(?:a|an|the|one|two|photo of|picture of|image of|cute|little|my|our) )*"
JOIN = r"(?:and|&|with|plus)"
BARE = re.compile(rf"^{LEAD}{CAT} {JOIN} {LEAD}{DOG}$|^{LEAD}{DOG} {JOIN} {LEAD}{CAT}$")
# the two animal words next to each other, with at most one joining word between them
ADJ = re.compile(rf"\b({CAT}|{DOG})\b(?:\s*(and|&|with|or|vs\.?|versus|x|plus|\+|,)\s*|\s+)(?:a |an |the )?\b({CAT}|{DOG})\b")

CLASSES = {
    "bare": "only the two animals joined by a word (a cat and a dog)",
    "short": "12 words or fewer, not bare",
    "random": "an unbiased draw from every caption naming both",
    "long": "30 words or more",
}


def norm(c: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z& ]", " ", c.lower())).strip()


def rows(parts: Path):
    for f in sorted(glob.glob(str(parts / "[0-9]*.parquet"))):
        t = pq.read_table(f, columns=["url", "caption", "similarity", "width", "height"]).to_pylist()
        for r in t:
            r["file"] = Path(f).stem
            yield r


def kind(caption: str, nw: int) -> str:
    if BARE.match(norm(caption)):
        return "bare"
    return "short" if nw <= 12 else ("long" if nw >= 30 else "mid")


def count(parts: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    nw_hist = collections.Counter()
    kinds = collections.Counter()
    joiners = collections.Counter()
    adjacent = 0
    per_file = collections.Counter()
    for r in rows(parts):
        n += 1
        per_file[r["file"]] += 1
        nw = len(r["caption"].split())
        nw_hist[min(nw, 60)] += 1
        kinds[kind(r["caption"], nw)] += 1
        m = ADJ.search(r["caption"].lower())
        if m:
            adjacent += 1
            joiners[(m.group(2) or "(none)")] += 1
    ws = sorted(nw_hist.elements())
    q = lambda p: ws[int(p * (len(ws) - 1))]
    res = {
        "files": len(per_file),
        "captions_naming_both": n,
        "words_median": q(0.5), "words_p10": q(0.1), "words_p90": q(0.9),
        "share_by_form": {k: round(v / n, 5) for k, v in kinds.items()},
        "count_by_form": dict(kinds),
        "adjacent_pair_captions": adjacent,
        "joiner_between_adjacent_animals": dict(joiners.most_common()),
        "words_note": "word counts above 60 are folded into 60",
    }
    (OUT / "counts.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def sample(parts: Path, per_class: int = 60) -> None:
    rng = random.Random(SEED)
    pools = {k: [] for k in CLASSES}
    seen = {k: 0 for k in CLASSES}
    for r in rows(parts):
        nw = len(r["caption"].split())
        ks = ["random"]
        k = kind(r["caption"], nw)
        if k in ("bare", "short", "long"):
            ks.append(k)
        for c in ks:
            seen[c] += 1
            item = {"url": r["url"], "caption": r["caption"], "words": nw, "file": r["file"], "class": c}
            if len(pools[c]) < per_class:
                pools[c].append(item)
            else:
                j = rng.randrange(seen[c])
                if j < per_class:
                    pools[c][j] = item
    (OUT / "sample.json").write_text(json.dumps({"seen": seen, "pools": pools}, indent=1))
    print({k: (seen[k], len(v)) for k, v in pools.items()})


def pid(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def fetch() -> None:
    import requests
    from PIL import Image
    s = json.loads((OUT / "sample.json").read_text())
    (OUT / "images").mkdir(exist_ok=True)
    items = [i for v in s["pools"].values() for i in v]

    def one(i):
        p = OUT / "images" / f"{pid(i['url'])}.jpg"
        if p.exists():
            return i["url"], True
        try:
            r = requests.get(i["url"], timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.thumbnail((360, 360))
            im.save(p, quality=80)
            return i["url"], True
        except Exception:
            return i["url"], False

    with ThreadPoolExecutor(48) as ex:
        ok = dict(ex.map(one, items))
    print(f"loaded {sum(ok.values())} of {len(ok)}")
    (OUT / "fetched.json").write_text(json.dumps(ok))


def page(keep: int = 30) -> None:
    s = json.loads((OUT / "sample.json").read_text())
    ok = json.loads((OUT / "fetched.json").read_text()) if (OUT / "fetched.json").exists() else {}
    c = json.loads((OUT / "counts.json").read_text())
    L = ["# Does any caption say a cat and a dog, and does the picture show both?", "",
         f"Searched {c['captions_naming_both']:,} captions that name a cat word and a dog word, out of "
         "2,097,693,557 in Re-LAION-2B-en (128 files). Counts are in `counts.json` beside this file.", "",
         "**How to check by eye.** Click a thumbnail to open the original picture, read the caption, and "
         "write one letter in the last column: `B` both a cat and a dog are in the picture, `O` only one "
         "of them, `N` neither or not a photo. Count the letters per section at the end.", ""]
    for k, desc in CLASSES.items():
        pool = s["pools"][k]
        L += [f"## {k}: {desc}", "", f"{s['seen'][k]:,} captions in this class; {len(pool)} drawn at random.", "",
              "| # | Picture | Caption | Words | B / O / N |", "|---|---|---|---|---|"]
        shown = 0
        for i in pool:
            f = f"images/{pid(i['url'])}.jpg"
            has = ok.get(i["url"], False)
            if ok and not has:
                continue
            shown += 1
            if shown > keep:
                break
            cap = i["caption"].replace("|", "/").replace("\n", " ")
            pic = f"[![]({f})]({i['url']})" if has else f"[original picture]({i['url']})"
            L.append(f"| {shown} | {pic} | {cap} | {i['words']} | |")
        L += ["", f"Pictures that loaded: {shown} of {len(pool)} drawn.", ""]
    (OUT / "caption-check.md").write_text("\n".join(L))
    print("wrote", OUT / "caption-check.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["count", "sample", "fetch", "page"])
    ap.add_argument("--parts", type=Path)
    a = ap.parse_args()
    {"count": lambda: count(a.parts), "sample": lambda: sample(a.parts), "fetch": fetch, "page": page}[a.stage]()
