#!/usr/bin/env python
"""Pull Re-LAION-2B photos whose caption names both a cat and a dog, as checker data for the Claude API to vet.

Re-LAION-2B (laion/relaion2B-en-research-safe, gated; the account's saved Hugging Face token must have
accepted its terms) holds captions and image links, no images. This reads only four columns of randomly
chosen parquet files (url, caption, punsafe, pwatermark) and keeps a row when its caption names a cat (cat,
kitten, kitty) and a dog (dog, puppy, pup), its unsafe score is under 0.1 and its watermark score under 0.5.
It stops at --target candidates, then downloads each image, keeps it when the short side is at least 384 px,
and saves a centre square crop. A caption is a weak label: the crops go through api_label.py --strict
--real-only, which decides clean pair or reject.

Runs on the laptop in its own environment (~/.venvs/laion: pyarrow, huggingface_hub, pillow).
Writes data/relaion-catdog/candidates.jsonl and data/relaion-catdog/images/<key>.jpg.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

ROOT = Path(__file__).resolve().parents[3] / "data/relaion-catdog"
REPO = "datasets/laion/relaion2B-en-research-safe"
CAT = re.compile(r"\b(cat|cats|kitten|kittens|kitty)\b", re.I)
DOG = re.compile(r"\b(dog|dogs|puppy|puppies|pup|pups)\b", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    cand_path = ROOT / "candidates.jsonl"
    seen = {json.loads(l)["key"] for l in cand_path.read_text().splitlines()} if cand_path.exists() else set()
    fs = HfFileSystem()
    files = sorted(f for f in fs.glob(f"{REPO}/**/*.parquet"))
    random.Random(0).shuffle(files)
    print(f"{len(files)} parquet files; {len(seen)} candidates already", flush=True)
    done_files = ROOT / "files_read.txt"
    read = set(done_files.read_text().split()) if done_files.exists() else set()
    rows_read = 0
    for f in files:
        if len(seen) >= args.target:
            break
        if f in read:
            continue
        with fs.open(f) as fh:
            t = pq.read_table(fh, columns=["url", "caption", "punsafe", "pwatermark"]).to_pylist()
        rows_read += len(t)
        new = 0
        with cand_path.open("a") as out:
            for r in t:
                cap = r["caption"] or ""
                if not (CAT.search(cap) and DOG.search(cap)):
                    continue
                if (r["punsafe"] or 1) >= 0.1 or (r["pwatermark"] or 1) >= 0.5:
                    continue
                key = hashlib.sha1(r["url"].encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps({"key": key, "url": r["url"], "caption": cap[:300], "file": f}) + "\n")
                new += 1
        with done_files.open("a") as d:
            d.write(f + "\n")
        print(f"{f.split('/')[-1]}: {len(t)} rows, {new} candidates (total {len(seen)}, "
              f"one per {rows_read // max(1, len(seen))} rows)", flush=True)

    rows = [json.loads(l) for l in cand_path.read_text().splitlines()]
    (ROOT / "images").mkdir(exist_ok=True)

    def fetch(r):
        dest = ROOT / "images" / f"{r['key']}.jpg"
        if dest.exists():
            return "had"
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0 (research image fetch)"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                im = Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return f"failed: {type(e).__name__}"
        w, h = im.size
        s = min(w, h)
        if s < 384:
            return "too small"
        im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).save(dest, quality=95)
        return "got"

    from collections import Counter
    with ThreadPoolExecutor(args.workers) as ex:
        res = Counter(ex.map(fetch, rows))
    print("downloads", dict(res), flush=True)
    print("LAION CATDOG DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
