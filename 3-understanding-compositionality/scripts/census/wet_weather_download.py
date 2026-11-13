#!/usr/bin/env python
"""Download the pictures behind every camel and dog caption with a wet word or a weather word, and write markdown
pages that show each caption, its picture URL and the picture.

Reads the slice's hits files (3-understanding-compositionality/scripts/census/laptop_slice.py count: the 2 downloaded files, 32,776,409 captions).
Wet words and weather words are the same whole-word patterns that script uses (FAMILIES).

    /home/molef/miniconda3/bin/python3 -I 3-understanding-compositionality/scripts/census/wet_weather_download.py

Writes, under 3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/wet-and-weather/:
    README.md                      counts per group, pictures loaded, links to the pages
    <group>/images/NNNN.jpg        thumbnails (longest side 384)
    <group>/fetch.json             caption, URL and fetch status of every hit
    <group>/page-NN.md             50 hits per page: caption, URL, thumbnail
The groups are camel-wet, camel-weather, dog-wet and dog-weather. A caption with both kinds of word is in both.
"""
from __future__ import annotations

import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laptop_slice import FAMILIES, SLICE  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/wet-and-weather"
PER_PAGE = 50
GROUPS = [("camel-wet", "camel", "wet"), ("camel-weather", "camel", "weather"),
          ("dog-wet", "dog", "wet"), ("dog-weather", "dog", "weather")]
WORDS = {"wet": "wet, soaked, drenched, soaking, dripping or damp",
         "weather": "rain, raining, rainy, storm, downpour or puddle"}


def fetch_one(row: dict, dst: Path) -> str:
    import requests
    from PIL import Image
    try:
        r = requests.get(row["url"], timeout=8, headers={"User-Agent": "Mozilla/5.0 (research)"})
        if r.status_code != 200:
            return f"http {r.status_code}"
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(im.size) < 64:
            return "too small"
        im.thumbnail((384, 384))
        im.save(dst, quality=88)
        return "ok"
    except Exception as e:
        return type(e).__name__


def main() -> None:
    summary = []
    for name, subject, family in GROUPS:
        t = pq.read_table(SLICE / f"hits-{subject}.parquet")
        t = t.filter(pc.match_substring_regex(pc.utf8_lower(t["caption"]), FAMILIES[family]))
        rows = t.to_pylist()
        d = OUT / name
        (d / "images").mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(32) as ex:
            status = list(ex.map(lambda ir: fetch_one(ir[1], d / "images" / f"{ir[0]:04d}.jpg"), enumerate(rows)))
        rec = [{"i": i, "status": s, "caption": r["caption"], "url": r["url"]} for i, (r, s) in enumerate(zip(rows, status))]
        (d / "fetch.json").write_text(json.dumps(rec, indent=1))
        ok = sum(s == "ok" for s in status)
        pages = max(1, math.ceil(len(rec) / PER_PAGE))
        for p in range(pages):
            chunk = rec[p * PER_PAGE:(p + 1) * PER_PAGE]
            md = [f"# {name}, page {p + 1} of {pages}\n",
                  f"Captions naming a {subject} with {WORDS[family]}; hits {p * PER_PAGE + 1} to {p * PER_PAGE + len(chunk)} "
                  f"of {len(rec)}. A picture that did not load is marked with the reason.\n"]
            for r in chunk:
                cap = r["caption"].replace("\n", " ").strip()[:240]
                md.append(f"**{r['i'] + 1}.** {cap}\n")
                md.append(f"[picture URL]({r['url']}) · {r['status']}\n")
                if r["status"] == "ok":
                    md.append(f'<img src="images/{r["i"]:04d}.jpg" width="220">\n')
            (d / f"page-{p + 1:02d}.md").write_text("\n".join(md))
        print(f"{name}: {ok} of {len(rec)} pictures loaded, {pages} pages", flush=True)
        summary.append((name, subject, family, len(rec), ok, pages))
    lines = ["# Wet and weather captions, with their pictures\n",
             "Camel and dog captions with a wet word or a weather word, from the 2 downloaded files of Re-LAION-2B-en "
             "(32,776,409 captions), the public stand-in for SDXL's unpublished training data. Whole-word, case-ignoring "
             "matches, so a caption can be about a camel-coloured jacket or a dog-food can. A caption with both kinds of word "
             "is in both groups.\n",
             "| Group | Words | Captions | Pictures loaded | Pages |", "|---|---|---|---|---|"]
    for name, subject, family, n, ok, pages in summary:
        links = ", ".join(f"[{p + 1}]({name}/page-{p + 1:02d}.md)" for p in range(pages))
        lines.append(f"| {name} | {WORDS[family]} | {n:,} | {ok:,} | {links} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
