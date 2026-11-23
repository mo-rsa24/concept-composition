#!/usr/bin/env python
"""The check that every page of this ladder can be trusted: links resolve and no decimal is invented.

For each markdown page in the ladder it reports
  - a link or image whose local target does not exist, or whose #anchor is not a heading on that page;
  - a decimal that appears in neither the miniature's key, the numbers file, the paper, nor the two checked notes;
  - an em dash or en dash, which the writing rules forbid in prose.

Exit code 1 if anything is reported. Run:
  PYTHONDONTWRITEBYTECODE=1 python3 -I check_pages.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = Path(__file__).resolve().parent / "sources" / "learn-to-steer-paged.txt"
NOTES = Path("/home/molef/PhD/concept-composition/artifacts/notes/progress/abstract-checkers/01-reading")
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w]|\.\d)")
LINK = re.compile(r"(?<!\!)\]\(([^)\s]+)\)|(?:href|src)=\"([^\"]+)\"")


def floats_in(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            floats_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            floats_in(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(round(abs(float(obj)), 6))
    elif isinstance(obj, str):
        out.update(round(float(m), 6) for m in DECIMAL.findall(obj))


def known_values():
    values: set[float] = set()
    for path in (ROOT / "2-miniature/miniature_answer_key.json", ROOT / "numbers/numbers.json"):
        floats_in(json.loads(path.read_text()), values)
    for path in (PAPER, NOTES / "learn-to-steer.md", NOTES / "detector-guidance.md"):
        values.update(round(float(m), 6) for m in DECIMAL.findall(path.read_text(errors="ignore")))
    return values


def slug(heading):
    text = re.sub(r"[`*$\\]", "", heading.strip()).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text.strip())


def anchors(path):
    return {slug(m) for m in re.findall(r"^#{1,6}\s+(.+)$", path.read_text(), flags=re.M)}


def strip_blocks(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


def in_key(value, known):
    """A page may print a key value to fewer places, so a rounded match counts."""
    places = len(str(value).split(".")[1])
    return any(round(k, places) == round(float(value), places) for k in known)


def main():
    known = known_values()
    problems = []
    pages = sorted(p for p in ROOT.rglob("*.md") if "drop" not in p.parts)
    for page in pages:
        raw = page.read_text()
        name = page.relative_to(ROOT)
        for m in LINK.finditer(raw):
            target = m.group(1) or m.group(2)
            if re.match(r"(https?:|mailto:)", target):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (page.parent / unquote(file_part)).resolve() if file_part else page
            if not dest.exists():
                problems.append(f"{name}: missing {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                problems.append(f"{name}: no heading for #{anchor} in {dest.name}")
        body = strip_blocks(raw)
        body = re.sub(r"\]\([^)]*\)|(?:href|src)=\"[^\"]*\"|alt=\"[^\"]*\"", "", body)
        body = re.sub(r"(§\s*|[Cc]hapter\s+)\d+(\.\d+)*", "", body)     # a textbook section number is not a value
        for value in sorted(set(DECIMAL.findall(body))):
            if not in_key(value, known):
                problems.append(f"{name}: decimal {value} is in no source")
        for i, line in enumerate(raw.splitlines(), 1):
            if "—" in line or "–" in line:
                problems.append(f"{name}:{i}: dash used as punctuation")
    print(f"checked {len(pages)} pages against {len(known)} known values")
    for p in problems:
        print(p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
