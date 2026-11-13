#!/usr/bin/env python
"""How big the text in each paper figure prints, against the layout sheet's lettering rule.

For every live \\includegraphics in the tex, the printed width comes from its width option
(a fraction of \\textwidth or \\linewidth, 397.48 pt in this template). A vector PDF is measured
from its own text layer (pdftotext -bbox: word box height in the file's points, scaled by printed
width over the file's page width). A raster image is measured by OCR (tesseract word boxes in
pixels, scaled by printed width over image width); OCR boxes span ascender to descender, so the
font size is estimated as box height / 0.85 and marked an estimate.

    python3 scripts/figure_text_audit.py [paper/iclr-56173/iclr2027_conference.tex]

Prints one row per figure: file, printed width, words measured, smallest and median text in pt,
and PASS when the smallest is at least 7 pt.
"""
from __future__ import annotations

import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

TEXTWIDTH_PT = 397.48
MIN_PT = 7.0


def includes(tex: Path) -> list[tuple[str, float]]:
    out, wrap = [], None
    for line in tex.read_text().splitlines():
        if line.lstrip().startswith("%"):
            continue
        w = re.search(r"\\begin\{(?:wrapfigure\}(?:\[[^\]]*\])?\{[a-z]\}|minipage\}(?:\[[^\]]*\])?)\{([0-9.]+)\\textwidth\}", line)
        if w:
            wrap = float(w.group(1))
        r = re.search(r"\\reviewslot(?:\[([^\]]+)\])?\{(R\d+)\}\{\\linewidth\}", line)
        if r and (r.group(1) or wrap):  # a slot inside a box: its filled file if any, else its fallback
            filled = [f"slots/{r.group(2)}{e}" for e in (".pdf", ".png") if (tex.parent / f"slots/{r.group(2)}{e}").exists()]
            if filled or r.group(1):
                out.append(((filled or [r.group(1)])[0], wrap or 1.0))
            continue
        if "\\includegraphics" not in line or "#" in line:
            continue
        m = re.search(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", line)
        if not m:
            continue
        w = re.search(r"width=([0-9.]*)\\(textwidth|linewidth)", m.group(1))
        out.append((m.group(2), float(w.group(1) or 1) if w else 1.0))
    return out


def resolve(root: Path, name: str) -> Path | None:
    for base in (root, root / "figures"):
        p = base / name
        if p.exists():
            return p
    return None


def pdf_sizes(p: Path, printed_pt: float) -> list[float]:
    """One size per word: the height of its line's box. A line whose longest word (three or
    more letters) has a box much taller than wide is set on its side, so its size is the line box's width."""
    html = subprocess.run(["pdftotext", "-bbox-layout", str(p), "-"], capture_output=True, text=True).stdout
    page = re.search(r'<page width="([0-9.]+)"', html)
    if not page:
        return []
    scale = printed_pt / float(page.group(1))
    num = r'"([0-9.]+)"'
    sizes = []
    for x1, y1, x2, y2, body in re.findall(rf'<line xMin={num} yMin={num} xMax={num} yMax={num}>(.*?)</line>', html, re.S):
        w, h = float(x2) - float(x1), float(y2) - float(y1)
        words = [(float(a) - float(b), float(c) - float(d), t) for b, d, a, c, t in
                 re.findall(rf'<word xMin={num} yMin={num} xMax={num} yMax={num}>([^<]*)</word>', body)]
        long = [wd for wd in words if len(wd[2]) >= 3]  # a digit is taller than wide upright too
        on_side = bool(long) and max(long, key=lambda wd: len(wd[2]))[1] > 1.5 * max(long, key=lambda wd: len(wd[2]))[0]
        sizes += [(w if on_side else h) * scale] * len(words)
    return sizes


def raster_sizes(p: Path, printed_pt: float) -> list[float]:
    from PIL import Image
    w = Image.open(p).width
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "ocr"
        subprocess.run(["tesseract", str(p), str(base), "tsv"], capture_output=True)
        rows = (base.with_suffix(".tsv")).read_text().splitlines()[1:]
    sizes = []
    for r in rows:
        f = r.split("\t")
        if len(f) == 12 and f[11].strip() and len(f[11].strip()) >= 2 and float(f[10]) >= 60:
            sizes.append(int(f[9]) / 0.85 * printed_pt / w)
    return sizes


def main() -> int:
    tex = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/iclr-56173/iclr2027_conference.tex")
    root = tex.parent
    print(f"{'figure':58s} {'printed':>8s} {'words':>6s} {'min pt':>7s} {'median':>7s}  verdict")
    for name, frac in includes(tex):
        p = resolve(root, name)
        printed = frac * TEXTWIDTH_PT
        if p is None:
            print(f"{name:58s} {printed:8.0f} {'':>6s} {'':>7s} {'':>7s}  file missing")
            continue
        raster = p.suffix.lower() in (".png", ".jpg", ".jpeg")
        sizes = raster_sizes(p, printed) if raster else pdf_sizes(p, printed)
        if not sizes:
            print(f"{name:58s} {printed:8.0f} {0:6d} {'':>7s} {'':>7s}  no text found")
            continue
        lo, med = min(sizes), statistics.median(sizes)
        verdict = ("PASS" if lo >= MIN_PT else "FAIL") + (" (OCR estimate)" if raster else "")
        print(f"{name:58s} {printed:8.0f} {len(sizes):6d} {lo:7.1f} {med:7.1f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
