"""List every figure caption in the 19 rival PDFs, by page.

Reads each page in both pdftotext modes, because two-column papers put captions mid-line in one
mode and at line start in the other. The list is a floor: each figure reader also reports any
figure it sees that is missing here.

Usage: python3 -I artifacts/notes/progress/targets/reads/figure_inventory.py
"""
import csv
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figure-inventory.csv'
PDFS = HERE / 'rivals.tsv'

LABEL = r'((?:[A-Z]\.?)?\d+(?:\.\d+)?)'
UPPER = re.compile(r'F\s?IGURE\s+' + LABEL + r'\b\s*[:.]?\s*(.{0,110})')
COLON = re.compile(r'\b(?:Figure|Fig\.)\s+' + LABEL + r'\s*:\s*(.{0,110})')
LINE_START = re.compile(r'^\s*(?:Figure|Fig\.)\s+' + LABEL + r'\s*\.\s+([A-Z(].{0,110})')


def pages(path, *flags):
    out = subprocess.run(['pdftotext', *flags, path, '-'], capture_output=True, text=True).stdout
    return out.split('\f')


rows = []
names = []
for line in PDFS.open():
    name, arxiv, path = line.rstrip('\n').split('\t')
    names.append(name)
    found = {}
    for flags in (['-layout'], []):
        for number, page in enumerate(pages(path, *flags), 1):
            for raw in page.splitlines():
                for pattern in (UPPER, COLON, LINE_START):
                    hit = pattern.search(raw)
                    if hit and hit.group(1) not in found:
                        found[hit.group(1)] = (number, re.sub(r'\s+', ' ', hit.group(2)).strip())
    for label, (number, caption) in sorted(found.items(), key=lambda kv: kv[1][0]):
        rows.append({'paper': name, 'arxiv': arxiv, 'figure': label, 'page': number,
                     'caption_start': caption})

with OUT.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['paper', 'arxiv', 'figure', 'page', 'caption_start'],
                            lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

for name in names:
    mine = [r for r in rows if r['paper'] == name]
    print(f"{name:20s} {len(mine):3d}  pages {min((r['page'] for r in mine), default=0)}-"
          f"{max((r['page'] for r in mine), default=0)}")
print('total', len(rows))
