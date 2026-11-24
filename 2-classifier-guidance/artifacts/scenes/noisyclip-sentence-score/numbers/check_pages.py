"""Checks the ladder's markdown pages: every local link and anchor resolves, and every decimal appears in a source.

A decimal passes if it equals a value in numbers.json, the miniature's key, the W2 design or the paper's text, allowing for rounding
to fewer places. Fenced code is skipped, because the code walk's quotes have their own check.
Run from anywhere: PYTHONDONTWRITEBYTECODE=1 python3 numbers/check_pages.py
"""
import re
import sys
from pathlib import Path
from urllib.parse import unquote

L = Path(__file__).resolve().parents[1]
REPO = L.parents[3]
PAPER = Path('/home/molef/.claude/jobs/bed4762f/tmp/papers/noisyclip/paged.txt')
PAGES = [L / n for n in ('README.md', 'NOTATION.md', 'read-through-a-small-network.md', 'rebuild-from-fundamentals.md', 'route-zoom-ladder-noisyclip.md',
                         'style-reference.md', '0.5-network-view/README.md', '1-illustrations/README.md', '1-illustrations/diagram-prompts.md',
                         '2-miniature/README.md', '3-code/walk-noisyclip.md', 'overview/README.md')]
NUMBER = re.compile(r'(?<![\w.#/])\d+\.\d+(?![\w.]*\d)')
sources = [L / 'numbers/numbers.json', L / '2-miniature/miniature_answer_key.json',
           REPO / 'artifacts/notes/progress/noise-text-critic/02-can-it-read/design.md',
           REPO / 'artifacts/notes/progress/noise-text-critic/01-reading/noisyclip.md', L / '2-miniature/word_check_toy_key.json',
           REPO / 'artifacts/notes/progress/noise-text-critic/02-can-it-read/word-check/word-check.json',
           REPO / 'artifacts/notes/progress/noise-text-critic/02-can-it-read/review.md']
text = '\n'.join(p.read_text() for p in sources) + (PAPER.read_text() if PAPER.exists() else '')
pool = set()
for tok in re.findall(r'\d+\.\d+', text):
    v = float(tok)
    pool.add(v)
    for k in range(1, len(tok.split('.')[1])):
        pool.add(round(v, k))


def slug(h):
    h = re.sub(r'\$[^$]*\$', lambda m: m.group(0).strip('$'), h)
    h = re.sub(r'[`*_]|\[|\]\([^)]*\)', '', h).strip().lower()
    h = re.sub(r'[^\w\- ]', '', h)
    return h.replace(' ', '-')


def anchors(path):
    out, fenced = set(), False
    for line in path.read_text(errors='ignore').split('\n'):
        if line.startswith('```'):
            fenced = not fenced
        if not fenced and re.match(r'#{1,6} ', line):
            out.add(slug(re.sub(r'^#+ ', '', line)))
    out |= set(re.findall(r'id="([^"]+)"', path.read_text(errors='ignore')))
    return out


bad = 0
links = 0
for page in PAGES:
    raw = page.read_text()
    body = re.sub(r'```.*?```', '', raw, flags=re.S)
    targets = re.findall(r'\]\(([^)\s]+)\)', body) + re.findall(r'(?:href|src)="([^"]+)"', body)
    for t in targets:
        if re.match(r'(https?:|mailto:)', t):
            continue
        links += 1
        p, _, a = t.partition('#')
        f = (page.parent / unquote(p)).resolve() if p else page
        if not f.exists():
            bad += 1
            print(f'{page.relative_to(L)}: missing {t}')
        elif a and f.suffix == '.md' and unquote(a) not in anchors(f):
            bad += 1
            print(f'{page.relative_to(L)}: no anchor {t}')
    if page.name != 'diagram-prompts.md':
        prose = re.sub(r'`[^`\n]*`', '', re.sub(r'<[^>]+>', '', re.sub(r'\]\([^)]*\)', ']', body)))
        for tok in NUMBER.findall(prose):
            if float(tok) not in pool:
                bad += 1
                print(f'{page.relative_to(L)}: decimal {tok} is in no source')
print(f'{len(PAGES)} pages, {links} local links, {bad} problems')
sys.exit(1 if bad else 0)
