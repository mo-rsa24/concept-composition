"""Join the 19 figure reads into targets/failures.md, one section per rival.

A row belongs to a rival when its method is the rival's short name or starts with it, so
`PiCo without the exclusive mask` sits under PiCo. Every other method is listed by the paper that
printed it. Cells are copied unchanged. Stops if a paper is missing or a row names no page.

Usage: python3 -I artifacts/notes/progress/targets/reads/build_failures.py
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'
OUT = HERE.parent / 'failures.md'
RIVAL_ROW = re.compile(r'^\| ([A-Z][\w-]*) \| \d{4}\.\d{5}v\d+ \|')
NO_OWN_METHOD = {
    'Catastrophic': 'This paper analyses composition and proposes no sampler of its own. What its figures show of FKC is under FKC, and its other methods are under "Other methods".',
    'CFGRevisited': 'This paper re-evaluates guidance methods and proposes none of its own. Its figures of CFG, CFG++, APG and the rest are under "Other methods".',
}
HEADER = """# How each rival fails

This page records what each method draws well and where it breaks, as seen in the pictures the papers print. It comes
from {figures} figures in 19 papers, {qualitative} of them generated samples, and {rows} rows of per-method findings. The
scoreboard's `failure` line for each rival is taken from here.

**Authors choose their own figures.** A method's successes are over-represented, and a failure its authors did not print
stays unseen. `no failure shown in this paper's figures` means exactly that, and says nothing about how the method
behaves elsewhere. A method drawn by a rival's authors may also be run less carefully than by its own.

**Two-object pictures use one ladder:** one thing, a fused hybrid, two bodies or heads, two clear things.

What our own runs show on our pairs is kept apart, in [the record of our runs](reads/figures/on-our-data.md), because
those are our numbers and never published ones. Each paper's tile-by-tile calls are in `reads/figures/<Paper>.observations.csv`.
[The reading instructions](reads/how-the-figures-were-read.md) say how a tile was judged.
"""
COLUMNS = '| Dataset | Method as shown | Succeeds | Fails | On which | Where | Printed by |\n|---|---|---|---|---|---|---|'


def cell(text):
    return re.sub(r'\s+', ' ', text).replace('|', '\\|').strip()


def line(row):
    return (f"| {cell(row['dataset'])} | {cell(row['method'])} | {cell(row['succeeds'])} | {cell(row['fails'])} | "
            f"{cell(row['on_which'])} | {cell(row['figures'])} | {row['printed_by']} |")


def rival_of(method, rivals):
    for name in sorted(rivals, key=len, reverse=True):
        if method == name or method.startswith(name + ' '):
            return name
    return None


def main():
    rivals = [hit.group(1) for text in (HERE.parent / 'rivals.md').read_text().splitlines()
              if (hit := RIVAL_ROW.match(text))]
    if len(rivals) != 19:
        sys.exit(f'expected 19 rivals, found {len(rivals)}')

    rows, kinds = [], Counter()
    for paper in rivals:
        path = FIGURES / f'{paper}.failures.csv'
        if not path.exists():
            sys.exit(f'missing {path.name}')
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                row['printed_by'] = paper
                rows.append(row)
        with (FIGURES / f'{paper}.coverage.csv').open(newline='') as handle:
            kinds.update(row['kind'] for row in csv.DictReader(handle))

    no_page = [row for row in rows if not re.search(r'p\.?\s?\d+', row['figures'])]
    if no_page:
        sys.exit(f'{len(no_page)} rows name no page, first: {no_page[0]["printed_by"]} {no_page[0]["method"]}')

    by_rival, others = defaultdict(list), defaultdict(list)
    for row in rows:
        owner = rival_of(row['method'], rivals)
        (by_rival[owner] if owner else others[row['printed_by']]).append(row)

    parts = [HEADER.format(figures=sum(kinds.values()), qualitative=kinds['qualitative'], rows=len(rows))]
    parts.append('## Rivals\n')
    for name in rivals:
        parts.append(f'### {name}\n')
        if name in NO_OWN_METHOD:
            parts.append(NO_OWN_METHOD[name] + '\n')
        if by_rival[name]:
            ordered = sorted(by_rival[name], key=lambda r: (r['dataset'], r['method']))
            parts.append(COLUMNS)
            parts.extend(line(row) for row in ordered)
            parts.append('')
        elif name not in NO_OWN_METHOD:
            sys.exit(f'{name} has no rows and no stated reason')
    parts.append('## Other methods, by the paper that printed them\n')
    for name in rivals:
        if others[name]:
            parts.append(f'### Printed by {name}\n')
            parts.append(COLUMNS)
            parts.extend(line(row) for row in sorted(others[name], key=lambda r: (r['method'], r['dataset'])))
            parts.append('')

    OUT.write_text('\n'.join(parts).rstrip() + '\n')
    placed = sum(len(v) for v in by_rival.values()) + sum(len(v) for v in others.values())
    print(f'{OUT.name}: {placed} of {len(rows)} rows placed, {sum(len(v) for v in by_rival.values())} under a rival, '
          f'{sum(1 for n in rivals if by_rival[n])} rivals with rows, {len(NO_OWN_METHOD)} with no method of their own')
    return 0 if placed == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
