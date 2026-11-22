"""Check one paper's figure read against the figure inventory and the column rules.

Usage: python3 -I artifacts/notes/progress/targets/reads/check_figure_read.py <Paper>
"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'
RIVAL_ROW = re.compile(r'^\| ([A-Z][\w-]*) \| (\d{4}\.\d{5})v\d+ \| \[[^\]]+\]\(([^)]+)\) \| (\d+) \|')
COVERAGE = ['figure', 'page', 'kind', 'reason']
OBSERVATIONS = ['figure', 'page', 'dataset', 'method', 'prompt_or_pair', 'outcome', 'ladder', 'how']
FAILURES = ['method', 'dataset', 'succeeds', 'fails', 'on_which', 'failure_line', 'figures']
KINDS = {'qualitative', 'plot', 'diagram', 'other'}
OUTCOMES = {'succeeds', 'fails', 'mixed'}
LADDER = {'one thing', 'fused hybrid', 'two bodies or heads', 'two clear things', 'not applicable'}


def page_count(name):
    for line in (HERE.parent / 'rivals.md').read_text().splitlines():
        hit = RIVAL_ROW.match(line)
        if hit and hit.group(1) == name:
            return int(hit.group(4))
    return None


def load(path, columns, problems):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            problems.append(f'{path.name}: columns are {reader.fieldnames}')
        return list(reader)


def main(name):
    pages = page_count(name)
    if pages is None:
        sys.exit(f'{name} is not a rival in rivals.md')
    problems = []
    paths = {key: FIGURES / f'{name}.{key}.csv' for key in ('coverage', 'observations', 'failures')}
    for path in paths.values():
        if not path.exists():
            problems.append(f'missing {path.name}')
    if problems:
        print('MISS\n  ' + '\n  '.join(problems))
        return 1

    coverage = load(paths['coverage'], COVERAGE, problems)
    observations = load(paths['observations'], OBSERVATIONS, problems)
    failures = load(paths['failures'], FAILURES, problems)

    with (HERE / 'figure-inventory.csv').open(newline='') as handle:
        listed = [row['figure'] for row in csv.DictReader(handle) if row['paper'] == name]
    covered = {row['figure']: row for row in coverage}
    for label in listed:
        if label not in covered:
            problems.append(f'Figure {label} is in the inventory and not in the coverage file')
    for row in coverage:
        if row['kind'] not in KINDS:
            problems.append(f'Figure {row["figure"]}: kind {row["kind"]!r}')
        if not row['page'].isdigit() or not 1 <= int(row['page']) <= pages:
            problems.append(f'Figure {row["figure"]}: page {row["page"]!r} outside 1 to {pages}')
        if row['kind'] != 'qualitative' and not row['reason'].strip():
            problems.append(f'Figure {row["figure"]}: a {row["kind"]} figure needs a reason')

    qualitative = {row['figure'] for row in coverage if row['kind'] == 'qualitative'}
    seen = {row['figure'] for row in observations}
    for label in sorted(qualitative - seen):
        problems.append(f'Figure {label} is qualitative and has no observation')
    for label in sorted(seen - set(covered)):
        problems.append(f'Figure {label} has observations and no coverage line')
    for number, row in enumerate(observations, 2):
        where = f'observations line {number}'
        for column in OBSERVATIONS:
            if not (row.get(column) or '').strip():
                problems.append(f'{where}: empty {column}')
        if row.get('outcome') not in OUTCOMES:
            problems.append(f'{where}: outcome {row.get("outcome")!r}')
        if row.get('ladder') not in LADDER:
            problems.append(f'{where}: ladder {row.get("ladder")!r}')

    if qualitative and not failures:
        problems.append('the paper has qualitative figures and no failures rows')
    shown = {(row['method'], row['dataset']) for row in observations}
    written = {(row['method'], row['dataset']) for row in failures}
    for method, dataset in sorted(shown - written):
        problems.append(f'no failures row for {method} on {dataset}')
    for number, row in enumerate(failures, 2):
        where = f'failures line {number}'
        for column in FAILURES:
            if not (row.get(column) or '').strip():
                problems.append(f'{where}: empty {column}')
        if len(row.get('failure_line') or '') > 140:
            problems.append(f'{where}: failure_line is {len(row["failure_line"])} characters, limit 140')

    if problems:
        print(f'MISS: {len(problems)} problems')
        for problem in problems[:40]:
            print(' ', problem)
        return 1
    print(f'PASS {name}: {len(coverage)} figures ({len(qualitative)} qualitative), '
          f'{len(observations)} observations, {len(failures)} failure rows')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
