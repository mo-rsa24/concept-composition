"""Check one first-read file against the table inventory and the column rules.

Usage: python3 -I artifacts/notes/progress/targets/reads/check_first_read.py <Paper>
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLUMNS = ('id,kind,paper,arxiv,dataset,task,metric,better,value,interval,samples,model,protocol,'
           'source_page,source_table,read_by,second_read').split(',')
NUMBER = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
REQUIRED = ('id', 'kind', 'paper', 'arxiv', 'dataset', 'metric', 'value', 'protocol', 'source_page',
            'source_table', 'read_by')
RIVAL_ROW = re.compile(r'^\| ([A-Z][\w-]*) \| (\d{4}\.\d{5})v\d+ \| \[[^\]]+\]\(([^)]+)\) \| (\d+) \|')


def rival(name):
    for line in (HERE.parent / 'rivals.md').read_text().splitlines():
        hit = RIVAL_ROW.match(line)
        if hit and hit.group(1) == name:
            return hit.group(2), int(hit.group(4))
    return None, None


def main(name):
    problems = []
    arxiv, page_count = rival(name)
    if arxiv is None:
        sys.exit(f'{name} is not a rival in rivals.md')

    data = HERE / 'first' / f'{name}.csv'
    coverage = HERE / 'first' / f'{name}.coverage.csv'
    notes = HERE / 'first' / f'{name}.notes.md'
    for path in (data, coverage, notes):
        if not path.exists():
            problems.append(f'missing {path.name}')
    if problems:
        print('MISS\n  ' + '\n  '.join(problems))
        return 1

    with data.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            problems.append(f'columns differ from the required list: {reader.fieldnames}')
        rows = list(reader)

    seen = set()
    for number, row in enumerate(rows, 2):
        where = f'line {number} ({row.get("id")})'
        for column in REQUIRED:
            if not (row.get(column) or '').strip():
                problems.append(f'{where}: empty {column}')
        if row.get('id') in seen:
            problems.append(f'{where}: duplicate id')
        seen.add(row.get('id'))
        if row.get('kind') not in ('published', 'baseline', 'plot'):
            problems.append(f'{where}: kind {row.get("kind")!r}')
        if row.get('better') not in ('higher', 'lower', ''):
            problems.append(f'{where}: better {row.get("better")!r}')
        if not NUMBER.match(row.get('value') or ''):
            problems.append(f'{where}: value {row.get("value")!r} is not a bare number')
        if row.get('arxiv') != arxiv:
            problems.append(f'{where}: arxiv {row.get("arxiv")!r}, expected {arxiv}')
        if not (row.get('source_page') or '').isdigit() or not 1 <= int(row['source_page']) <= page_count:
            problems.append(f'{where}: source_page {row.get("source_page")!r} outside 1 to {page_count}')
        if not (row.get('source_table') or '').startswith(f'{name} '):
            problems.append(f'{where}: source_table {row.get("source_table")!r} must start with "{name} "')
        if row.get('second_read'):
            problems.append(f'{where}: second_read must stay empty in a first read')

    with coverage.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['table', 'page', 'status', 'rows', 'reason']:
            problems.append(f'coverage columns: {reader.fieldnames}')
        covered = {line['table']: line for line in reader}

    with (HERE / 'table-inventory.csv').open(newline='') as handle:
        listed = [line['table'] for line in csv.DictReader(handle) if line['paper'] == name]
    for label in listed:
        if label not in covered:
            problems.append(f'Table {label} is in the inventory and not in the coverage file')

    for label, line in covered.items():
        count = sum(row.get('source_table') == f'{name} Table {label}' for row in rows)
        if line['status'] == 'read':
            if not line['rows'].isdigit() or int(line['rows']) != count or count == 0:
                problems.append(f'Table {label}: coverage says {line["rows"]} rows, the CSV has {count}')
        elif line['status'] == 'no results':
            if count or not line['reason'].strip():
                problems.append(f'Table {label}: "no results" needs a reason and no rows (has {count})')
        else:
            problems.append(f'Table {label}: status {line["status"]!r}')

    if problems:
        print(f'MISS: {len(problems)} problems')
        for problem in problems[:40]:
            print(' ', problem)
        return 1
    read = sum(line['status'] == 'read' for line in covered.values())
    print(f'PASS {name}: {len(rows)} rows, {read} tables read, {len(covered) - read} with no results')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
