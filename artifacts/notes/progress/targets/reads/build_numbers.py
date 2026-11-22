"""Join the first reads into targets/numbers.csv, in the order rivals.md lists the papers.

Copies every cell unchanged, so no value can drift between a first read and the joined table.
Stops if a paper is missing or an id repeats.

Usage: python3 -I artifacts/notes/progress/targets/reads/build_numbers.py
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'numbers.csv'
COLUMNS = ('id,kind,paper,arxiv,dataset,task,metric,better,value,interval,samples,model,protocol,'
           'source_page,source_table,read_by,second_read').split(',')
RIVAL_ROW = re.compile(r'^\| ([A-Z][\w-]*) \| \d{4}\.\d{5}v\d+ \|')


def previous_second_reads():
    """Keep any second read already recorded, so a rebuild never erases one."""
    if not OUT.exists():
        return {}
    with OUT.open(newline='') as handle:
        return {row['id']: row['second_read'] for row in csv.DictReader(handle) if row['second_read']}


def main():
    papers = [hit.group(1) for line in (HERE.parent / 'rivals.md').read_text().splitlines()
              if (hit := RIVAL_ROW.match(line))]
    if not papers:
        sys.exit('no papers found in rivals.md')

    kept = previous_second_reads()
    rows = []
    for paper in papers:
        path = HERE / 'first' / f'{paper}.csv'
        if not path.exists():
            sys.exit(f'missing first read: {path.name}')
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != COLUMNS:
                sys.exit(f'{path.name}: columns differ from the required list')
            rows.extend(reader)

    repeated = [key for key, count in Counter(row['id'] for row in rows).items() if count > 1]
    if repeated:
        sys.exit(f'{len(repeated)} repeated ids, first: {repeated[:3]}')

    for row in rows:
        if row['id'] in kept:
            row['second_read'] = kept[row['id']]

    with OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    kinds = Counter(row['kind'] for row in rows)
    print(f'{OUT.name}: {len(rows)} rows from {len(papers)} papers; ' +
          ', '.join(f'{kind} {count}' for kind, count in sorted(kinds.items())) +
          f'; second reads kept {sum(bool(row["second_read"]) for row in rows)}')


if __name__ == '__main__':
    main()
