"""Compare the blind second read with the first read, cell by cell.

A cell agrees only when the two values are the same text and the two intervals match once spaces
are removed. Agreement is written to `second_read` in numbers.csv. Every other cell goes to
reads/disagreements.csv for someone to open the page, and its `second_read` stays empty.

Usage: python3 -I artifacts/notes/progress/targets/reads/compare_second_read.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBERS = HERE.parent / 'numbers.csv'
SECOND = HERE / 'second'
DISAGREE = HERE / 'disagreements.csv'
SETTLED = HERE / 'settled.csv'
LABEL = 'gpt-6-astra via codex, blind, agrees'
COLUMNS = ['id', 'source_page', 'source_table', 'paper', 'dataset', 'task', 'metric',
           'first_value', 'second_value', 'first_interval', 'second_interval', 'why', 'note']


def squash(text):
    return ''.join((text or '').split())


def main():
    with NUMBERS.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    by_id = {row['id']: row for row in rows}

    settled = {}
    if SETTLED.exists():
        with SETTLED.open(newline='') as handle:
            settled = {row['id']: row for row in csv.DictReader(handle)}

    second = {}
    retries = sorted((SECOND / 'retry').glob('*.csv')) if (SECOND / 'retry').exists() else []
    for path in sorted(SECOND.glob('*.csv')) + retries:
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['id'] not in by_id:
                    sys.exit(f'{path.name}: unknown id {row["id"]}')
                second[row['id']] = row

    outcome = Counter()
    open_cells = []
    for key, reply in second.items():
        row = by_id[key]
        if key in settled:
            row['second_read'] = settled[key]['second_read']
            outcome['settled on the page'] += 1
            continue
        if not reply['value']:
            why = 'second reader could not place the cell'
        elif reply['value'] != row['value']:
            why = 'value differs'
        elif squash(reply['interval']) != squash(row['interval']):
            why = 'interval differs'
        else:
            row['second_read'] = LABEL
            outcome['agree'] += 1
            continue
        row['second_read'] = ''
        outcome[why] += 1
        open_cells.append({'id': key, 'source_page': row['source_page'],
                           'source_table': row['source_table'], 'paper': row['paper'],
                           'dataset': row['dataset'], 'task': row['task'], 'metric': row['metric'],
                           'first_value': row['value'], 'second_value': reply['value'],
                           'first_interval': row['interval'], 'second_interval': reply['interval'],
                           'why': why, 'note': reply['note']})

    with NUMBERS.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with DISAGREE.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(open_cells)

    print(f'{len(second)} cells compared: ' + ', '.join(f'{k} {v}' for k, v in outcome.most_common()))
    print(f'{len(open_cells)} cells to open on the page, listed in {DISAGREE.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
