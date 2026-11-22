"""Check the limits rows against numbers.csv and the PDFs, then render targets/limits.md.

Each row in reads/limits/<Paper N>.csv names the best published number on one benchmark by its id, so
limits.md takes every value from numbers.csv and none is typed by hand. The check fails when:
  - number_id is not a published, twice-read row, or a better published row exists on the same
    table, dataset, task, metric and protocol;
  - a number in `breaks_at` is not the value of a row named in `break_ids`;
  - a number in `cost` is not printed on the PDF page named in `cost_source`.

Usage: python3 -I artifacts/notes/progress/targets/reads/build_limits.py [--check]
"""
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMITS = HERE / 'limits'
OUT = HERE.parent / 'limits.md'
PDFS = Path(os.environ.get('CLAUDE_JOB_DIR', '/home/molef/.claude/jobs/f541e553')) / 'tmp' / 'rivals.tsv'
COLUMNS = ['paper', 'family', 'dataset', 'task', 'metric', 'number_id', 'outside_ids', 'dispute', 'breaks_at',
           'break_ids', 'seen_in', 'cost', 'cost_source']
NO_NUMBER = 'no rival number'
NUMBER = re.compile(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])')
SAME = ('source_table', 'dataset', 'task', 'metric', 'protocol')
WIDE = ('source_table', 'dataset', 'task', 'metric')


def numbers_in(text):
    return [hit.replace(',', '') for hit in NUMBER.findall(text or '')]


def same_number(a, b):
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def page_text(paper, page, pdfs, cache={}):
    key = (paper, page)
    if key not in cache:
        out = subprocess.run(['pdftotext', '-f', str(page), '-l', str(page), pdfs[paper], '-'],
                             capture_output=True, text=True).stdout
        cache[key] = out.replace('−', '-').replace(',', '')
    return cache[key]


def split_ids(text):
    return [part.strip() for part in (text or '').split(';') if part.strip()]


def check(rows, numbers, pdfs):
    problems = []
    groups, wide = defaultdict(list), defaultdict(list)
    for row in numbers.values():
        if row['kind'] == 'plot':
            continue
        groups[tuple(row[c] for c in SAME)].append(row)
        wide[tuple(row[c] for c in WIDE)].append(row)

    for line, row in rows:
        where = f'{row["_file"]} line {line}'
        for column in ('paper', 'family', 'dataset', 'metric', 'number_id', 'breaks_at'):
            if not row[column].strip():
                problems.append(f'{where}: empty {column}')
        for key in split_ids(row['break_ids']) + split_ids(row['outside_ids']):
            if key not in numbers:
                problems.append(f'{where}: unknown id {key}')

        if row['number_id'] != NO_NUMBER:
            best = numbers.get(row['number_id'])
            if best is None:
                problems.append(f'{where}: unknown number_id {row["number_id"]}')
                continue
            if best['kind'] != 'published':
                problems.append(f'{where}: {best["id"]} is {best["kind"]}, and a target must rest on a published row')
            if not best['second_read']:
                problems.append(f'{where}: {best["id"]} has no second read')
            if best['better'] not in ('higher', 'lower'):
                problems.append(f'{where}: {best["id"]} has no better direction')
            else:
                sign = 1 if best['better'] == 'higher' else -1
                for other in groups[tuple(best[c] for c in SAME)]:
                    if other['kind'] == 'published' and sign * float(other['value']) > sign * float(best['value']):
                        problems.append(f'{where}: {other["id"]} ({other["paper"]} {other["value"]}) beats '
                                        f'{best["id"]} ({best["value"]}) on the same benchmark')
                        break
                # A quoted score carries its own protocol text, so outside leaders are searched table-wide
                # on the same base model: this is what catches CoMat above ToMe in ToMe's Table 1.
                for other in wide[tuple(best[c] for c in WIDE)]:
                    same_model = not other['model'] or not best['model'] or other['model'] == best['model']
                    if (other['kind'] == 'baseline' and same_model
                            and sign * float(other['value']) > sign * float(best['value'])
                            and other['id'] not in split_ids(row['outside_ids'])):
                        problems.append(f'{where}: {other["id"]} ({other["paper"]} {other["value"]}) leads and is '
                                        'not named in outside_ids')
                        break

        allowed = [numbers[k]['value'] for k in split_ids(row['break_ids']) if k in numbers]
        for value in numbers_in(row['breaks_at']):
            if not any(same_number(value, a) or same_number(value.lstrip('-'), a.lstrip('-')) for a in allowed):
                problems.append(f'{where}: {value} in breaks_at is not the value of a row in break_ids')

        if row['cost'].strip():
            source = re.fullmatch(r'([A-Z][\w-]*) p(\d+)', row['cost_source'].strip())
            if not source or source.group(1) not in pdfs:
                problems.append(f'{where}: cost_source must read "<Paper> p<page>", got {row["cost_source"]!r}')
            else:
                text = page_text(source.group(1), source.group(2), pdfs)
                for value in numbers_in(row['cost']):
                    if value not in text:
                        problems.append(f'{where}: {value} in cost is not on {row["cost_source"]}')
    return problems


def cell(text):
    return re.sub(r'\s+', ' ', text or '').replace('|', '\\|').strip()


def describe(row, numbers):
    if row['number_id'] == NO_NUMBER:
        return NO_NUMBER, '', ''
    best = numbers[row['number_id']]
    value = best['value'] + (f' {best["interval"]}' if best['interval'] else '')
    return value, best['paper'], f'{best["source_table"]}, p{best["source_page"]}'


def outside(row, numbers):
    parts = [f'{numbers[k]["paper"]} {numbers[k]["value"]} ({numbers[k]["source_table"]})'
             for k in split_ids(row['outside_ids'])]
    return '; '.join(parts)


def render(rows, numbers):
    parts = ["""# Where each method stands and where it breaks

For each dataset and metric this page names the best published result, who holds it, the condition where the methods
stop working, and what matching the benchmark costs. The targets on the scoreboard are chosen from these rows.

**How to read a row.** A number is only comparable with another on the same table, subset, metric and protocol, so one
dataset can carry several rows. `Printed above it` names a method outside the 22 rivals that the same table shows ahead of
the best rival. `Disputed` means the paper contradicts itself about that number, and a disputed number cannot set a target
until the page settles it. Every value is copied from `numbers.csv`, which holds its page and both reads. The best
published row of every benchmark, including datasets this page does not discuss, is in `reads/limits-all.csv`.
"""]
    by_paper = defaultdict(lambda: defaultdict(list))
    for _, row in rows:
        by_paper[row['paper']][row['family']].append(row)
    for paper in sorted(by_paper):
        parts.append(f'## {paper}\n')
        for family, members in by_paper[paper].items():
            parts.append(f'### {family}\n')
            parts.append('| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |')
            parts.append('|---|---|---|---|---|---|---|')
            for row in members:
                value, holder, where = describe(row, numbers)
                bench = cell(row['dataset']) + (f', {cell(row["task"])}' if row['task'].strip() else '')
                parts.append(f'| {bench} | {cell(row["metric"])} | {value} | {holder} | {where} | '
                             f'{cell(outside(row, numbers))} | {cell(row["dispute"])} |')
            parts.append('')
            seen = []
            for row in members:
                text = cell(row['breaks_at'])
                if row['seen_in'].strip():
                    text += f' Seen in {cell(row["seen_in"])}.'
                if text not in seen:
                    seen.append(text)
            parts.append('**Where it breaks.** ' + ' '.join(seen) + '\n')
            costs = []
            for row in members:
                if row['cost'].strip():
                    text = f'{cell(row["cost"])} ({cell(row["cost_source"])})'
                    if text not in costs:
                        costs.append(text)
            if costs:
                parts.append('**What matching it costs.** ' + ' '.join(costs) + '\n')
    OUT.write_text('\n'.join(parts).rstrip() + '\n')


def catalogue(numbers):
    """Write the best published row of every benchmark in numbers.csv, for the datasets limits.md does not discuss."""
    groups = defaultdict(list)
    for row in numbers.values():
        if row['kind'] == 'published':
            groups[tuple(row[c] for c in SAME)].append(row)
    lines = []
    for key, members in groups.items():
        direction = members[0]['better']
        if direction in ('higher', 'lower'):
            sign = 1 if direction == 'higher' else -1
            best = max(members, key=lambda r: sign * float(r['value']))
        else:
            best = members[0]
        lines.append({'dataset': best['dataset'], 'task': best['task'], 'metric': best['metric'], 'better': direction
                      or 'not stated', 'best_value': best['value'] if direction else '', 'interval': best['interval']
                      if direction else '', 'holder': best['paper'] if direction else '', 'number_id': best['id']
                      if direction else '', 'rival_rows': len(members), 'source_table': best['source_table'],
                      'source_page': best['source_page'], 'protocol': best['protocol']})
    lines.sort(key=lambda r: (r['dataset'], r['metric'], r['source_table']))
    with (HERE / 'limits-all.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lines[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(lines)


def main():
    with (HERE.parent / 'numbers.csv').open(newline='') as handle:
        numbers = {row['id']: row for row in csv.DictReader(handle)}
    pdfs = {line.split('\t')[0]: line.rstrip('\n').split('\t')[2] for line in PDFS.open()}
    files = sorted(LIMITS.glob('*.csv'))
    only = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--only=')), None)
    if only:
        files = [path for path in files if path.name == only]
    if not files:
        sys.exit('no limits files yet')

    rows = []
    for path in files:
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != COLUMNS:
                sys.exit(f'{path.name}: columns are {reader.fieldnames}')
            for line, row in enumerate(reader, 2):
                row['_file'] = path.name
                rows.append((line, row))

    problems = check(rows, numbers, pdfs)
    if problems:
        print(f'MISS: {len(problems)} problems')
        for problem in problems[:50]:
            print(' ', problem)
        return 1
    if '--check' not in sys.argv:
        if only:
            sys.exit('--only is for checking one file, so pass --check with it')
        render(rows, numbers)
        catalogue(numbers)
    families = {row['family'] for _, row in rows}
    print(f'PASS: {len(rows)} rows across {len(families)} datasets from {len(files)} files'
          + ('' if '--check' in sys.argv else f'; wrote {OUT.name}'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
