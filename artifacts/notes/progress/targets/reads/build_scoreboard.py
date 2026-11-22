"""Write the scoreboard's targets and side-by-side rows from the chosen headlines.

Every copied field comes from the number's row in numbers.csv, so a target cannot drift from its
page. The target itself is left out, because the drawing script computes it. Anything already
measured on a row (our result, a reproduced number, side-by-side counts) is kept on a rebuild.

Usage: python3 -I artifacts/notes/progress/targets/reads/build_scoreboard.py
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = HERE.parent
BOARD = TARGETS.parent / 'scoreboard.json'
KEEP_TARGET = ('reproduced', 'reproduced_id', 'ours', 'ours_interval', 'held_out_question')
KEEP_SIDE = ('pairs', 'better', 'tied', 'worse', 'why', 'skip_question')


def bound(text):
    return float(text) if text.strip() else None


def whole(value):
    return int(value) if float(value).is_integer() else float(value)


def interval(row):
    hit = re.fullmatch(r'±\s*([\d.]+)', row['interval'].strip())
    if not hit:
        return None
    middle, half = float(row['value']), float(hit.group(1))
    return [round(middle - half, 6), round(middle + half, 6)]


def main():
    with (TARGETS / 'numbers.csv').open(newline='') as handle:
        numbers = {row['id']: row for row in csv.DictReader(handle)}
    data = json.loads(BOARD.read_text())
    old_targets = {r['number_id']: r for r in data.get('targets', [])}
    old_sides = {(r['paper'], r['dataset'], r['rival']): r for r in data.get('qualitative', [])}

    targets = []
    with (HERE / 'headlines.csv').open(newline='') as handle:
        for line in csv.DictReader(handle):
            row = numbers.get(line['number_id'])
            if row is None:
                sys.exit(f'unknown number_id {line["number_id"]}')
            entry = {'paper': line['paper'], 'dataset': row['dataset'], 'metric': row['metric'], 'unit': line['unit'],
                     'better': row['better'], 'rival': row['paper'], 'published': float(row['value']), 'reproduced': None,
                     'step': whole(line['step']), 'range': [bound(line['range_low']), bound(line['range_high'])],
                     'protocol': row['protocol'], 'number_id': row['id'], 'reproduced_id': None, 'ours': None,
                     'ours_interval': None, 'rival_interval': interval(row)}
            previous = old_targets.get(row['id'], {})
            entry.update({key: previous[key] for key in KEEP_TARGET if previous.get(key) is not None})
            targets.append(entry)

    sides = []
    with (HERE / 'side-by-side-rows.csv').open(newline='') as handle:
        for line in csv.DictReader(handle):
            if len(line['failure']) > 140:
                sys.exit(f'{line["rival"]} on {line["dataset"]}: failure is {len(line["failure"])} characters')
            entry = {'paper': line['paper'], 'dataset': line['dataset'], 'rival': line['rival'],
                     'failure': line['failure'], 'pairs': 0, 'better': 0, 'tied': 0, 'worse': 0, 'why': ''}
            previous = old_sides.get((line['paper'], line['dataset'], line['rival']), {})
            entry.update({key: previous[key] for key in KEEP_SIDE if key in previous})
            sides.append(entry)

    data['targets'], data['qualitative'] = targets, sides
    BOARD.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(f'{BOARD.name}: {len(targets)} targets, {len(sides)} side-by-side rows')
    return 0


if __name__ == '__main__':
    sys.exit(main())
