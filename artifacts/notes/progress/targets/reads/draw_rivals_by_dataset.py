"""Draw which rival is scored on which candidate dataset, from the first reads.

A filled dot means the paper prints that dataset in its own tables. A ring means another paper's
table measured that rival's method on it. A rival whose first read has not landed uses the datasets
stated in rivals.md and is labelled as still being read.

Usage: ~/.claude/venvs/figures/bin/python -I artifacts/notes/progress/targets/reads/draw_rivals_by_dataset.py
"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'rivals-by-dataset.png'
INK, MUTED, RULE, BAND = '#1f2328', '#6b7280', '#d9dce1', '#f3f4f6'

COLUMNS = ['CLEVR', 'Colored\nMNIST', 'CelebA', 'Shapes3D', 'CIFAR-10\nsplit',
           'Attend-and-\nExcite', 'T2I-\nCompBench', 'GenEval', 'RareBench', 'COCO']
KEYS = [('clevr', 0), ('mnist', 1), ('celeba', 2), ('shapes3d', 3), ('cifar', 4), ('attend', 5),
        ('t2i', 6), ('geneval', 7), ('rarebench', 8), ('coco', 9)]
STATED = {'LiftScores': {0, 5}, 'LogicalGuidance': {1, 2, 3}, 'CountLoop': {6, 9}}
ALSO_PAPER_2 = {'SuperDiff', 'LiftScores'}


def column(dataset):
    text = dataset.lower()
    for key, index in KEYS:
        if key in text:
            return index
    return None


def rivals_by_paper():
    paper, groups = None, {'Paper 1': [], 'Paper 2': []}
    for line in (HERE.parent / 'rivals.md').read_text().splitlines():
        if line.startswith('## Paper 1'):
            paper = 'Paper 1'
        elif line.startswith('## Paper 2'):
            paper = 'Paper 2'
        elif line.startswith('## '):
            paper = None
        hit = re.match(r'^\| ([A-Z][\w-]*) \| \d{4}\.\d{5}v\d', line)
        if hit and paper:
            groups[paper].append(hit.group(1))
    return groups


def cells(rivals):
    own, ring, pending = {r: set() for r in rivals}, {r: set() for r in rivals}, set()
    for rival in rivals:
        path = HERE / 'first' / f'{rival}.csv'
        if not path.exists():
            own[rival] |= STATED.get(rival, set())
            pending.add(rival)
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['kind'] == 'plot':
                    continue
                index = column(row['dataset'])
                if index is None:
                    continue
                own[rival].add(index)
                if row['paper'] in ring and row['paper'] != rival:
                    ring[row['paper']].add(index)
    return own, {r: ring[r] - own[r] for r in rivals}, pending


def main():
    groups = rivals_by_paper()
    rivals = groups['Paper 1'] + groups['Paper 2']
    own, ring, pending = cells(rivals)
    size = lambda r: len(own[r] | ring[r])
    ordered = {k: sorted(v, key=lambda r: (-size(r), rivals.index(r))) for k, v in groups.items()}

    rows, y, spans = [], 0, {}
    for name, members in ordered.items():
        start = y
        for rival in members:
            rows.append((rival, y))
            y += 1
        spans[name] = (start, y - 1)
        y += 0.9
    height = y - 0.9

    fig, ax = plt.subplots(figsize=(12.2, 10.6), dpi=170)
    fig.patch.set_facecolor('white')
    ax.add_patch(plt.Rectangle((4.5, -3.62), 5.0, height + 3.77, facecolor=BAND, edgecolor='none', zorder=0))
    ax.text(2.0, -3.32, 'datasets a model is trained on', ha='center', va='center', fontsize=10.5, color=MUTED)
    ax.text(7.0, -3.32, 'prompts for a pretrained text-to-image model', ha='center', va='center', fontsize=10.5,
            color=MUTED)
    for rival, row in rows:
        ax.plot([-0.5, 9.5], [row, row], color=RULE, lw=0.6, zorder=1)
        notes = (['also Paper 2'] if rival in ALSO_PAPER_2 else []) + (['being read'] if rival in pending else [])
        label = rival + (f' ({", ".join(notes)})' if notes else '')
        ax.text(-0.72, row, label, ha='right', va='center', fontsize=10.5, color=INK)
        for x in own[rival]:
            ax.scatter(x, row, s=150, color=INK, zorder=3, linewidths=0)
        for x in ring[rival]:
            ax.scatter(x, row, s=130, facecolor='white', edgecolor=INK, linewidths=1.7, zorder=3)
        if not own[rival] and not ring[rival]:
            ax.text(-0.38, row, 'none of these datasets', ha='left', va='center', fontsize=9.5, color=MUTED,
                    style='italic', bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=2)

    for name, (first, last) in spans.items():
        ax.text(-5.05, (first + last) / 2, name, rotation=90, ha='center', va='center', fontsize=12,
                fontweight='bold', color=INK)
        ax.plot([-4.82, -4.82], [first - 0.35, last + 0.35], color=INK, lw=1.4, clip_on=False)

    for x, title in enumerate(COLUMNS):
        count = sum(x in own[r] or x in ring[r] for r in rivals)
        ax.text(x, -1.0, title, ha='center', va='bottom', fontsize=10, color=INK, linespacing=1.05)
        ax.text(x, -2.25, str(count), ha='center', va='bottom', fontsize=15, fontweight='bold', color=INK)
    ax.text(-0.72, -2.2, 'rivals scored on it', ha='right', va='bottom', fontsize=10, color=MUTED)

    ax.scatter(-0.2, -4.2, s=120, color=INK, clip_on=False)
    ax.text(0.0, -4.2, 'in its own tables', va='center', fontsize=10.5, color=INK)
    ax.scatter(2.3, -4.2, s=105, facecolor='white', edgecolor=INK, linewidths=1.7, clip_on=False)
    ax.text(2.5, -4.2, "in another paper's table", va='center', fontsize=10.5, color=INK)
    ax.text(9.5, -4.2, 'a shared column can still mean different protocols', ha='right', va='center', fontsize=10.5,
            color=MUTED, style='italic')
    ax.text(-5.2, -5.0, 'Which rival is scored on which dataset', fontsize=17, fontweight='bold', color=INK)

    ax.set_xlim(-5.3, 9.6)
    ax.set_ylim(height + 0.2, -5.5)
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT, facecolor='white')

    for x, title in enumerate(COLUMNS):
        print(title.replace('\n', ' '), sum(x in own[r] for r in rivals), 'own,',
              sum(x in ring[r] for r in rivals), 'ring')
    print('rivals drawn', len(rows), '| still being read', sorted(pending))


if __name__ == '__main__':
    main()
