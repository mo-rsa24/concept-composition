"""The score-path family for the zoom ladder's two styles, kept beside this ladder.

Importing this module registers family "score-path" in ~/.claude/skills/zoom-ladder/scripts/style_samples.py's FAMILIES
at run time (the skill's file is not edited). In the annotated-explainer style it draws this ladder's overview: one real
digit canvas and one caption followed through the frozen digit U-Net's three calls, the score path, three encoders and a
planned shared space, with the U-Net opened in an inset. In any other style it draws a small sample of the family.

Run from this folder:
    PYTHONDONTWRITEBYTECODE=1 python3 score_path_family.py overview   # overview/ov0-the-score-path-as-a-third-space.{png,svg}
    PYTHONDONTWRITEBYTECODE=1 python3 score_path_family.py sample     # style-sample-score-path.png, one panel per style

Both refuse to save when a label overlaps another label or a line runs through a label.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.path.expanduser('~/.claude/skills/zoom-ladder/scripts'))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

import style_samples as ss  # noqa: E402
from annotated_explainer import ANNOTATED, bracket, card, inset, line, strip, trace, trapezoid  # noqa: E402

N = json.loads((HERE / 'numbers' / 'numbers.json').read_text())
SCORE, SCORE_EDGE, SCORE_INK = '#f2b8c6', '#c76b84', '#9c3f5a'
CANVAS = HERE / 'overview' / 'canvas-3-7-native.png'


def _canvas(ax, x, y, w, s, sqrt_abar, noise_seed=10):
    """The real canvas noised with the real signal scale of one step and one numpy draw."""
    from PIL import Image
    clean = np.asarray(Image.open(CANVAS).convert('L'), dtype=float) / 127.5 - 1
    z = np.random.default_rng(noise_seed).standard_normal(clean.shape)
    xt = sqrt_abar * clean + np.sqrt(1 - sqrt_abar ** 2) * z
    ax.imshow(np.clip((xt + 3) / 6, 0, 1), extent=(x, x + w, y, y + w / 2), cmap='Greys_r', vmin=0, vmax=1,
              zorder=3, interpolation='nearest')
    ax.add_patch(Rectangle((x, y), w, w / 2, fill=False, ec=s['cin'], lw=1.8, zorder=4))


def _clean(ax, x, y, w, s):
    from PIL import Image
    img = np.asarray(Image.open(CANVAS).convert('L'), dtype=float) / 255.0
    ax.imshow(img, extent=(x, x + w, y, y + w / 2), cmap='Greys_r', vmin=0, vmax=1, zorder=3, interpolation='nearest')
    ax.add_patch(Rectangle((x, y), w, w / 2, fill=False, ec=s['cin'], lw=1.8, zorder=4))


def _dashed_trap(ax, x0, x1, y, h0, h1, lines_, s):
    ax.add_patch(Polygon([(x0, y + h0 / 2), (x1, y + h1 / 2), (x1, y - h1 / 2), (x0, y - h0 / 2)], closed=True,
                         edgecolor=SCORE_EDGE, facecolor='#fbe6ec', lw=1.6, ls=(0, (4, 3)), zorder=3))
    for i, (t, weight) in enumerate(lines_):
        ax.text((x0 + x1) / 2, y + (len(lines_) - 1) * 0.15 - i * 0.3, t, ha='center', va='center',
                fontsize=10.5 if weight == 'bold' else 8.6, weight=weight, color=SCORE_INK, family=s['font'], zorder=4)


def _tiles(ax, x, y, w, h, n, colour, dx=0.07):
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + k * dx, y + k * dx), w, h, facecolor=colour, edgecolor=SCORE_EDGE, lw=0.6,
                               alpha=0.35 + 0.65 * (1 - k / n), zorder=3))


def fam_score_path_overview(ax, s):
    U, W4, win = N['unet'], N['w4_text_with_latent'], N['sampler']['window_steps_0_to_15']
    grey, grey_fill = s['fixed'], '#e8ebee'
    for x0, x1, name in ((0.2, 2.95, 'What goes in'), (3.2, 6.2, 'Digit U-Net, three calls'), (6.45, 9.0, 'The score path'),
                         (9.25, 12.05, 'Three encoders'), (12.3, 15.95, 'One space, two tests')):
        bracket(ax, x0, x1, 9.3, name, s)
    # what goes in
    card(ax, 0.2, 7.65, 2.75, 0.95, 'the digit three and the\ndigit seven, apart', s, size=9.4, weight='bold')
    ax.text(1.57, 7.38, f"one of {W4['caption_bank']} captions (W4)", ha='center', va='center', fontsize=8.0, color=s['ink'])
    _clean(ax, 0.2, 5.55, 2.0, s)
    ax.text(1.2, 6.82, 'x0: a real 7 and 3', ha='center', va='center', fontsize=8.6, color=s['ink'])
    _canvas(ax, 0.2, 3.9, 2.0, s, win[0]['sqrt_abar'])
    ax.text(1.5, 3.62, 'x_t at step 0,', ha='left', va='center', fontsize=8.2, color=s['ink'])
    ax.text(1.5, 3.36, f"signal x {win[0]['sqrt_abar']}", ha='left', va='center', fontsize=8.2, color=s['ink'])
    line(ax, [(1.2, 5.5), (1.2, 4.95)], s['edge'])
    ax.add_patch(Rectangle((0.38, 4.1), 0.32, 0.45, fill=False, ec=s['cout'], lw=1.8, zorder=6))
    # digit U-Net, three calls
    trapezoid(ax, 3.55, 5.35, 4.55, 2.2, 1.2, grey, grey_fill,
              [('Digit U-Net', 'bold'), (f"{U['weights']:,} weights", 'normal'), ('frozen', 'normal')], s)
    line(ax, [(2.24, 4.4), (3.52, 4.4)], s['edge'])
    for i, (lab, y) in enumerate((('label 10: no digit', 6.1), ('label 3', 6.55), ('label 7', 7.0))):
        ax.text(4.45, y, lab, ha='center', va='center', fontsize=8.4, color=s['ink'], zorder=4,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=grey, lw=0.8))
    line(ax, [(4.45, 5.9), (4.45, 5.68)], s['edge'], ms=8)
    for i, (lab, y) in enumerate(((r'$\epsilon_\varnothing$', 5.2), (r'$\epsilon_3$', 4.55), (r'$\epsilon_7$', 3.9))):
        line(ax, [(5.38, 4.55 + (y - 4.55) * 0.35), (5.62, y)], s['edge'], ms=8)
        strip(ax, 5.65, y, 0.5, 0.3, grey, n=4, seed=i)
        ax.text(5.9, y + 0.3, lab, ha='center', va='center', fontsize=10, color=s['ink'])
    # the score path
    ax.text(7.72, 6.85, r'$d_t(c) = \epsilon_c - \epsilon_\varnothing$', ha='center', va='center', fontsize=12.5, color=s['ink'])
    for j, (lab, y) in enumerate(((r'$d_t(3)$', 5.85), (r'$d_t(7)$', 4.65))):
        _tiles(ax, 6.55, y - 0.25, 0.7, 0.35, 8, SCORE)
        ax.text(6.55 + 0.35, y + 0.55, lab, ha='left', va='center', fontsize=10, color=SCORE_INK)
    line(ax, [(6.2, 4.55), (6.5, 4.55)], s['edge'], ms=8)
    line(ax, [(6.2, 4.75), (6.52, 5.6)], s['edge'], ms=8)
    ax.text(8.05, 6.28, 'steps 0\nto 15', ha='center', va='center', fontsize=8.2, color=SCORE_INK)
    _tiles(ax, 7.75, 3.65, 0.95, 0.5, 8, SCORE, dx=0.06)
    ax.text(8.2, 3.35, 'S(3) + S(7): what', ha='center', va='center', fontsize=8.2, color=SCORE_INK)
    ax.text(8.2, 3.08, 'the product adds', ha='center', va='center', fontsize=8.2, color=SCORE_INK)
    line(ax, [(7.62, 4.7), (7.9, 4.4)], SCORE_EDGE, ms=8)
    ax.text(8.32, 5.05, f"16 x 32 x 64\n= {N['score_path']['numbers']:,}", ha='center', va='center', fontsize=8.0, color=s['ink'])
    # three encoders
    trapezoid(ax, 9.3, 11.25, 7.9, 1.5, 0.95, grey, grey_fill, [('Text encoder', 'bold'), ('CLIP-L, frozen', 'normal')], s)
    line(ax, [(2.99, 8.05), (9.27, 8.05)], s['edge'])
    trapezoid(ax, 9.3, 11.25, 5.85, 1.5, 0.95, s['chid'], '#bfe5ca', [('W4 encoder', 'bold'), ('Loop 9, trained', 'normal')], s)
    line(ax, [(2.24, 4.75), (2.6, 4.75), (2.6, 7.45), (8.95, 7.45), (8.95, 5.95), (9.27, 5.95)], s['edge'])
    _dashed_trap(ax, 9.3, 11.25, 3.95, 1.5, 0.95, [('Score-path', 'bold'), ('encoder, planned', 'normal')], s)
    line(ax, [(8.92, 3.95), (9.27, 3.95)], SCORE_EDGE)
    for y, colour, seed in ((7.9, s['cin'], 1), (5.85, s['chid'], 2), (3.95, SCORE_EDGE, 3)):
        strip(ax, 11.4, y, 0.6, 0.28, colour, n=5, seed=seed)
    ax.text(10.27, 2.92, 'illustrative, not yet designed', ha='center', va='center', fontsize=8.0, color=SCORE_INK,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=SCORE_EDGE, lw=0.8))
    # one space, two tests
    ax.add_patch(FancyBboxPatch((12.35, 4.55), 3.55, 4.0, boxstyle='round,pad=0.02,rounding_size=0.12', fc='#fdf3f6',
                                ec=SCORE_EDGE, lw=1.3, ls=(0, (4, 3)), zorder=1))
    ax.text(14.12, 8.25, 'one shared space (planned)', ha='center', va='center', fontsize=10, weight='bold', color=SCORE_INK)
    for (x, y), colour, lab in (((13.0, 7.1), s['cin'], 'e(3)'), ((15.15, 7.2), s['cin'], 'e(7)'),
                                ((14.0, 6.45), s['cin'], 'e(joint)'), ((14.25, 5.75), SCORE_EDGE, 'z(S(3)+S(7))'),
                                ((13.05, 5.2), s['chid'], 'canvas')):
        ax.plot([x], [y], marker='o', ms=9, color=colour, zorder=5)
        ax.text(x + 0.17, y + 0.02, lab, ha='left', va='center', fontsize=8.4, color=s['ink'], zorder=5)
    ax.text(14.12, 4.8, 'dots placed by hand: illustrative', ha='center', va='center', fontsize=7.8, color=grey)
    for y, colour in ((7.9, s['cin']), (5.85, s['chid']), (3.95, SCORE_EDGE)):
        line(ax, [(12.04, y), (12.32, min(max(y, 4.75), 8.3))], colour, ms=8)
    base = 1.25
    for i, (x, lab) in enumerate(((12.9, 'test 1\nshuffled'), (13.7, 'test 1:\njoint top-1'), (14.6, 'test 2\nshuffled'), (15.4, 'test 2:\ngap cosine'))):
        ax.add_patch(Rectangle((x - 0.3, base), 0.6, 1.2, facecolor='none', edgecolor=s['ink'],
                               lw=0.9, ls=(0, (3, 2)), zorder=3))
        ax.text(x, base + 0.6, '?', ha='center', va='center', fontsize=13, color=grey)
        ax.text(x, base - 0.32, lab, ha='center', va='center', fontsize=7.6, color=s['ink'], linespacing=1.1)
    ax.plot([12.45, 15.85], [base, base], color=s['ink'], lw=1.1, zorder=3)
    ax.text(14.15, 3.55, 'not run: hollow bars, no heights', ha='center', va='center', fontsize=8.4, color=grey)
    ax.text(14.15, 3.2, 'bars written in step X4 first', ha='center', va='center', fontsize=8.4, color=grey)
    # inset: inside the U-Net
    inset(ax, 3.25, 0.35, 4.65, 2.35, 'Inside the digit U-Net', (3.55, 3.45), (5.35, 3.95), s)
    lv = U['levels']
    xs = [(3.45, lv[0], 0.95), (4.0, lv[1], 0.62), (4.55, lv[2], 0.36), (5.1, lv[2], 0.36), (5.65, lv[1], 0.62), (6.2, lv[0], 0.95)]
    for x, shp, h in xs:
        ax.add_patch(Rectangle((x, 1.55 - h / 2 + (0.95 - h) * 0.0), 0.42, h, facecolor=s['chid'],
                               alpha=0.25 + 0.5 * shp[0] / 128, lw=0, zorder=3))
    for x, shp, h in xs[:3]:
        ax.text(x + 0.21, 1.55 - h / 2 - 0.2, 'x'.join(map(str, shp)), ha='center', va='center', fontsize=6.6, color=s['ink'], zorder=5)
    ax.text(5.03, 2.2, 'attention at 8 x 16', ha='center', va='center', fontsize=7.4, color=s['ink'], zorder=5)
    ax.text(7.25, 1.95, 'emb = MLP(t)\n+ table[c]', ha='center', va='center', fontsize=7.6, color=s['ink'], zorder=5)
    ax.text(7.25, 1.25, f"11 rows:\n0-9 and none", ha='center', va='center', fontsize=7.2, color=s['ink'], zorder=5)
    trace(ax, (0.70, 4.1), (3.45, 1.1), s['cout'])
    return (-0.05, 16.1, 0.05, 10.05)


def fam_score_path(ax, s):
    """A small sample of the family: sixteen d tiles stacked, a planned encoder, a strip, and its score against a caption."""
    if s.get('layout') == 'annotated':
        return fam_score_path_overview(ax, s)
    for k in range(7, -1, -1):
        ax.add_patch(Rectangle((0.1 + k * 0.06, 1.2 + k * 0.06), 0.9, 0.5, facecolor=SCORE, edgecolor=SCORE_EDGE, lw=0.6,
                               alpha=0.35 + 0.65 * (1 - k / 8)))
    ss.note(ax, 0.75, 0.9, 'd_t, steps 0-15', s, color=s['ink'], size=7.5)
    ax.add_patch(Polygon([(1.7, 2.1), (2.9, 1.85), (2.9, 1.25), (1.7, 1.0)], closed=True, facecolor='#fbe6ec',
                         edgecolor=SCORE_EDGE, lw=1.2, ls=(0, (4, 3))))
    ss.note(ax, 2.3, 1.55, 'planned', s, color=SCORE_INK, size=7.5)
    ss.arrow(ax, (1.45, 1.55), (1.7, 1.55), s)
    for i in range(5):
        ax.add_patch(Rectangle((3.1 + i * 0.18, 1.45), 0.18, 0.2, facecolor=SCORE, alpha=0.3 + 0.14 * i, ec='white', lw=0.5))
    for i in range(5):
        ax.add_patch(Rectangle((3.1 + i * 0.18, 2.6), 0.18, 0.2, facecolor=s.get('text', s['cin']), alpha=0.3 + 0.14 * i, ec='white', lw=0.5))
    ss.note(ax, 3.55, 3.05, 'caption', s, color=s['ink'], size=7.5)
    ss.arrow(ax, (2.9, 1.55), (3.1, 1.55), s)
    ss.note(ax, 3.55, 0.75, r'$\cos(z_S, e_T)/\tau$', s, color=s['ink'], size=9)
    return (-0.1, 4.3, 0.4, 3.4)


ss.FAMILIES['score-path'] = fam_score_path
ss.STYLES['clip-contrastive']['families'] = tuple(ss.STYLES['clip-contrastive']['families']) + ('score-path',)
ANNOTATED['families'] = tuple(ANNOTATED['families']) + ('score-path',)


def check(fig, ax):
    """Refuse a figure where two labels overlap or a drawn line or arrow runs through a label."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(t, t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
    bad = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ba.x0 < bb.x1 - 1 and bb.x0 < ba.x1 - 1 and ba.y0 < bb.y1 - 1 and bb.y0 < ba.y1 - 1:
                bad.append(f'labels overlap: {a.get_text()!r} and {b.get_text()!r}')
    segs = []
    for ln in ax.lines:
        pts = ax.transData.transform(ln.get_xydata())
        segs += [(p, q, 'line') for p, q in zip(pts[:-1], pts[1:]) if len(pts) > 1]
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch):
            v = p.get_path().vertices
            segs += [(a, b, 'arrow') for a, b in zip(v[:-1], v[1:])]
    for p, q, kind in segs:
        for t, bb in boxes:
            for u in np.linspace(0, 1, 40):
                x, y = p + u * (q - p)
                if bb.x0 + 1.5 < x < bb.x1 - 1.5 and bb.y0 + 1.5 < y < bb.y1 - 1.5:
                    bad.append(f'{kind} through label {t.get_text()!r}')
                    break
    if bad:
        raise SystemExit('layout check failed:\n' + '\n'.join(sorted(set(bad))))
    print(f'layout check: {len(boxes)} labels, {len(segs)} segments, 0 overlaps, 0 lines through labels')


def overview():
    s = ANNOTATED
    fig, ax = plt.subplots(figsize=s['panel'], dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg'])
    ax.set_facecolor(s['bg'])
    x0, x1, y0, y1 = fam_score_path_overview(ax, s)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1 + 0.4); ax.set_aspect('equal'); ax.axis('off')
    check(fig, ax)
    out = HERE / 'overview' / 'ov0-the-score-path-as-a-third-space'
    for ext in ('png', 'svg'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight', facecolor=s['bg'])
    print(f'wrote {out.name}.png and .svg')


def sample():
    styles = {n: s for n, s in ss.STYLES.items() if s.get('layout') != 'annotated'}
    fig, axes = plt.subplots(1, len(styles), figsize=(4.4 * len(styles), 3.4), dpi=110)
    for ax, (name, s) in zip(axes, styles.items()):
        s = dict(s, layout=None)
        ax.set_facecolor(s['bg'])
        x0, x1, y0, y1 = fam_score_path(ax, s)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1 + 0.4); ax.set_aspect('equal'); ax.axis('off')
        ax.text((x0 + x1) / 2, y1 + 0.25, s['label'], ha='center', fontsize=12, weight='bold', color=s['ink'])
    fig.savefig(HERE / 'style-sample-score-path.png', bbox_inches='tight', facecolor='white')
    print('wrote style-sample-score-path.png')


if __name__ == '__main__':
    {'overview': overview, 'sample': sample}[sys.argv[1]]()
