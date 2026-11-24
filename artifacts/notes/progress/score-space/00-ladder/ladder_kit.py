"""Helpers shared by this ladder's figure scripts, on top of figure_kit.py (copied unchanged from the NoisyCLIP ladder).

Colour key, kept from 2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/ladder.json, plus one new part:
text lilac, image (canvas) green, matching pairs pale blue, frozen or fixed grey, the push orange (unused: this loop does
not push), and the score path rose, which no earlier ladder used.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from figure_kit import *  # noqa: E402,F401,F403
from figure_kit import _width  # noqa: E402,F401

SCORE = '#f2b8c6'
SCORE_EDGE = '#c76b84'
SCORE_INK = '#9c3f5a'
PLANNED_DASH = '7 5'
N = json.loads((ROOT / 'numbers' / 'numbers.json').read_text())


def go(f, *pts, color=INK, width=1.4, head=10, allowed=(), name='link', gap=4.6):
    """An arrow or elbow that stops just short of the things it joins."""
    pts = [tuple(map(float, p)) for p in pts]
    for idx, nb in ((0, 1), (-1, -2)):
        (x, y), (u, v) = pts[idx], pts[nb]
        d = math.hypot(u - x, v - y)
        pts[idx] = (x + (u - x) * gap / d, y + (v - y) * gap / d)
    if len(pts) == 2:
        f.arrow(*pts[0], *pts[1], color, width, head, allowed, name)
    else:
        f.route(pts, color, width, head, allowed, name)


def wire(f, *pts, color=INK, width=1.4, gap=4.6, allowed=()):
    pts = [tuple(map(float, p)) for p in pts]
    for idx, nb in ((0, 1), (-1, -2)):
        (x, y), (u, v) = pts[idx], pts[nb]
        d = math.hypot(u - x, v - y)
        pts[idx] = (x + (u - x) * gap / d, y + (v - y) * gap / d)
    for p, q in zip(pts, pts[1:]):
        f.line(*p, *q, color, width, allowed, 'wire')


def boxed(f, x, y, w, h, lines, fill=FIXED_FILL, stroke=FIXED, size=18, sub=15, oid=None, dash=None, ink=INK):
    """A box holding one main line and smaller grey lines under it."""
    oid = oid or f'box-{len(f.obstacles)}'
    f.rect(x, y, w, h, fill, stroke, 1.3, 9, oid=oid, dash=dash)
    total = size + sum(sub + 6 for _ in lines[1:])
    yy = y + (h - total) / 2 + size * 0.85
    f.words(x + w / 2, yy, lines[0], size, ink, anchor='middle')
    for extra in lines[1:]:
        yy += sub + 6
        f.words(x + w / 2, yy, extra, sub, FROZEN_TEXT, anchor='middle')
    return oid


def cells(f, x, y, values, fill=WHITE, stroke=INK, size=18, cw=40, ch=36, oid=None, signed=False):
    """A row of number cells; with signed=True the fill shades rose for negatives and green for positives."""
    oid = f.obstacle(oid, x, y, cw * len(values), ch, 'rect')
    for i, v in enumerate(values):
        c = fill
        if signed and v != 0:
            c = mix(SCORE_EDGE if v < 0 else IMAGE_EDGE, max(0.25, 0.85 - 0.22 * abs(v)))
        f.raw(R(x + i * cw, y, cw, ch, c, stroke, 0.9, 0))
        f.raw(T(x + i * cw + cw / 2, y + ch / 2 + size * 0.35, str(v), size, INK, 500, 'middle')
              .replace('<text ', f'<text font-family="{FONT}" ', 1))
    return oid


def title(f, text, sub=None):
    f.words(36, 52, text, 30, INK, 700)
    if sub:
        f.words(36, 88, sub, 17, FROZEN_TEXT)


def footer(f, text, y=None):
    f.words(36, y or f.H - 22, text, 14, FROZEN_TEXT)


def planned_tag(f, cx, cy):
    """The mark every planned part carries."""
    f.pill(cx, cy, 'illustrative, not yet designed', color=SCORE_INK, size=14)
