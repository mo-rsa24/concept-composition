"""Drawing primitives and layout checks for the NoisyCLIP ladder, in the approved clip-contrastive style.

The Fig class and its two checks come from the word-map checker ladder's figure_kit.py, unchanged in behaviour. Every helper
records what it draws, and two checks run before a figure is written; either one refuses to write it:

- crossings: every line and arrow against every box, node, picture and label it is not allowed to touch;
- overlaps: every label against every other label, and against any box it overlaps without sitting inside.

Colours are this ladder's key from ladder.json: text lilac, image green, matching pairs pale blue, the push orange,
frozen or fixed grey. Only the push is orange.
"""
from __future__ import annotations

import base64
import math
import re
import io
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, '/home/molef/.claude/skills/render-diagrams/scripts')
from mathsvg import T, R, Ln, Ar, wrap, tex, _width, FONT  # noqa: E402
import cairosvg  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

INK = '#2b3440'
TEXT = '#b9a7e8'
IMAGE = '#a8dcb5'
IMAGE_EDGE = '#5fa877'
MATCH = '#bfe3f5'
MATCH_EDGE = '#6fb5d6'
PUSH = '#c86a0a'
FIXED = '#6b7a88'
FIXED_FILL = '#e6e9ec'
FAINT = '#f6f7f9'
FAINT_LINE = '#c9d0d7'
FROZEN_TEXT = '#5f6b78'
WHITE = '#ffffff'
GREY = FIXED
TOLERANCE = 2.0
PNG_WIDTH = 2000


def mix(colour, t):
    """Blend a #rrggbb colour toward white by share t."""
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % tuple(round(v + (255 - v) * t) for v in (r, g, b))


def tex_number(n):
    """An integer with TeX thousands separators: 5907436 -> 5{,}907{,}436."""
    return format(int(n), ',').replace(',', r'{,}')


class Fig:
    def __init__(self, stem, width, height):
        self.stem, self.W, self.H = stem, width, height
        self.segments, self.obstacles, self.ids, self.parts = [], [], set(), []
        self.obstacle('canvas', 0, 0, width, height, 'canvas')
        self.parts.append('<defs><filter id="soft-halo" x="-30%" y="-30%" width="160%" height="160%">'
                          '<feGaussianBlur stdDeviation="9"/></filter></defs>')

    # ------------------------------------------------------------ records
    def obstacle(self, oid, x, y, w, h, kind):
        if oid is None:
            oid = f'{kind}-{len(self.obstacles)}'
        if oid in self.ids:
            raise ValueError(f'Duplicate obstacle id: {oid}')
        self.ids.add(oid)
        self.obstacles.append({'id': oid, 'box': (float(x), float(y), float(x + w), float(y + h)), 'kind': kind})
        return oid

    def segment(self, name, p, q, allowed=(), width=1.2):
        self.segments.append({'id': f'{name}:{len(self.segments)}', 'p': tuple(map(float, p)), 'q': tuple(map(float, q)),
                              'allowed': frozenset(allowed) | {'canvas'}, 'width': float(width)})

    def add(self, *fragments):
        self.parts.extend(fragments)

    # ------------------------------------------------------------ drawing helpers
    def words(self, x, y, label, size=22, color=INK, weight=400, anchor='start', oid=None):
        label = str(label)
        w = _width(label, size, bold=weight >= 600)
        left = x - (w / 2 if anchor == 'middle' else w if anchor == 'end' else 0)
        self.obstacle(oid, left, y - size, w, size * 1.25, 'text')
        frag = T(x, y, escape(label), size, color, weight, anchor)
        self.add(frag.replace('<text ', f'<text font-family="{FONT}" ', 1))

    def maths(self, source, x, y, size=24, color=INK, anchor='middle', oid=None):
        result = tex(source, x, y, size, color, anchor)
        frag, measured = result[:2]
        tag = frag[:frag.index('>') + 1]
        a = dict(re.findall(r'([\w:-]+)="([^"]*)"', tag))
        left, top = float(a['x']), float(a['y'])
        w, h = max(float(a['width']), float(measured)), float(a['height'])
        self.obstacle(oid, left - 0.1, top - 0.1, w + 0.2, h + 0.2, 'tex')
        self.add(frag)
        return left, top, w, h

    def rect(self, x, y, w, h, fill=WHITE, stroke=GREY, sw=1.2, radius=0, oid=None, kind='rect', dash=None):
        self.obstacle(oid, x, y, w, h, kind)
        self.add(R(x, y, w, h, fill, stroke, sw, radius, dash) if dash else R(x, y, w, h, fill, stroke, sw, radius))

    def card(self, x, y, w, h, fill=WHITE, stroke=GREY, sw=1.2, radius=0, oid=None, dash=None):
        """A frame that labels and lines sit inside; edges drawn within must name it in allowed."""
        self.rect(x, y, w, h, fill, stroke, sw, radius, oid, 'card', dash)

    def circle(self, x, y, r, fill, stroke, width=1.2, oid=None):
        self.obstacle(oid, x - r, y - r, 2 * r, 2 * r, 'circle')
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def line(self, x1, y1, x2, y2, color=GREY, width=1.2, allowed=(), name='line', dash=None):
        self.segment(name, (x1, y1), (x2, y2), allowed, width)
        self.add(Ln(x1, y1, x2, y2, color, width, dash) if dash else Ln(x1, y1, x2, y2, color, width))

    def arrow(self, x1, y1, x2, y2, color=GREY, width=1.2, head=6, allowed=(), name='arrow'):
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        left = (bx + head * 0.4 * math.sin(ang), by - head * 0.4 * math.cos(ang))
        right = (bx - head * 0.4 * math.sin(ang), by + head * 0.4 * math.cos(ang))
        tip = (x2, y2)
        self.segment(name + '.shaft', (x1, y1), (bx, by), allowed, width)
        for i, (p, q) in enumerate(((tip, left), (left, right), (right, tip))):
            self.segment(f'{name}.head-{i}', p, q, allowed, 0)
        self.add(Ar(x1, y1, x2, y2, color, width, head))

    def route(self, points, color=GREY, width=1.2, head=6, allowed=(), name='route'):
        for i, (p, q) in enumerate(zip(points[:-2], points[1:-1])):
            self.line(*p, *q, color, width, allowed, f'{name}.leg-{i}')
        self.arrow(*points[-2], *points[-1], color, width, head, allowed, name + '.end')

    def halo(self, x, y, w, h, color, radius=10, oid=None):
        self.obstacle(oid, x, y, w, h, 'halo')
        for pad, op in ((14, .035), (9, .055), (4, .08)):
            self.add(f'<g opacity="{op}">' + R(x - pad, y - pad, w + 2 * pad, h + 2 * pad, color, 'none', 0, radius + pad) + '</g>')
        self.add('<g filter="url(#soft-halo)" opacity="0.35">' + R(x, y, w, h, color, 'none', 0, radius) + '</g>')

    def image(self, path, x, y, w, h, frame=INK, oid=None):
        enc = base64.b64encode(Path(path).read_bytes()).decode('ascii')
        oid = self.obstacle(oid, x, y, w, h, 'image')
        self.add(f'<image href="data:image/png;base64,{enc}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/>')
        self.add(R(x, y, w, h, 'none', frame, 1.4, 0))
        return oid

    # ------------------------------------------------------------ the clip-contrastive vocabulary
    def poly(self, points, fill, stroke, sw=1.4, oid=None):
        xs, ys = zip(*points)
        oid = self.obstacle(oid, min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), 'rect')
        pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
        self.add(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        return oid

    def encoder(self, x, y, w, h, name, colour, frozen=False, tag=None, shape='trapezoid', size=22, oid=None):
        """A trapezoid narrowing to the right, or a rounded block. Frozen parts fade and carry their tag."""
        fill = mix(colour, 0.55) if frozen else colour
        stroke = FIXED if frozen else (IMAGE_EDGE if colour == IMAGE else colour)
        ink = FROZEN_TEXT if frozen else INK
        oid = oid or f'enc-{len(self.obstacles)}'
        if shape == 'trapezoid':
            n = h * 0.22
            self.poly([(x, y), (x + w, y + n), (x + w, y + h - n), (x, y + h)], fill, stroke, oid=oid)
            self.words(x + w / 2, y + h / 2 + size * 0.35, name, size, ink, anchor='middle')
            if tag:
                self.pill(x + w / 2, y - 21, tag)
        else:
            self.rect(x, y, w, h, fill, stroke, 1.5, 14, oid=oid)
            mid = y + h / 2 - (15 if tag else 0)
            self.words(x + w / 2, mid + size * 0.35, name, size, ink, anchor='middle')
            if tag:
                self.pill(x + w / 2, y + h - 25, tag)
        return oid

    def pill(self, cx, cy, label, color=FIXED, size=15):
        w = _width(label, size) + 22
        self.rect(cx - w / 2, cy - 14, w, 28, WHITE, color, 1.0, 13)
        self.words(cx, cy + size * 0.35, label, size, color, anchor='middle')

    def strip(self, x, y, w, h, colour, n=6, vertical=False, oid=None):
        """Thin embedding strip whose cells shade from light to dark."""
        oid = self.obstacle(oid, x, y, w, h, 'rect')
        for i in range(n):
            c = mix(colour, 0.72 * (1 - i / (n - 1)))
            if vertical:
                self.raw(R(x, y + i * h / n, w, h / n, c, INK, 0.6, 0))
            else:
                self.raw(R(x + i * w / n, y, w / n, h, c, INK, 0.6, 0))
        return oid

    def product(self, x, y, r=17, oid=None):
        """The dot product node."""
        oid = self.obstacle(oid, x - r, y - r, 2 * r, 2 * r, 'circle')
        q = r * 0.62
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{WHITE}" stroke="{INK}" stroke-width="1.4"/>')
        self.raw(Ln(x - q, y - q, x + q, y + q, INK, 1.3) + Ln(x - q, y + q, x + q, y - q, INK, 1.3))
        return oid

    def chip(self, x, y, w, h, label, fill=WHITE, stroke=FIXED, size=20, color=INK, oid=None):
        oid = oid or f'chip-{len(self.obstacles)}'
        self.rect(x, y, w, h, fill, stroke, 1.3, 8, oid=oid)
        self.words(x + w / 2, y + h / 2 + size * 0.35, label, size, color, anchor='middle')
        return oid

    def stack(self, x, y, w, h, seed=7, oid=None, step=7):
        """A small stack of noisy thumbnails: synthetic Gaussian noise, never a generated picture."""
        oid = self.obstacle(oid, x, y - 2 * step, w + 2 * step, h + 2 * step, 'image')
        for k in (2, 1):
            self.raw(R(x + k * step, y - k * step, w, h, '#eef0f2', INK, 0.9, 0))
        rng = np.random.default_rng(seed)
        a = 0.56 + 0.15 * rng.standard_normal((24, 24, 1)) + 0.035 * rng.standard_normal((24, 24, 3))
        buf = io.BytesIO()
        Image.fromarray((np.clip(a, 0, 1) * 255).astype('uint8')).resize((192, 192), Image.NEAREST).save(buf, 'PNG')
        enc = base64.b64encode(buf.getvalue()).decode('ascii')
        self.raw(f'<image href="data:image/png;base64,{enc}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none"/>')
        self.raw(R(x, y, w, h, 'none', INK, 1.1, 0))
        return oid

    def cards(self, x, y, w, h, label, size=18, oid=None, step=6):
        oid = self.obstacle(oid, x, y - 2 * step, w + 2 * step, h + 2 * step, 'rect')
        for k in (2, 1):
            self.raw(R(x + k * step, y - k * step, w, h, WHITE, INK, 0.9, 0))
        self.raw(R(x, y, w, h, WHITE, INK, 1.1, 0))
        self.words(x + w / 2, y + h / 2 + size * 0.35, label, size, INK, anchor='middle')
        return oid

    def frame(self, x, y, w, h, label):
        """The bold CLIP frame with its lighter inner frame. Lines may enter it, so it is not an obstacle."""
        self.raw(R(x + 7, y + 7, w - 14, h - 14, '#fbfcfd', FAINT_LINE, 1.3, 9))
        self.raw(R(x, y, w, h, 'none', FIXED, 3.2, 12))
        self.words(x + 22, y + 31, label, 20, FIXED, 700)

    def brace(self, x0, x1, y, label=None, size=18, d=11):
        xm, r = (x0 + x1) / 2, 12
        path = (f'M{x0},{y} Q{x0},{y + d} {x0 + r},{y + d} L{xm - r},{y + d} Q{xm},{y + d} {xm},{y + 2 * d} '
                f'Q{xm},{y + d} {xm + r},{y + d} L{x1 - r},{y + d} Q{x1},{y + d} {x1},{y}')
        self.raw(f'<path d="{path}" fill="none" stroke="{INK}" stroke-width="1.1"/>')
        if label:
            self.words(xm, y + 2 * d + size + 3, label, size, INK, anchor='middle')

    def dotted(self, x0, y, x1):
        self.raw(Ln(x0, y, x1, y, FIXED, 1.0, '2 5'))

    def dot(self, x, y, color=PUSH, r=6.5, oid=None):
        oid = self.obstacle(oid, x - r, y - r, 2 * r, 2 * r, 'circle')
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{WHITE}" stroke-width="1.2"/>')
        return oid

    def raw(self, fragment):
        """Drawn content that is neither a line nor a label (cells of a heat map); never checked."""
        self.add(fragment)

    # ------------------------------------------------------------ checks
    @staticmethod
    def _hits(p, q, box, pad):
        xmin, ymin, xmax, ymax = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
        x, y = p
        dx, dy = q[0] - x, q[1] - y
        enter, leave = 0.0, 1.0
        for d, dist in ((-dx, x - xmin), (dx, xmax - x), (-dy, y - ymin), (dy, ymax - y)):
            if abs(d) < 1e-12:
                if dist < 0:
                    return False
                continue
            r = dist / d
            if d < 0:
                enter = max(enter, r)
            else:
                leave = min(leave, r)
            if enter > leave:
                return False
        return True

    def check(self):
        bad = []
        for e in self.segments:
            unknown = e['allowed'] - self.ids
            if unknown:
                raise ValueError(f"{e['id']}: unknown permitted obstacles {sorted(unknown)}")
            for o in self.obstacles:
                if o['id'] in e['allowed'] or o['kind'] in ('halo', 'canvas'):
                    continue
                if o['kind'] == 'card' and self._inside_seg(e, o['box']):
                    continue
                if self._hits(e['p'], e['q'], o['box'], TOLERANCE + e['width'] / 2):
                    bad.append(f"{e['id']} {e['p']}->{e['q']} crosses {o['id']} ({o['kind']}) {o['box']}")
        labels = [o for o in self.obstacles if o['kind'] in ('text', 'tex')]
        solids = [o for o in self.obstacles if o['kind'] in ('rect', 'circle', 'image', 'card')]
        overlaps = []
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if self._overlap(a['box'], b['box']):
                    overlaps.append(f"label {a['id']} {a['box']} overlaps label {b['id']} {b['box']}")
            for s in solids:
                if self._overlap(a['box'], s['box']) and not self._contains(s['box'], a['box']):
                    overlaps.append(f"label {a['id']} {a['box']} overlaps {s['kind']} {s['id']} {s['box']}")
        if bad or overlaps:
            raise RuntimeError(f'layout check: {self.stem}: {len(bad)} crossings, {len(overlaps)} overlaps\n' + '\n'.join(bad + overlaps))
        print(f'crossing check: {self.stem}: 0 crossings over {len(self.segments)} segments and '
              f'{len(self.obstacles)} obstacles; 0 label overlaps over {len(labels)} labels')

    @staticmethod
    def _overlap(a, b, pad=0.5):
        return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad

    @staticmethod
    def _contains(outer, inner):
        return outer[0] <= inner[0] + 0.5 and outer[1] <= inner[1] + 0.5 and outer[2] >= inner[2] - 0.5 and outer[3] >= inner[3] - 0.5

    @staticmethod
    def _inside_seg(e, box):
        """A segment lying wholly inside a card does not cross the card's frame."""
        (x1, y1), (x2, y2) = e['p'], e['q']
        pad = TOLERANCE + e['width'] / 2
        return all(box[0] + pad < x < box[2] - pad and box[1] + pad < y < box[3] - pad for x, y in ((x1, y1), (x2, y2)))

    def write(self, folder, description):
        self.check()
        doc = wrap(self.W, self.H, '\n'.join(self.parts), escape(description, quote=True))
        svg, png = Path(folder) / f'{self.stem}.svg', Path(folder) / f'{self.stem}.png'
        svg.write_text(doc, encoding='utf-8')
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=PNG_WIDTH)
        print(f'wrote {svg.name} and {png.name}')
