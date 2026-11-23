"""Drawing primitives and the layout checks shared by the word-map checker ladder's figures.

Copied unchanged in behaviour from journey 13's Readout Guidance figures
(artifacts/figures/readout-guidance/network-view/rg_draw.py). Every helper records what it draws, and two
checks run before a figure is written; either one refuses to write it:

- crossings: every line and arrow against every box, node, picture and label it is not allowed to touch;
- overlaps: every label against every other label, and against any box it overlaps without sitting inside.

Colours are this ladder's key from ladder.json: blue inputs, green hidden units, red outputs, orange the
backward signal and the push, grey what is off, fixed or broken.
"""
from __future__ import annotations

import base64
import math
import re
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, '/home/molef/.claude/skills/render-diagrams/scripts')
from mathsvg import T, R, Ln, Ar, wrap, tex, _width, FONT  # noqa: E402
import cairosvg  # noqa: E402

INK = '#20313c'
INPUT = '#1f5fd0'
INPUT_FILL = '#dbe6fa'
HIDDEN = '#1d7a3e'
HIDDEN_FILL = '#d6eddd'
OUTPUT = '#c0262d'
OUTPUT_FILL = '#f6d5d6'
ORANGE = '#c86a0a'
ORANGE_FILL = '#f8e3cc'
GREY = '#6b7a88'
FADED_FILL = '#eceef1'
FADED_TEXT = '#9aa3ad'
WHITE = '#ffffff'
TOLERANCE = 2.0
PNG_WIDTH = 2000


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

    def image(self, path, x, y, w, h, frame=INPUT, oid=None):
        enc = base64.b64encode(Path(path).read_bytes()).decode('ascii')
        oid = self.obstacle(oid, x, y, w, h, 'image')
        self.add(f'<image href="data:image/png;base64,{enc}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/>')
        self.add(R(x, y, w, h, 'none', frame, 1.4, 0))
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
