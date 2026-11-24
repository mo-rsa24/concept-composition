"""Drawing helpers and the layout check shared by the digit-map checker ladder's figures.

The same contract as the word-map checker ladder's figure_kit.py, rebuilt on matplotlib because the laptop's figures
environment (~/.claude/venvs/figures) has matplotlib and numpy and no cairosvg. Coordinates are pixels with y pointing
down. Every helper records what it draws, and write() refuses to save a figure where

- a line or arrow crosses a box, circle, picture or label it is not allowed to touch, or
- a label overlaps another label, or overlaps a box it does not sit wholly inside.

Colours are the ladder's key from ladder.json: blue inputs, green hidden units, red outputs, orange the backward
signal and the push, grey what is frozen, fixed or off. Purple marks the count branch and teal the identity branch.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "none"

INK = "#20313c"
INPUT, INPUT_FILL = "#1f5fd0", "#dbe6fa"
HIDDEN, HIDDEN_FILL = "#1d7a3e", "#d6eddd"
OUTPUT, OUTPUT_FILL = "#c0262d", "#f6d5d6"
ORANGE, ORANGE_FILL = "#c86a0a", "#f8e3cc"
GREY, FADED_FILL, FADED_TEXT = "#6b7a88", "#eceef1", "#9aa3ad"
COUNT, COUNT_FILL = "#7b3fa0", "#eadcf3"
IDENT, IDENT_FILL = "#0f7f86", "#d3eef0"
WHITE = "#ffffff"
TOLERANCE = 2.0
DPI = 100


class Fig:
    def __init__(self, stem, width, height):
        self.stem, self.W, self.H = stem, width, height
        self.fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor(WHITE)
        self.segments, self.obstacles, self.texts, self.ids = [], [], [], set()
        self.z = 2

    # ------------------------------------------------------------ records
    def _obstacle(self, oid, x, y, w, h, kind):
        if oid is None:
            oid = f"{kind}-{len(self.obstacles) + len(self.texts)}"
        if oid in self.ids:
            raise ValueError(f"duplicate obstacle id {oid}")
        self.ids.add(oid)
        if kind not in ("text",):
            self.obstacles.append({"id": oid, "box": (x, y, x + w, y + h), "kind": kind})
        return oid

    def _segment(self, name, p, q, allowed, width):
        self.segments.append({"id": f"{name}:{len(self.segments)}", "p": p, "q": q,
                              "allowed": frozenset(allowed), "width": width})

    # ------------------------------------------------------------ drawing
    def text(self, x, y, s, size=13, color=INK, weight="normal", ha="left", va="baseline", oid=None, style="normal"):
        oid = oid or f"text-{len(self.texts)}"
        if oid in self.ids:
            raise ValueError(f"duplicate obstacle id {oid}")
        self.ids.add(oid)
        t = self.ax.text(x, y, s, fontsize=size * 0.75, color=color, fontweight=weight, ha=ha, va=va,
                         zorder=10, fontstyle=style)
        self.texts.append((oid, t))
        return oid

    def box(self, x, y, w, h, fill=WHITE, stroke=GREY, lw=1.2, radius=6, oid=None, kind="rect", dash=None, alpha=1.0):
        oid = self._obstacle(oid, x, y, w, h, kind)
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                           edgecolor=stroke, linewidth=lw, linestyle=dash or "solid", zorder=self.z, alpha=alpha)
        self.ax.add_patch(p)
        return oid

    def card(self, x, y, w, h, fill=WHITE, stroke=GREY, lw=1.2, radius=8, oid=None, dash=None):
        """A frame that labels and lines may sit inside."""
        return self.box(x, y, w, h, fill, stroke, lw, radius, oid, "card", dash)

    def halo(self, x, y, w, h, color, radius=10):
        for pad, a in ((12, .05), (8, .08), (4, .12)):
            self.ax.add_patch(FancyBboxPatch((x - pad, y - pad), w + 2 * pad, h + 2 * pad,
                                             boxstyle=f"round,pad=0,rounding_size={radius + pad}", facecolor=color,
                                             edgecolor="none", alpha=a, zorder=1))

    def circle(self, x, y, r, fill, stroke, lw=1.2, oid=None):
        oid = self._obstacle(oid, x - r, y - r, 2 * r, 2 * r, "circle")
        self.ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=self.z))
        return oid

    def image(self, arr, x, y, w, h, cmap="gray", vmin=-1, vmax=1, frame=INPUT, oid=None, lw=1.4):
        oid = self._obstacle(oid, x, y, w, h, "image")
        self.ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + w, y + h, y), interpolation="nearest",
                       zorder=self.z)
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=frame, linewidth=lw, zorder=self.z + 1))
        return oid

    def volume(self, x, y, w, h, depth, fill, stroke, oid=None, lw=1.2):
        """A CNN volume: a front face w x h and a receding top and side of `depth` pixels; the whole is one obstacle."""
        oid = self._obstacle(oid, x, y - depth, w + depth, h + depth, "rect")
        d = depth
        top = plt.Polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)], closed=True, facecolor=fill,
                          edgecolor=stroke, linewidth=lw, zorder=self.z, alpha=0.75)
        side = plt.Polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)], closed=True,
                           facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=self.z, alpha=0.55)
        front = Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=self.z)
        for p in (top, side, front):
            self.ax.add_patch(p)
        return oid

    def line(self, x1, y1, x2, y2, color=GREY, lw=1.2, allowed=(), name="line", dash=None):
        self._segment(name, (x1, y1), (x2, y2), allowed, lw)
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=dash or "solid", zorder=self.z,
                     solid_capstyle="butt")

    def arrow(self, x1, y1, x2, y2, color=GREY, lw=1.2, allowed=(), name="arrow", head=7):
        self._segment(name, (x1, y1), (x2, y2), allowed, lw)
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=f"-|>,head_length={head * .1},head_width={head * .055}",
                                          mutation_scale=10, color=color, linewidth=lw, zorder=self.z,
                                          shrinkA=0, shrinkB=0))

    def route(self, points, color=GREY, lw=1.2, allowed=(), name="route", head=7):
        for i, (p, q) in enumerate(zip(points[:-2], points[1:-1])):
            self.line(*p, *q, color, lw, allowed, f"{name}.leg{i}")
        self.arrow(*points[-2], *points[-1], color, lw, allowed, f"{name}.end", head)

    def raw(self):
        """The axes, for drawing that is checked by nothing (cell grids inside an image box)."""
        return self.ax

    # ------------------------------------------------------------ checks
    def _text_boxes(self):
        renderer = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        out = []
        for oid, t in self.texts:
            bb = t.get_window_extent(renderer)
            (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            out.append({"id": oid, "box": (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), "kind": "text"})
        return out

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

    @staticmethod
    def _overlap(a, b, pad=0.5):
        return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad

    @staticmethod
    def _contains(outer, inner):
        return (outer[0] <= inner[0] + .5 and outer[1] <= inner[1] + .5 and outer[2] >= inner[2] - .5
                and outer[3] >= inner[3] - .5)

    def check(self):
        self.fig.canvas.draw()
        texts = self._text_boxes()
        everything = self.obstacles + texts
        bad = []
        for e in self.segments:
            unknown = e["allowed"] - self.ids
            if unknown:
                raise ValueError(f"{e['id']}: unknown allowed ids {sorted(unknown)}")
            for o in everything:
                if o["id"] in e["allowed"]:
                    continue
                pad = TOLERANCE + e["width"] / 2
                if o["kind"] == "card":
                    b = o["box"]
                    inside = all(b[0] + pad < x < b[2] - pad and b[1] + pad < y < b[3] - pad for x, y in (e["p"], e["q"]))
                    if inside:
                        continue
                if self._hits(e["p"], e["q"], o["box"], pad):
                    bad.append(f"{e['id']} {e['p']}->{e['q']} crosses {o['id']} ({o['kind']}) {tuple(round(v) for v in o['box'])}")
        solids = [o for o in self.obstacles if o["kind"] in ("rect", "circle", "image", "card")]
        overlaps = []
        for i, a in enumerate(texts):
            if a["box"][0] < 0 or a["box"][2] > self.W or a["box"][1] < 0 or a["box"][3] > self.H:
                overlaps.append(f"label {a['id']} leaves the canvas {tuple(round(v) for v in a['box'])}")
            for b in texts[i + 1:]:
                if self._overlap(a["box"], b["box"]):
                    overlaps.append(f"label {a['id']} overlaps label {b['id']}")
            for s in solids:
                if self._overlap(a["box"], s["box"]) and not self._contains(s["box"], a["box"]):
                    overlaps.append(f"label {a['id']} {tuple(round(v) for v in a['box'])} overlaps {s['kind']} {s['id']}")
        if bad or overlaps:
            raise RuntimeError(f"layout check: {self.stem}: {len(bad)} crossings, {len(overlaps)} overlaps\n"
                               + "\n".join(bad + overlaps))
        print(f"layout check: {self.stem}: 0 crossings over {len(self.segments)} segments, "
              f"0 overlaps over {len(texts)} labels")

    def write(self, folder):
        self.check()
        folder = Path(folder)
        self.fig.savefig(folder / f"{self.stem}.svg", facecolor=WHITE)
        self.fig.savefig(folder / f"{self.stem}.png", dpi=DPI * 2, facecolor=WHITE)
        plt.close(self.fig)
        print(f"wrote {self.stem}.svg and {self.stem}.png")
