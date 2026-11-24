"""Shared drawing kit for the CO3 ladder's static figures.

One colour key (the same hexes ladder.json records), helpers that register every box, node, label and line
they draw, and a check that refuses to save a figure where a line crosses a node, box or label, or where two
labels overlap. Lines may cross other lines. A label drawn on its own edge (an edge number) is exempt only
for that edge.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "co3_numbers.json").read_text())

KEY = dict(
    input="#1f5fd0",      # x_t, the starting noise, the joint prompt C
    denoiser="#1d7a3e",   # the SDXL U-Net (frozen)
    output="#c0262d",     # x_t^(2), x_{t-1}, the final picture
    method="#c86a0a",     # CO3's own parts: weights, composition, its passes
    fixed="#6b7a88",      # the unconditional prediction, plain steps, what CO3 leaves alone
    broken="#b0186e",     # where the code departs from the paper
    concept1="#7b4fb8",   # c_1 (eagle)
    concept2="#0f8b8d",   # c_2 (fox)
)
FILL = dict(input="#e4edfb", denoiser="#dff1e5", output="#fbe3e3", method="#fdeedd", fixed="#eceef1",
            broken="#fbe6f1", concept1="#efe8f8", concept2="#e0f3f3", white="#ffffff")
INK, BG, MONO = "#1b2430", "#fbfaf6", "DejaVu Sans Mono"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "svg.fonttype": "none"})


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


class Canvas:
    def __init__(self, w, h, scale=0.82, dpi=130):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * scale, h * scale), dpi=dpi, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_axis_off()
        self.ax.set_facecolor(BG)
        self.obstacles = []   # (artist, name, shape)
        self.labels = []      # (text artist, name, edge id or None)
        self.segments = []    # (p0, p1, edge id)
        self._edge = 0

    # ---- drawing helpers that register what they draw -------------------------------------------------
    def box(self, x, y, w, h, text="", kind="white", ec=None, size=10, weight="normal", ls="-", lw=1.5,
            family=None, ha="center", name=None, container=False, round_=0.08, color=INK, linespacing=1.3):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={round_}", lw=lw, ls=ls,
                           edgecolor=ec or KEY.get(kind, INK), facecolor=FILL.get(kind, kind), zorder=3 if not container else 1)
        self.ax.add_patch(p)
        if text:
            tx = x + w / 2 if ha == "center" else x + 0.12
            self.ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, weight=weight, color=color,
                         family=family or "DejaVu Sans", zorder=4, linespacing=linespacing)
        if not container:
            self.obstacles.append((p, name or text[:20], "rect"))
        return (x, y, w, h)

    def node(self, x, y, r, text="", kind="white", ec=None, size=12, lw=1.8, name=None, color=INK):
        c = Circle((x, y), r, lw=lw, edgecolor=ec or KEY.get(kind, INK), facecolor=FILL.get(kind, kind), zorder=3)
        self.ax.add_patch(c)
        if text:
            self.ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=4)
        self.obstacles.append((c, name or text, ("circle", x, y, r)))

    def image(self, path, x, y, w, h, ec=None):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB").resize((400, int(400 * h / w))))
        self.ax.imshow(img, extent=(x, x + w, y, y + h), zorder=3, interpolation="bilinear")
        r = Rectangle((x, y), w, h, fill=False, ec=ec or KEY["output"], lw=1.8, zorder=4)
        self.ax.add_patch(r)
        self.obstacles.append((r, f"image {Path(path).name}", "rect"))
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def text(self, x, y, s, size=9.5, ha="center", va="center", color=INK, weight="normal", style="normal",
             rotation=0, family=None, on_edge=None, bg=False, name=None, linespacing=1.25):
        t = self.ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, weight=weight, style=style,
                         rotation=rotation, family=family or "DejaVu Sans", zorder=6, linespacing=linespacing)
        if bg or on_edge is not None:
            t.set_bbox(dict(boxstyle="round,pad=0.12", fc=BG, ec="none"))
        self.labels.append((t, name or s[:24], on_edge))
        return t

    def line(self, pts, color=INK, lw=1.4, arrow=True, ls="-", ms=12):
        """A polyline; returns its edge id so a label can sit on it."""
        self._edge += 1
        eid = self._edge
        xs, ys = zip(*pts)
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=2)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if arrow else "-", mutation_scale=ms,
                                          lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))
        for a, b in zip(pts[:-1], pts[1:]):
            self.segments.append((a, b, eid))
        return eid

    def curve(self, xs, ys, color=INK, lw=1.8, ls="-"):
        self._edge += 1
        eid = self._edge
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2)
        for a, b in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            self.segments.append((a, b, eid))
        return eid

    def bracket(self, x0, x1, y, text, size=12, color=INK):
        self.ax.plot([x0, x0, x1, x1], [y - 0.13, y, y, y - 0.13], color=color, lw=1.3, zorder=2)
        self._edge += 1
        for a, b in (((x0, y - 0.13), (x0, y)), ((x0, y), (x1, y)), ((x1, y), (x1, y - 0.13))):
            self.segments.append((a, b, self._edge))
        self.text((x0 + x1) / 2, y + 0.08, text, size=size, weight="bold", va="bottom", color=color)

    # ---- the check ------------------------------------------------------------------------------------
    def check(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        to_px = self.ax.transData.transform
        px_per_unit = to_px((1, 0))[0] - to_px((0, 0))[0]
        rects = []
        for art, name, shape in self.obstacles:
            if shape == "rect":
                bb = art.get_window_extent(r)
                rects.append(("box", name, (bb.x0, bb.y0, bb.x1, bb.y1), None))
            else:
                _, cx, cy, rad = shape
                c = to_px((cx, cy))
                rects.append(("node", name, None, (c[0], c[1], rad * px_per_unit)))
        lab = []
        for t, name, eid in self.labels:
            bb = t.get_window_extent(r)
            lab.append((name, (bb.x0, bb.y0, bb.x1, bb.y1), eid))
        problems = []
        m = 2.5  # px of tolerance where a line ends on an edge
        for a, b, eid in self.segments:
            pa, pb = to_px(a), to_px(b)
            for kind, name, rc, circ in rects:
                if rc is not None and _seg_rect(pa, pb, (rc[0] + m, rc[1] + m, rc[2] - m, rc[3] - m)):
                    problems.append(f"line {eid} {tuple(np.round(a, 2))}->{tuple(np.round(b, 2))} crosses box '{name}'")
                if circ is not None and _seg_circle(pa, pb, circ[:2], circ[2] - m):
                    problems.append(f"line {eid} {tuple(np.round(a, 2))}->{tuple(np.round(b, 2))} crosses node '{name}'")
            for name, rc, own in lab:
                if own == eid:
                    continue
                if _seg_rect(pa, pb, (rc[0] + 1, rc[1] + 1, rc[2] - 1, rc[3] - 1)):
                    problems.append(f"line {eid} {tuple(np.round(a, 2))}->{tuple(np.round(b, 2))} crosses label '{name}'")
        for i, (n1, r1, _) in enumerate(lab):
            for n2, r2, _ in lab[i + 1:]:
                if _overlap(r1, r2, 1):
                    problems.append(f"label '{n1}' overlaps label '{n2}'")
            for kind, name, rc, circ in rects:
                box = rc if rc is not None else (circ[0] - circ[2], circ[1] - circ[2], circ[0] + circ[2], circ[1] + circ[2])
                if _overlap(r1, box, 1):
                    problems.append(f"label '{n1}' overlaps {kind} '{name}'")
            W, H = self.fig.canvas.get_width_height()
            if r1[0] < 0 or r1[1] < 0 or r1[2] > W or r1[3] > H:
                problems.append(f"label '{n1}' runs off the canvas")
        return problems

    def save(self, stem):
        problems = self.check()
        if problems:
            raise SystemExit(f"{stem}: refusing to write, {len(problems)} crossing(s):\n  " + "\n  ".join(problems))
        for ext in ("png", "svg"):
            self.fig.savefig(HERE / f"{stem}.{ext}", facecolor=BG)
        plt.close(self.fig)
        print(f"wrote {stem}.png and {stem}.svg: no line crosses a node, box or label")


def _overlap(a, b, m=0):
    return a[0] < b[2] - m and b[0] < a[2] - m and a[1] < b[3] - m and b[1] < a[3] - m


def _seg_rect(p0, p1, rc):
    """Liang-Barsky: does the segment enter the open rectangle?"""
    x0, y0, x1, y1 = rc
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if p == 0:
            if q <= 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 >= t1:
                return False
    return True


def _seg_circle(p0, p1, c, rad):
    p0, p1, c = np.asarray(p0), np.asarray(p1), np.asarray(c)
    d = p1 - p0
    L = float(d @ d)
    t = 0.0 if L == 0 else float(np.clip((c - p0) @ d / L, 0, 1))
    return float(np.linalg.norm(p0 + t * d - c)) < rad


def code_line(path, n, must_contain):
    text = Path(path).read_text().splitlines()[n - 1]
    assert must_contain in text, f"{path}:{n} no longer holds {must_contain!r}: {text!r}"
    return text


def wrap_code(n, text, width):
    """One numbered code line, stripped and wrapped at width characters with a hanging indent."""
    s = text.strip()
    head = f"{n:>3} "
    out, cur = [], s
    while len(cur) > width - len(head):
        cut = cur.rfind(" ", 0, width - len(head))
        cut = cut if cut > 10 else width - len(head)
        out.append(cur[:cut])
        cur = "  " + cur[cut:].lstrip()
    out.append(cur)
    return "\n".join([head + out[0]] + ["    " + o for o in out[1:]])
