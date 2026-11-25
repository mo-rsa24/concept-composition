"""The ladder's rule that no line runs over a node, a box, a picture or a label, checked on the drawn figure.

check(fig) draws the figure, then tests every connector (Line2D and FancyArrowPatch) against every text label, every
box-like patch and every image, in display pixels. A connector may touch what it starts or ends at, so 4 pixels are
trimmed from each end of every segment and 2 pixels from every box. Patches given gid="container" (an inset frame, a
panel) and artists given gid="free" (contour lines, scatter, field arrows inside a plot) are not checked. It also
reports two labels that overlap each other. A figure with any finding is refused: the script raises before saving.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage

TRIM, SHRINK = 4.0, 2.0


def _clip(p, q, box):
    """Liang-Barsky: does segment p-q pass through the open box (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0:
            if qk <= 0:
                return False
        else:
            r = qk / pk
            if pk < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 >= t1:
                return False
    return True


def _segments(art, ax):
    if isinstance(art, Line2D):
        xy = art.get_transform().transform(np.column_stack(art.get_data()))
        if art.get_linestyle() in ("None", "none", " ", "") or len(xy) < 2:
            return []
        return [(xy[i], xy[i + 1]) for i in range(len(xy) - 1)]
    if isinstance(art, FancyArrowPatch):
        pts = getattr(art, "_posA_posB", None)
        if pts is None:
            return []
        xy = art.get_transform().transform(np.array(pts))
        return [(xy[0], xy[1])]
    return []


def _trim(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    n = np.hypot(*d)
    if n <= 2 * TRIM:
        return None
    u = d / n
    return p + u * TRIM, q - u * TRIM


def check(fig, name="figure"):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    findings = []
    for ax in fig.axes:
        texts, boxes = [], []
        for t in ax.texts + [a for a in ax.get_children() if isinstance(a, Text) and a not in ax.texts]:
            if not t.get_visible() or not t.get_text().strip() or t.get_gid() == "free":
                continue
            if t is ax.title or t in (ax.xaxis.label, ax.yaxis.label):
                pass
            bb = t.get_window_extent(r)
            if bb.width < 1:
                continue
            texts.append((t.get_text()[:40].replace("\n", " "), (bb.x0 + 1, bb.y0 + 1, bb.x1 - 1, bb.y1 - 1)))
        for p in ax.patches:
            if p.get_gid() in ("container", "free") or not p.get_visible():
                continue
            if isinstance(p, (FancyBboxPatch, Rectangle, Polygon, Circle)):
                bb = p.get_window_extent(r)
                boxes.append((f"{type(p).__name__}@{bb.x0:.0f},{bb.y0:.0f}", (bb.x0 + SHRINK, bb.y0 + SHRINK, bb.x1 - SHRINK, bb.y1 - SHRINK)))
        for im in ax.images:
            if im.get_gid() in ("container", "free"):
                continue
            bb = im.get_window_extent(r)
            boxes.append((f"image@{bb.x0:.0f},{bb.y0:.0f}", (bb.x0 + SHRINK, bb.y0 + SHRINK, bb.x1 - SHRINK, bb.y1 - SHRINK)))
        lines = [a for a in ax.lines if a.get_gid() != "free"] + [p for p in ax.patches if isinstance(p, FancyArrowPatch) and p.get_gid() != "free"]
        for ln in lines:
            for p, q in _segments(ln, ax):
                tq = _trim(p, q)
                if tq is None:
                    continue
                for label, box in texts:
                    if _clip(tq[0], tq[1], box):
                        findings.append(f"line {np.round(p).astype(int).tolist()}->{np.round(q).astype(int).tolist()} crosses label '{label}'")
                for label, box in boxes:
                    if _clip(tq[0], tq[1], box):
                        findings.append(f"line {np.round(p).astype(int).tolist()}->{np.round(q).astype(int).tolist()} crosses {label}")
        # A label whose centre sits inside a box (containers included) must fit inside that box.
        holders = []
        for p in ax.patches:
            if p.get_gid() == "free" or not p.get_visible() or not isinstance(p, (FancyBboxPatch, Rectangle)):
                continue
            bb = p.get_window_extent(r)
            holders.append((bb.x0, bb.y0, bb.x1, bb.y1))
        for label, (x0, y0, x1, y1) in texts:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            for (bl, bb) in boxes:
                holds = bb[0] - SHRINK < cx < bb[2] + SHRINK and bb[1] - SHRINK < cy < bb[3] + SHRINK
                if not holds and x0 < bb[2] and bb[0] < x1 and y0 < bb[3] and bb[1] < y1:
                    findings.append(f"label '{label}' overlaps {bl}")
            inside = [h for h in holders if h[0] < cx < h[2] and h[1] < cy < h[3]]
            if not inside:
                continue
            h = min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            if x0 < h[0] - 1 or x1 > h[2] + 1 or y0 < h[1] - 1 or y1 > h[3] + 1:
                findings.append(f"label spills out of its box: '{label}'")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = texts[i][1], texts[j][1]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    findings.append(f"labels overlap: '{texts[i][0]}' and '{texts[j][0]}'")
    if findings:
        raise SystemExit(f"{name}: {len(findings)} crossing(s), figure not written:\n  " + "\n  ".join(findings[:40]))
    return True
