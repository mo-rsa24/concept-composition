"""Shared drawing pieces for Paper 1's ladder, and the check every figure must pass before it is written.

The check: every line drawn through `line()` is tested, segment by segment, against the box of every node (patch) and every
text on the figure. A line may touch the box it starts or ends at; any other crossing stops the script with a list of what
crossed what. Two texts whose boxes overlap stop it too. Containers (an inset's frame, a panel's background) are marked with
gid "frame" and are not nodes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.home() / ".claude/skills/zoom-ladder/scripts"))
import annotated_explainer as ae  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
REPO = SCENE.parents[3]
NUMBERS = json.loads((SCENE / "data" / "paper1_numbers.json").read_text())

# The ladder's colour key, shared with the PoE ladder and the DCFM ladder.
KEY = {"inputs": "#1f5fd0", "experts": "#1d6b3a", "outputs": "#c0262d", "push": "#c8680a", "fixed": "#6b7a88"}
S = dict(ae.ANNOTATED)
S.update(cin=KEY["inputs"], chid=KEY["experts"], cout=KEY["outputs"], accent=KEY["push"], fixed=KEY["fixed"])

SHEET = REPO / "artifacts/notes/progress/method-search/01-beat-the-baseline/paper-1-four-rows-first-two-seeds.png"
SHEET_ROWS = ("paste", "method, no push", "method with the checker push", "method with a random push")
SHEET_PAIRS = ("1 and 9", "1 and 9", "3 and 6", "3 and 6", "4 and 7", "4 and 7", "2 and 8", "2 and 8", "0 and 5", "0 and 5")
CLEVR_SHEET = REPO / "artifacts/notes/progress/method-search/05-two-experts-by-colour/smoke-renders-by-colour-and-count.png"

_LINES = []


def reset():
    _LINES.clear()


def line(ax, pts, color, lw=1.3, arrow=True, ls="-", ms=11, name=""):
    """A polyline that the crossing check will test."""
    _LINES.append((name or f"line from {pts[0]} to {pts[-1]}", [tuple(p) for p in pts], ax))
    ae.line(ax, pts, color, lw=lw, arrow=arrow, ls=ls, ms=ms)


def sheet_tile(row, col):
    """One 32 x 64 canvas from step M1's sheet, as a 0-to-1 grey array. row: 0 paste, 1 no push, 2 checker push, 3 random."""
    a = np.asarray(Image.open(SHEET).convert("L"), dtype=float) / 255.0
    x0, y0 = 253 + 198 * col, 37 + 102 * row
    tile = a[y0:y0 + 90, x0:x0 + 186]
    return np.asarray(Image.fromarray((tile * 255).astype(np.uint8)).resize((64, 32), Image.BILINEAR), dtype=float) / 255.0


def clevr_tile(row, col):
    """One render from the two-colour CLEVR smoke sheet. Rows: r1, r3, b3, c21, c33; columns 0 to 7."""
    a = np.asarray(Image.open(CLEVR_SHEET).convert("RGB"), dtype=float) / 255.0
    x0, y0 = 111 + 203 * col, 4 + 153 * row
    return a[y0 + 14:y0 + 148, x0:x0 + 198]


def noisy(img, amount, seed=1):
    z = np.random.default_rng(seed).standard_normal(img.shape) * 0.32 + 0.4
    return np.clip((1 - amount) * img + amount * z, 0, 1)


def _seg_hits_box(p, q, box):
    """Liang-Barsky: does the segment p-q enter the box (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _inside(pt, box, pad):
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def check(fig, shrink=2.5, end_pad=9.0, text_overlap=1.5):
    """Stop if a registered line crosses a node or a text it does not start or end at, or if two texts overlap."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                b = t.get_window_extent(rend)
                boxes.append((f"text {t.get_text()[:40]!r}", (b.x0, b.y0, b.x1, b.y1), "text"))
        for p in ax.patches:
            if p.get_gid() == "frame" or type(p).__name__ == "FancyArrowPatch" or not p.get_visible():
                continue
            b = p.get_window_extent(rend)
            boxes.append((f"node {type(p).__name__} at {tuple(round(v) for v in (b.x0, b.y0))}", (b.x0, b.y0, b.x1, b.y1), "node"))
        for im in ax.images:
            b = im.get_window_extent(rend)
            boxes.append((f"picture at {tuple(round(v) for v in (b.x0, b.y0))}", (b.x0, b.y0, b.x1, b.y1), "node"))
    for t in fig.texts:
        b = t.get_window_extent(rend)
        boxes.append((f"text {t.get_text()[:40]!r}", (b.x0, b.y0, b.x1, b.y1), "text"))
    bad = []
    for name, pts, ax in _LINES:
        disp = ax.transData.transform(np.array(pts, float))
        start, end = disp[0], disp[-1]
        for bname, box, kind in boxes:
            if _inside(start, box, end_pad) or _inside(end, box, end_pad):
                continue
            inner = (box[0] + shrink, box[1] + shrink, box[2] - shrink, box[3] - shrink)
            if inner[0] >= inner[2] or inner[1] >= inner[3]:
                continue
            if any(_seg_hits_box(disp[i], disp[i + 1], inner) for i in range(len(disp) - 1)):
                bad.append(f"{name} crosses {bname}")
    texts = [b for b in boxes if b[2] == "text"]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            if min(a[2], b[2]) - max(a[0], b[0]) > text_overlap and min(a[3], b[3]) - max(a[1], b[1]) > text_overlap:
                bad.append(f"{texts[i][0]} overlaps {texts[j][0]}")
    if bad:
        raise SystemExit("crossing check failed:\n  " + "\n  ".join(bad))
    return len(_LINES), len(boxes)


def save(fig, out_stem):
    n_lines, n_boxes = check(fig)
    fig.savefig(f"{out_stem}.png", dpi=S["dpi"], facecolor=fig.get_facecolor())
    fig.savefig(f"{out_stem}.svg", facecolor=fig.get_facecolor())
    print(f"wrote {out_stem}.png and .svg; {n_lines} lines checked against {n_boxes} nodes and texts")
