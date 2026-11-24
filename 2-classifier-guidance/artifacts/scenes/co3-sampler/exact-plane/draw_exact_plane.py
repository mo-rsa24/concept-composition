"""CO3's move drawn exactly on the flat sheet through the three clean-picture guesses, one panel per step, both pairs.

At each step's last pass, the joint prompt's and the two animals' guided Tweedie means are three points in the
65,536-number latent space. They fix one flat sheet. Each panel draws that sheet with real lengths: the joint's point at
the centre of its axes, the first axis toward the first animal's point, the second at right angles within the sheet. On a
corrector step the orange point is CO3's composed mean 2 x joint + w1 x A + w2 x B (Algorithm 3 line 4), before
any rescale. On a resampler step the orange arrow is the composed noise term the resampler adds, a direction within the
sheet. Every coordinate and every off-sheet distance is read from exact_plane_numbers.json.

  python3 draw_exact_plane.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from co3_figkit import BG, FILL, INK, KEY, Canvas, _overlap, _seg_rect  # noqa: E402

J = json.loads((HERE / "exact_plane_numbers.json").read_text())
ORDER = ["a_cheetah__x__a_cougar", "an_eagle__x__a_fox"]
ROWNAME = {"a_cheetah__x__a_cougar": "cheetah x cougar\n(borrowed: the cougar\nhas cheetah spots)",
           "an_eagle__x__a_fox": "eagle x fox\n(each keeps\nits own parts)"}
BAR = {"a_cheetah__x__a_cougar": 0.5, "an_eagle__x__a_fox": 1.0}

PW, PH = 3.3, 3.0           # one panel's drawing area
GAPX = 0.25
X0 = 2.6
TXT = 1.25                  # text lines under each panel
ROWH = PH + TXT + 0.75
W = X0 + 6 * PW + 5 * GAPX + 0.3
H = 2.9 + 2 * ROWH
c = Canvas(W, H, scale=0.8)


def fmt_small(v):
    return "0" if v == 0 else f"{v:.0e}".replace("e-0", "e-")


def place(text, px, py, color, size=8.5, weight="normal", avoid=None):
    """Put a label next to a point at the first of several offsets where it touches nothing already drawn."""
    c.fig.canvas.draw()
    r = c.fig.canvas.get_renderer()
    to_px = c.ax.transData.transform
    boxes = []
    for art, _, shape in c.obstacles:
        if shape == "rect":
            bb = art.get_window_extent(r)
            boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
        else:
            _, cx, cy, rad = shape
            p, q = to_px((cx - rad, cy - rad)), to_px((cx + rad, cy + rad))
            boxes.append((p[0], p[1], q[0], q[1]))
    for t, _, _ in c.labels:
        bb = t.get_window_extent(r)
        boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
    segs = [(to_px(a), to_px(b)) for a, b, _ in c.segments]
    d = 0.11
    for dx, dy, ha, va in ((d, d, "left", "bottom"), (d, -d, "left", "top"), (-d, d, "right", "bottom"),
                           (-d, -d, "right", "top"), (0, 1.4 * d, "center", "bottom"), (0, -1.4 * d, "center", "top"),
                           (1.4 * d, 0, "left", "center"), (-1.4 * d, 0, "right", "center"),
                           (2.2 * d, 2.2 * d, "left", "bottom"), (-2.2 * d, -2.2 * d, "right", "top"),
                           (2.2 * d, -2.2 * d, "left", "top"), (-2.2 * d, 2.2 * d, "right", "bottom")):
        t = c.text(px + dx, py + dy, text, size=size, ha=ha, va=va, color=color, weight=weight)
        bb = t.get_window_extent(r)
        rc = (bb.x0, bb.y0, bb.x1, bb.y1)
        ok = all(not _overlap(rc, b, 0) for b in boxes) and all(not _seg_rect(a, b, rc) for a, b in segs)
        if avoid is not None:
            ok = ok and avoid[0] <= rc[0] and rc[2] <= avoid[2] and avoid[1] <= rc[1] and rc[3] <= avoid[3]
        if ok:
            return t
        c.labels.pop()
        t.remove()
    raise SystemExit(f"no free place for label {text!r}")


c.text(0.3, H - 0.4, "CO3's move drawn exactly: the flat sheet through the joint prompt's and the two animals' "
       "clean-picture guesses, at each step", size=15, weight="bold", ha="left")
c.text(0.3, H - 0.95,
       r"Each panel: at the step's last pass, the three guided Tweedie means $x_t-\sqrt{1-\bar\alpha_t}\,\epsilon^{\lambda,c}$ "
       r"(joint, first animal, second animal) are three points among 65,536 numbers, and they fix one flat sheet.",
       size=10, ha="left", color="#3b4552")
c.text(0.3, H - 1.35,
       "The panel is that sheet at true lengths (no t-SNE, no rescaling of axes): the joint's guess sits at the crossing "
       "of the faint axes, the first animal's lies along the horizontal one. The scale bar gives the length unit.",
       size=10, ha="left", color="#3b4552")
c.text(0.3, H - 1.75,
       "Orange point: the corrector's composed mean $2\\,\\hat x^{C}+w_1\\hat x^{c_1}+w_2\\hat x^{c_2}$ before any rescale. "
       "Orange arrow: the resampler's added noise term $-\\sqrt{1-\\bar\\alpha_t}\\,\\lambda\\,(\\epsilon^{C}+w_1\\epsilon^{c_1}"
       "+w_2\\epsilon^{c_2})$, a direction, drawn from the singles' weighted average (grey) to the joint.",
       size=10, ha="left", color="#3b4552")
c.text(0.3, H - 2.15,
       "Hollow magenta ring: where the code's rescale (attn_utils.py line 255, which matches largest entries) puts the "
       "orange point, shown by its shadow on the sheet. Under each panel: how far each drawn point sits off the sheet.",
       size=10, ha="left", color="#3b4552")

for ri, pair in enumerate(ORDER):
    P = J["pairs"][pair]
    A, B = P["single_prompts"]
    ytop = H - 2.75 - ri * ROWH
    c.text(X0 - 0.2, ytop - PH / 2, ROWNAME[pair], size=10.5, ha="right", weight="bold")
    # one scale per row, so the six steps of a pair compare
    allpts = []
    for row in P["steps"]:
        S = row["sheet"]
        allpts += [S["joint"]["xy"], S["a"]["xy"], S["b"]["xy"], S["singles_average"]["xy"]]
        if "combination" in S:
            allpts += [S["combination"]["xy"], S["after_code_rescale"]["xy"]]
    allpts = np.array(allpts)
    span = max(np.ptp(allpts[:, 0]) / (PW - 0.9), np.ptp(allpts[:, 1]) / (PH - 0.9))
    k = 1.0 / span
    for si, row in enumerate(P["steps"]):
        S = row["sheet"]
        x0 = X0 + si * (PW + GAPX)
        frame = c.box(x0, ytop - PH, PW, PH, "", kind="white", ec="#c9ced6", lw=1.0, container=True)
        pts = [S["joint"]["xy"], S["a"]["xy"], S["b"]["xy"], S["singles_average"]["xy"]]
        if "combination" in S:
            pts += [S["combination"]["xy"], S["after_code_rescale"]["xy"]]
        pts = np.array(pts)
        mid = (pts.min(0) + pts.max(0)) / 2
        cx, cy = x0 + PW / 2, ytop - PH / 2 - 0.08

        def to(p):
            return (cx + k * (p[0] - mid[0]), cy + k * (p[1] - mid[1]))
        J0, PA, PB, AV = to(S["joint"]["xy"]), to(S["a"]["xy"]), to(S["b"]["xy"]), to(S["singles_average"]["xy"])
        # faint axes of the sheet through the joint's point
        c.ax.plot([x0 + 0.08, x0 + PW - 0.08], [J0[1], J0[1]], color="#e3e6ea", lw=0.8, zorder=1)
        c.ax.plot([J0[0], J0[0]], [ytop - PH + 0.08, ytop - 0.08], color="#e3e6ea", lw=0.8, zorder=1)
        r0 = 0.075
        circles = [(AV, 0.045), (J0, r0), (PA, r0), (PB, r0)]
        if row["kind"] == "correct":
            CB, RS = to(S["combination"]["xy"]), to(S["after_code_rescale"]["xy"])
            circles += [(CB, r0), (RS, 0.1)]

        def seg(p, q, **kw):
            """A line from p to q, trimmed back to the rim of any marker it starts or ends in."""
            p, q = np.array(p, float), np.array(q, float)
            d = q - p
            L = float(np.linalg.norm(d))
            u = d / L
            ra = max([rad for cc, rad in circles if np.linalg.norm(np.array(cc) - p) < rad + 0.03] + [0]) + 0.02
            rb = max([rad for cc, rad in circles if np.linalg.norm(np.array(cc) - q) < rad + 0.03] + [0]) + 0.02
            if L > ra + rb + 0.03:
                c.line([tuple(p + u * ra), tuple(q - u * rb)], **kw)
        # the triangle of the three guesses; the singles' average lies on the edge between the two animals
        seg(J0, PA, color="#b9c0c9", lw=0.9, arrow=False)
        seg(J0, PB, color="#b9c0c9", lw=0.9, arrow=False)
        seg(PA, AV, color="#b9c0c9", lw=0.9, arrow=False)
        seg(AV, PB, color="#b9c0c9", lw=0.9, arrow=False)
        if row["kind"] == "correct":
            seg(AV, J0, color=KEY["method"], lw=1.1, arrow=False, ls="--")
            seg(J0, CB, color=KEY["method"], lw=1.1, arrow=False, ls="--")
        else:
            seg(AV, J0, color=KEY["method"], lw=2.0, arrow=True, ms=11)
        c.node(*AV, 0.045, kind="fixed", ec=KEY["fixed"], name="average", lw=1.0)
        c.node(*J0, r0, kind=KEY["input"], ec=KEY["input"], name="joint", lw=1.6)
        c.node(*PA, r0, kind=KEY["concept1"], ec=KEY["concept1"], name="A", lw=1.6)
        c.node(*PB, r0, kind=KEY["concept2"], ec=KEY["concept2"], name="B", lw=1.6)
        if row["kind"] == "correct":
            c.node(*CB, r0, kind=KEY["method"], ec=KEY["method"], name="combination", lw=1.6)
            from matplotlib.patches import Circle
            ring = Circle(RS, 0.1, fill=False, ec=KEY["broken"], lw=1.6, zorder=5)
            c.ax.add_patch(ring)
            if np.linalg.norm(np.array(RS) - np.array(CB)) > 0.2:
                c.obstacles.append((ring, "rescaled", ("circle", RS[0], RS[1], 0.1)))
        # scale bar, bottom right of the panel
        L = BAR[pair] * k
        bx1, by = x0 + PW - 0.15, ytop - PH + 0.2
        c.line([(bx1 - L, by), (bx1, by)], color=INK, lw=1.6, arrow=False)
        c.text(bx1 - L / 2, by + 0.04, f"{BAR[pair]:g}", size=8, va="bottom", on_edge=c._edge)
        area = c.ax.transData.transform([(x0 + 0.03, ytop - PH + 0.03), (x0 + PW - 0.03, ytop - 0.03)])
        avoid = (area[0][0], area[0][1], area[1][0], area[1][1])
        place("joint", *J0, KEY["input"], weight="bold", avoid=avoid)
        place(A, *PA, KEY["concept1"], weight="bold", avoid=avoid)
        place(B, *PB, KEY["concept2"], weight="bold", avoid=avoid)
        if row["kind"] == "correct":
            place(r"$\tilde x$", *CB, KEY["method"], size=11, weight="bold", avoid=avoid)
            if np.linalg.norm(np.array(RS) - np.array(CB)) > 0.2:
                place("rescaled", *RS, KEY["broken"], size=8, avoid=avoid)
        # header and lines under the panel
        verb = "resamples" if row["kind"] == "resample" else "corrects"
        c.text(x0 + PW / 2, ytop + 0.2, f"step {row['step']}, t = {row['t']}: {verb}", size=9.5, weight="bold",
               color=KEY["method"])
        w1, w2 = row["last_pass"]["w"]
        y1 = ytop - PH - 0.25
        c.text(x0 + 0.05, y1, f"$w_1$ = {w1:.2f}, $w_2$ = {w2:.2f}", size=8.5, ha="left", color=KEY["method"])
        g3 = max(S[q]["off_sheet"] for q in ("joint", "a", "b"))
        if row["kind"] == "correct":
            c.text(x0 + 0.05, y1 - 0.33, f"off sheet: $\\tilde x$ {fmt_small(S['combination']['off_sheet'])}, "
                   f"guesses {fmt_small(g3)}", size=8.5, ha="left")
            c.text(x0 + 0.05, y1 - 0.66, f"ring: {S['after_code_rescale']['off_sheet']:.2f} off sheet "
                   f"(scale {S['after_code_rescale']['scale']:.2f})", size=8.5, ha="left", color=KEY["broken"])
        else:
            c.text(x0 + 0.05, y1 - 0.33, f"off sheet: arrow {fmt_small(S['resampler_term']['off_sheet'])}, "
                   f"guesses {fmt_small(g3)}", size=8.5, ha="left")
            c.text(x0 + 0.05, y1 - 0.66, f"arrow length {S['resampler_term']['length']:.2f}",
                   size=8.5, ha="left", color=KEY["method"])

c.text(0.3, 0.35,
       "Off-sheet distances are in the same units as the scale bar. The orange point and arrow sit off by at most "
       + fmt_small(max(r['sheet'][q]['off_sheet'] for p in J['pairs'].values() for r in p['steps'] for q in ('combination', 'resampler_term') if q in r['sheet']))
       + ", left by the logged float32 weights summing to $-1$ only to 7 digits; on passes where they sum exactly it is about 1e-15.",
       size=9, ha="left", color="#3b4552")
c.save("exact-plane/co3-exact-plane")
