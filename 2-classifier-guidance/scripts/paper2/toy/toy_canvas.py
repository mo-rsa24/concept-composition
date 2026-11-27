"""The canvases, the conditions and the exact reader for Paper 2's miniature toy rung.

A canvas is 16 by 16 RGB on the -1 to 1 scale, flat grey background (0, 0, 0), one to three objects. Every object is a 5 by 5 shape
(a filled square of 25 pixels, or a diamond |dy| + |dx| <= 2 of 13 pixels) in one of four flat colours, red (1, -1, -1), green
(-1, 1, -1), blue (-1, -1, 1) or yellow (1, 1, -1), placed so the 5 by 5 boxes of any two objects are at least GAP_MIN (2) pixels apart.

A combination is (shape, colour), indexed shape * 4 + colour: RS GS BS YS RD GD BD YD = 0..7. The training canvases never hold a square
and a diamond together (the look-alike pair never shares a canvas, plan rung 1) and never hold yellow square or green diamond (the two
held-out combinations that give the binding dial something to fix).

The condition is 14 numbers: the count of each of the 8 combinations, then the count of each shape (2), then of each colour (4).

The reader snaps every pixel to the nearest palette entry, finds the 8-connected components of each colour, calls a component of
fewer than MIN_PIX pixels a speck (not an object), and names each object's shape by the template (placed at the best of 9 offsets
around the component's box centre) it overlaps most by IoU: "blend" when the two IoUs are within BLEND_WITHIN, "other" when the best
is below SHAPE_MIN_IOU.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import torch

H = 16
OBJ = 5
GAP_MIN = 2
MIN_PIX = 4
BLEND_WITHIN = 0.05
SHAPE_MIN_IOU = 0.6
POS_TOL = 0.05 * H                     # the plan's "within 0.05 of the canvas width" for positions, 0.8 pixels here

SHAPES = ("square", "diamond", "blend", "other")
COLOURS = ("red", "green", "blue", "yellow")
PALETTE = torch.tensor([[0.0, 0.0, 0.0],      # background
                        [1.0, -1.0, -1.0],    # red
                        [-1.0, 1.0, -1.0],    # green
                        [-1.0, -1.0, 1.0],    # blue
                        [1.0, 1.0, -1.0]])    # yellow
NAMES = ("RS", "GS", "BS", "YS", "RD", "GD", "BD", "YD")
HELD_OUT = (NAMES.index("YS"), NAMES.index("GD"))
ALLOWED = tuple(k for k in range(8) if k not in HELD_OUT)
COND_DIM = 14
BLUE = 2

_yy, _xx = np.mgrid[-2:3, -2:3]
TEMPLATES = np.stack([np.ones((OBJ, OBJ), bool), (np.abs(_yy) + np.abs(_xx)) <= 2])     # square, diamond


def combo(shape, colour):
    return shape * 4 + colour


def shape_of(k):
    return k // 4


def colour_of(k):
    return k % 4


def parse(s):
    """'RS+BD' -> (0, 6), sorted."""
    return tuple(sorted(NAMES.index(t) for t in s.split("+")))


def name(cond):
    return "+".join(NAMES[k] for k in cond)


def cond_vector(conds):
    """(N, 14) float tensor from a list of combination tuples."""
    v = torch.zeros(len(conds), COND_DIM)
    for i, c in enumerate(conds):
        for k in c:
            v[i, k] += 1
            v[i, 8 + shape_of(k)] += 1
            v[i, 10 + colour_of(k)] += 1
    return v


# -------------------------------------------------------------------------------------------------------------------------------------
# Condition sets. Every list is fixed here, before any run.

def training_conditions():
    """Every condition a training canvas can carry: 1 to 3 objects, all one shape, colours from that shape's allowed three."""
    out = []
    for shape in (0, 1):
        ks = [k for k in ALLOWED if shape_of(k) == shape]
        for n in (1, 2, 3):
            out += [tuple(c) for c in itertools.combinations_with_replacement(ks, n)]
    return out


def _p(*names):
    return [parse(n) for n in names]


DIAL_CONDS = {
    # dial: (training conditions, test conditions); tuning uses the training conditions on tuning seeds
    "identity": (_p("RS+RD", "RS+YD", "BS+RD", "BS+YD"), _p("GS+RD", "GS+BD", "GS+YD", "RS+BD", "BS+BD")),
    "binding": (_p("GD", "RD+GD", "GD+BD"), _p("YS", "RS+YS", "GS+YS")),
    "count": (_p("BD", "RD+BD", "BS", "GS+BS"), _p("BD+YD", "RS+BS")),
    "relation": (_p("RS+GS", "GS+BS", "RD+YD", "BD+YD"), _p("RS+BS", "RD+BD")),
}
DIALS = ("identity", "binding", "count", "relation")

PAIRS = {
    # (dial a, dial b): (training conditions, test conditions) for the stack of the two
    ("identity", "count"): (_p("BS+RD", "BS+YD"), _p("RS+BD", "GS+BD")),
    ("identity", "binding"): (_p("RS+GD", "BS+GD"), _p("YS+RD", "YS+BD")),
    ("identity", "relation"): (_p("RS+YD", "BS+RD"), _p("GS+YD", "RS+BD")),
    ("binding", "count"): (_p("GD+BD"), _p("YS+BS")),
    ("count", "relation"): (_p("GS+BS", "BD+YD"), _p("RS+BS", "RD+BD")),
    ("binding", "relation"): (_p("RD+GD", "GD+BD"), _p("RS+YS", "GS+YS")),
}
DECIDING = _p("RS+BD", "GS+BD")          # the deciding test: identity's held-out colours, the count dial copies the blue diamond

# The relation dial's bins: strength -> (low, high) for the horizontal gap in canvas widths between the two named objects.
RELATION_BINS = {0.5: (0.05, 0.30), 1.0: (0.30, 10.0)}

SEEDS = {"train_canvases": 0, "reader_check": 1, "tuning": 10_000, "test": 20_000, "kept": 1_000_000, "corr_states": 3_000_000,
         "corr_tuning": 3_500_000}


def named_pair(cond):
    """The two named objects of a condition for the relation measurement: A is the lower combination index, B the other. None when
    the condition does not hold two different combinations."""
    u = sorted(set(cond))
    if len(u) < 2:
        return None
    return u[0], u[1]


def asked_blue(cond):
    """The asked object the count dial copies: the condition's blue combination (one per condition by construction)."""
    b = [k for k in cond if colour_of(k) == BLUE]
    return b[0] if b else None


# -------------------------------------------------------------------------------------------------------------------------------------
# Layout and rendering


def sample_positions(n_objs, rng):
    """Top-left corners for len(n_objs) canvases, each with n_objs[i] objects, boxes at least GAP_MIN apart."""
    out = []
    lim = H - OBJ + 1
    for n in n_objs:
        while True:
            p = rng.integers(0, lim, size=(n, 2))
            ok = True
            for a, b in itertools.combinations(range(n), 2):
                if abs(p[a, 0] - p[b, 0]) < OBJ + GAP_MIN and abs(p[a, 1] - p[b, 1]) < OBJ + GAP_MIN:
                    ok = False
                    break
            if ok:
                out.append(p)
                break
    return out


def render(conds, positions):
    """Palette-index maps (N, H, H) uint8: 0 background, 1..4 the colours."""
    idx = np.zeros((len(conds), H, H), np.uint8)
    for i, (c, p) in enumerate(zip(conds, positions)):
        for k, (r, q) in zip(c, p):
            t = TEMPLATES[shape_of(k)]
            sub = idx[i, r:r + OBJ, q:q + OBJ]
            sub[t] = colour_of(k) + 1
    return idx


def to_rgb(idx):
    return PALETTE[torch.as_tensor(idx, dtype=torch.long)].permute(0, 3, 1, 2).contiguous()


def training_canvases(n, seed=SEEDS["train_canvases"]):
    """(index maps, conditions): a condition drawn uniformly from training_conditions() per canvas, then its layout."""
    rng = np.random.default_rng(seed)
    pool = training_conditions()
    conds = [pool[i] for i in rng.integers(0, len(pool), size=n)]
    conds = [tuple(rng.permutation(c)) for c in conds]        # object order on the canvas carries no meaning
    pos = sample_positions([len(c) for c in conds], rng)
    return render(conds, pos), [tuple(sorted(c)) for c in conds]


# -------------------------------------------------------------------------------------------------------------------------------------
# The exact reader


def snap(x):
    """Nearest palette index per pixel, (N, H, H) numpy int."""
    x = torch.as_tensor(x).float().cpu()
    d = ((x.permute(0, 2, 3, 1)[..., None, :] - PALETTE.view(1, 1, 1, 5, 3)) ** 2).sum(-1)
    return d.argmin(-1).numpy()


def _placed(t, r, q):
    m = np.zeros((H, H), bool)
    r0, q0 = max(r, 0), max(q, 0)
    r1, q1 = min(r + OBJ, H), min(q + OBJ, H)
    if r1 > r0 and q1 > q0:
        m[r0:r1, q0:q1] = t[r0 - r:r1 - r, q0 - q:q1 - q]
    return m


def shape_ious(mask):
    """Best IoU of a component mask with each template over 9 offsets around the box centre."""
    ys, xs = np.nonzero(mask)
    cr, cq = int(round((ys.min() + ys.max()) / 2)), int(round((xs.min() + xs.max()) / 2))
    best = [0.0, 0.0]
    for s in (0, 1):
        for dr in (-1, 0, 1):
            for dq in (-1, 0, 1):
                m = _placed(TEMPLATES[s], cr - 2 + dr, cq - 2 + dq)
                iou = (m & mask).sum() / (m | mask).sum()
                best[s] = max(best[s], iou)
    return best


def call_shape(ious):
    if max(ious) < SHAPE_MIN_IOU:
        return 3
    if abs(ious[0] - ious[1]) < BLEND_WITHIN:
        return 2
    return int(np.argmax(ious))


def read_one(idx, keep_masks=False):
    """One palette-index map -> {"objects": [(shape, colour, cy, cx, npix)], "specks": n, ("masks": [...])}."""
    from scipy import ndimage
    objs, masks, specks = [], [], 0
    for c in range(1, 5):
        lab, n = ndimage.label(idx == c, structure=np.ones((3, 3)))
        for j in range(1, n + 1):
            m = lab == j
            npix = int(m.sum())
            if npix < MIN_PIX:
                specks += 1
                continue
            ys, xs = np.nonzero(m)
            objs.append((call_shape(shape_ious(m)), c - 1, float(ys.mean()), float(xs.mean()), npix))
            masks.append(m)
    out = {"objects": objs, "specks": specks}
    if keep_masks:
        out["masks"] = masks
    return out


def read(x, keep_masks=False):
    idx = snap(x)
    return [read_one(i, keep_masks) for i in idx]


def pairs_of(reading):
    """Multiset of (shape, colour) as combination indices; a blend or other shape gets index 100 + colour (never asked)."""
    return Counter(combo(s, c) if s < 2 else 100 + s * 10 + c for s, c, *_ in reading["objects"])


def gap(reading, ab):
    """(smallest centre x of an object matching B) - (largest centre x of an object matching A), in canvas widths; None if missing."""
    if ab is None:
        return None
    a, b = ab
    xa = [cx for s, c, cy, cx, n in reading["objects"] if s < 2 and combo(s, c) == a]
    xb = [cx for s, c, cy, cx, n in reading["objects"] if s < 2 and combo(s, c) == b]
    if not xa or not xb:
        return None
    return (min(xb) - max(xa)) / H


def positions_kept(reading, base):
    """Every object of the base reading has an object in this reading whose centre is within POS_TOL in both coordinates."""
    for _, _, by, bx, _ in base["objects"]:
        if not any(abs(by - y) <= POS_TOL and abs(bx - x) <= POS_TOL for _, _, y, x, _ in reading["objects"]):
            return False
    return True


# -------------------------------------------------------------------------------------------------------------------------------------
# The four headline measurements, one per dial kind


def m_count(reading, cond):
    """Blue objects drawn minus blue objects asked."""
    return sum(1 for s, c, *_ in reading["objects"] if c == BLUE) - sum(1 for k in cond if colour_of(k) == BLUE)


def m_shape(reading, cond):
    """1 when there is at least one object, every object's shape is a shape the condition asks in that object's colour, and every
    colour drawn shows each shape asked in it (so a red square and red diamond asked must not come out as two red squares). Counts
    of each shape are not read here, so an added copy does not move it."""
    if not reading["objects"]:
        return 0
    drawn = {(c, s) for s, c, *_ in reading["objects"]}
    asked = {(colour_of(k), shape_of(k)) for k in cond}
    colours = {c for c, _ in drawn}
    return int(drawn <= asked and all(a in drawn for a in asked if a[0] in colours))


def m_binding(reading, cond):
    """1 when every asked combination appears at least once among the drawn objects."""
    p = pairs_of(reading)
    return int(all(p[k] >= 1 for k in set(cond)))


def m_gap(reading, cond):
    return gap(reading, named_pair(cond))


# -------------------------------------------------------------------------------------------------------------------------------------
# Meets every limit


def _edit(objs, cond, attr):
    """objs: list of [shape, colour]. attr 'shape' (the identity dial): within each colour, drawn objects take the asked shapes of
    that colour, objects already right first, leftovers unchanged. attr 'colour' (the binding dial): the same within each shape,
    setting colours. Objects of a colour (or shape) the condition does not ask are left as drawn; no object is created or removed."""
    key, val = (1, 0) if attr == "shape" else (0, 1)
    out = [list(o) for o in objs]
    asked = [(shape_of(k), colour_of(k)) for k in cond]
    for g in {o[key] for o in out}:
        want = [a[val] for a in asked if a[key] == g]
        mine = [i for i, o in enumerate(out) if o[key] == g]
        rest = []
        for i in mine:
            if out[i][val] in want:
                want.remove(out[i][val])
            else:
                rest.append(i)
        for i, v in zip(rest, want):
            out[i][val] = v
    return out


def target_pairs(base, cond, stack):
    """The (shape, colour) multiset a stack must draw on a seed, built from what the frozen model drew on that seed: the identity dial
    at strength 1 gives every object the asked shape for its colour, the binding dial at strength 1 the asked colour for its shape,
    the count dial at strength 1 adds one copy of the asked blue object. None when a strength is not one these dials were trained to
    hit (0.5 or 1.5)."""
    objs = [[s, c] for s, c, *_ in base["objects"]]
    for d, attr in (("identity", "shape"), ("binding", "colour")):
        s = stack.get(d, 0)
        if s:
            if s != 1:
                return None
            objs = _edit(objs, cond, attr)
    p = Counter(combo(s, c) if s < 2 else 100 + s * 10 + c for s, c in objs)
    s = stack.get("count", 0)
    if s:
        if s != 1:
            return None
        p = p + Counter([asked_blue(cond)])
    return p


def asked_exactly(reading, cond, add=()):
    """The deciding test's success: no specks, and the drawn (shape, colour) multiset is exactly the condition plus the added objects."""
    return not reading["specks"] and pairs_of(reading) == Counter(cond) + Counter(add)


def meets(reading, base, cond, stack):
    """True when the reading meets every limit for this stack (dict dial -> strength), False when not, None when undefined."""
    t = target_pairs(base, cond, stack)
    if t is None:
        return None
    if reading["specks"] or pairs_of(reading) != t:
        return False
    s = stack.get("relation", 0)
    if s:
        lo_hi = RELATION_BINS.get(float(s))
        if lo_hi is None:
            return None
        g = m_gap(reading, cond)
        return g is not None and lo_hi[0] <= g < lo_hi[1]
    return positions_kept(reading, base)
