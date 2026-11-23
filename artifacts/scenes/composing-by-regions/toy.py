"""Compute every number the reading 7 page shows, in exact fractions, and write toy.json.

The toy is invented. Eight canvas positions hold one brightness value each. Window A wants data
drawn from N(2, 1) at each of its positions and window B wants N(8, 1). Each window uses the exact
denoiser for its Gaussian, D(x, s) = (x + s^2 mu) / (1 + s^2), and takes one Euler step from noise
level s to the next level s', I = D + (s'/s)(x - D).
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction as F
from pathlib import Path

import numpy as np

X0 = [F(v) for v in (5, 3, 6, 4, 5, 7, 4, 6)]
SIGMAS = [F(2), F(1), F(1, 2), F(1, 4), F(1, 8), F(0)]
MU_A, MU_B = F(2), F(8)
WEIGHTS = [F(1, 4), F(1, 2), F(1), F(2), F(3), F(4)]
OVERLAPS = [1, 2, 3, 4]
DEFAULT_K = 2
N = len(X0)


def fmt(v):
    """Two decimals, half up, dropping one trailing zero so 3.20 prints 3.2 and 5.00 prints 5.0."""
    d = (Decimal(v.numerator) / Decimal(v.denominator)).quantize(Decimal("0.01"), ROUND_HALF_UP)
    s = f"{d:.2f}"
    return s[:-1] if s.endswith("0") else s


def fmt2(v):
    d = (Decimal(v.numerator) / Decimal(v.denominator)).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return f"{d:.2f}"


def windows(k):
    a_end = 3 + k // 2
    b_start = a_end - k + 1
    return (0, a_end), (b_start, N - 1)


def euler(x, mu, s, s_next):
    denoised = (x + s * s * mu) / (1 + s * s)
    return denoised + (s_next / s) * (x - denoised)


def propose(canvas, win, mu, s, s_next):
    return [euler(canvas[p], mu, s, s_next) if win[0] <= p <= win[1] else None for p in range(N)]


def reconcile(a, b, w=F(1)):
    out = []
    for va, vb in zip(a, b):
        if va is not None and vb is not None:
            out.append((va + w * vb) / (1 + w))
        else:
            out.append(va if va is not None else vb)
    return out


def strs(values):
    return [None if v is None else fmt(v) for v in values]


def first_step_cases():
    cases = {}
    for k in OVERLAPS:
        win_a, win_b = windows(k)
        a = propose(X0, win_a, MU_A, SIGMAS[0], SIGMAS[1])
        b = propose(X0, win_b, MU_B, SIGMAS[0], SIGMAS[1])
        shared = [p for p in range(N) if a[p] is not None and b[p] is not None]
        assert len(shared) == k
        p = shared[0]
        for wi, w in enumerate(WEIGHTS):
            joined = reconcile(a, b, w)
            va, vb, j = a[p], b[p], joined[p]
            cost = lambda v: (v - va) ** 2 + w * (v - vb) ** 2
            j_eq4 = (va + w * w * vb) / (1 + w * w)
            # Check the closed form against a numerical weighted least-squares solve.
            design = np.array([[1.0], [float(w) ** 0.5]])
            target = np.array([float(va), float(w) ** 0.5 * float(vb)])
            solved = np.linalg.lstsq(design, target, rcond=None)[0][0]
            assert abs(solved - float(j)) < 1e-9
            cases[f"{k}_{wi}"] = {
                "k": k, "w": fmt(w), "winA": win_a, "winB": win_b, "shared": shared, "p": p,
                "a": strs(a), "b": strs(b), "J": strs(joined),
                "va": fmt(va), "vb": fmt(vb), "vJ": fmt(j), "vJeq4": fmt(j_eq4),
                "sum": fmt(va + w * vb), "den": fmt(1 + w),
                "costJ": fmt(cost(j)), "costA": fmt(cost(va)), "costB": fmt(cost(vb)),
                "gap": fmt(vb - va),
                "gaps": [fmt(b[q] - a[q]) if q in shared else None for q in range(N)],
            }
    return cases


def walk():
    win_a, win_b = windows(DEFAULT_K)
    shared = [p for p in range(N) if win_a[0] <= p <= win_a[1] and win_b[0] <= p <= win_b[1]]
    p = shared[0]
    canvas, run_a, run_b = list(X0), list(X0), list(X0)
    on, off = [], []
    for s, s_next in zip(SIGMAS, SIGMAS[1:]):
        a = propose(canvas, win_a, MU_A, s, s_next)
        b = propose(canvas, win_b, MU_B, s, s_next)
        joined = reconcile(a, b)
        loss = sum((joined[q] - a[q]) ** 2 + (joined[q] - b[q]) ** 2 for q in shared)
        on.append({"from": fmt(s) if s < 1 else str(int(s)), "to": str(s_next) if s_next == 0 else fmt(s_next),
                   "a": strs(a), "b": strs(b), "J": strs(joined), "gap": fmt2(b[p] - a[p]),
                   "gapValue": float(b[p] - a[p]), "loss": fmt2(loss)})
        canvas = joined
        run_a = [euler(run_a[q], MU_A, s, s_next) if win_a[0] <= q <= win_a[1] else None for q in range(N)]
        run_b = [euler(run_b[q], MU_B, s, s_next) if win_b[0] <= q <= win_b[1] else None for q in range(N)]
        off.append({"a": strs(run_a), "b": strs(run_b), "gap": fmt2(run_b[p] - run_a[p]),
                    "gapValue": float(run_b[p] - run_a[p])})
    return {"p": p, "shared": shared, "on": on, "off": off}


def sigma_label(s):
    if s == 0 or s.denominator == 1:
        return str(int(s))
    return str(float(s))


def main(out):
    w = walk()
    for row, s, s_next in zip(w["on"], SIGMAS, SIGMAS[1:]):
        row["from"], row["to"] = sigma_label(s), sigma_label(s_next)
    data = {
        "x0": [fmt(v) for v in X0],
        "muA": fmt(MU_A), "muB": fmt(MU_B),
        "weights": [fmt(v) for v in WEIGHTS],
        "cases": first_step_cases(),
        "walk": w,
    }
    Path(out).write_text(json.dumps(data, indent=1))
    d = data["cases"][f"{DEFAULT_K}_2"]
    print("positions 3 and 4, step 1:", d["a"][3], d["b"][3], "->", d["J"][3], "|", d["a"][4], d["b"][4], "->", d["J"][4])
    print("hand check at position 3: cost", d["costJ"], "at J; A's value", d["costA"], "; B's value", d["costB"])
    d3 = data["cases"][f"{DEFAULT_K}_4"]
    print("weight 3 at position 3: Eq. 5", d3["vJ"], "; Eq. 4 as printed", d3["vJeq4"])
    print("gap reconciled:", [r["gap"] for r in w["on"]])
    print("gap never reconciled:", [r["gap"] for r in w["off"]])
    print("loss left in the overlap:", [r["loss"] for r in w["on"]])
    print("final canvas:", w["on"][-1]["J"])


if __name__ == "__main__":
    main(sys.argv[1])
