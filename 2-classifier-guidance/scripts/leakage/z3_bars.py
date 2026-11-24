"""Bars for Loop 12 step Z3, fixed before any picture of the fix exists.

Design: artifacts/notes/progress/leakage/03-one-fix/design.md. Measure (board question 92): two clear subjects with their own
parts, drawing style ignored, on 48 matched pictures (six borrowing pairs x seeds 6001-6008).
"""
from math import comb

MIN_LEAD = 4          # of 48
ALPHA = 0.05          # exact one-sided McNemar
MAX_EXTRA_BROKEN = 3  # the guard: broken anatomy, drawing style not counted
STEPS = tuple(range(15, 31))  # sampler steps 15 to 30 of 50


def mcnemar_one_sided(a: list, b: list) -> dict:
    """a, b: 0/1 two_clear calls on the same seeds. p that b beats a at least this often among discordant seeds."""
    assert len(a) == len(b)
    up = sum(1 for x, y in zip(a, b) if y and not x)
    down = sum(1 for x, y in zip(a, b) if x and not y)
    n = up + down
    p = sum(comb(n, k) for k in range(up, n + 1)) / 2 ** n if n else 1.0
    return {"lead": sum(b) - sum(a), "b_only": up, "a_only": down, "p": p}


def leads(a, b):
    t = mcnemar_one_sided(a, b)
    return t["lead"] >= MIN_LEAD and t["p"] < ALPHA, t


def verdict(co3, whole, boxes, broken_co3, broken_boxes, controls_pass=True):
    """co3, whole, boxes: 0/1 two_clear per matched seed for rows 1, 2, 3; broken_*: counts of the guard mark."""
    if not controls_pass:
        return "inconclusive", {}
    r3, t3 = leads(co3, boxes)
    r2, t2 = leads(co3, whole)
    r32, t32 = leads(whole, boxes)
    detail = {"boxes_vs_co3": t3, "whole_vs_co3": t2, "boxes_vs_whole": t32}
    if not r3:
        return "null", detail
    if broken_boxes > broken_co3 + MAX_EXTRA_BROKEN:
        return "broken", detail
    if r2 and not r32:
        return "extra correction explains it", detail
    return "support", detail


if __name__ == "__main__":
    base = [0] * 48
    print(verdict(base, base, [1] * 5 + [0] * 43, 10, 11))
    print(verdict(base, base, [1] * 2 + [0] * 46, 10, 10))
