#!/usr/bin/env python3
"""The bars for step M1, fixed before any run. The design is
artifacts/notes/progress/method-search/01-beat-the-baseline/design.md.

A lead counts only when it exceeds Z standard errors of the matched difference. Paper 2 also treats a lead under
PAPER2_MIN_LEAD pictures of 96 as a tie, because repeating a pushed render has moved its count by about 2 of 96.
"""
import numpy as np

Z = 1.96
ZERO_PUSH_MAX_ABS = 1e-5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
PAPER2_MIN_LEAD = 5


def se_within_groups(differences_by_group):
    """Standard error of the summed difference, resampling items inside each group (Paper 1's five pairs)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = [np.asarray(g, dtype=float) for g in differences_by_group]
    totals = np.zeros(BOOTSTRAP_DRAWS)
    for g in groups:
        picks = rng.integers(len(g), size=(BOOTSTRAP_DRAWS, len(g)))
        totals += g[picks].sum(1)
    return float(totals.std(ddof=1))


def se_over_groups(group_totals):
    """Standard error of the summed difference, resampling whole groups (Paper 2's 12 pairs)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    g = np.asarray(group_totals, dtype=float)
    picks = rng.integers(len(g), size=(BOOTSTRAP_DRAWS, len(g)))
    return float(g[picks].sum(1).std(ddof=1))


def clears(lead, se, min_lead=0):
    return lead >= min_lead and lead > Z * se


def paper1_verdict(method, paste, no_push, random_push, se_method_paste, se_random_no_push, zero_max_abs):
    """Counts of clean digit pairs of 320, for rows 3, 1, 2 and 4 of the design."""
    if zero_max_abs >= ZERO_PUSH_MAX_ABS:
        return "inconclusive: the zero push differs from the method without a push"
    if clears(random_push - no_push, se_random_no_push):
        return "inconclusive: the random push beats the method without a push"
    if clears(method - paste, se_method_paste):
        return "support"
    if clears(paste - method, se_method_paste):
        return "loses"
    return "null"


def paper2_verdict(pushed, co3, random_push, se_pushed_co3, se_random_co3, zero_max_abs):
    """Counts of faithful pairs of 96, for rows 2, 1 and 3 of the design."""
    if zero_max_abs >= ZERO_PUSH_MAX_ABS:
        return "inconclusive: the zero push differs from CO3's own loop"
    if clears(random_push - co3, se_random_co3, PAPER2_MIN_LEAD):
        return "inconclusive: the random push beats CO3 alone"
    if clears(pushed - co3, se_pushed_co3, PAPER2_MIN_LEAD):
        return "support"
    if clears(co3 - pushed, se_pushed_co3, PAPER2_MIN_LEAD):
        return "loses"
    return "null"
