#!/usr/bin/env python3
"""The rule that chooses one checker architecture (step D3), fixed before any candidate trains.

Input is a JSON list with one object per candidate:
    {"name", "parameters", "zero_push_max_abs", "neither_below_zero", "neither_total",
     "faithful", "damaged", "random_push_faithful", "pictures"}

A candidate is eligible when a zero push reproduces the plain product, it rejects pictures with neither subject, and
its size-matched random push does not produce the pairs. Among eligible candidates the most faithful pairs wins. A lead
under MIN_FAITHFUL_LEAD is treated as a tie, and ties go to fewer damaged pictures, then to fewer parameters.
"""
import argparse
import json
from pathlib import Path

ZERO_PUSH_MAX_ABS = 1e-5
NEITHER_MIN_REJECTED = 0.80
RANDOM_PUSH_MAX_FAITHFUL = 2
MIN_FAITHFUL_LEAD = 5


def eligible(candidate):
    reasons = []
    if candidate["zero_push_max_abs"] >= ZERO_PUSH_MAX_ABS:
        reasons.append("zero push differs from the plain product")
    if candidate["neither_below_zero"] / candidate["neither_total"] < NEITHER_MIN_REJECTED:
        reasons.append("accepts pictures with neither subject")
    if candidate["random_push_faithful"] > RANDOM_PUSH_MAX_FAITHFUL:
        reasons.append("its random push also makes faithful pairs")
    return reasons


def choose(candidates):
    passed = [candidate for candidate in candidates if not eligible(candidate)]
    if not passed:
        return None, "no candidate passed the checks"
    best = max(candidate["faithful"] for candidate in passed)
    tied = [candidate for candidate in passed if best - candidate["faithful"] < MIN_FAITHFUL_LEAD]
    tied.sort(key=lambda candidate: (candidate["damaged"], candidate["parameters"]))
    if len(tied) == 1:
        return tied[0], f"most faithful pairs, leading by at least {MIN_FAITHFUL_LEAD}"
    return tied[0], f"{len(tied)} candidates within {MIN_FAITHFUL_LEAD} faithful pairs; fewest damaged, then fewest parameters"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    candidates = json.loads(args.results.read_text())
    for candidate in candidates:
        print(candidate["name"], "eligible" if not eligible(candidate) else eligible(candidate))
    chosen, why = choose(candidates)
    print("chosen:", chosen["name"] if chosen else None, "|", why)


if __name__ == "__main__":
    main()
