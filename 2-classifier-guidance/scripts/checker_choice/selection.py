#!/usr/bin/env python3
"""The rule that chooses one checker architecture (step D3), fixed before any candidate trains.

Input is a JSON list with one object per candidate:
    {"name", "parameters", "zero_push_max_abs", "co3_zero_push_max_abs", "neither_below_zero", "neither_total",
     "random_push_two_clear", "two_clear", "damaged", "pictures",
     "co3_two_clear", "co3_damaged", "co3_alone_two_clear", "co3_lead_se"}

Fields without a prefix are the plain product's rows. Fields starting with co3_ are the CO3 rows, counted blind in one shuffled
set with CO3 alone, so a candidate and CO3 alone are read by the same reviewers.

A picture counts when it holds two clear subjects, one clear A and one clear B as separate bodies with no borrowed features,
with drawing style ignored. Damage is reported beside the count and decides nothing. The owner set this bar on 2026-11-24, on board
question 87, before any result of the search existed; it replaced faithful pairs, which also required an intact photograph.

A candidate is eligible when a zero push reproduces both the plain product and CO3's own loop, it rejects pictures with neither
subject, and its size-matched random push on the plain product does not produce the pairs. Among eligible candidates the most
pictures with two clear subjects on the CO3 row wins (board question 57). A lead under MIN_TWO_CLEAR_LEAD is treated as a tie,
and ties go to fewer parameters. The chosen candidate is also reported as beating CO3 alone or not.
"""
import argparse
import json
from pathlib import Path

ZERO_PUSH_MAX_ABS = 1e-5
NEITHER_MIN_REJECTED = 0.80
RANDOM_PUSH_MAX_TWO_CLEAR = 2
MIN_TWO_CLEAR_LEAD = 5
Z = 1.96


def eligible(candidate):
    reasons = []
    if candidate["zero_push_max_abs"] >= ZERO_PUSH_MAX_ABS:
        reasons.append("zero push differs from the plain product")
    if candidate["co3_zero_push_max_abs"] >= ZERO_PUSH_MAX_ABS:
        reasons.append("zero push differs from CO3's own loop")
    if candidate["neither_below_zero"] / candidate["neither_total"] < NEITHER_MIN_REJECTED:
        reasons.append("accepts pictures with neither subject")
    if candidate["random_push_two_clear"] > RANDOM_PUSH_MAX_TWO_CLEAR:
        reasons.append("its random push also makes pictures with two clear subjects")
    return reasons


def beats_co3_alone(candidate):
    lead = candidate["co3_two_clear"] - candidate["co3_alone_two_clear"]
    return lead >= MIN_TWO_CLEAR_LEAD and lead > Z * candidate["co3_lead_se"]


def choose(candidates):
    passed = [candidate for candidate in candidates if not eligible(candidate)]
    if not passed:
        return None, "no candidate passed the checks"
    best = max(candidate["co3_two_clear"] for candidate in passed)
    tied = [candidate for candidate in passed if best - candidate["co3_two_clear"] < MIN_TWO_CLEAR_LEAD]
    tied.sort(key=lambda candidate: candidate["parameters"])
    if len(tied) == 1:
        return tied[0], f"most pictures with two clear subjects on the CO3 row, leading by at least {MIN_TWO_CLEAR_LEAD}"
    return tied[0], (f"{len(tied)} candidates within {MIN_TWO_CLEAR_LEAD} pictures with two clear subjects on the CO3 row; "
                     "fewest parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    candidates = json.loads(args.results.read_text())
    for candidate in candidates:
        print(candidate["name"], "eligible" if not eligible(candidate) else eligible(candidate),
              f"| CO3 row: {candidate['co3_two_clear']} with two clear subjects, {candidate['co3_damaged']} damaged")
    chosen, why = choose(candidates)
    print("chosen:", chosen["name"] if chosen else None, "|", why)
    if chosen:
        print("beats CO3 alone:", beats_co3_alone(chosen))


if __name__ == "__main__":
    main()
