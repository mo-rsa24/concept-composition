#!/usr/bin/env python3
"""Step M12's second reader: score the size-corrected per-place reader on the fresh trajectories against fresh blind calls.

The design is artifacts/notes/progress/method-search/08-size-corrected-identity-reader/design.md, with the bar written before any run.
Reads, from one folder pulled from the cluster: perplace/reads.jsonl (each step's first-animal share and the animal count on the clean
estimate) and expected/expected.json (each pair's expected first-animal share); and the blind calls on the 96 fresh finished pictures.

  size-corrected   min(share_A / expected_A, share_B / expected_B), capped at 1; a count above 2 on an animal pair sets it to 0
  without the rule the same score with no count rule
  uncorrected      the first reader's score, the smaller share, called at 0.30

Usage: python m12_sizescore.py --run <folder> --calls <csv with id,rung> --key <json id -> pair, seed> --out <json>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m5_read_co3 import auc  # noqa: E402

STEPS = ("5", "10", "15", "20", "25", "30")
CALL, CALL_UNCORRECTED = 0.60, 0.30
BAR, FLOOR, FIRST_READER_BEST_AUC = 0.75, 0.65, 0.742


def scores(row, expected, rule):
    e_a = expected[row["pair"]]["first_expected_share"]
    out = {}
    for st in STEPS:
        a = row["first_animal_closer_share"][st]
        value = min(a / e_a, (1 - a) / (1 - e_a), 1.0)
        if rule and row.get("animal_count_on_clean_estimate", {}).get(st, 0) > 2:
            value = 0.0
        out[st] = value
    return out


def table(rows, labels, values, call):
    per_step = {}
    for st in STEPS:
        s = [values[r["key"]][st] for r in rows]
        lab = [labels[r["key"]] for r in rows]
        pos = [x >= call for x, l in zip(s, lab) if l]
        neg = [x < call for x, l in zip(s, lab) if not l]
        per_step[st] = {"balanced_accuracy": round(float((np.mean(pos) + np.mean(neg)) / 2), 4), "area_under_curve": round(auc(s, lab), 4)}
    return per_step


def verdict(per_step):
    best = max(v["balanced_accuracy"] for v in per_step.values())
    best_auc = max(v["area_under_curve"] for v in per_step.values())
    if best >= BAR:
        return "the reader sees the failure"
    if best_auc >= BAR:
        return "on ranking only (question 96)"
    if best < FLOOR and best_auc <= FIRST_READER_BEST_AUC:
        return "it does not"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (args.run / "perplace/reads.jsonl").read_text().splitlines() if l.strip()]
    expected = json.loads((args.run / "expected/expected.json").read_text())["pairs"]
    key = json.loads(args.key.read_text())
    labels = {}
    for c in csv.DictReader(open(args.calls, newline="")):
        k = key[c["id"]]
        labels[f"{k['pair']}|{int(k['seed'])}"] = c["rung"] == "two_clear"
    rows = [r for r in rows if r["key"] in labels]
    corrected = {r["key"]: scores(r, expected, True) for r in rows}
    no_rule = {r["key"]: scores(r, expected, False) for r in rows}
    uncorrected = {r["key"]: {st: r["reads"][st][0] for st in STEPS} for r in rows}
    result = {"pictures": len(rows), "two_clear": sum(labels[r["key"]] for r in rows), "call": CALL, "bar": BAR, "floor": FLOOR}
    result["size_corrected"] = table(rows, labels, corrected, CALL)
    result["verdict"] = verdict(result["size_corrected"])
    result["beside"] = {"without_the_extra_animal_rule": table(rows, labels, no_rule, CALL),
                        "uncorrected_first_reader": table(rows, labels, uncorrected, CALL_UNCORRECTED)}
    per_pair = {}
    for pair in sorted({r["pair"] for r in rows}):
        rs = [r for r in rows if r["pair"] == pair]
        per_pair[pair] = {"first_expected_share": expected[pair]["first_expected_share"],
                          "two_clear_of_8": int(sum(labels[r["key"]] for r in rs)),
                          "mean_corrected_step20": round(float(np.mean([corrected[r["key"]]["20"] for r in rs])), 3),
                          "mean_uncorrected_step20": round(float(np.mean([uncorrected[r["key"]]["20"] for r in rs])), 3),
                          "steps_with_more_than_two_animals": int(sum(v > 2 for r in rs
                                                                    for v in r.get("animal_count_on_clean_estimate", {}).values()))}
    result["per_pair"] = per_pair
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("pictures", "two_clear", "verdict")}))
    for name, t in [("size-corrected", result["size_corrected"])] + list(result["beside"].items()):
        print(name, {st: (v["balanced_accuracy"], v["area_under_curve"]) for st, v in t.items()})


if __name__ == "__main__":
    main()
