#!/usr/bin/env python3
"""Step M12's diagnostic: the judge reads CO3's one-step clean estimates, after a positive control on the finished pictures.

The design is artifacts/notes/progress/method-search/07-a-checker-that-reads-identity/design.md. Runs on the laptop, where the Anthropic
key is (2-classifier-guidance/scripts/rung2/api_label.py reads it and never prints it). A picture is "two clear" to the judge when it holds
two bodies, a clear first subject, a clear second subject and no hybrid; the photograph requirement is dropped, as M1's blind rubric
ignores drawing style.

  control   label the 96 finished CO3 pictures; balanced accuracy against M1's blind two-clear calls must reach 0.75
  estimates only after the control passes: label the clean estimates at steps 5 to 30 and score each step the same way

Spend: the owner's answer to board question 101 allows about 1.5 million input tokens, counted against question 15's unspent 1.46 million
input and 67,000 output. The run stops before a batch that would take the total past 5% over that.

Usage: python m12_judge.py control --finals <dir of <pair>/seed-N.png> --out <dir>
       python m12_judge.py estimates --estimates <clean-estimates dir> --out <dir>
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "rung2"))
NOTES = REPO / "artifacts/notes/progress/method-search/01-beat-the-baseline"
BAR = 0.75
STEPS = (5, 10, 15, 20, 25, 30)
BUDGET_INPUT = 1_460_000 * 1.05
BUDGET_OUTPUT = 67_000 * 1.05


def blind_two_clear():
    key = json.loads((NOTES / "paper-2-blind-key.json").read_text())
    clear = {}
    for row in csv.DictReader(open(NOTES / "paper-2-blind-calls.csv", newline="")):
        k = key[row["id"]]
        if k["row"] == "co3":
            clear[(k["pair"], int(k["seed"]))] = row["rung"] == "two_clear"
    return clear


def judge_two_clear(r):
    return r.get("bodies") == 2 and r.get("a_present") and r.get("b_present") and not r.get("hybrid")


def spent(out):
    rows = []
    for f in out.glob("labels-*.jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return sum(r.get("input_tokens", 0) for r in rows), sum(r.get("output_tokens", 0) for r in rows)


def label(items, path, out):
    import api_label
    used_in, used_out = spent(out)
    rows = [json.loads(l) for f in out.glob("labels-*.jsonl") for l in f.read_text().splitlines() if l.strip()]
    per_in = 1.05 * used_in / len(rows) if rows else 2_300      # the measured rate per picture, with a 5% margin
    per_out = 1.05 * used_out / len(rows) if rows else 110
    todo = len(items)
    if used_in + todo * per_in > BUDGET_INPUT or used_out + todo * per_out > BUDGET_OUTPUT:
        raise SystemExit(f"stopping: {todo} more pictures would pass question 15's budget (used {used_in} input, {used_out} output)")
    asyncio.run(api_label.run(items, path, 12))


def balanced(rows, clear):
    pos = [judge_two_clear(r) for r in rows if clear[(r["pair"], r["seed"])]]
    neg = [not judge_two_clear(r) for r in rows if not clear[(r["pair"], r["seed"])]]
    return round(float((np.mean(pos) + np.mean(neg)) / 2), 4), len(pos), len(neg)


def control(args):
    clear = blind_two_clear()
    items = [{"id": f"final|{p}|{s}", "pair": p, "seed": s, "path": args.finals / p / f"seed-{s}.png"} for p, s in sorted(clear)]
    missing = [it for it in items if not it["path"].exists()]
    if missing:
        raise SystemExit(f"{len(missing)} finished pictures missing, first {missing[0]['path']}")
    path = args.out / "labels-finals.jsonl"
    label(items, path, args.out)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "verdict" in r]
    ba, npos, nneg = balanced(ok, clear)
    result = {"labelled": len(ok), "errors": len(rows) - len(ok), "two_clear_blind": npos, "others_blind": nneg,
              "balanced_accuracy": ba, "bar": BAR, "passes": bool(ba >= BAR), "tokens": dict(zip(("input", "output"), spent(args.out)))}
    (args.out / "judge-control.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


def estimates(args):
    control_result = json.loads((args.out / "judge-control.json").read_text())
    if not control_result["passes"]:
        raise SystemExit("the judge missed its control on the finished pictures, so its reads count for nothing and none are made")
    clear = blind_two_clear()
    items = [{"id": f"step{st}|{p}|{s}", "pair": p, "seed": s, "step": st,
              "path": args.estimates / f"step-{st:02d}" / p / f"seed-{s}.png"} for st in STEPS for p, s in sorted(clear)]
    path = args.out / "labels-estimates.jsonl"
    label(items, path, args.out)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    per_step = {}
    for st in STEPS:
        ok = [r for r in rows if "verdict" in r and r["step"] == st]
        ba, _, _ = balanced(ok, clear)
        per_step[st] = {"labelled": len(ok), "balanced_accuracy": ba,
                        "judge_two_clear": int(sum(bool(judge_two_clear(r)) for r in ok))}
    reached = [st for st, v in per_step.items() if v["balanced_accuracy"] >= BAR]
    result = {"bar": BAR, "per_step": per_step, "first_step_at_bar": reached[0] if reached else None,
              "tokens": dict(zip(("input", "output"), spent(args.out)))}
    (args.out / "judge-estimates.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["control", "estimates"])
    ap.add_argument("--finals", type=Path)
    ap.add_argument("--estimates", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"control": control, "estimates": estimates}[args.stage](args)


if __name__ == "__main__":
    main()
