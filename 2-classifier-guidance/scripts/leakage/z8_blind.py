#!/usr/bin/env python3
"""Loop 12 step Z8: put the 99 pictures under neutral ids for the blind calls, and score the calls against the bar.

Design: artifacts/notes/progress/leakage/08-first-correction-test/design.md, "The bar". The thresholds sit here so they cannot
move after the calls are read: a base passes when the push gives at least LEAD_TO_PASS more two-clear pictures than the random
push over its failure cases' draws; the go decision also needs the push to leave no more of the already-right CO3 cases'
draws short of two clear than the random push does.

  python3 z8_blind.py pack <cases folder> <blind dir>      copy to <blind dir>/pictures/<id>.png, write sheet.csv and key.json
  python3 z8_blind.py score <blind dir> <calls1.csv> <calls2.csv> [<calls3.csv>]
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

LEAD_TO_PASS = 4  # of 12 draws (three failure cases, four draws each)
CASES = {
    "poe": [("a_puma__x__a_condor", 6001), ("a_zebra__x__a_flamingo", 6001), ("a_seal__x__a_gull", 6001)],
    "joint": [("a_puma__x__a_condor", 6001), ("a_zebra__x__a_flamingo", 6001), ("a_seal__x__a_gull", 6001)],
    "co3": [("a_cheetah__x__a_cougar", 6001), ("a_puma__x__a_condor", 6001), ("a_duck__x__a_goose", 6002),
            ("a_zebra__x__a_flamingo", 6001), ("an_eagle__x__a_fox", 6001)],
}
CO3_RIGHT = {("a_zebra__x__a_flamingo", 6001), ("an_eagle__x__a_fox", 6001)}
ARMS = ["without"] + [f"{a}-{k}" for k in range(4) for a in ("push", "random")]


def prompt(pair):
    a, b = pair.split("__x__")
    return f"{a.replace('_', ' ')} and {b.replace('_', ' ')}"


def pack(cases, out):
    out = Path(out); (out / "pictures").mkdir(parents=True, exist_ok=True)
    items = [(base, pair, seed, arm, Path(cases) / base / pair / f"seed-{seed}" / f"{arm}.png")
             for base, cs in CASES.items() for pair, seed in cs for arm in ARMS]
    missing = [str(p) for *_, p in items if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pictures missing, first: {missing[0]}")
    rng = random.Random(20261125 + 8)
    rng.shuffle(items)
    key, rows = {}, []
    for base, pair, seed, arm, src in items:
        pid = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(12))
        shutil.copy(src, out / "pictures" / f"{pid}.png")
        key[pid] = {"base": base, "pair": pair, "seed": seed, "arm": arm}
        rows.append({"id": pid, "prompt": prompt(pair)})
    with open(out / "sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "prompt"]); w.writeheader(); w.writerows(rows)
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"packed {len(rows)} pictures")


def read(path):
    return {r["id"]: r for r in csv.DictReader(open(path))}


def score(out, c1, c2, c3=None):
    out = Path(out); key = json.loads((out / "key.json").read_text())
    a, b = read(c1), read(c2)
    third = read(c3) if c3 else {}
    clear = lambda r: r["rung"] == "two_clear"
    agree = sum(clear(a[i]) == clear(b[i]) for i in key)
    final, unsettled = {}, []
    for i in key:
        if clear(a[i]) == clear(b[i]):
            final[i] = {"two_clear": int(clear(a[i])), "broken": int(a[i]["broken"] == "1" and b[i]["broken"] == "1")}
        elif i in third:
            final[i] = {"two_clear": int(clear(third[i])), "broken": int(third[i]["broken"] == "1")}
        else:
            unsettled.append(i)
    if unsettled:
        print(f"{len(unsettled)} pictures need a third call: {','.join(unsettled)}")
        (out / "need-third.txt").write_text("\n".join(unsettled) + "\n")
        return
    tally = {}
    for i, k in key.items():
        arm = k["arm"].split("-")[0]
        right = k["base"] == "co3" and (k["pair"], k["seed"]) in CO3_RIGHT
        group = (k["base"] + ("_right" if right else ""), arm)
        t = tally.setdefault(group, {"pictures": 0, "two_clear": 0, "broken": 0})
        t["pictures"] += 1; t["two_clear"] += final[i]["two_clear"]; t["broken"] += final[i]["broken"]
    per_case = {}
    for i, k in key.items():
        c = per_case.setdefault(f"{k['base']} {k['pair']} {k['seed']}", {})
        arm = k["arm"].split("-")[0]
        c[arm] = c.get(arm, 0) + final[i]["two_clear"]
    get = lambda g, arm, m="two_clear": tally[(g, arm)][m]
    bases = {}
    for base in ("poe", "joint", "co3"):
        lead = get(base, "push") - get(base, "random")
        bases[base] = {"push": get(base, "push"), "random": get(base, "random"), "without": get(base, "without"),
                       "of": tally[(base, "push")]["pictures"], "lead": lead, "passes": lead >= LEAD_TO_PASS,
                       "broken_push": get(base, "push", "broken"), "broken_random": get(base, "random", "broken")}
    short_push = tally[("co3_right", "push")]["pictures"] - get("co3_right", "push")
    short_random = tally[("co3_right", "random")]["pictures"] - get("co3_right", "random")
    right = {"short_of_two_clear_push": short_push, "short_of_two_clear_random": short_random,
             "of": tally[("co3_right", "push")]["pictures"], "push_no_worse": short_push <= short_random}
    go = bases["poe"]["passes"] and bases["joint"]["passes"] and right["push_no_worse"]
    result = {"agreement_two_clear": agree / len(key), "bases": bases, "co3_already_right": right, "per_case_two_clear": per_case,
              "call": "go to the four-way test" if go else "parked", "lead_to_pass": LEAD_TO_PASS}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"pack": lambda: pack(*sys.argv[2:4]), "score": lambda: score(*sys.argv[2:6])}[sys.argv[1]]()
