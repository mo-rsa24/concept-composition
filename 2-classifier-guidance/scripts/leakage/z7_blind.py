#!/usr/bin/env python3
"""Loop 12 step Z7: put the 144 pictures (CO3 alone, whole-picture correction, own prediction in its box; 48 each) under neutral ids for the blind calls, and read the calls back.

Design: artifacts/notes/progress/leakage/07-own-prediction/design.md, "How pictures are counted". The key that names each picture's
row stays in key.json, which no reviewer opens until every call is saved.

  python3 z3_blind.py pack <co3 folder> <z7 full folder> <blind dir>       copy to <blind dir>/pictures/<id>.png, write sheet.csv and key.json
  python3 z3_blind.py score <blind dir> <calls1.csv> <calls2.csv> [<calls3.csv>]   agreement, settled calls, and the bar
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z3_bars import verdict  # noqa: E402

PAIRS = ["a_cheetah__x__a_cougar", "a_duck__x__a_goose", "a_puma__x__a_condor",
         "a_boar__x__a_pheasant", "a_bobcat__x__a_quail", "a_seal__x__a_gull"]
SEEDS = list(range(6001, 6009))


def prompt(pair):
    a, b = pair.split("__x__")
    return f"{a.replace('_', ' ')} and {b.replace('_', ' ')}"


def pack(co3, z7full, out):
    out = Path(out); (out / "pictures").mkdir(parents=True, exist_ok=True)
    items = []
    for pair in PAIRS:
        for seed in SEEDS:
            items.append(("co3", pair, seed, Path(co3) / pair / f"seed-{seed}.png"))
            for row in ("whole", "own"):
                items.append((row, pair, seed, Path(z7full) / row / pair / f"seed-{seed}.png"))
    missing = [str(p) for *_, p in items if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pictures missing, first: {missing[0]}")
    rng = random.Random(20261125)
    rng.shuffle(items)
    key, rows = {}, []
    for row, pair, seed, src in items:
        pid = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(12))
        shutil.copy(src, out / "pictures" / f"{pid}.png")
        key[pid] = {"row": row, "pair": pair, "seed": seed}
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
    agree = sum(a[i]["rung"] == b[i]["rung"] for i in key)
    agree_clear = sum((a[i]["rung"] == "two_clear") == (b[i]["rung"] == "two_clear") for i in key)
    final, unsettled = {}, []
    for i in key:
        if (a[i]["rung"] == "two_clear") == (b[i]["rung"] == "two_clear"):
            final[i] = {"two_clear": int(a[i]["rung"] == "two_clear"), "broken": int(a[i]["broken"] == "1" and b[i]["broken"] == "1")}
        elif i in third:
            final[i] = {"two_clear": int(third[i]["rung"] == "two_clear"), "broken": int(third[i]["broken"] == "1")}
        else:
            unsettled.append(i)
    for i in key:
        if i in final:
            for m in ("a_own", "b_own"):
                final[i][m] = int(a[i].get(m) == "1" and b[i].get(m) == "1")
    if unsettled:
        print(f"{len(unsettled)} pictures need a third call: {','.join(unsettled)}")
        (out / "need-third.txt").write_text("\n".join(unsettled) + "\n")
        return
    by = {row: {} for row in ("co3", "whole", "own")}
    for i, k in key.items():
        by[k["row"]][(k["pair"], k["seed"])] = final[i]
    order = [(p, s) for p in PAIRS for s in SEEDS]
    col = lambda row, m: [by[row][x][m] for x in order]
    call, detail = verdict(col("co3", "two_clear"), col("whole", "two_clear"), col("own", "two_clear"),
                           sum(col("co3", "broken")), sum(col("own", "broken")))
    result = {"agreement_rung": agree / len(key), "agreement_two_clear": agree_clear / len(key),
              "two_clear": {row: sum(col(row, "two_clear")) for row in by}, "broken": {row: sum(col(row, "broken")) for row in by},
              "per_pair_two_clear": {row: {p: sum(by[row][(p, s)]["two_clear"] for s in SEEDS) for p in PAIRS} for row in by},
              "per_subject_both_reviewers": {row: {m: sum(col(row, m)) for m in ("a_own", "b_own")} for row in by},
              "call": call, "detail": detail}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"pack": lambda: pack(*sys.argv[2:5]), "score": lambda: score(*sys.argv[2:6])}[sys.argv[1]]()
