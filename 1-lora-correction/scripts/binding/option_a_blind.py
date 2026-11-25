#!/usr/bin/env python3
"""The binding plan's option A: blind calls on all four rows in one shuffled batch, and the bar.

Plan: 1-lora-correction/binding-plan.md, section 2. All 128 pictures (8 pairs by 4 seeds by 4 rows) are called fresh in one batch
under neutral ids, so no row is called in a different sitting from another; Z10's earlier calls on rows 1 and 4 are not reused.

  pack   <z10 renders> <option-a out> <batch dir> <key dir>   neutral ids, one shuffled sheet; the key kept apart
  count  <batch dir> <key dir> <calls1> <calls2> [<calls3>]   settled rung and broken flag per picture
  bar    <key dir> <settled json>                             the bar written in the plan before the run

Rows: lora (Z10 weight 1.00), own (row 2), both (row 3), plain (Z10 weight 0.00). Each call file has columns id, rung (one, hybrid,
two_bodies, two_clear), broken (0 or 1), note. A picture is settled when the two calls agree on the rung; a third call settles the
rest. Broken is settled the same way.
"""
import csv
import json
import random
import shutil
import sys
from math import comb
from pathlib import Path

SEEDS = (6001, 6002, 6003, 6004)
ROWS = ("lora", "own", "both", "plain")
RUNGS = ("one", "hybrid", "two_bodies", "two_clear")
PAIRS = ("a_bee__x__an_owl", "a_butterfly__x__a_crab", "a_flamingo__x__a_sloth", "a_fox__x__a_llama", "a_goat__x__a_rhinoceros",
         "a_kangaroo__x__a_squid", "a_ladybug__x__an_octopus", "a_lobster__x__a_whale")


def source(z10, opt, row, pair, seed):
    if row == "lora":
        return Path(z10) / pair / f"seed-{seed}-w1.00.png"
    if row == "plain":
        return Path(z10) / pair / f"seed-{seed}-w0.00.png"
    return Path(opt) / row / pair / f"seed-{seed}.png"


def pack(z10, opt, batch, keydir):
    batch, keydir = Path(batch), Path(keydir)
    (batch / "pictures").mkdir(parents=True, exist_ok=True); keydir.mkdir(parents=True, exist_ok=True)
    items = [(r, p, s, source(z10, opt, r, p, s)) for r in ROWS for p in PAIRS for s in SEEDS]
    missing = [str(x[-1]) for x in items if not x[-1].exists()]
    if missing:
        raise SystemExit(f"{len(missing)} missing, first {missing[0]}")
    rng = random.Random("lora-binding-option-a"); rng.shuffle(items)
    key, rows = {}, []
    for r, p, s, src in items:
        pid = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(12))
        shutil.copy(src, batch / "pictures" / f"{pid}.png")
        a, b = p.split("__x__")
        key[pid] = {"row": r, "pair": p, "seed": s}
        rows.append({"id": pid, "prompt": f"{a.replace('_', ' ')} and {b.replace('_', ' ')}"})
    with open(batch / "sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["id", "prompt"]); w.writeheader(); w.writerows(rows)
    (keydir / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"packed {len(rows)} into {batch}; key {keydir / 'key.json'}")


def read(p):
    return {r["id"]: r for r in csv.DictReader(open(p))}


def count(batch, keydir, c1, c2, c3=None):
    key = json.loads((Path(keydir) / "key.json").read_text())
    a, b = read(c1), read(c2); third = read(c3) if c3 else {}
    settled, need = {}, []
    for i in key:
        if a[i]["rung"] == b[i]["rung"] and a[i]["broken"] == b[i]["broken"]:
            settled[i] = {"rung": a[i]["rung"], "broken": int(a[i]["broken"])}
        elif i in third:
            rung = a[i]["rung"] if a[i]["rung"] == b[i]["rung"] else third[i]["rung"]
            broken = int(a[i]["broken"]) if a[i]["broken"] == b[i]["broken"] else int(third[i]["broken"])
            settled[i] = {"rung": rung, "broken": broken}
        else:
            need.append(i)
    if need:
        (Path(batch) / "need-third.txt").write_text("\n".join(need) + "\n")
        print(f"{len(need)} need a third call"); return
    out = {"rung_agreement": sum(a[i]["rung"] == b[i]["rung"] for i in key) / len(key),
           "settled": {i: {**key[i], **settled[i]} for i in key}}
    (Path(batch) / "settled.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"rung_agreement": out["rung_agreement"]}))


def bar(keydir, settled_file):
    sf = lambda k, n: sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n  # exact one-sided binomial, P(X >= k)
    rows = json.loads(Path(settled_file).read_text())["settled"].values()
    by = {(r["row"], r["pair"], r["seed"]): r for r in rows}
    cnt = {row: {rg: sum(1 for k, v in by.items() if k[0] == row and v["rung"] == rg) for rg in RUNGS} for row in ROWS}
    broken = {row: sum(1 for k, v in by.items() if k[0] == row and v["broken"]) for row in ROWS}
    T = {row: cnt[row]["two_clear"] for row in ROWS}

    def mcnemar(x, y):
        won = sum(1 for p in PAIRS for s in SEEDS if by[(x, p, s)]["rung"] == "two_clear" and by[(y, p, s)]["rung"] != "two_clear")
        lost = sum(1 for p in PAIRS for s in SEEDS if by[(y, p, s)]["rung"] == "two_clear" and by[(x, p, s)]["rung"] != "two_clear")
        return {"lead": T[x] - T[y], "won": won, "lost": lost, "p": float(sf(won, won + lost)) if won + lost else 1.0}
    vs_lora, vs_own = mcnemar("both", "lora"), mcnemar("both", "own")
    ok = lambda m: m["lead"] >= 5 and m["p"] < 0.05
    passes = ok(vs_lora) and ok(vs_own) and broken["both"] <= broken["lora"] + 3
    if passes:
        call = "pass"
    elif ok(mcnemar("own", "lora")) and not ok(vs_own):
        call = "identity comes from the box rule; the LoRA adds nothing"
    else:
        call = "fail"
    res = {"pictures": len(by), "counts_per_row": cnt, "two_clear": T, "broken": broken, "both_vs_lora": vs_lora, "both_vs_own": vs_own,
           "own_vs_lora": mcnemar("own", "lora"), "passes": passes, "call": call}
    print(json.dumps(res, indent=1)); return res


if __name__ == "__main__":
    {"pack": lambda: pack(*sys.argv[2:6]), "count": lambda: count(*sys.argv[2:7]), "bar": lambda: bar(*sys.argv[2:4])}[sys.argv[1]]()
