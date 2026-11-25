#!/usr/bin/env python3
"""Loop 12 step Z12, stage A: blind packing, settling, the futility rule and the bar.

Design: artifacts/notes/progress/leakage/12-aligned-classifier-on-co3/design.md, "How pictures are counted", "The bar" and
"A rule that can only stop".

  pack   <z12 out> <batch dir> <key dir> half|rest|all   neutral ids; both rows of the chosen pairs at seeds 6017 to 6024, one shuffle
  count  <batch dir> <key dir> <calls1> <calls2> [<calls3>]   settled rung and broken per picture (a third call settles any split)
  half   <settled json> [...]                             the futility rule on the first half's 48 matched pictures
  bar    <settled json> [...]                             the bar on all 96 matched pictures

Call files: id, rung (one, hybrid, two_bodies, two_clear), broken (0 or 1), faithful, damaged, a_own, b_own, note.
"""
import csv
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

SEEDS = tuple(range(6017, 6025))
ROWS = ("co3", "box")
# the first two pairs of each of the three render jobs (pairs[shard::3]); the design names these six as the first half
HALF = ("a_puma__x__a_condor", "a_hippopotamus__x__a_swallow", "a_zebra__x__a_flamingo", "a_giraffe__x__a_tortoise",
        "a_seal__x__a_gull", "a_bobcat__x__a_quail")
REST = ("a_duck__x__a_goose", "a_lighthouse__x__a_sailboat", "a_crow__x__a_raven", "a_boar__x__a_pheasant",
        "an_eagle__x__a_fox", "a_cheetah__x__a_cougar")


def pack(out, batch, keydir, which):
    pairs = {"half": HALF, "rest": REST, "all": HALF + REST}[which]
    batch, keydir = Path(batch), Path(keydir)
    (batch / "pictures").mkdir(parents=True, exist_ok=True); keydir.mkdir(parents=True, exist_ok=True)
    items = [(r, p, s, Path(out) / r / p / f"seed-{s}.png") for r in ROWS for p in pairs for s in SEEDS]
    missing = [str(x[-1]) for x in items if not x[-1].exists()]
    if missing:
        raise SystemExit(f"{len(missing)} missing, first {missing[0]}")
    rng = random.Random(f"z12-{which}"); rng.shuffle(items)
    key, rows = {}, []
    for r, p, s, src in items:
        pid = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(12))
        shutil.copy(src, batch / "pictures" / f"{pid}.png")
        a, b = p.split("__x__")
        key[pid] = {"row": r, "pair": p, "seed": s}
        rows.append({"id": pid, "prompt": f"{a.replace('_', ' ')} and {b.replace('_', ' ')}"})
    with open(batch / "sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["id", "prompt"]); w.writeheader(); w.writerows(rows)
    (keydir / f"key-{which}.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"packed {len(rows)} into {batch}")


def read(p):
    return {r["id"]: r for r in csv.DictReader(open(p))}


def count(batch, keydir, c1, c2, c3=None):
    batch = Path(batch)
    key = {}
    for f in Path(keydir).glob("key-*.json"):
        key.update(json.loads(f.read_text()))
    ids = [r["id"] for r in csv.DictReader(open(batch / "sheet.csv"))]
    a, b = read(c1), read(c2); third = read(c3) if c3 else {}
    settled, need = {}, []
    for i in ids:
        if a[i]["rung"] == b[i]["rung"]:
            rec = {k: a[i].get(k) for k in ("rung", "broken", "faithful", "damaged", "a_own", "b_own")}
            rec["broken"] = int(a[i]["broken"]) if a[i]["broken"] == b[i]["broken"] else int(int(a[i]["broken"]) and int(b[i]["broken"]))
            settled[i] = rec
        elif i in third:
            settled[i] = {k: third[i].get(k) for k in ("rung", "broken", "faithful", "damaged", "a_own", "b_own")}
            settled[i]["broken"] = int(third[i]["broken"])
        else:
            need.append(i)
    if need:
        (batch / "need-third.txt").write_text("\n".join(need) + "\n")
        print(f"{len(need)} need a third call"); return
    out = {"rung_agreement": sum(a[i]["rung"] == b[i]["rung"] for i in ids) / len(ids), "settled": {i: {**key[i], **settled[i]} for i in ids}}
    (batch / "settled.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"rung_agreement": out["rung_agreement"], "pictures": len(ids)}))


def merged(files):
    by = {}
    for f in files:
        for r in json.loads(Path(f).read_text())["settled"].values():
            by[(r["row"], r["pair"], r["seed"])] = r
    return by


def compare(by, pairs):
    d = []
    for p in pairs:
        for s in SEEDS:
            x = int(by[("box", p, s)]["rung"] == "two_clear"); y = int(by[("co3", p, s)]["rung"] == "two_clear")
            d.append(x - y)
    T = {r: sum(1 for p in pairs for s in SEEDS if by[(r, p, s)]["rung"] == "two_clear") for r in ROWS}
    lost = sum(1 for p in pairs for s in SEEDS if by[("co3", p, s)]["rung"] == "two_clear" and by[("box", p, s)]["rung"] != "two_clear")
    sd = statistics.stdev(d) if len(set(d)) > 1 else 0.0
    se = sd / len(d) ** 0.5 if sd else 0.0
    per_pair = {p: {r: sum(1 for s in SEEDS if by[(r, p, s)]["rung"] == "two_clear") for r in ROWS} for p in pairs}
    return {"n": len(d), "two_clear": T, "lead": T["box"] - T["co3"], "won": d.count(1), "lost_two_clear": lost,
            "lead_in_se": (sum(d) / len(d)) / se if se else (float("inf") if sum(d) > 0 else 0.0), "per_pair": per_pair}


def half(*files):
    res = compare(merged(files), HALF)
    res["stop"] = res["lead"] <= 0
    res["rule"] = "stop for futility at a two-clear lead of 0 or less on the first half's 48 matched pictures"
    print(json.dumps(res, indent=1)); return res


def bar(*files):
    by = merged(files)
    res = compare(by, HALF + REST)
    res["broken"] = {r: sum(1 for k, v in by.items() if k[0] == r and int(v["broken"])) for r in ROWS}
    res["passes"] = res["lead"] >= 5 and res["lead_in_se"] > 1.96 and res["lost_two_clear"] <= 2
    print(json.dumps(res, indent=1)); return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"pack": lambda: pack(*sys.argv[2:6]), "count": lambda: count(*sys.argv[2:7]), "half": lambda: half(*sys.argv[2:]),
     "bar": lambda: bar(*sys.argv[2:])}[cmd]()
