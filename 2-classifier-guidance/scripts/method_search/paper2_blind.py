#!/usr/bin/env python3
"""The blind count for step M1, Paper 2.

prepare  copies each render to a neutral name and writes tiles.csv (id and prompt) for whoever makes the calls. The key that
         names each picture's method is written to a separate folder, and is read only by `score`.
score    joins the saved calls to the key, counts faithful pairs per row, and applies the bars in beat_baseline_bars.py.

Each call is one row of a CSV with the columns id, rung, faithful, damaged and note. rung is one, hybrid, two_bodies or
two_clear. faithful and damaged are 0 or 1. A faithful pair must sit on the two_clear rung and must not be damaged.

Usage: python paper2_blind.py prepare --renders <dir> --blind <dir> --key <dir>
       python paper2_blind.py score --calls <csv> --key <dir> --check <check.json> --out <result.json>
"""
import argparse
import csv
import hashlib
import json
import random
import secrets
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_baseline_bars as bars  # noqa: E402

ROWS = ("co3", "push", "random")
RUNGS = ("one", "hybrid", "two_bodies", "two_clear")


def prepare(args):
    items = []
    for row in ROWS:
        for meta_path in sorted((args.renders / row).glob("*/seed-*.json")):
            meta = json.loads(meta_path.read_text())
            items.append({"row": row, "pair": meta["pair"], "seed": meta["seed"], "prompt": meta["prompt"],
                          "image": meta_path.with_suffix(".png")})
    counts = Counter((i["pair"], i["seed"]) for i in items)
    incomplete = sorted(k for k, n in counts.items() if n != len(ROWS))
    if incomplete:
        raise SystemExit(f"pair and seed cells missing a row: {incomplete[:5]}")
    salt = secrets.token_hex(16)
    for item in items:
        name = f"{salt}|{item['row']}|{item['pair']}|{item['seed']}"
        item["id"] = hashlib.sha256(name.encode()).hexdigest()[:12]
    random.Random(salt).shuffle(items)
    args.blind.mkdir(parents=True, exist_ok=True)
    args.key.mkdir(parents=True, exist_ok=True)
    with open(args.blind / "tiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt"])
        for item in items:
            shutil.copyfile(item["image"], args.blind / f"{item['id']}.png")
            writer.writerow([item["id"], item["prompt"]])
    key = {i["id"]: {"row": i["row"], "pair": i["pair"], "seed": i["seed"]} for i in items}
    (args.key / "key.json").write_text(json.dumps(key))
    print(json.dumps({"pictures": len(items), "pairs": len({i["pair"] for i in items}),
                      "seeds": len({i["seed"] for i in items})}))


def read_calls(path, key):
    calls = {}
    with open(path, newline="") as f:
        for line in csv.DictReader(f):
            tile = line["id"].strip()
            if tile in calls:
                raise SystemExit(f"two calls for {tile}")
            rung, faithful, damaged = line["rung"].strip(), int(line["faithful"]), int(line["damaged"])
            if rung not in RUNGS:
                raise SystemExit(f"{tile}: unknown rung {rung}")
            if faithful and (rung != "two_clear" or damaged):
                raise SystemExit(f"{tile}: a faithful pair must be two_clear and undamaged")
            calls[tile] = {"rung": rung, "faithful": faithful, "damaged": damaged}
    missing, extra = set(key) - set(calls), set(calls) - set(key)
    if missing or extra:
        raise SystemExit(f"{len(missing)} pictures have no call and {len(extra)} calls name no picture")
    return calls


def score(args):
    key = json.loads((args.key / "key.json").read_text())
    calls = read_calls(args.calls, key)
    zero_max = json.loads(args.check.read_text())["zero_push_max_abs"]
    table = {row: {} for row in ROWS}
    for tile, where in key.items():
        table[where["row"]].setdefault(where["pair"], {})[where["seed"]] = calls[tile]
    pairs = sorted(table["co3"])
    faithful = {row: {p: sum(c["faithful"] for c in table[row][p].values()) for p in pairs} for row in ROWS}
    total = {row: sum(faithful[row].values()) for row in ROWS}
    lead = lambda a, b: [faithful[a][p] - faithful[b][p] for p in pairs]
    se_push, se_random = bars.se_over_groups(lead("push", "co3")), bars.se_over_groups(lead("random", "co3"))
    verdict = bars.paper2_verdict(total["push"], total["co3"], total["random"], se_push, se_random, zero_max)
    result = {"pictures_per_row": {row: sum(len(v) for v in table[row].values()) for row in ROWS}, "faithful": total,
              "faithful_per_pair": faithful,
              "rungs": {row: dict(Counter(c["rung"] for v in table[row].values() for c in v.values())) for row in ROWS},
              "damaged": {row: sum(c["damaged"] for v in table[row].values() for c in v.values()) for row in ROWS},
              "lead_push_over_co3": total["push"] - total["co3"], "se_push_minus_co3": se_push,
              "lead_random_over_co3": total["random"] - total["co3"], "se_random_minus_co3": se_random,
              "min_lead": bars.PAPER2_MIN_LEAD, "z": bars.Z, "zero_push_max_abs": zero_max, "verdict": verdict}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("faithful", "damaged", "lead_push_over_co3", "se_push_minus_co3", "verdict")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--blind", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--calls", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--check", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (prepare if args.stage == "prepare" else score)(args)


if __name__ == "__main__":
    main()
