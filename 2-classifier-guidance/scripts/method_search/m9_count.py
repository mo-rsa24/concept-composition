#!/usr/bin/env python3
"""The blind count for step M9, rung 1.

prepare  copies the 48 renders to neutral names, writes tiles.csv for the reviewers, and keeps the key in a separate folder.
differ   lists the pictures where the two reviewers disagree, by id only.
score    joins the final calls to the key and applies the bar written in the M9 design.

A reviewer's CSV has the columns id, rung, faithful, damaged, animals and note. animals is the number of separate, clear animals.
The third-call CSV has the columns id, two_clear and two_animals, with 0 or 1 in each.

Usage: python m9_count.py prepare --renders <rung1 folder> --blind <dir> --key <dir>
       python m9_count.py differ --first <csv> --second <csv>
       python m9_count.py score --first <csv> --second <csv> [--third <csv>] --key <dir> --out <json>
"""
import argparse
import csv
import hashlib
import json
import random
import secrets
import shutil
from pathlib import Path

SEEDS = (9, 12, 16, 19, 22, 25, 27, 30)
PAIR = "pairs/a_cat__x__a_dog"
ROWS = {"joint": "joint/seed{seed:02d}.png",
        "plain": PAIR + "/seed_{seed}/poe/poe.png",
        "right": PAIR + "/seed_{seed}/poe_regional_oracle_even/poe_regional_oracle_even.png",
        "random": PAIR + "/seed_{seed}/poe_regional_random_even/poe_regional_random_even.png",
        "swapped": PAIR + "/seed_{seed}/poe_regional_swapped_even/poe_regional_swapped_even.png",
        "same": PAIR + "/seed_{seed}/poe_regional_same_even/poe_regional_same_even.png"}
BAR = 4
RUNGS = ("one", "hybrid", "two_bodies", "two_clear")


def prepare(args):
    salt = secrets.token_hex(16)
    items = []
    for row, template in ROWS.items():
        for seed in SEEDS:
            path = args.renders / template.format(seed=seed)
            if not path.exists():
                raise SystemExit(f"missing {path}")
            items.append({"id": hashlib.sha256(f"{salt}|{row}|{seed}".encode()).hexdigest()[:12], "row": row, "seed": seed, "path": path})
    random.Random(salt).shuffle(items)
    args.blind.mkdir(parents=True, exist_ok=True)
    args.key.mkdir(parents=True, exist_ok=True)
    with open(args.blind / "tiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt"])
        for item in items:
            shutil.copyfile(item["path"], args.blind / f"{item['id']}.png")
            writer.writerow([item["id"], "a cat and a dog"])
    (args.key / "key.json").write_text(json.dumps({i["id"]: {"row": i["row"], "seed": i["seed"]} for i in items}))
    print(json.dumps({"pictures": len(items), "rows": len(ROWS), "seeds": len(SEEDS)}))


def read_calls(path):
    calls = {}
    for line in csv.DictReader(open(path, newline="")):
        tile, rung = line["id"].strip(), line["rung"].strip()
        if rung not in RUNGS or tile in calls:
            raise SystemExit(f"{path}: bad row for {tile}")
        calls[tile] = {"two_clear": int(rung == "two_clear"), "two_animals": int(int(line["animals"]) >= 2), "rung": rung,
                       "faithful": int(line["faithful"]), "damaged": int(line["damaged"])}
    return calls


def disagreements(first, second):
    if set(first) != set(second):
        raise SystemExit("the two reviewers called different pictures")
    return sorted(t for t in first if first[t]["two_clear"] != second[t]["two_clear"]
                  or first[t]["two_animals"] != second[t]["two_animals"])


def differ(args):
    first, second = read_calls(args.first), read_calls(args.second)
    tiles = disagreements(first, second)
    agree = sum(first[t]["two_clear"] == second[t]["two_clear"] for t in first)
    print(json.dumps({"pictures": len(first), "agree_on_two_clear": agree, "to_decide": tiles}))


def verdict(right, random_row, swapped, same):
    if right < 2:
        return "null"
    if right < BAR:
        return "inconclusive"
    if random_row >= right - 1:
        return "layout content irrelevant"
    if swapped >= right - 1 or same >= right - 1:
        return "regions without names"
    return "support"


def score(args):
    first, second = read_calls(args.first), read_calls(args.second)
    key = json.loads((args.key / "key.json").read_text())
    tiles = disagreements(first, second)
    third = {}
    if args.third:
        third = {l["id"].strip(): {"two_clear": int(l["two_clear"]), "two_animals": int(l["two_animals"])}
                 for l in csv.DictReader(open(args.third, newline=""))}
    missing = sorted(set(tiles) - set(third))
    if missing or set(first) != set(key):
        raise SystemExit(f"{len(missing)} disagreements have no third call, or the calls do not match the key")
    final = {t: (third[t] if t in third else {"two_clear": first[t]["two_clear"], "two_animals": first[t]["two_animals"]}) for t in key}
    rows = {row: {"clear_pairs": sum(final[t]["two_clear"] for t, k in key.items() if k["row"] == row),
                  "two_separate_animals": sum(final[t]["two_animals"] for t, k in key.items() if k["row"] == row),
                  "clear_pair_seeds": sorted(k["seed"] for t, k in key.items() if k["row"] == row and final[t]["two_clear"])}
            for row in ROWS}
    count = {row: rows[row]["clear_pairs"] for row in ROWS}
    count["same"] = rows["same"]["two_separate_animals"]
    result = {"of": len(SEEDS), "rows": rows, "counts_used_by_the_bar": count, "bar_on_right_boxes": BAR,
              "verdict": verdict(count["right"], count["random"], count["swapped"], count["same"]),
              "rung_2_runs": count["right"] >= BAR, "pictures_the_reviewers_differed_on": len(tiles),
              "reviewers_agree_on_two_clear": sum(first[t]["two_clear"] == second[t]["two_clear"] for t in key)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("counts_used_by_the_bar", "verdict", "rung_2_runs", "reviewers_agree_on_two_clear")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--blind", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    d = sub.add_parser("differ")
    d.add_argument("--first", type=Path, required=True)
    d.add_argument("--second", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--first", type=Path, required=True)
    s.add_argument("--second", type=Path, required=True)
    s.add_argument("--third", type=Path)
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"prepare": prepare, "differ": differ, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
