#!/usr/bin/env python3
"""Build reproductions/table.csv from the scoreboard, the numbers table and each rival's recorded results.

One row per `qualitative` row per `targets` row with the same `paper` and `dataset`. The published number for a
rival is its row in numbers.csv with the same dataset, metric, protocol and task as the row behind the target. The
target's own rival uses the target's number_id. A rival with zero or several such rows gets no number and the row
counts as `not comparable` until the owner answers the filed question.

Results are read from <rival>/rows.json, a list of objects keyed by `target_id` (the target's number_id), holding
any of: ours, interval_low, interval_high, verdict, commit, run_id, render_folder, note.

    python3 build_table.py            write table.csv and print a summary
    python3 build_table.py --check    also exit 1 unless every row meets step D2's proof
    python3 build_table.py --unmatched   print the rows with no matching number and their nearest candidates
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRESS = HERE.parent
COLUMNS = ["rival", "number_id", "dataset", "metric", "protocol", "samples", "published", "ours", "interval_low",
           "interval_high", "verdict", "commit", "run_id", "render_folder", "note"]
VERDICTS = {"reproduced", "not reproduced", "not comparable", "blocked"}
# Published prompts the subject guard keeps out of every render. The owner's answer to board question 14 makes each
# row on these benchmarks `not comparable`, with our score on the allowed prompts recorded beside the published one.
GUARD_KEPT_OUT = {"T2I-CompBench Complex": (12, 1000), "RareBench Concat": (7, 40), "T2I-CompBench Color": (12, 300),
                  "T2I-CompBench Shape": (7, 300)}
NO_NUMBER_COMMIT = "c3462133-b235ea8"  # the commits that built the rows question 13 settled
GUARD_COMMIT = "e39a31ca-0739e7e"  # the commits that applied the subject guard


def load():
    board = json.loads((PROGRESS / "scoreboard.json").read_text())
    with open(PROGRESS / "targets" / "numbers.csv", newline="", encoding="utf-8") as handle:
        numbers = list(csv.DictReader(handle))
    return board, numbers, {row["id"]: row for row in numbers}


def pairs(board):
    for target in board["targets"]:
        for row in board["qualitative"]:
            if row["paper"] == target["paper"] and row["dataset"] == target["dataset"]:
                yield target, row["rival"]


def matches(numbers, source, rival):
    keys = ("dataset", "metric", "protocol", "task")
    return [r for r in numbers if r["paper"] == rival and all(r[k] == source[k] for k in keys)]


def recorded():
    found = {}
    for path in sorted(HERE.glob("*/rows.json")):
        rival = path.parent.name
        for row in json.loads(path.read_text()):
            found[(rival, row["target_id"])] = row
    return found


def build():
    board, numbers, by_id = load()
    results = recorded()
    table = []
    for target, rival in pairs(board):
        source = by_id[target["number_id"]]
        if rival == target["rival"]:
            hit = [source]
        else:
            hit = matches(numbers, source, rival)
        row = dict.fromkeys(COLUMNS, "")
        row["rival"] = rival
        base = hit[0] if len(hit) == 1 else source
        row.update(dataset=base["dataset"], metric=base["metric"], protocol=base["protocol"], samples=base["samples"])
        if len(hit) == 1:
            row.update(number_id=hit[0]["id"], published=hit[0]["value"])
        else:
            row["verdict"] = "not comparable"
            row["note"] = f"{len(hit)} rows in numbers.csv match this rival under the target's protocol"
            row.update(run_id="question 13", commit=NO_NUMBER_COMMIT)
        extra = results.get((rival, target["number_id"]), {})
        for key in COLUMNS:
            if key in extra:
                row[key] = extra[key]
        if row["dataset"] in GUARD_KEPT_OUT:
            kept_out, total = GUARD_KEPT_OUT[row["dataset"]]
            guard_note = f"the subject guard keeps {kept_out} of {total} published prompts out"
            row["verdict"] = "not comparable"
            row["note"] = "; ".join(part for part in (row["note"], guard_note) if part)
            row["run_id"] = row["run_id"] or "question 14"
            row["commit"] = row["commit"] or GUARD_COMMIT
        row["_target"] = target["number_id"]
        table.append(row)
    return table


def check(table):
    problems = []
    for row in table:
        name = f"{row['rival']} on {row['dataset']}"
        if row["verdict"] not in VERDICTS:
            problems.append(f"{name}: no verdict")
            continue
        if "-" not in row["commit"]:
            problems.append(f"{name}: both commits missing")
        if not row["run_id"]:
            problems.append(f"{name}: no run id or filed question")
    return problems


def unmatched():
    board, numbers, by_id = load()
    for target, rival in pairs(board):
        if rival == target["rival"]:
            continue
        source = by_id[target["number_id"]]
        if len(matches(numbers, source, rival)) == 1:
            continue
        print(f"\n{rival} | {target['dataset']} | target metric: {source['metric']} | target row {source['id']}")
        for row in numbers:
            if row["paper"] == rival and row["dataset"] == source["dataset"]:
                same = "same protocol" if row["protocol"] == source["protocol"] else "different protocol"
                print(f"  {row['id']}  {row['metric']}  task={row['task']}  value={row['value']}  "
                      f"{row['source_table']} p{row['source_page']}  {same}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--unmatched", action="store_true")
    args = parser.parse_args()
    if args.unmatched:
        unmatched()
        return
    table = build()
    with open(HERE / "table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    counts = {}
    for row in table:
        counts[row["verdict"] or "waiting"] = counts.get(row["verdict"] or "waiting", 0) + 1
    print(f"table.csv: {len(table)} rows, " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if args.check:
        problems = check(table)
        for line in problems:
            print("  " + line)
        if problems:
            sys.exit(1)
        print("every row meets the proof")


if __name__ == "__main__":
    main()
