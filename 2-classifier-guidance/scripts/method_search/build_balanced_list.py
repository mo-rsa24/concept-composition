#!/usr/bin/env python3
"""Build the two training lists that step M5 compares, and print what each one holds.

Both lists start from training_set_multipair.json, drop every pair the subject guard refuses, and hold out the pairs whose
name hash is 0 modulo 4 (board question 37).

current    keeps the trainer's sampling: pick one of the ten sources, then a picture. Six sources hold only cat and dog.
balanced   keeps at most CAP_PER_LABEL clean and CAP_PER_LABEL reject cat and dog pictures, the first by SHA-256 of their id, and
           sets each picture's source to its pair, so the trainer's "pick a source" step becomes "pick a pair".

Usage: python build_balanced_list.py --out <folder for the two lists> --summary <composition.json>
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "shared"))
from disallowed_subjects import is_allowed  # noqa: E402

SOURCE = REPO / "2-classifier-guidance" / "scripts" / "rung2" / "training_set_multipair.json"
SPLITS = REPO / "2-classifier-guidance" / "scripts" / "checker_choice" / "splits.json"
CAT_DOG = "a_cat__x__a_dog"
CAP_PER_LABEL = 8   # 16 pictures in all, the size of the largest other pair


def name_hash(text):
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def held_out_by_name(pair):
    return name_hash(pair) % 4 == 0


def describe(tiles):
    """What the list holds, and the share of a batch's draws that land on cat and dog under the trainer's sampling."""
    by = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}
    for tile in tiles:
        by[tile["label"]][tile["source"]].append(tile)

    def cat_dog_share(label):
        groups = by[label]
        return sum(sum(t["pair"] == CAT_DOG for t in g) / len(g) for g in groups.values()) / len(groups)

    return {"pictures": len(tiles), "clean": sum(t["label"] for t in tiles), "rejects": sum(1 - t["label"] for t in tiles),
            "pairs": len({t["pair"] for t in tiles}), "pairs_with_a_clean_picture": len({t["pair"] for t in tiles if t["label"]}),
            "cat_and_dog_pictures": sum(t["pair"] == CAT_DOG for t in tiles),
            "sampling_groups": {"clean": len(by[1]), "reject": len(by[0])},
            "cat_and_dog_share_of_clean_draws": round(cat_dog_share(1), 4),
            "cat_and_dog_share_of_reject_draws": round(cat_dog_share(0), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(SOURCE.read_text())
    splits = json.loads(SPLITS.read_text())
    allowed = [t for t in data["tiles"] if not t.get("check") and is_allowed(t["pair"].replace("_", " "))]
    held = sorted({t["pair"] for t in allowed if held_out_by_name(t["pair"])})
    if CAT_DOG in held:
        raise SystemExit("cat and dog falls in the held-out quarter, so this rule needs the owner's decision")
    current = [t for t in allowed if t["pair"] not in held]

    balanced = []
    for label in (0, 1):
        cat_dog = sorted((t for t in current if t["pair"] == CAT_DOG and t["label"] == label), key=lambda t: name_hash(t["id"]))
        balanced += [dict(t, orig_source=t["source"], source=t["pair"]) for t in cat_dog[:CAP_PER_LABEL]]
    balanced += [dict(t, orig_source=t["source"], source=t["pair"]) for t in current if t["pair"] != CAT_DOG]

    evaluation = set(splits["tuning_pairs"]) | set(splits["held_out_pairs"])
    leaked = sorted(evaluation & {t["pair"] for t in balanced + current})
    if leaked:
        raise SystemExit(f"evaluation pairs in a training list: {leaked}")

    summary = {"source": str(SOURCE.relative_to(REPO)), "rule": "guard, then hold out pairs whose SHA-256 is 0 modulo 4",
               "pairs_before_holdout": len({t["pair"] for t in allowed}), "pairs_held_out": len(held), "held_out_pairs": held,
               "cat_and_dog_cap_per_label": CAP_PER_LABEL, "evaluation_pairs_in_either_list": 0,
               "current": describe(current), "balanced": describe(balanced)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, tiles in (("current", current), ("balanced", balanced)):
            payload = {"rule": summary["rule"], "sources": sorted({t["source"] for t in tiles}),
                       "held_out_pairs": sorted(set(data["held_out_pairs"]) | set(held)), "tiles": tiles}
            (args.out / f"training_set_{name}.json").write_text(json.dumps(payload))
    print(json.dumps({k: summary[k] for k in ("pairs_before_holdout", "pairs_held_out", "current", "balanced")}, indent=1))


if __name__ == "__main__":
    main()
