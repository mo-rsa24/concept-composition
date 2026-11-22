#!/usr/bin/env python3
"""Fix the pairs and seeds of the checker comparison (step D3) before any candidate trains.

An evaluation pair must be outside the 275-pair benchmark pool and outside the kept training list, pass the subject
guard, name no close relative of a guarded subject, and share no word with any pair in those two sets. The eligible
pairs are ordered by the SHA-256 of their name, so the split depends on nothing but the names. Walking that order, a
pair is taken when it shares no word with a pair already taken: the first 12 taken are tuning pairs and the next 12 are
held out. The remaining eligible pairs supply pictures with neither subject.

    python3 build_splits.py --pool <benchmark-pool index.json> --runs <pair-runs-pool index.json>
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "shared"))
from disallowed_subjects import is_allowed  # noqa: E402

TRAINING_LIST = REPO / "2-classifier-guidance" / "scripts" / "rung2" / "training_set_multipair.json"
TUNING_PAIRS, HELD_OUT_PAIRS = 12, 12
TUNING_SEEDS = list(range(6001, 6009))
HELD_OUT_SEEDS = list(range(7001, 7009))
# Not on the guard's lists, but a crocodilian and a close relative of the slug, so they stay out of new sheets.
NEAR_GUARD = {"caiman", "snail"}


def words(pair):
    return {word for word in re.split("_+", pair) if word not in {"a", "an", "the", "x"}}


def readable(pair):
    return pair.replace("_", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "splits.json")
    args = parser.parse_args()

    pool = {row["pair"] for row in json.loads(args.pool.read_text())}
    runs = {row["pair"] for row in json.loads(args.runs.read_text())}
    tiles = json.loads(TRAINING_LIST.read_text())["tiles"]
    trained = {tile["pair"] for tile in tiles}
    seen_words = set().union(*(words(pair) for pair in pool | trained))

    eligible = [p for p in runs - pool - trained
                if is_allowed(readable(p)) and not words(p) & seen_words and not words(p) & NEAR_GUARD]
    eligible.sort(key=lambda pair: hashlib.sha256(pair.encode()).hexdigest())
    kept_tiles = [tile for tile in tiles if is_allowed(readable(tile["pair"]))]

    chosen, used = [], set()
    for pair in eligible:
        if len(chosen) < TUNING_PAIRS + HELD_OUT_PAIRS and not words(pair) & used:
            chosen.append(pair)
            used |= words(pair)
    if len(chosen) < TUNING_PAIRS + HELD_OUT_PAIRS:
        raise SystemExit(f"only {len(chosen)} pairs share no word with each other")

    result = {
        "rule": __doc__.split("\n\n")[1].replace("\n", " "),
        "tuning_pairs": chosen[:TUNING_PAIRS],
        "held_out_pairs": chosen[TUNING_PAIRS:],
        "neither_source_pairs": [pair for pair in eligible if pair not in chosen],
        "tuning_seeds": TUNING_SEEDS,
        "held_out_seeds": HELD_OUT_SEEDS,
        "counts": {
            "pool_pairs": len(pool), "pair_run_pairs": len(runs), "eligible_pairs": len(eligible),
            "training_tiles_before_guard": len(tiles), "training_tiles_after_guard": len(kept_tiles),
            "training_pairs_after_guard": len({tile["pair"] for tile in kept_tiles}),
            "clean_tiles_after_guard": sum(tile["label"] == 1 for tile in kept_tiles),
            "reject_tiles_after_guard": sum(tile["label"] == 0 for tile in kept_tiles),
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))
    print("tuning:", ", ".join(result["tuning_pairs"]))


if __name__ == "__main__":
    main()
