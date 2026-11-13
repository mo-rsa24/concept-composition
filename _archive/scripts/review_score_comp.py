"""Comp. for the paper-review renders: the validated instance count on animal prompts only.

Calls poe_repair.experiments.compose_scorer_validation.detection_scorer.count_instances unchanged
(GroundingDINO asked for "animal", small boxes dropped, overlaps merged); comp is 1 when it finds
two or more animals. It is validated on animal pairs only, so a row is counted when it is in R1's
animals set or is a held-out animal pair (the ANIMAL_PAIRS of scripts/review_step4_count.py);
every other row is left out, never guessed. Rows already in --out are skipped, so a run resumes.

Usage (co3 env, repo on PYTHONPATH, non-Blackwell GPU or CPU):
  python scripts/review_score_comp.py --in rows_a.csv rows_b.csv --out comp.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ANIMAL_PAIRS = {
    "a_baboon__x__a_gazelle", "a_bear__x__a_salmon", "a_bear__x__a_squirrel", "a_bison__x__a_horse",
    "a_cat__x__a_horse", "a_cat__x__a_lion", "a_cow__x__a_horse", "a_deer__x__a_hedgehog",
    "a_dog__x__a_horse", "a_fox__x__a_wolf", "a_giraffe__x__a_lion", "a_lion__x__a_dog",
    "a_lion__x__a_horse", "a_peacock__x__a_sheep", "a_raccoon__x__a_salmon", "a_tiger__x__a_dog",
    "a_wolf__x__a_horse", "a_wolf__x__a_husky",
    "an_eagle__x__a_hawk", "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "a_goose__x__a_swan",
    "a_seal__x__a_walrus",
}


def wanted(r: dict) -> bool:
    if r["run"].startswith("r1_") or r["run"].endswith("/r1"):  # baseline folders name their set as a subfolder
        return r["group"] == "animals"
    return r["pair"] in ANIMAL_PAIRS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    rows = [r for f in a.inp for r in csv.DictReader(open(f)) if wanted(r)]
    done = {r["image"] for r in csv.DictReader(open(a.out))} if os.path.exists(a.out) else set()
    todo = [r for r in rows if r["image"] not in done]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["image", "comp_count", "comp"])
        for n, r in enumerate(todo):
            k, _ = count_instances(Path(r["image"]), device=device)
            w.writerow([r["image"], int(k), int(k >= 2)])
            if n % 50 == 0:
                f.flush(); print(f"{n + 1}/{len(todo)}", flush=True)
    print(f"done: {len(todo)} counted, {len(done)} already in {a.out}, {len(rows)} animal rows in all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
