#!/usr/bin/env python
"""Count animals in the route step 4 census pictures, for the competing animal pairs.

Uses the validated instance count (poe_repair/experiments/compose_scorer_validation/
detection_scorer.py: GroundingDINO asked for "animal", boxes under a quarter of the image's
longer side dropped, overlapping boxes merged; two or more boxes means two animals). It is
validated on animal pairs only, so the 12 competing object pairs are judged by eye instead
(report/paper-review/protocol.md and the step 4 walk map).

One row per picture: pair, seed, column (mono is the joint prompt, poe the plain product),
count. Reads the census written by scripts/review_step4_renders.sh; writes nothing else.

    python scripts/review_step4_count.py <census dir> <out.csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ANIMAL_PAIRS = [
    "a_baboon__x__a_gazelle", "a_bear__x__a_salmon", "a_bear__x__a_squirrel", "a_bison__x__a_horse",
    "a_cat__x__a_horse", "a_cat__x__a_lion", "a_cow__x__a_horse", "a_deer__x__a_hedgehog",
    "a_dog__x__a_horse", "a_fox__x__a_wolf", "a_giraffe__x__a_lion", "a_lion__x__a_dog",
    "a_lion__x__a_horse", "a_peacock__x__a_sheep", "a_raccoon__x__a_salmon", "a_tiger__x__a_dog",
    "a_wolf__x__a_horse", "a_wolf__x__a_husky",
]
SEEDS = range(9, 17)


def main() -> int:
    import torch
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    census, out = Path(sys.argv[1]), Path(sys.argv[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, missing = [], 0
    for pair in ANIMAL_PAIRS:
        for seed in SEEDS:
            for column in ("mono", "poe"):
                png = census / pair / f"seed{seed:02d}" / f"{column}.png"
                if not png.exists():
                    missing += 1
                    continue
                n, _ = count_instances(png, device=device)
                rows.append({"pair": pair, "seed": seed, "column": column, "count": int(n)})
        print(pair, sum(1 for r in rows if r["pair"] == pair), flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "seed", "column", "count"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}; {missing} pictures missing")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
