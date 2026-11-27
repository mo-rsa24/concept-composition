"""List the renamed wrong-species pictures, for when natural wrong species runs short (species judge step 0.8).

The plan ("The labelled check set", "If wrong species is short") makes them by renaming: a picture the reviewers
settled as two clear, on a pair holding a species in paper2_bars.RENAMING, is shown with that asked name replaced by
its stand-in: "a cougar" as "a leopard", "a goose" as "a swan", "a fox" as "a coyote", "a flamingo" as "a spoonbill".
The picture is unchanged; only what the reviewers and the judges are asked changes, so a reviewer who sees a cougar
where a leopard is asked calls the body OTHER (rubric version 4: a close relative is OTHER), and the picture settles as
wrong species, or as whatever the reviewers do settle it as.

Reads the labelled set (paper2_label_set.py's labelled-set.csv; a confirmation picture's kind is sealed there, so it is
never renamed) and the collector's candidates.csv for each picture's own columns. Writes renamed-list.csv: the
collector's columns with a or b replaced, pair_type recomputed, renamed True and renaming "<species> as <stand-in>",
every pair checked by the subject guard, ordered by RENAMING_SEED (5), as many as --count (all by default). The pair
slug stays the picture's own, so a renamed picture always falls on the same side of the split as its original.
paper2_sheets.py builds it as a blind batch; paper2_label_set.py keeps a renamed picture apart from its original, holds
the renamed to at most RENAMED_PER_KIND_MAX (20) of a kind's 40 read, and reports natural and renamed apart.

    "$PY" paper2_renamed.py --labelled-set <paper2-label-set>/labelled-set.csv \\
        --candidates <paper2-collect>/candidates.csv --out <dir>/renamed-list.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts/shared"))

from paper2_bars import RENAMING, RENAMING_SEED, pair_type  # noqa: E402
from paper2_common import with_article  # noqa: E402
from paper2_kinds import TWO_CLEAR  # noqa: E402


def renamed_rows(labelled: list[dict], candidates: dict[str, dict]) -> list[dict]:
    from disallowed_subjects import is_allowed
    out = []
    for r in labelled:
        if r.get("kind") != TWO_CLEAR or r.get("renamed") == "True" or r.get("part") == "confirmation":
            continue
        a, b = r["a"], r["b"]
        hit = [s for s in (a, b) if s in RENAMING]
        if len(hit) != 1:
            continue
        old = hit[0]
        new_a, new_b = (RENAMING[a], b) if old == a else (a, RENAMING[b])
        if not is_allowed(f"{with_article(new_a)} and {with_article(new_b)}"):
            continue
        base = dict(candidates.get(r["picture"], {"picture": r["picture"], "pair": r["pair"], "seed": r["seed"],
                                                  "source": r.get("source", "")}))
        base.update({"a": new_a, "b": new_b, "pair_type": pair_type(new_a, new_b), "renamed": True,
                     "renaming": f"{old} as {RENAMING[old]}", "earlier_call": "", "earlier_scale": "renamed two clear",
                     "unselected": False})
        out.append(base)
    out.sort(key=lambda r: r["picture"])
    random.Random(RENAMING_SEED).shuffle(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labelled-set", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="renamed-list.csv")
    ap.add_argument("--count", type=int, default=None, help="the first N in seed order (default: all)")
    a = ap.parse_args(argv)
    with open(a.labelled_set, newline="") as f:
        labelled = list(csv.DictReader(f))
    with open(a.candidates, newline="") as f:
        cands = {r["picture"]: r for r in csv.DictReader(f)}
    rows = renamed_rows(labelled, cands)
    chosen = rows[: a.count] if a.count else rows
    cols = list(next(iter(cands.values())).keys()) if cands else ["picture", "a", "b", "pair", "seed", "source"]
    cols = [c for c in cols if c not in ("renamed", "renaming")] + ["renamed", "renaming"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(chosen)
    print(f"{len(rows)} settled two-clear pictures can be renamed; wrote {len(chosen)} to {a.out}")
    print("by renaming entry: " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(r["renaming"] for r in rows).items())))
    print("by pair type after renaming: " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(r["pair_type"] for r in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
