"""Protocol rule 5's rows for the render grids R4, R10 and R12: per group, ours' highest- and lowest-BLIP-VQA cell.

R4 and R10 take R3's cells (the 53 held-out pairs, seeds 9 to 16), one pair of rows per held-out group; R12 takes
R1's cells (the 276 benchmark prompts, seeds 9 to 12), one pair of rows per benchmark set. A tie goes to list order
(scripts/review_held_out_pairs.txt or scripts/review_benchmark_pairs.txt), then the lower seed. Reads only the
scores; nobody looks at a picture to choose.

    python3 scripts/review_rule5_picks.py --scores <scores_long.csv> --repo <repo> > picks.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

GROUPS = {"r3_ours": ["hard", "easy but unseen", "not competing"],
          "r1_ours": ["animals", "animals_objects", "objects"]}


def order(repo: str) -> dict[str, int]:
    out = {}
    for name in ("review_held_out_pairs.txt", "review_benchmark_pairs.txt"):
        rows = [ln for ln in open(f"{repo}/scripts/{name}") if ln.strip() and not ln.startswith("#")]
        for i, ln in enumerate(rows):
            a, b = [c.strip() for c in ln.split("|")[:2]]
            out.setdefault(f"{a.replace(' ', '_')}__x__{b.replace(' ', '_')}", i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    rank = order(a.repo)
    cells = [r for r in csv.DictReader(open(a.scores)) if r["run"] in GROUPS and r["blip"] not in ("", "nan")]
    w = csv.writer(sys.stdout)
    w.writerow(["slot", "group", "which", "pair", "seed", "blip", "prompt"])
    for run, groups in GROUPS.items():
        slot = "R4,R10" if run == "r3_ours" else "R12"
        for g in groups:
            rows = [r for r in cells if r["run"] == run and r["group"] == g]
            key = lambda r: (rank.get(r["pair"], 10**6), int(r["seed"]))
            hi = max(float(r["blip"]) for r in rows)
            lo = min(float(r["blip"]) for r in rows)
            for which, v in (("highest", hi), ("lowest", lo)):
                r = min((r for r in rows if float(r["blip"]) == v), key=key)
                w.writerow([slot, g, which, r["pair"], r["seed"], f"{v:.4f}", r["prompt"]])


if __name__ == "__main__":
    main()
