"""Tables 1 and 3 restricted to prompts and pairs that share no concept with the training pairs.

Reads the same per-image file as 1-lora-correction/scripts/review_build_tables.py (scores_long.csv) and the training
pool (pair_pool_v58.yaml). A concept is a noun phrase of the prompt with its article and colour word
removed ("a red car" -> "car"). An image is kept when neither of its two concepts appears in any
training pair. The full-set means are printed first, so the numbers can be checked against
main_table.csv and held_out_ablation.csv before the subset is read.

Usage:
  python3 1-lora-correction/scripts/review_no_overlap_tables.py --scores <scores_long.csv> --out <results dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from statistics import mean

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from review_build_tables import GROUPS, NAME, ORDER, SETS, two_level  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml")
COLOURS = {"red", "blue", "green", "yellow", "purple", "pink", "orange", "black", "white", "brown",
           "gray", "grey", "gold"}


def slug_concepts(slug: str) -> tuple[str, ...]:
    out = []
    for part in slug.split("__x__"):
        words = [w for w in part.split("_") if w not in ("a", "an") and w not in COLOURS]
        out.append(" ".join(words))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    train = yaml.safe_load(open(POOL))["train"]
    train_concepts = {c for s in train for c in slug_concepts(s)}
    rows = list(csv.DictReader(open(a.scores)))
    for r in rows:
        r["shared"] = sum(c in train_concepts for c in slug_concepts(r["pair"]))

    result = {"training_concepts": len(train_concepts), "t1": {}, "t3": {}, "bar": {}}
    rng = random.Random(0)
    for subset, keep in (("all", lambda r: True), ("no shared concept", lambda r: r["shared"] == 0)):
        sub = [r for r in rows if keep(r)]
        for m in ORDER:
            for s in SETS:
                cell = [r for r in sub if r["table"] == "R1" and r["method"] == m and r["group"] == s]
                if not cell:
                    continue
                vals = {k: [float(r[k]) for r in cell if r[k] != ""] for k in ("clip", "blip", "ir", "comp")}
                result["t1"][f"{subset} | {NAME[m]} | {s}"] = {
                    "prompts": len({r["pair"] for r in cell}), "images": len(cell),
                    **{k: round(mean(v), 4) for k, v in vals.items() if v}}
            for g in GROUPS:
                by_pair = defaultdict(list)
                for r in sub:
                    if r["table"] == "R3" and r["method"] == m and r["group"] == g and r["blip"] != "":
                        by_pair[r["pair"]].append(float(r["blip"]))
                if not by_pair:
                    continue
                lo, hi = two_level(by_pair, rng, a.draws)
                allv = [v for vs in by_pair.values() for v in vs]
                result["t3"][f"{subset} | {NAME[m]} | {g}"] = {
                    "pairs": len(by_pair), "images": len(allv), "blip": round(mean(allv), 4),
                    "ci_half": round((hi - lo) / 2, 4)}
        idx = {(r["method"], r["pair"], r["seed"]): r for r in sub if r["table"] == "R1" and r["blip"] != ""}
        for s in SETS:
            by_prompt = defaultdict(list)
            for (m, pair, seed), r in idx.items():
                if m == "ours" and r["group"] == s and ("poe", pair, seed) in idx:
                    by_prompt[pair].append(float(r["blip"]) - float(idx[("poe", pair, seed)]["blip"]))
            if by_prompt:
                lo, hi = two_level(by_prompt, rng, a.draws)
                result["bar"][f"{subset} | {s}"] = {
                    "prompts": len(by_prompt), "ours_minus_poe": round(mean([v for vs in by_prompt.values() for v in vs]), 4),
                    "ci": [round(lo, 4), round(hi, 4)]}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "no_overlap_tables.json"), "w") as f:
        json.dump(result, f, indent=1)
    for part in ("t1", "t3", "bar"):
        for k, v in result[part].items():
            print(part, k, v)


if __name__ == "__main__":
    main()
