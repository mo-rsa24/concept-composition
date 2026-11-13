"""Pick the rows of Figure 8 (slot R12) by score: per Table 1 set and per Table 3 group, the pair and
seed where our adapter's BLIP-VQA is highest, among cells scored for all four methods.

The rule is fixed here so the caption can state it: "rows are where our adapter scores highest on
BLIP-VQA in each set and group". Ties go to the higher mean BLIP-VQA of the other three methods, then
to the alphabetically first pair and lowest seed.

    python3 scripts/review_pick_by_blip.py <dir of blip_*.csv> [--top N]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AE = REPO / "data/prompt-sets/attend-and-excite/a.e_prompts.json"
HELD_OUT = REPO / "report/paper-review/held-out-set.md"
FILES = ["blip_batch1a.csv", "blip_batch1b.csv", "blip_batch2.csv", "blip_batch3.csv", "blip_co3.csv"]
METHODS = ("joint", "poe", "co3", "ours")


def method_of(path: str) -> tuple[str, str] | None:
    """(table, method) from an image path, or None for an image outside Tables 1 and 3."""
    if "/r1_joint/" in path: return "r1", "joint"
    if "/r1_poe/" in path: return "r1", "poe"
    if "/r1_ours/" in path: return "r1", "ours"
    if "/co3_default/r1/" in path: return "r1", "co3"
    if "/co3_default/r3/" in path: return "r3", "co3"
    if "/r3_ours/" in path: return "r3", "ours"
    if "/review_step4/census/" in path:
        return ("r3", "joint") if path.endswith("/mono.png") else ("r3", "poe") if path.endswith("/poe.png") else None
    return None


def slug(prompt: str) -> str:
    a, b = re.split(r" and | with ", prompt, maxsplit=1)
    return "__x__".join(re.sub(r"[^a-z0-9]+", "_", x.strip().lower()) for x in (a, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path)
    ap.add_argument("--top", type=int, default=1)
    a = ap.parse_args()
    score: dict[tuple, float] = {}
    for f in FILES:
        for row in csv.DictReader(open(a.scores / f)):
            m = method_of(row["image"])
            seed = re.search(r"/seed(\d+)/", row["image"])
            pair = re.search(r"/(an?_[^/]+__x__[^/]+)/seed", row["image"])
            if m and seed and pair:
                score[(m[0], pair[1], int(seed[1]), m[1])] = float(row["blip_vqa"])
    group: dict[tuple[str, str], str] = {}
    for name, prompts in json.load(open(AE)).items():
        for p in prompts:
            group[("r1", slug(p))] = name
    for line in HELD_OUT.read_text().splitlines():
        m = re.match(r"\| (hard|easy but unseen|easy|not competing) \| (.+?) × (.+?) \|", line)
        if m:
            s = "__x__".join(re.sub(r"[^a-z0-9]+", "_", ("a " + x if not x.startswith(("a ", "an ")) else x).lower()) for x in (m[2], m[3]))
            group[("r3", s)] = m[1]
    cells = defaultdict(dict)
    for (t, pair, seed, meth), v in score.items():
        cells[(t, pair, seed)][meth] = v
    by_group = defaultdict(list)
    for (t, pair, seed), d in cells.items():
        g = group.get((t, pair)) or group.get((t, pair.replace("an_", "a_")))
        if g and all(k in d for k in METHODS):
            others = sum(d[k] for k in ("joint", "poe", "co3")) / 3
            by_group[(t, g)].append((-d["ours"], -others, pair, seed, d))
    for key in sorted(by_group):
        rows = sorted(by_group[key])[: a.top]
        print(f"{key[0]} {key[1]}: {len(by_group[key])} complete cells")
        for _, _, pair, seed, d in rows:
            print(f"   {pair} seed {seed}: " + "  ".join(f"{k} {d[k]:.3f}" for k in METHODS))


if __name__ == "__main__":
    main()
