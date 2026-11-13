#!/usr/bin/env python
"""Pick the one adapter by the protocol's showcase rule, from the calls, and write the choice.

Reads 1-lora-correction/report/paper-review/showcase-calls.csv (one row per picture: candidate, pair, seed, call
"clear" or "not clear"). The winner has the most seeds on which both showcase pairs are clear; a
tie goes to the most clear pictures out of 16, then to the most optimiser steps. The shared seed is
the lowest seed on which the winner is clear for both pairs (1-lora-correction/report/paper-review/protocol.md, rule 1).

Writes the choice to artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml, the one file
later scripts read. To change your mind, either edit a call in the CSV and rerun, or set
`override_candidate` and/or `override_seed` in the yaml and rerun; the override wins and the yaml
records that it was used.

    python3 1-lora-correction/scripts/review_showcase_pick.py            rank, write the yaml, draw R8's figure
    python3 1-lora-correction/scripts/review_showcase_pick.py --dry-run  rank only
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CALLS = REPO / "1-lora-correction/report/paper-review/showcase-calls.csv"
CHOICE = REPO / "artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml"
FIGURE = REPO / "1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/showcase_calls.png"
CKPT_ROOT = "/datasets/mmolefe/poe_repair_min/outputs/showcase"
COHORT = {"v57-11b-x0-self-decay": "cohort2", "v58f-05-self-contrast": "cohort4"}
PAIRS = ("a_cat__x__a_dog", "an_elephant__x__a_penguin")
SEEDS = range(9, 17)


def read_overrides() -> dict:
    out = {}
    if CHOICE.exists():
        for line in CHOICE.read_text().splitlines():
            m = re.match(r"^(override_candidate|override_seed):\s*(\S+)\s*$", line)
            if m and m.group(2) not in ("null", "~", "''"):
                out[m.group(1)] = m.group(2)
    return out


def rank(rows: list[dict]) -> list[tuple]:
    clear = defaultdict(lambda: {p: set() for p in PAIRS})
    for r in rows:
        if r["call"] == "clear":
            clear[r["candidate"]][r["pair"]].add(int(r["seed"]))
    table = []
    for cand, by_pair in clear.items():
        both = sorted(by_pair[PAIRS[0]] & by_pair[PAIRS[1]])
        total = sum(len(v) for v in by_pair.values())
        step = int(re.search(r"_s(\d+)$", cand).group(1))
        table.append((cand, both, total, step))
    for r in rows:  # candidates with no clear picture at all still rank
        if r["candidate"] not in clear:
            table.append((r["candidate"], [], 0, int(re.search(r"_s(\d+)$", r["candidate"]).group(1))))
    table = list({t[0]: t for t in table}.values())
    return sorted(table, key=lambda t: (-len(t[1]), -t[2], -t[3]))


def checkpoint(cand: str) -> str:
    run, step = re.match(r"^(.*)_s(\d+)$", cand).groups()
    return f"{CKPT_ROOT}/{COHORT.get(run, 'cohort3')}/{run}/checkpoints/lora_step_{step}.pt"


# The figure prints at the full text width; every label is sized so it reads at 7 pt there.
PRINT_WIDTH_IN = 5.5
MIN_PT = 7


def short(cand: str) -> str:
    run, step = re.match(r"^(.*)_s(\d+)$", cand).groups()
    run = run.replace("-x0-self-decay", "").replace("-self-contrast", " contrast").replace("-self", " self").replace("-plain", " plain").replace("-contrast", " contrast")
    k = int(step) / 1000
    return f"{run} {k:g}k"


def draw(rows: list[dict], order: list[str], winner: str, seed: int) -> None:
    from PIL import Image, ImageDraw, ImageFont
    T, G, GAP = 96, 3, 14
    cols = [(p, s) for p in PAIRS for s in SEEDS]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    grid_w = len(cols) * (T + G) + GAP
    px = 26
    for _ in range(3):  # label width and font size depend on each other; settle in a few passes
        f = ImageFont.truetype(font_path, px); fb = ImageFont.truetype(bold_path, px)
        L = max(fb.getlength(short(c)) for c in order) + 16
        W = int(L + grid_w + 8)
        px = int(-(-MIN_PT / 72 * W / PRINT_WIDTH_IN // 1))
    f = ImageFont.truetype(font_path, px); fb = ImageFont.truetype(bold_path, px)
    L = int(max(fb.getlength(short(c)) for c in order) + 16); W = L + grid_w + 8
    H = px + 14
    im = Image.new("RGB", (W, H + len(order) * (T + G) + 8), "white")
    d = ImageDraw.Draw(im)
    d.text((L, 2), "cat and dog, seeds 9 to 16", fill="black", font=fb)
    d.text((L + 8 * (T + G) + GAP, 2), "elephant and penguin, seeds 9 to 16", fill="black", font=fb)
    by = {(r["candidate"], r["pair"], int(r["seed"])): r for r in rows}
    for i, cand in enumerate(order):
        y = H + i * (T + G)
        d.text((4, y + T // 2 - px // 2), short(cand), fill="black", font=fb if cand == winner else f)
        for j, (p, s) in enumerate(cols):
            x = L + j * (T + G) + (GAP if j >= 8 else 0)
            r = by[(cand, p, s)]
            im.paste(Image.open(REPO / r["picture"]).convert("RGB").resize((T, T)), (x, y))
            colour = (40, 150, 70) if r["call"] == "clear" else (200, 60, 50)
            d.rectangle([x, y, x + T - 1, y + T - 1], outline=colour, width=4 if s != seed else 8)
        if cand == winner:
            d.rectangle([L - 3, y - 2, W - 6, y + T + 1], outline=(0, 0, 0), width=2)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    im.save(FIGURE)
    print(f"figure {W} x {im.height} px, labels at {px} px = {MIN_PT} pt at {PRINT_WIDTH_IN} in wide")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(CALLS.open()))
    table = rank(rows)
    for cand, both, total, _ in table:
        print(f"{cand:34s} both clear on {len(both)} seeds {both}  clear pictures {total}/16")
    ov = read_overrides()
    winner = ov.get("override_candidate", table[0][0])
    both = next(t[1] for t in table if t[0] == winner)
    seed = int(ov.get("override_seed", both[0] if both else 9))
    rule_winner = table[0][0]
    print(f"chosen: {winner}, shared seed {seed}" + ("  (override)" if ov else "  (by the rule)"))
    if a.dry_run:
        return 0
    CHOICE.write_text(
        "# The one adapter for every adapter number and image in the review response, and the shared seed.\n"
        "# Written by 1-lora-correction/scripts/review_showcase_pick.py from 1-lora-correction/report/paper-review/showcase-calls.csv under\n"
        "# protocol rule 1. To change it, edit a call in the CSV, or set an override below, then rerun.\n"
        f"candidate: {winner}\n"
        f"checkpoint: {checkpoint(winner)}\n"
        f"shared_seed: {seed}\n"
        f"chosen_by: {'override' if ov else 'the showcase rule'}\n"
        f"rule_winner: {rule_winner}\n"
        f"seeds_both_clear: {both}\n"
        f"override_candidate: {ov.get('override_candidate', 'null')}\n"
        f"override_seed: {ov.get('override_seed', 'null')}\n")
    draw(rows, [t[0] for t in table], winner, seed)
    print(f"wrote {CHOICE.relative_to(REPO)} and {FIGURE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
