#!/usr/bin/env python3
"""One grid per held-out pair at one checkpoint, with every detector box drawn.

Green boxes pass the instance-count scorer's size filter (a box must span at least
MIN_BOX_FRACTION of the image's long side); red boxes are dropped by it. Each tile's
label gives the scorer's call with the filter and without it, so the tiles whose call
the filter flips can be read by eye.

    PYTHONPATH=$REPO $PY 1-lora-correction/scripts/animals_compose_transfer/filter_call_grids.py \
        --run-dir <pooled run> --step 60000
Writes <run-dir>/filter-calls-step-<step>/<pair>.png and calls.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
    MIN_BOX_FRACTION, count_instances,
)

TILE = 384


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, required=True)
    a = ap.parse_args()
    snap = next((a.run_dir / "samples/per_epoch").glob(f"epoch_*_step_{a.step:06d}"))
    out = a.run_dir / f"filter-calls-step-{a.step:06d}"
    out.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    font = ImageFont.load_default()

    by_pair = defaultdict(list)
    for png in sorted(snap.glob("out_out__*__seed*.png")):
        m = re.fullmatch(r"out_out__(.+)__seed(\d+)", png.stem)
        if m:  # the plain render only; skip variants such as __seed09__cmp
            by_pair[m.group(1)].append((int(m.group(2)), png))

    calls = {}
    for pair, items in sorted(by_pair.items()):
        tiles, rows = [], []
        for seed, png in items:
            n_all, boxes_all = count_instances(png, device=dev, min_box_fraction=0.0)
            n_kept, _ = count_instances(png, device=dev)
            im = Image.open(png).convert("RGB")
            side = MIN_BOX_FRACTION * max(im.size)
            d = ImageDraw.Draw(im)
            for b in boxes_all:
                x0, y0, x1, y1 = b["box"]
                kept = max(x1 - x0, y1 - y0) >= side
                d.rectangle([x0, y0, x1, y1], outline=(40, 200, 80) if kept else (230, 40, 40), width=6)
            im.thumbnail((TILE, TILE))
            label = f"seed {seed}: {n_kept} kept -> {'compose' if n_kept >= 2 else 'fused'}"
            if (n_all >= 2) != (n_kept >= 2):
                label += f"  (FLIPPED: {n_all} without filter)"
            canvas = Image.new("RGB", (TILE, TILE + 22), (20, 20, 20))
            canvas.paste(im, (0, 22))
            ImageDraw.Draw(canvas).text((6, 5), label, fill=(255, 220, 120) if "FLIPPED" in label else (230, 230, 230), font=font)
            tiles.append(canvas)
            rows.append({"seed": seed, "n_without_filter": n_all, "n_with_filter": n_kept})
        cols = 4
        grid = Image.new("RGB", (cols * TILE, math.ceil(len(tiles) / cols) * (TILE + 22)), (20, 20, 20))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * TILE, (i // cols) * (TILE + 22)))
        grid.save(out / f"{pair}.png")
        calls[pair] = rows
        print(pair, sum(r["n_with_filter"] >= 2 for r in rows), "/", len(rows), "compose with filter;",
              sum(r["n_without_filter"] >= 2 for r in rows), "without")
    (out / "calls.json").write_text(json.dumps(calls, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
