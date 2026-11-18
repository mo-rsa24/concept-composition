#!/usr/bin/env python
"""Sheet of the held-out rejects a fine-tuned checker called clean, so the label or the network can be judged by eye.

Reads probe/finetune_<model>.json (per-picture logits on eval_set_catdog_v2.json's 220 pictures) for every
model given, and draws each reject that any of them scored above 0 (called clean), grouped by reject kind,
captioned with its id, its kind and each model's logit. A second sheet shows the clean pairs any model
called a reject. Tiles are 512 px; every tile is also saved at full size beside the sheet.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/probe/finetune-misses/:
    rejects-called-clean.png, clean-called-reject.png, tiles/<id>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import augment_encode  # noqa: E402
import probe_frozen_features as pff  # noqa: E402

TILE, COLS = 512, 6


def sheet(rows, logits, models, path):
    if not rows:
        return 0
    font = ImageFont.load_default(size=22)
    cap = 64
    n_rows = (len(rows) + COLS - 1) // COLS
    out = Image.new("RGB", (COLS * TILE, n_rows * (TILE + cap)), "white")
    d = ImageDraw.Draw(out)
    for k, t in enumerate(rows):
        x, y = (k % COLS) * TILE, (k // COLS) * (TILE + cap)
        im = augment_encode.square(Image.open(augment_encode.picture_path(t)).convert("RGB"), TILE)
        out.paste(im, (x, y))
        scores = "  ".join(f"{m} {logits[m][t['id']]:+.1f}" for m in models)
        d.text((x + 6, y + TILE + 4), f"{k + 1}. {t['kind']}", fill="black", font=font)
        d.text((x + 6, y + TILE + 32), scores, fill="black", font=font)
    out.save(path)
    return len(rows)


def main() -> int:
    models = sys.argv[1:] or ["dinov2", "clip"]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    logits = {m: json.loads((pff.OUT / f"finetune_{m}.json").read_text())["logits"] for m in models}
    out = pff.OUT / "finetune-misses"
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    rej = sorted([t for t in evals if t["label"] == 0 and any(logits[m][t["id"]] > 0 for m in models)],
                 key=lambda t: (t["kind"], -max(logits[m][t["id"]] for m in models)))
    pos = [t for t in evals if t["label"] == 1 and any(logits[m][t["id"]] <= 0 for m in models)]
    for t in rej + pos:
        Image.open(augment_encode.picture_path(t)).convert("RGB").save(out / "tiles" / f"{t['id']}.png")
    a = sheet(rej, logits, models, out / "rejects-called-clean.png")
    b = sheet(pos, logits, models, out / "clean-called-reject.png")
    for m in models:
        print(m, "rejects called clean:", sum(logits[m][t["id"]] > 0 for t in evals if t["label"] == 0),
              "clean called reject:", sum(logits[m][t["id"]] <= 0 for t in evals if t["label"] == 1))
    print(f"sheet 1: {a} rejects any model called clean; sheet 2: {b} clean pairs any model called reject")
    print("index:", json.dumps([[i + 1, t["id"], t["kind"]] for i, t in enumerate(rej)]))
    print("MISSES DONE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
