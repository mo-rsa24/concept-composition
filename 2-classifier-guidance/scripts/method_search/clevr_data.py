#!/usr/bin/env python3
"""Step M7: the shared 64 by 64 picture rule and the control on the paste window.

The design is artifacts/notes/progress/method-search/01c-clevr-paste-first/design.md.

  control  on 256 real single-object renders per position, the share of changed pixels that fall inside that position's window

A pixel counts as changed when any colour channel differs from the mean empty scene by more than 0.1 on a 0-to-1 scale. The
share is pooled over the position's 256 renders, and every position must reach 95%.

Usage: python clevr_data.py control --renders <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
WINDOW = 18
POSITIONS = {"p1": (0.30, 0.30), "p2": (0.60, 0.30), "p3": (0.40, 0.60)}
CONTROL_IMAGES = 256
CHANGED_ABOVE = 0.1
MIN_INSIDE_SHARE = 0.95


def to_64(path):
    """A 320 by 240 render as a 64 by 64 float array in [0, 1], squashed with Lanczos."""
    with Image.open(path) as image:
        if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
            raise ValueError(f"{path} has transparent pixels")
        rgb = image.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def window_box(x, y):
    """(x0, y0, x1, y1) in pixels of the 18 by 18 window centred on a position given as fractions of the image."""
    x0, y0 = int(round(x * SIZE - WINDOW / 2)), int(round(y * SIZE - WINDOW / 2))
    if x0 < 0 or y0 < 0 or x0 + WINDOW > SIZE or y0 + WINDOW > SIZE:
        raise ValueError(f"the window at ({x}, {y}) leaves the image")
    return x0, y0, x0 + WINDOW, y0 + WINDOW


def first_images(renders, name, count):
    paths = [renders / name / "images" / f"{name}_{index:06d}.png" for index in range(count)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {count} {name} renders are missing")
    return paths


def control(args):
    args.out.mkdir(parents=True, exist_ok=True)
    empty = np.mean([to_64(p) for p in first_images(args.renders, "empty", CONTROL_IMAGES)], axis=0)
    rows, tiles = {}, []
    for name, (x, y) in POSITIONS.items():
        x0, y0, x1, y1 = window_box(x, y)
        inside, total, frequency, per_image = 0, 0, np.zeros((SIZE, SIZE)), []
        for path in first_images(args.renders, name, CONTROL_IMAGES):
            changed = (np.abs(to_64(path) - empty).max(-1) > CHANGED_ABOVE)
            here = int(changed[y0:y1, x0:x1].sum())
            inside, total = inside + here, total + int(changed.sum())
            frequency += changed
            per_image.append(here / max(int(changed.sum()), 1))
        share = inside / max(total, 1)
        rows[name] = {"position": [x, y], "window_pixels": [x0, y0, x1, y1], "changed_pixels": total, "inside_window": inside,
                      "inside_share": round(share, 4), "median_share_per_image": round(float(np.median(per_image)), 4),
                      "lowest_share_of_any_image": round(float(np.min(per_image)), 4), "passes": share >= MIN_INSIDE_SHARE}
        scaled = Image.fromarray(np.uint8(255 - 255 * frequency / CONTROL_IMAGES)).convert("RGB").resize((256, 256), Image.NEAREST)
        draw = ImageDraw.Draw(scaled)
        draw.rectangle([x0 * 4, y0 * 4, x1 * 4 - 1, y1 * 4 - 1], outline=(200, 30, 30), width=2)
        draw.text((4, 4), f"{name}: {share:.1%} inside", fill=(0, 0, 160))
        tiles.append(scaled)
    sheet = Image.new("RGB", (3 * 256 + 16, 256), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * 264, 0))
    sheet.save(args.out / "window-control.png")
    result = {"images_per_position": CONTROL_IMAGES, "changed_above": CHANGED_ABOVE, "min_inside_share": MIN_INSIDE_SHARE,
              "positions": rows, "passes": all(r["passes"] for r in rows.values())}
    (args.out / "window-control.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passes"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["control"])
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    control(args)


if __name__ == "__main__":
    main()
