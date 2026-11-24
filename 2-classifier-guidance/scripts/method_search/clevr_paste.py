#!/usr/bin/env python3
"""Step M7: sample and paste, and the paste rate that decides whether M7 stops.

The design is artifacts/notes/progress/method-search/01c-clevr-paste-first/design.md. For image i, a window around each asked position is
taken from expert sample i and laid on background sample i. The window's outer 2 pixels fade linearly into the background.

  paste  builds 1,024 pasted images for experts 1 and 2, and 1,024 for experts 1, 2 and 3
  score  scores both sets with the position scorer and applies the bar: a paste rate of at least 0.906 predicts 7 objects

The window sizes are named, so the choice on board question 66 is one argument: `written` keeps 18 by 18 everywhere, and `tall-p3`
gives position 3 a window 18 wide and 24 tall that starts on the row where the other two windows end.

Usage: python clevr_paste.py paste --samples <dir> --out <dir> --windows written|tall-p3
       python clevr_paste.py score --pasted <dir> --scorer <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clevr_data import POSITIONS, SIZE, window_box  # noqa: E402

IMAGES = 1024
FADE = 2
PASTE_BAR = 0.906   # 0.906 ** 7 is about 0.50, Bradley's rule for 7 objects with independent objects
WINDOWS = {"written": {name: window_box(*xy) for name, xy in POSITIONS.items()}}
WINDOWS["tall-p3"] = {**WINDOWS["written"], "p3": (17, 28, 35, 52)}
COUNTS = {"two": ("p1", "p2"), "three": ("p1", "p2", "p3")}


def window_weight(box):
    """1 inside the window, falling linearly to 1/3 over its outer 2 pixels, 0 outside."""
    x0, y0, x1, y1 = box
    weight = np.zeros((SIZE, SIZE), dtype=np.float32)
    for y in range(y0, y1):
        for x in range(x0, x1):
            edge = min(x - x0, x1 - 1 - x, y - y0, y1 - 1 - y)
            weight[y, x] = min(1.0, (edge + 1) / (FADE + 1))
    return weight


def check_windows(windows):
    names = list(windows)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p, q = windows[a], windows[b]
            if min(p[2], q[2]) > max(p[0], q[0]) and min(p[3], q[3]) > max(p[1], q[1]):
                raise SystemExit(f"windows {a} and {b} overlap")


def load(folder, index):
    paths = sorted(folder.glob("*.png"))
    if len(paths) < IMAGES:
        raise SystemExit(f"{folder}: {len(paths)} samples, {IMAGES} needed")
    with Image.open(paths[index]) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def paste(args):
    windows = WINDOWS[args.windows]
    check_windows(windows)
    weights = {name: window_weight(box)[..., None] for name, box in windows.items()}
    for count, experts in COUNTS.items():
        folder = args.out / args.windows / count
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(IMAGES):
            image = load(args.samples / "empty", index)
            for name in experts:
                image = image * (1 - weights[name]) + load(args.samples / name, index) * weights[name]
            Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255))).save(folder / f"{index:05d}.png")
    (args.out / args.windows / "windows.json").write_text(json.dumps({"windows": windows, "fade_pixels": FADE}, indent=1) + "\n")
    print(json.dumps({"windows": args.windows, "images_per_count": IMAGES}), flush=True)


def score(args):
    from clevr_scorer import Scorer, call_samples
    state = torch.load(args.scorer / "scorer.pt", map_location="cpu", weights_only=False)
    model = Scorer()
    model.load_state_dict(state["state"])
    result = {}
    for count, experts in COUNTS.items():
        calls, _ = call_samples(model, args.pasted / count, IMAGES)
        asked = [int(name[1]) - 1 for name in experts]
        rates = {f"p{p + 1}": round(float(calls[:, p].float().mean()), 4) for p in asked}
        result[count] = {"images": IMAGES, "in_place_rate": rates,
                         "share_with_all_asked": round(float(calls[:, asked].all(1).float().mean()), 4),
                         "objects_where_not_asked": int(calls.sum() - calls[:, asked].sum())}
    paste_rate = float(np.mean(list(result["three"]["in_place_rate"].values())))
    result.update({"paste_rate": round(paste_rate, 4), "bar": PASTE_BAR,
                   "verdict": "paste predicted to reach 7, M7 stops" if paste_rate >= PASTE_BAR else "paste falls short, M6 runs"})
    (args.pasted / "paste-score.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["paste", "score"])
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--windows", choices=sorted(WINDOWS))
    ap.add_argument("--pasted", type=Path)
    ap.add_argument("--scorer", type=Path)
    args = ap.parse_args()
    {"paste": paste, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
