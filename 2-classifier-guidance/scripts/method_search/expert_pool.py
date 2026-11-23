#!/usr/bin/env python3
"""Write a digit pool drawn only from the two MNIST experts, in MNIST's file format (step M1, Paper 1).

Board question 37 limits Paper 1's checker to expert samples. Each expert draws its own digits from seed 1,000 up, the ink
is cropped to 28 x 28, and the label is the digit the expert was asked for. No classifier filters or relabels a sample, so
pixel_checker.py can train on this folder through its --mnist argument with no change.

Usage: python expert_pool.py --a <e04 unet.pt> --b <e59 unet.pt> --train 3000 --held 500 --out <dir>
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnist"))
from canvas_unet import sample, to_uint8  # noqa: E402
from checker import load_unet  # noqa: E402

D = 28
SEED_BASE = 1000
SEEDS_PER_DIGIT = 100_000


def crop(canvas):
    """The 28 x 28 window centred on the ink, kept inside the canvas. None for a canvas with no ink."""
    ys, xs = np.nonzero(canvas > 128)
    if len(ys) == 0:
        return None
    h, w = canvas.shape
    y0 = int(np.clip((ys.min() + ys.max() + 1) // 2 - D // 2, 0, h - D))
    x0 = int(np.clip((xs.min() + xs.max() + 1) // 2 - D // 2, 0, w - D))
    return canvas[y0:y0 + D, x0:x0 + D]


def draw(expert, digit, seeds, batch, device):
    def eps(x, t):
        return expert(x, t, torch.full((x.shape[0],), digit, device=x.device))

    kept, empty = [], 0
    for i in range(0, len(seeds), batch):
        canvases = to_uint8(sample(None, seeds[i:i + batch], [], eps_fn=eps, device=device, eta=1.0)[0])
        for canvas in canvases:
            tile = crop(canvas)
            if tile is None:
                empty += 1
            else:
                kept.append(tile)
    return kept, empty


def write_idx(folder, prefix, images, labels):
    with open(folder / f"{prefix}-images-idx3-ubyte", "wb") as f:
        f.write(struct.pack(">IIII", 2051, len(images), D, D))
        f.write(images.tobytes())
    with open(folder / f"{prefix}-labels-idx1-ubyte", "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(labels.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--held", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda"
    expert_a, expert_b = load_unet(args.a, device), load_unet(args.b, device)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    parts = {"train": ([], []), "t10k": ([], [])}
    record = {"experts": [str(args.a), str(args.b)], "per_digit": {}}
    for digit in range(10):
        expert = expert_a if digit <= 4 else expert_b
        first = SEED_BASE + digit * SEEDS_PER_DIGIT
        seeds = list(range(first, first + args.train + args.held))
        tiles, empty = draw(expert, digit, seeds, args.batch, device)
        train_tiles, held_tiles = tiles[:args.train], tiles[args.train:]
        for key, chosen in (("train", train_tiles), ("t10k", held_tiles)):
            parts[key][0].extend(chosen)
            parts[key][1].extend([digit] * len(chosen))
        record["per_digit"][str(digit)] = {"train": len(train_tiles), "held": len(held_tiles), "no_ink": empty,
                                           "first_seed": first}
        print(json.dumps({"digit": digit, **record["per_digit"][str(digit)],
                          "minutes": round((time.time() - start) / 60, 2)}), flush=True)
    for key, (images, labels) in parts.items():
        write_idx(args.out, key, np.stack(images).astype(np.uint8), np.array(labels, dtype=np.uint8))
    record["minutes"] = round((time.time() - start) / 60, 2)
    (args.out / "pool.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"done": True, "minutes": record["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
