#!/usr/bin/env python3
"""Loop 12 step Z9: Colored MNIST built to DCFM's description, the seen-and-novel split, and the two data holders.

Design: artifacts/notes/progress/leakage/09-dcfm-colored-mnist/design.md, "The data". Numpy only, so it runs anywhere.

A picture is an MNIST digit padded from 28 to 32 pixels, its stroke (the grey level, 0 to 1) multiplied by one of ten
colours on black, three channels in [-1, 1]. Combination (d, c) is digit d in colour c. 50 of the 100 combinations are
seen, drawn once with seed 20261125 so that every digit and every colour is in some seen combination; the 50 seen are dealt
to two holders, 25 each.

  python z9_data.py split --out <dir>                       write split.json
  python z9_data.py sheet --mnist <raw dir> --split <json> --out <png>   one real picture per combination, seen ones framed
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20261125
# matplotlib's tab10, as RGB in 0..1
PALETTE = np.array([[0.122, 0.467, 0.706], [1.000, 0.498, 0.055], [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
                    [0.580, 0.404, 0.741], [0.549, 0.337, 0.294], [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
                    [0.737, 0.741, 0.133], [0.090, 0.745, 0.812]], dtype=np.float32)
COLOUR_NAMES = ["blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "olive", "cyan"]


def load_mnist(folder, split="train"):
    pre = "train" if split == "train" else "t10k"
    folder = Path(folder)
    images = np.fromfile(folder / f"{pre}-images-idx3-ubyte", dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.fromfile(folder / f"{pre}-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    return images, labels


def colourise(grey, colours):
    """grey: (n, 28, 28) uint8; colours: (n,) ints. Returns (n, 3, 32, 32) float32 in [-1, 1]."""
    g = np.pad(grey.astype(np.float32) / 255.0, ((0, 0), (2, 2), (2, 2)))
    rgb = g[:, None] * PALETTE[np.asarray(colours)][:, :, None, None]
    return rgb * 2.0 - 1.0


def make_split(seed=SEED):
    rng = np.random.default_rng(seed)
    combos = [(d, c) for d in range(10) for c in range(10)]
    for _ in range(10_000):
        pick = rng.choice(100, 50, replace=False)
        seen = sorted(combos[i] for i in pick)
        if len({d for d, _ in seen}) == 10 and len({c for _, c in seen}) == 10:
            break
    else:
        raise SystemExit("no split covers every digit and colour")
    order = rng.permutation(50)
    holders = [sorted(seen[i] for i in order[:25]), sorted(seen[i] for i in order[25:])]
    novel = sorted(set(combos) - set(seen))
    return {"seed": seed, "seen": seen, "novel": novel, "holders": holders,
            "share_seen": len(seen) / 100, "colours": COLOUR_NAMES, "palette": PALETTE.round(3).tolist(),
            "digits_per_holder": [sorted({d for d, _ in h}) for h in holders],
            "colours_per_holder": [sorted({c for _, c in h}) for h in holders]}


class Stream:
    """Training pictures for one holder: a combination drawn uniformly from its list, a random training digit of that class."""

    def __init__(self, images, labels, combos, seed):
        self.images, self.combos = images, np.asarray(combos)
        self.by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}
        self.rng = np.random.default_rng(seed)

    def batch(self, n):
        k = self.combos[self.rng.integers(0, len(self.combos), n)]
        idx = np.array([self.rng.choice(self.by_digit[int(d)]) for d in k[:, 0]])
        return colourise(self.images[idx], k[:, 1]), k[:, 0], k[:, 1]


def reference(images, labels, combos, per, seed):
    """per real pictures of each combination, from the given digits (the test set for FID references)."""
    rng = np.random.default_rng(seed)
    out, dd, cc = [], [], []
    for d, c in combos:
        idx = rng.choice(np.flatnonzero(labels == d), per, replace=False)
        out.append(colourise(images[idx], np.full(per, c))); dd += [d] * per; cc += [c] * per
    return np.concatenate(out), np.array(dd), np.array(cc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("what", choices=["split", "sheet"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mnist", type=Path)
    ap.add_argument("--split", type=Path)
    args = ap.parse_args()
    if args.what == "split":
        args.out.mkdir(parents=True, exist_ok=True)
        s = make_split()
        (args.out / "split.json").write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps({k: s[k] for k in ("share_seen", "digits_per_holder", "colours_per_holder")}))
        return
    from PIL import Image, ImageDraw
    s = json.loads(args.split.read_text())
    images, labels = load_mnist(args.mnist, "test")
    holder = {tuple(c): i for i, h in enumerate(s["holders"]) for c in h}
    cell, pad = 64, 6
    W = Image.new("RGB", (10 * (cell + pad) + 40, 10 * (cell + pad) + 40), "white")
    dr = ImageDraw.Draw(W)
    for d in range(10):
        dr.text((8, 40 + d * (cell + pad) + cell // 2 - 6), str(d), fill="black")
        dr.text((40 + d * (cell + pad) + cell // 2 - 12, 10), COLOUR_NAMES[d], fill="black")
        for c in range(10):
            x = colourise(images[np.flatnonzero(labels == d)[:1]], [c])[0]
            tile = Image.fromarray(((x.transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)).resize((cell, cell), Image.NEAREST)
            px, py = 40 + c * (cell + pad), 40 + d * (cell + pad)
            W.paste(tile, (px, py))
            if (d, c) in holder:
                dr.rectangle([px - 3, py - 3, px + cell + 2, py + cell + 2], outline=("#1f77b4", "#d62728")[holder[(d, c)]], width=3)
    W.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
