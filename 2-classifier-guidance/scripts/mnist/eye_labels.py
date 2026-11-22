"""Gather Claude's eye calls on drawn canvases into one labelled file for the judge loop (plan 24, loop 1).

Each sheet is 64 drawn canvases asked for one pair, with the seeds Claude called "two clear, right
digits" by eye, blind to the judge. The split is fixed in the A2b review file: 12 sheets to tune
on, 13 held back. Writes data/mnist-canvas/24-mnist-one-model/eye-labels.npz.
"""
import json
from pathlib import Path
import numpy as np

B = Path("data/mnist-canvas/24-mnist-one-model")
F = B / "five-versions"
eye = json.load(open("artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/five-versions/eye_counts.json"))["calls"]

def five(key):           # a sheet from the cap-0.1 five-version run, with its eye list
    col, pair = key.split("|")
    return F / "cap-0.1/canvases.npz", f"{col}|{pair.replace('&', '')}", eye[key]["seeds"], pair

SWEEP = {"pair-data-0.01": [26, 30, 31, 32, 61, 62],
         "pair-flip-0.2": [4, 6, 11, 13, 17, 18, 20, 25, 29, 30, 31, 32, 49, 59, 60, 63],
         "pair-data-0.25": [0, 2, 3, 6, 10, 11, 12, 18, 20, 22, 23, 24, 25, 28, 29, 39, 40, 43, 45, 49, 50, 52,
                            53, 56, 59, 60, 62],
         "pair-data-0.05": [1, 2, 4, 13, 14, 15, 17, 18, 21, 22, 23, 24, 29, 30, 31, 32, 33, 36, 41, 43, 49, 51,
                            53, 60, 61, 62],
         "pair-flip-0.1": [0, 2, 3, 4, 6, 11, 14, 16, 17, 19, 20, 27, 33, 34, 49, 51, 53, 56, 57, 58, 60, 62]}
CAP1 = [0, 2, 4, 5, 6, 7, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34,
        36, 37, 40, 41, 44, 45, 46, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63]

tune = [(B / "plain-product/share-0.01/canvases.npz", "g1.0_78", [6], "7&8", "plain share 0.01"),
        (B / "plain-product/share-0.1/canvases.npz", "g1.0_78", [18, 31, 42, 48], "7&8", "plain share 0.1"),
        (F / "cap-1.0-s10/canvases.npz", "pair s=10|78", CAP1, "7&8", "pair s=10 cap 1.0")]
for k in ["pair s=3|7&8", "pair s=10|7&8", "separate s=10|7&8", "separate s=30|7&8", "random s=10|7&8",
          "best-of-2|7&8", "best-of-16|7&8"]:
    f, key, seeds, pair = five(k)
    tune.append((f, key, seeds, pair, k))
for h in ("pair-data-0.01", "pair-flip-0.2"):
    tune.append((F / f"sweep1-{h}/canvases.npz", "pair s=10|78", SWEEP[h], "7&8", h))
test = []
for p in ("0&3", "1&9", "3&4", "3&6", "4&7"):
    for s in ("3", "10"):
        f, key, seeds, pair = five(f"pair s={s}|{p}")
        test.append((f, key, seeds, pair, f"pair s={s}|{p}"))
for h in ("pair-data-0.25", "pair-data-0.05", "pair-flip-0.1"):
    test.append((F / f"sweep1-{h}/canvases.npz", "pair s=10|78", SWEEP[h], "7&8", h))

out = {}
for split, sheets in (("tune", tune), ("test", test)):
    xs, ys, pairs, names = [], [], [], []
    for f, key, seeds, pair, name in sheets:
        x = np.load(f)[key]
        assert len(x) == 64, (f, key)
        xs.append(x); ys.append(np.isin(np.arange(64), seeds)); names.append(name)
        pairs.append(tuple(int(d) for d in pair.split("&")))
    out[f"{split}_x"], out[f"{split}_y"] = np.stack(xs), np.stack(ys)
    out[f"{split}_pair"], out[f"{split}_name"] = np.array(pairs), np.array(names)
    print(split, len(sheets), "sheets,", int(np.stack(ys).sum()), "eye positives")
np.savez_compressed(B / "eye-labels.npz", **out)
