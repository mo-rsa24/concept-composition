"""Build the abstract-checker test bed: pasted MNIST canvases with exact plurality labels (step V1).

Digits 0 to 4 train and digits 5 to 9 are held out for the unseen test. Every asked pair gets the same mix of five canvas
kinds, so the asked pair alone carries no information about the label. The data hash covers the build version, the four MNIST
files and every array, and the same command gives the same hash.

Usage:
    python build_env.py --mnist <raw MNIST dir> --out <dir> [--twice] [--scale 0.01] [--sheet <png>]
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

VERSION = "abstract-checkers-env-v1"
ROOT_SEED = 8001
H, W, D = 32, 64, 28
SEEN, UNSEEN = tuple(range(5)), tuple(range(5, 10))
KINDS = ("apart", "overlaid", "same_half", "single", "wrong")
PER_UNIT = {"apart": 4, "overlaid": 1, "same_half": 1, "single": 1, "wrong": 1}      # one unit is 8 canvases
SPLITS = {
    "train": {"file": "train", "digits": SEEN, "units": 250},
    "seen_test": {"file": "t10k", "digits": SEEN, "units": 25},
    "unseen_test": {"file": "t10k", "digits": UNSEEN, "units": 25},
}
FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
ARRAYS = ("x", "asked", "holds", "corner", "src", "kind", "target", "n_digits", "apart")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_source(folder, name):
    images = np.fromfile(folder / f"{name}-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
    labels = np.fromfile(folder / f"{name}-labels-idx1-ubyte", np.uint8, offset=8)
    if len(images) != len(labels):
        raise ValueError(f"{name}: {len(images)} images and {len(labels)} labels")
    return {"images": images, "labels": labels, "by_digit": {d: np.flatnonzero(labels == d) for d in range(10)}}


def draw(source, rng, digit):
    i = int(source["by_digit"][digit][rng.integers(len(source["by_digit"][digit]))])
    return i, source["images"][i]


def half_corner(rng, side):
    return int(rng.integers(0, H - D + 1)), 32 * side + int(rng.integers(0, 5))


def paste(canvas, image, corner):
    y, x = corner
    canvas[y:y + D, x:x + D] = np.maximum(canvas[y:y + D, x:x + D], image)


def make_canvas(kind, a, b, digits, source, rng):
    """One canvas of a kind for the asked pair (a, b). For apart and wrong, holds[0] is the left digit."""
    if kind == "apart":
        shown = (a, b) if rng.random() < 0.5 else (b, a)
        corners = [half_corner(rng, 0), half_corner(rng, 1)]
    elif kind == "wrong":
        keep = a if rng.random() < 0.5 else b
        others = [d for d in digits if d not in (a, b)]
        other = others[int(rng.integers(len(others)))]
        shown = (keep, other) if rng.random() < 0.5 else (other, keep)
        corners = [half_corner(rng, 0), half_corner(rng, 1)]
    elif kind == "overlaid":
        shown = (a, b)
        spot = (int(rng.integers(0, H - D + 1)), 16 + int(rng.integers(0, 5)))
        corners = [spot, spot]
    elif kind == "same_half":
        shown = (a, b)
        side = int(rng.integers(2))
        corners = [half_corner(rng, side), half_corner(rng, side)]
    elif kind == "single":
        shown = (a if rng.random() < 0.5 else b, -1)
        corners = [half_corner(rng, int(rng.integers(2))), (-1, -1)]
    else:
        raise ValueError(kind)
    canvas = np.zeros((H, W), np.uint8)
    src = [-1, -1]
    for j, digit in enumerate(shown):
        if digit >= 0:
            src[j], image = draw(source, rng, digit)
            paste(canvas, image, corners[j])
    apart = int(kind in ("apart", "wrong"))
    return {"x": canvas, "asked": (a, b), "holds": shown, "corner": corners, "src": src, "kind": KINDS.index(kind),
            "target": int(kind == "apart"), "n_digits": 1 if kind == "single" else 2, "apart": apart}


def build_split(index, spec, source, scale):
    units = max(1, round(spec["units"] * scale))
    rows = []
    for a, b in itertools.combinations(spec["digits"], 2):
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, index, a, b]))
        for kind in KINDS:
            rows += [make_canvas(kind, a, b, spec["digits"], source, rng) for _ in range(units * PER_UNIT[kind])]
    order = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, index, 99])).permutation(len(rows))
    rows = [rows[i] for i in order]
    types = {"x": np.uint8, "asked": np.int8, "holds": np.int8, "corner": np.int8, "src": np.int32, "kind": np.uint8,
             "target": np.uint8, "n_digits": np.uint8, "apart": np.uint8}
    return {name: np.array([r[name] for r in rows], types[name]) for name in ARRAYS}


def audit(name, spec, split, source):
    """Stop the build if a label, a digit range or a class balance is wrong; return the counts."""
    allowed = set(spec["digits"])
    if not set(np.unique(split["asked"]).tolist()) <= allowed:
        raise ValueError(f"{name}: an asked digit is outside {sorted(allowed)}")
    shown = split["holds"][split["holds"] >= 0]
    if not set(np.unique(shown).tolist()) <= allowed:
        raise ValueError(f"{name}: a pasted digit is outside {sorted(allowed)}")
    used = split["src"] >= 0
    if not np.array_equal(used, split["holds"] >= 0):
        raise ValueError(f"{name}: source indices and held digits disagree on which slots are filled")
    if not np.array_equal(source["labels"][split["src"][used]], split["holds"][used].astype(np.uint8)):
        raise ValueError(f"{name}: a pasted image's MNIST label differs from the stored digit")
    if not (split["x"].reshape(len(split["x"]), -1).max(1) > 0).all():
        raise ValueError(f"{name}: a canvas is blank")
    rates = {}
    for a, b in itertools.combinations(spec["digits"], 2):
        mask = (split["asked"][:, 0] == a) & (split["asked"][:, 1] == b)
        rates[f"{a}-{b}"] = float(split["target"][mask].mean())
    if set(rates.values()) != {0.5}:
        raise ValueError(f"{name}: positive rate per asked pair is not exactly one half: {rates}")
    apart = split["kind"] == KINDS.index("apart")
    if not np.array_equal(split["target"].astype(bool), apart):
        raise ValueError(f"{name}: the label disagrees with the canvas kind")
    return {"canvases": len(split["x"]), "pairs": len(rates), "positive_rate_per_pair": 0.5,
            "by_kind": {k: int((split["kind"] == i).sum()) for i, k in enumerate(KINDS)}}


def build(mnist, scale):
    sources = {n: load_source(mnist, n) for n in ("train", "t10k")}
    data, counts = {}, {}
    for index, (name, spec) in enumerate(SPLITS.items()):
        data[name] = build_split(index, spec, sources[spec["file"]], scale)
        counts[name] = audit(name, spec, data[name], sources[spec["file"]])
    return data, counts


def data_hash(data, file_hashes, scale):
    h = hashlib.sha256()
    h.update(json.dumps({"version": VERSION, "seed": ROOT_SEED, "scale": scale, "files": file_hashes}, sort_keys=True).encode())
    for split in sorted(data):
        for name in ARRAYS:
            a = data[split][name]
            h.update(f"{split}/{name}/{a.dtype.str}/{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def draw_sheet(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 6
    fig, axes = plt.subplots(len(KINDS), cols, figsize=(cols * 2.1, len(KINDS) * 1.35))
    for r, kind in enumerate(KINDS):
        for s, split in enumerate(("train", "unseen_test")):
            where = np.flatnonzero(data[split]["kind"] == KINDS.index(kind))[:3]
            for c, i in enumerate(where):
                ax = axes[r, s * 3 + c]
                ax.imshow(data[split]["x"][i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                a, b = data[split]["asked"][i]
                y = int(data[split]["target"][i])
                ax.set_title(f"asked {a},{b}: label {y}", fontsize=8, color="#15803d" if y else "#b91c1c", pad=2)
                ax.set_xticks([]), ax.set_yticks([])
        axes[r, 0].set_ylabel(kind, fontsize=9)
    fig.text(0.28, 0.995, "train: digits 0 to 4", ha="center", va="top", fontsize=10)
    fig.text(0.76, 0.995, "unseen test: digits 5 to 9", ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each split's canvases, for the ladder's small rungs")
    ap.add_argument("--twice", action="store_true", help="build a second time from scratch and compare the hashes")
    ap.add_argument("--sheet", type=Path, default=None, help="also save a picture sheet of each canvas kind")
    args = ap.parse_args()

    file_hashes = {f: sha256_file(args.mnist / f) for f in FILES}
    data, counts = build(args.mnist, args.scale)
    digest = data_hash(data, file_hashes, args.scale)
    print(f"data hash: {digest}", flush=True)
    second = None
    if args.twice:
        again, _ = build(args.mnist, args.scale)
        second = data_hash(again, file_hashes, args.scale)
        print(f"rebuild hash: {second}", flush=True)
        print(f"identical: {second == digest}", flush=True)
        if second != digest:
            sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())["data_hash"]
        if recorded != digest:
            print(f"REFUSED: {manifest_path} records hash {recorded}. Nothing was overwritten.", flush=True)
            sys.exit(2)
        print(f"already built at {args.out}, same hash", flush=True)
    else:
        temp = args.out / "dataset.tmp.npz"
        np.savez_compressed(temp, **{f"{s}__{n}": a for s, split in data.items() for n, a in split.items()})
        os.replace(temp, args.out / "dataset.npz")
        manifest_path.write_text(json.dumps({
            "version": VERSION, "data_hash": digest, "rebuild_hash": second, "seed": ROOT_SEED, "scale": args.scale,
            "numpy": np.__version__, "mnist_sha256": file_hashes, "kinds": KINDS, "counts": counts}, indent=1))
    if args.sheet:
        draw_sheet(data, args.sheet)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
