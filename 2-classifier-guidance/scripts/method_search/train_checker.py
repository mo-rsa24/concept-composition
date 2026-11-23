#!/usr/bin/env python3
"""Train a Paper 1 checker with no pair of a 0 to 4 digit and a 5 to 9 digit anywhere in its data (step M1).

pixel_checker.py's PC_NOCROSS removes those pairs from the pairs it asks for, but its "wrong pair" negatives still paste the
asked digit beside any other digit, which can put a 1 beside a 9. This wrapper draws that other digit from the asked pair's own
range and leaves everything else in pixel_checker.py as it is. Its arguments are pixel_checker.py's.

Usage: python train_checker.py train --kind pair|separate --mnist <pool dir> --out <dir> --steps 4000
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ["PC_NOCROSS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnist"))
import pixel_checker as pc  # noqa: E402
from canvas_sheet import H, W  # noqa: E402


def own_range(digit):
    return range(0, 5) if digit <= 4 else range(5, 10)


def pasted_within_range(cv, rng, n, kind):
    """pixel_checker.pasted, with the wrong-pair digit taken from the asked pair's own range."""
    xs, ps, present = [], [], np.zeros((n, 10), np.float32)
    for i in range(n):
        a, b = pc.TRAIN_PAIRS[rng.integers(len(pc.TRAIN_PAIRS))]
        if kind == "two":
            c = cv.two(a, b)
            present[i, [a, b]] = 1
        elif kind == "fused":
            c = cv.overlaid(a, b) if rng.random() < 0.3 else np.zeros((H, W), np.uint8)
            if not c.any():
                side = int(rng.integers(2))
                c = np.maximum(cv.one_half(a, side), cv.one_half(b, side))
            present[i, [a, b]] = 1
        elif kind == "single":
            alone = a if rng.random() < 0.5 else b
            c = cv.one_half(alone, int(rng.integers(2)))
            present[i, alone] = 1
        else:
            choices = [d for d in own_range(a) if d not in (a, b)]
            other = int(choices[rng.integers(len(choices))])
            c = cv.two(a, other)
            present[i, [a, other]] = 1
        xs.append(c)
        ps.append((a, b))
    return np.stack(xs), np.array(ps), present


pc.pasted = pasted_within_range

if __name__ == "__main__":
    crossing = [p for p in pc.TRAIN_PAIRS if p[0] <= 4 < p[1]]
    if crossing:
        raise SystemExit(f"cross-range pairs in the training list: {crossing}")
    print(f"checker pairs: {len(pc.TRAIN_PAIRS)}, none crossing the two experts' ranges", flush=True)
    pc.main()
