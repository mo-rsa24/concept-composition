"""Draw a sheet of tiles from a samples file: one block per row, the first N seeds, each tile titled by
its seed. These are model outputs, drawn as they came out of the sampler.

Usage:
    python -I sheet.py <samples .npz> <out .png> [--rows "7 alone" ...] [--n 32] [--title "..."]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", nargs="+")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    d = np.load(a.npz)
    rows = a.rows or list(d.files)
    per = a.n // a.cols
    fig = plt.figure(figsize=(a.cols * 1.9, len(rows) * (per * 1.0 + 0.6) + 0.6))
    outer = fig.add_gridspec(len(rows), 1, hspace=0.35)
    for r, name in enumerate(rows):
        sub = outer[r].subgridspec(per, a.cols, wspace=0.05, hspace=0.25)
        imgs = d[name][: a.n]
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(sub[i // a.cols, i % a.cols])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"seed {i}", fontsize=6, pad=1, color="#6b7a88")
        head = fig.add_subplot(outer[r], frameon=False)
        head.set_xticks([]), head.set_yticks([])
        head.set_title(f'{name}: seeds 0 to {a.n - 1} of 256', fontsize=11, weight="bold", loc="left", pad=14)
    fig.suptitle(a.title or a.npz.name, fontsize=12, weight="bold", y=1.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
