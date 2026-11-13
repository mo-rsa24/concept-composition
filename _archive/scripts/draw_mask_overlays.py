#!/usr/bin/env python
"""Draw both overlap readings of every selected seed's masks over the render they came from.

Why this exists. The regional composer gives each expert its own cells, and a mask in the wrong
place would make every count in the sweep meaningless while still producing a sheet that looks
like an answer. So the masks are looked at before a single composed render is made. This is the
figure that stop exists for.

What one panel shows. The joint prompt's own render, with the cat mask tinted blue and the dog
mask tinted red, each cell's tint strength being that expert's weight there. The masks are built
on the 128 x 128 latent grid and upsampled to the render's own size with nearest-neighbour, so
what you see is the grid itself and not a smoothed picture of it. A dashed outline marks where
the two boxes crossed, which is the region the two readings disagree about.

Both readings are drawn for every seed, side by side, because which one to use is a question the
renders answer rather than one to settle by argument.

Checked on the way: every pair sums to 1 at every one of the 16,384 cells. The worst deviation
over all seeds and both readings is printed and written into the JSON, and anything over 1e-5 is
a failure rather than a note.

    python scripts/draw_mask_overlays.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from poe_repair.composers._regional_masks import (  # noqa: E402
    EDGE_CELLS, LATENT, READINGS, masks_from_boxes, sigma_for,
)

OUT_DIR = REPO / "artifacts/results/is-the-layout-the-missing-piece"
MAX_SUM_DEVIATION = 1e-5
CAT_RGB = np.array([0.12, 0.47, 0.71])
DOG_RGB = np.array([0.84, 0.15, 0.16])
TINT = 0.62   # how far a fully-owned cell is pulled toward its colour


def boxes_for(record: dict) -> tuple[dict, dict]:
    """The cat box and the dog box of one seed's record, in that order."""
    by_call = {b["call"]: b for b in record["boxes"]}
    return by_call["cat"], by_call["dog"]


def tinted(render: Image.Image, m_cat: np.ndarray, m_dog: np.ndarray) -> np.ndarray:
    """The render pulled toward blue where the cat owns the cell and red where the dog does."""
    side = render.size[0]
    up = (side // LATENT) or 1
    cat = np.kron(m_cat, np.ones((up, up)))[:side, :side, None]
    dog = np.kron(m_dog, np.ones((up, up)))[:side, :side, None]
    if cat.shape[0] != side:   # a 320 px tile is not a whole multiple of 128
        cat = np.array(Image.fromarray((m_cat * 255).astype(np.uint8)).resize(
            (side, side), Image.NEAREST))[..., None] / 255.0
        dog = 1.0 - cat
    base = np.asarray(render, dtype=np.float64) / 255.0
    colour = cat * CAT_RGB + dog * DOG_RGB
    weight = TINT * np.abs(cat - 0.5) * 2.0    # untinted where the two experts share a cell
    return np.clip(base * (1 - weight) + colour * weight, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=Path, default=OUT_DIR / "boxes.json")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--edge-cells", type=float, default=EDGE_CELLS)
    args = ap.parse_args()

    data = json.loads(args.boxes.read_text())
    chosen = [r for r in data["seeds"] if r.get("selected")]
    if not chosen:
        raise SystemExit(f"no selected seeds in {args.boxes}")
    print(f"{len(chosen)} seeds, both readings, edge ramp {args.edge_cells:g} cells "
          f"(sigma {sigma_for(args.edge_cells):.2f})", flush=True)

    rows, worst, summary = len(chosen), 0.0, []
    fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.7), squeeze=False)
    for row, rec in enumerate(chosen):
        cat_box, dog_box = boxes_for(rec)
        with Image.open(rec["render_path"]) as im:
            render = im.convert("RGB")
        axes[row][0].imshow(render)
        axes[row][0].set_title(f"seed {rec['seed']}: the joint render", fontsize=9)
        for col, reading in enumerate(READINGS, start=1):
            pair = masks_from_boxes(cat_box["box_cells"], dog_box["box_cells"],
                                    reading=reading, edge_cells=args.edge_cells)
            worst = max(worst, pair.max_sum_deviation)
            ax = axes[row][col]
            ax.imshow(tinted(render, pair.a, pair.b))
            scale = render.size[0] / LATENT
            cx0, cy0, cx1, cy1 = pair.box_a_cells
            dx0, dy0, dx1, dy1 = pair.box_b_cells
            ox0, oy0 = max(cx0, dx0) * scale, max(cy0, dy0) * scale
            ox1, oy1 = min(cx1, dx1) * scale, min(cy1, dy1) * scale
            if ox1 > ox0 and oy1 > oy0:
                ax.add_patch(Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, fill=False,
                                       edgecolor="white", linewidth=1.6, linestyle="--"))
            ax.set_title(f"overlap {reading}: cat holds {pair.share_a:.0%} of the grid",
                         fontsize=9)
            summary.append({"seed": rec["seed"], "reading": reading,
                            "cat_box_cells": list(pair.box_a_cells),
                            "dog_box_cells": list(pair.box_b_cells),
                            "cat_share_of_grid": pair.share_a,
                            "max_sum_deviation": pair.max_sum_deviation})
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    # A tall figure needs the title's room reserved explicitly, or tight_layout puts the first
    # row's own titles underneath it.
    title_inches = 0.95
    fig_inches = rows * 3.7
    fig.tight_layout(rect=[0, 0, 1, 1 - title_inches / fig_inches])
    fig.suptitle("both overlap readings, drawn over the joint render each layout came from\n"
                 "blue where the cat expert owns the cell, red where the dog expert does, "
                 "untinted where they share it; the dashed box is where the two boxes crossed",
                 fontsize=12, y=1 - 0.18 / fig_inches, va="top")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "mask-overlays.png", dpi=120)
    plt.close(fig)

    (args.out / "masks.json").write_text(json.dumps({
        "what_this_is": "the two soft masks per seed per overlap reading, on the 128 x 128 "
                        "latent grid, normalised to sum to 1 at every cell",
        "latent_grid": LATENT,
        "edge_cells": args.edge_cells,
        "gaussian_sigma": sigma_for(args.edge_cells),
        "readings": list(READINGS),
        "max_sum_deviation_over_all": worst,
        "bar": MAX_SUM_DEVIATION,
        "per_seed": summary,
    }, indent=2))

    print(f"worst |m_cat + m_dog - 1| over all seeds and both readings: {worst:.2e} "
          f"(bar {MAX_SUM_DEVIATION:.0e})", flush=True)
    print(f"wrote {args.out / 'mask-overlays.png'} and {args.out / 'masks.json'}", flush=True)
    if worst > MAX_SUM_DEVIATION:
        raise SystemExit(f"the masks do not sum to 1: worst deviation {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
