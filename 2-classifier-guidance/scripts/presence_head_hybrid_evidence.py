#!/usr/bin/env python
"""Show what the two-way softmax did to the fused animals, as a sheet you can look at.

The counts say 32 of 33 cat x dog product renders were named a confident single species. This
draws those renders at a size where the fusion is visible, each labelled with the call the
instrument made and the margin it made it at, so the number and the picture sit together.

Reads rows.json from the labelling pass. No GPU, no model: it only re-draws what was recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402

SET_DIR = paths.MOUNT_ROOT / "outputs/presence_head/labelled_set"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=320)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.loads((SET_DIR / "rows.json").read_text())
    picked = [r for r in rows if r["source"] == "render" and r["pair"] == args.pair]
    picked.sort(key=lambda r: (r["cat_present"], r["dog_present"], r["seed"] or 0))
    if not picked:
        raise SystemExit(f"no rows for pair {args.pair}")

    cols = args.cols
    rows_n = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.5, rows_n * 2.8))
    axes = axes.ravel()
    for ax, r in zip(axes, picked):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((args.thumb, args.thumb)))
        margin = r["margins"][0] if r.get("margins") else float("nan")
        call = "cat" if r["cat_present"] else ("dog" if r["dog_present"] else "neither")
        colour = {"cat": "crimson", "dog": "crimson", "neither": "darkgreen"}[call]
        ax.set_title(f"seed {r['seed']}   labelled: {call}\nsoftmax margin {margin:+.3f}",
                     fontsize=8, color=colour)
        ax.axis("off")
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.suptitle(
        f"{args.pair}: {len(picked)} plain product-of-experts renders, every one a fused animal\n"
        "the label the two-way softmax rule gave each one, and the margin it gave it at\n"
        "red means the rule called a species on a chimera; green means it said neither",
        fontsize=12)
    fig.tight_layout()
    out = Path(args.out) if args.out else SET_DIR / f"evidence_{args.pair}.png"
    fig.savefig(out, dpi=100)
    print(f"wrote {out}  ({len(picked)} tiles)")


if __name__ == "__main__":
    main()
