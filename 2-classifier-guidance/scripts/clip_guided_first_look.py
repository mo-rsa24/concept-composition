#!/usr/bin/env python
"""A first look at CLIP-guided product on one seed, before any full run.

Renders seed 9 of cat x dog at weight 0 (which must match plain product byte for byte) and at
three nudge strengths, times each render and records peak GPU memory, then lays the pictures in
one row beside plain product and the joint prompt so the nudge can be judged by eye.

Writes, under MOUNT_ROOT/outputs/clip_guided/first_look/:
    first-look.png   one row: plain, joint, then the three strengths
    first_look.json  per render: seconds, peak memory, the score at each guided step
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers import mono as mono_composer  # noqa: E402
from poe_repair.composers import poe as poe_composer  # noqa: E402
from poe_repair.composers import poe_clip_guided  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.methods._clip_guided import CoOccurrenceScore  # noqa: E402
from poe_repair.rewards.plurality import PluralityReward  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell  # noqa: E402

SEED = 9
OUT_ROOT = paths.MOUNT_ROOT / "outputs/clip_guided"


def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--augment", type=int, default=0, help="random crops and flips to average over")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--tag", default="first_look")
    args = ap.parse_args()
    WEIGHTS = tuple(args.weights)
    OUT = OUT_ROOT / args.tag
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig()
    ctx = make_ctx(output_root=OUT / "raw")
    cell = PairSeedCell(pair_dir=cfg.paths.pilot_dir / f"seed_{SEED}" / "a_cat__x__a_dog",
                        pair_slug="a_cat__x__a_dog", prompt_a="a cat", prompt_b="a dog",
                        seed=SEED, regime="collision", height=1024, width=1024, grid_assets={})
    score = CoOccurrenceScore(PluralityReward(ctx.device), augment=args.augment)

    record = {}
    plain = poe_composer.run(cell, ctx, exp_name="plain", overwrite=True)
    joint = mono_composer.run(cell, ctx, exp_name="joint", overwrite=True)
    zero = poe_clip_guided.run(cell, ctx, score=score, weight=0.0, exp_name="w0", overwrite=True)
    record["canary_identical"] = md5(plain) == md5(zero)
    print(f"weight 0 identical to plain product: {record['canary_identical']}", flush=True)

    tiles = [("plain product\n(what goes wrong)", plain), ("\"a cat and a dog\"\n(the goal)", joint)]
    for w in WEIGHTS:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        path, out = poe_clip_guided.run(cell, ctx, score=score, weight=w, exp_name=f"w{w}",
                                        overwrite=True, return_outputs=True)
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        guided = [r for r in out.extras["per_step"] if r["guided"]]
        record[f"weight {w}"] = {"seconds": round(secs, 1), "peak_gib": round(peak, 1),
                                 "score_first": guided[0]["score"], "score_last": guided[-1]["score"]}
        print(f"weight {w}: {secs:.0f}s, peak {peak:.1f} GiB, CLIP score "
              f"{guided[0]['score']:+.4f} at step 0 -> {guided[-1]['score']:+.4f} at step "
              f"{guided[-1]['step']}", flush=True)
        tiles.append((f"CLIP nudge {w}", path))

    fig, axes = plt.subplots(1, len(tiles), figsize=(len(tiles) * 3.2, 3.9))
    for ax, (title, p) in zip(axes, tiles):
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB").resize((420, 420)))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    how = f"averaged over {args.augment} random crops and flips" if args.augment else "one framing"
    fig.suptitle(f"Seed {SEED}: plain product, the goal, and CLIP nudging toward two things ({how})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "first-look.png", dpi=100, bbox_inches="tight")
    (OUT / "first_look.json").write_text(json.dumps(record, indent=2))
    print("wrote", OUT / "first-look.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
