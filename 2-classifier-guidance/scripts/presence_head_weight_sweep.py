#!/usr/bin/env python
"""The registered test: eight held-out seeds across four classifier weights, with three references.

Every column renders inside one loop on one device, including the three references, because a
float32-against-float16 difference of the same combine costs about three grey levels over 50
steps here and one uncorrected cell reads differently on three different cards. A reference
quoted from an older run is a reference that moved.

The columns:
    plain product       the failure case, 0 of 8 on these seeds
    damped product      each expert held at a constant 3.0 instead of 7.5. It already puts both
                        animals in frame on 8 of 8 cells and ruins the picture doing it, so this
                        is what a new method has to beat, not plain product.
    joint prompt        "a cat and a dog" through the model the ordinary way, 8 of 8. The ceiling.
    four weights        the presence head's gradient at each strength

Calls are made by eye on the full-resolution tiles, never on the sheet: two renders already read
as a clear cat and a clear dog at sheet scale and as fused animals at full size. No statistic over
the gradient may stand in for looking, because a norm guard on an earlier corrector here reported
healthy numbers while the samples were being destroyed.

Writes, under MOUNT_ROOT/outputs/presence_head/sweep/:
    weight-sweep-sheet.png   the 8 by 7 grid
    tiles/                   every full-resolution tile, named by seed and column
    manifest.json            one row per tile with its path, column and per-step gradient norms
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import socket
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
from poe_repair.composers import poe_classifier_guided  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.experiments.presence_head.head import PresenceHead  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell  # noqa: E402

PAIR_SLUG = "a_cat__x__a_dog"
PROMPT_A, PROMPT_B = "a cat", "a dog"
HELD_OUT_SEEDS = tuple(range(9, 17))
WEIGHTS = (0.25, 0.5, 1.0, 2.0)
DAMPED_GUIDANCE = 3.0
FAMILY = "outputs/presence_head/sweep"
MIN_FREE_DISK_PERCENT = 10


def disk_guard(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_percent = 100.0 * usage.free / usage.total
    print(f"disk guard: {root} has {free_percent:.1f}% free", flush=True)
    if free_percent < MIN_FREE_DISK_PERCENT:
        raise SystemExit(f"guard: {root} is {100 - free_percent:.1f}% full")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                    help="head checkpoint; default is the last of presence-head-01")
    ap.add_argument("--run", default="presence-head-01")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY
    disk_guard(out_dir)
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else sorted(
        (paths.MOUNT_ROOT / f"outputs/presence_head/{args.run}/checkpoints").glob("head_step_*.pt")
    )[-1]
    print(f"node {socket.gethostname()}  checkpoint {ckpt_path}", flush=True)

    cfg = RunConfig()
    ctx = make_ctx(output_root=out_dir / "raw")
    damped_ctx = dataclasses.replace(ctx, guidance_scale=DAMPED_GUIDANCE)
    print(f"device {ctx.device}  dtype {ctx.dtype}  steps {ctx.num_inference_steps}  "
          f"guidance {ctx.guidance_scale}, damped {damped_ctx.guidance_scale}", flush=True)

    state = torch.load(ckpt_path, map_location="cpu")
    head = PresenceHead().to(ctx.device)
    head.load_state_dict(state["state_dict"])
    head.eval()
    train_step = int(state.get("step", -1))
    print(f"head loaded from training step {train_step}", flush=True)

    columns = (["plain product", "damped product", "joint prompt"]
               + [f"push {w}" for w in WEIGHTS])
    rows, started = [], time.time()

    for seed in HELD_OUT_SEEDS:
        cell = PairSeedCell(
            pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / PAIR_SLUG,
            pair_slug=PAIR_SLUG, prompt_a=PROMPT_A, prompt_b=PROMPT_B,
            seed=seed, regime="collision", height=1024, width=1024, grid_assets={})
        for column in columns:
            tile = out_dir / "tiles" / f"seed{seed:02d}__{column.replace(' ', '_')}.png"
            extras = {}
            try:
                if column == "plain product":
                    src = poe_composer.run(cell, ctx, exp_name="sweep/plain", overwrite=True)
                elif column == "damped product":
                    src = poe_composer.run(cell, damped_ctx, exp_name="sweep/damped",
                                           overwrite=True)
                elif column == "joint prompt":
                    src = mono_composer.run(cell, ctx, exp_name="sweep/joint", overwrite=True)
                else:
                    weight = float(column.split()[1])
                    src, out = poe_classifier_guided.run(
                        cell, ctx, head=head, weight=weight, train_step=train_step,
                        exp_name=f"sweep/push_{weight}", overwrite=True, return_outputs=True)
                    extras = {"mean_grad_norm": round(
                        sum(out.extras["grad_norm_per_step"])
                        / max(1, len(out.extras["grad_norm_per_step"])), 6)}
                shutil.copyfile(src, tile)
            except Exception as exc:
                print(f"  seed {seed} {column}: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append({"seed": seed, "column": column, "tile": str(tile),
                         "source": str(src), **extras})
            print(f"  seed {seed:2d}  {column:16s}  ({time.time() - started:.0f}s)", flush=True)

    by_key = {(r["seed"], r["column"]): r for r in rows}
    fig, axes = plt.subplots(len(HELD_OUT_SEEDS), len(columns),
                             figsize=(len(columns) * 2.6, len(HELD_OUT_SEEDS) * 2.8))
    for r, seed in enumerate(HELD_OUT_SEEDS):
        for c, column in enumerate(columns):
            ax = axes[r][c]
            row = by_key.get((seed, column))
            if row:
                with Image.open(row["tile"]) as im:
                    ax.imshow(im.convert("RGB").resize((300, 300)))
            if r == 0:
                ax.set_title(column, fontsize=10)
            if c == 0:
                ax.text(-0.08, 0.5, f"seed {seed}", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        f"cat x dog on eight held-out seeds: the presence head's push at four strengths, "
        f"beside the three references\nhead trained {train_step} steps; every tile rendered in "
        f"one loop on one device; calls are made on the full-resolution tiles, not on this sheet",
        fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    sheet = out_dir / "weight-sweep-sheet.png"
    fig.savefig(sheet, dpi=100)
    plt.close(fig)

    manifest = {
        "what_this_is": "eight held-out cat x dog seeds at four classifier weights, with plain "
                        "product, the damped product and the joint prompt rendered in the same "
                        "loop as references",
        "pair": PAIR_SLUG, "prompts": [PROMPT_A, PROMPT_B],
        "seeds": list(HELD_OUT_SEEDS), "weights": list(WEIGHTS),
        "damped_guidance": DAMPED_GUIDANCE,
        "guidance_scale": ctx.guidance_scale, "num_inference_steps": ctx.num_inference_steps,
        "head_checkpoint": str(ckpt_path), "head_train_step": train_step,
        "columns": columns, "n_tiles": len(rows), "tiles": rows,
        "node": socket.gethostname(), "elapsed_seconds": round(time.time() - started, 1),
        "how_calls_are_made": "by eye on the full-resolution tile, into two clear animals, one "
                              "hybrid, or one single animal; hybrid is its own category",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nsheet {sheet}\n{len(rows)} tiles in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
