#!/usr/bin/env python
"""Render the joint prompt for every candidate seed, here, from the cached starting noise.

Why this exists. The layout is read off the joint prompt's render and then handed to the product
on the same seed, so the layout and the render have to be the same picture. They were not. For
seeds 17 and up the layout was detected on a tile cropped from an old contact sheet, while the
sweep renders this repo's own cached noise through this repo's own sampler, and the two disagree:
on seed 19 the sheet tile and our render are 26.47 apart on a 0-to-255 mean absolute pixel scale,
a cat by a brick wall against a cat in a garden. A layout belonging to another picture makes
every count that follows meaningless without making the sheet look wrong.

So the joint prompt is rendered here, for every candidate seed, before anything is detected. Two
things follow. The layout is read off the picture the sweep will actually produce, and these same
files are the sheet's joint column, rendered by a process that never imports the regional
composer, which is what the plan asks for anyway.

Held-out seeds 9 to 16 already have 1024 px renders from the learned-and-token run. Those are
kept and checked rather than redrawn: seed 9's existing render against ours is 0.37 of 255, so
the cached noise reproduces them, and a seed whose existing render disagrees by more than
MAX_REPRODUCTION_DIFF is reported rather than silently overwritten.

    python scripts/render_joint_references.py --seeds 9 32
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments._eval_common import cell_for  # noqa: E402
from poe_repair.methods._sampling import run_cfg, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

RENDER_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/is_the_layout_the_missing_piece")
EXISTING_1024 = Path("/datasets/mmolefe/poe_repair_min/outputs/learned_and_token/oracle/joint")
PROMPT_A, PROMPT_B = "a cat", "a dog"

# Above this the existing render is not the one our noise produces, and the seed is reported.
MAX_REPRODUCTION_DIFF = 3.0
NEEDED_MIB = 12288
MIN_FREE_DISK_FRACTION = 0.10


def guard_disk(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    free = usage.free / usage.total
    print(f"disk guard: {out_dir} has {usage.free / 1024 ** 3:.1f} GiB free "
          f"({free:.1%})", flush=True)
    if free < MIN_FREE_DISK_FRACTION:
        raise SystemExit(f"guard: under {MIN_FREE_DISK_FRACTION:.0%} free; refusing to write")


def guard_device(max_foreign_mib: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; this would render silently on the CPU")
    free_b, total_b = torch.cuda.mem_get_info(0)
    used_mib, free_mib = (total_b - free_b) / 1024 ** 2, free_b / 1024 ** 2
    if free_mib < NEEDED_MIB:
        raise SystemExit(f"guard: {free_mib:.0f} MiB free, under the {NEEDED_MIB} MiB SDXL needs")
    if used_mib > max_foreign_mib:
        raise SystemExit(f"guard: {used_mib:.0f} MiB held by other processes, over the "
                         f"{max_foreign_mib} MiB limit; pass --max-foreign-mib to share")
    print(f"gpu guard: {used_mib:.0f} MiB foreign, {free_mib:.0f} MiB free", flush=True)


def mean_abs_diff(a: Path, b: Path) -> float:
    with Image.open(a) as ia, Image.open(b) as ib:
        x = np.asarray(ia.convert("RGB"), dtype=np.float64)
        y = np.asarray(ib.convert("RGB").resize(ia.size), dtype=np.float64)
    return float(np.abs(x - y).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs=2, default=[9, 32], metavar=("FIRST", "LAST"))
    ap.add_argument("--render-root", type=Path, default=RENDER_ROOT)
    ap.add_argument("--max-foreign-mib", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    first, last = args.seeds
    out_dir = args.render_root / "joint"
    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    guard_disk(args.render_root)
    guard_device(args.max_foreign_mib)

    ctx = make_ctx(output_root=args.render_root)
    print(f"guidance {ctx.guidance_scale}, {ctx.num_inference_steps} steps, dtype {ctx.dtype}",
          flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(first, last + 1):
        path = out_dir / f"seed{seed:02d}.png"
        if path.exists() and not args.overwrite:
            print(f"seed {seed:2d}  already rendered", flush=True)
        else:
            cell = cell_for(PROMPT_A, PROMPT_B, seed)
            init, sigma = init_latents_for_cell(cell, ctx)
            emb = encode_pair(cell, ctx)
            seq_j, pool_j = get_joint_embeds(cell, ctx)
            res = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                          seq_cond=seq_j, pool_cond=pool_j,
                          seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                          guidance_scale=ctx.guidance_scale,
                          num_inference_steps=ctx.num_inference_steps,
                          height=cell.height, width=cell.width,
                          euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
            write_decoded_image(res.image, path)
            del res; gc.collect(); torch.cuda.empty_cache()
            print(f"seed {seed:2d}  rendered", flush=True)

        existing = EXISTING_1024 / f"seed{seed:02d}.png"
        diff = mean_abs_diff(existing, path) if existing.exists() else None
        if diff is not None:
            verdict = "reproduced" if diff <= MAX_REPRODUCTION_DIFF else "DISAGREES"
            print(f"        against the learned-and-token render: {diff:.2f} of 255 -> {verdict}",
                  flush=True)
        rows.append({"seed": seed, "path": str(path),
                     "existing_1024": str(existing) if existing.exists() else None,
                     "mean_abs_diff_to_existing": diff})

    (args.render_root / "joint_references.json").write_text(json.dumps({
        "what_this_is": "the joint prompt rendered here, from the cached starting noise, for "
                        "every candidate seed; the layout is detected on these and the sheet's "
                        "joint column is these",
        "prompt": f"{PROMPT_A} and {PROMPT_B}",
        "guidance_scale": ctx.guidance_scale,
        "num_inference_steps": ctx.num_inference_steps,
        "reproduction_bar": MAX_REPRODUCTION_DIFF,
        "seeds": rows,
    }, indent=2))
    disagreed = [r["seed"] for r in rows
                 if r["mean_abs_diff_to_existing"] is not None
                 and r["mean_abs_diff_to_existing"] > MAX_REPRODUCTION_DIFF]
    print(f"\n{len(rows)} seeds in {out_dir}")
    if disagreed:
        print(f"seeds whose existing render our noise does not reproduce: {disagreed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
