#!/usr/bin/env python
"""The same product-of-experts composition on SDXL, with the sampler, step count and guidance scale
free to be set from the command line.

1-lora-correction/scripts/concept_multi_poe.py and 5-text-embedding-third-expert/scripts/and_vector_sheet.py both pin SDXL to DDIM at 50 steps and
guidance 7.5, which is what the project's own results were measured at. This script exists so the
same arithmetic can be run at another model's settings and nothing else changes:

    eps = eps_empty + gs * sum_i (eps_i - eps_empty)

The comparison it was written for is PixArt-alpha's defaults, 20 steps of DPM-Solver++ order 2 at
guidance 4.5, against SDXL's own 50 steps of DDIM at 7.5, on the same pair, the same cached starting
noise and the same held-out seeds.

    # the control: this script's own code path at SDXL's settings, which must reproduce the
    # published 0 of 8
    python 1-lora-correction/scripts/sdxl_matched_multi_poe.py --out <dir> --sampler ddim --steps 50 --guidance 7.5 \
        --seeds 9 10 11 12 13 14 15 16

    # the matched run
    python 1-lora-correction/scripts/sdxl_matched_multi_poe.py --out <dir> --sampler dpmpp --steps 20 --guidance 4.5 \
        --seeds 9 10 11 12 13 14 15 16

Tiles land as <out>/<a>__x__<b>/seedNN/<column>.png with a JSON sidecar naming the sampler, the step
count and the guidance scale; an existing tile is skipped. The starting latent is the cached x_T for
that seed, the same one every other SDXL result in this repo starts from.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NOISE_PAIR = "a_cat__x__a_dog"   # whose cached starting latent every seed borrows
COLUMNS = [("a", ["a"]), ("b", ["b"]), ("poe", ["a", "b"]), ("joint", ["joint"])]
MIN_FREE_GB = 2.0
MAX_USED_PCT = 90.0


def slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def disk_guard(path: Path) -> None:
    """Guard the filesystem `path` actually resolves to, never an assumed one (poe-disk-001)."""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1024 ** 3
    used_pct = 100.0 * usage.used / usage.total
    print(f"disk guard on {path.resolve()}: {free_gb:.1f} GB free, {used_pct:.0f}% used", flush=True)
    if free_gb < MIN_FREE_GB or used_pct >= MAX_USED_PCT:
        sys.exit(f"disk guard refused {path.resolve()}: need {MIN_FREE_GB} GB free and under {MAX_USED_PCT}% used")


def make_scheduler(name: str, ddim):
    """DDIM as the project uses it, or DPM-Solver++ order 2 built from the same model's config."""
    if name == "ddim":
        return ddim
    from diffusers import DPMSolverMultistepScheduler
    return DPMSolverMultistepScheduler.from_config(ddim.config, algorithm_type="dpmsolver++",
                                                   solver_order=2, use_karras_sigmas=False)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--a", default="a cat")
    ap.add_argument("--b", default="a dog")
    ap.add_argument("--joint", default="a cat and a dog")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--sampler", choices=["ddim", "dpmpp"], required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--guidance", type=float, required=True)
    args = ap.parse_args()

    disk_guard(args.out)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; this would run on the CPU with no error (poe-launch-002)")

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    ctx = make_ctx()
    sched = make_scheduler(args.sampler, ctx.scheduler)
    unet, gs = ctx.models["unet"], args.guidance
    print(f"sampler {type(sched).__name__}, {args.steps} steps, guidance {gs}", flush=True)

    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    e = {"a": enc(args.a), "b": enc(args.b), "joint": enc(args.joint)}
    e0 = enc("")
    root = args.out / f"{slug(args.a)}__x__{slug(args.b)}"

    for seed in args.seeds:
        cell = cell_from_slug(NOISE_PAIR, seed)
        init, sigma = init_latents_for_cell(cell, ctx)
        for key, names in COLUMNS:
            out = root / f"seed{seed:02d}" / f"{key}.png"
            if out.exists():
                print(f"{out}: already rendered", flush=True)
                continue
            embs = [e[c] for c in names] + [e0]
            seq = torch.cat([s for s, _ in embs], dim=0).to(ctx.dtype)
            pool = torch.cat([p for _, p in embs], dim=0).to(ctx.dtype)
            n = len(embs)
            cond = {"text_embeds": pool, "time_ids": add_time_ids(height=cell.height, width=cell.width,
                                                                  batch_size=n, device=ctx.device,
                                                                  dtype=ctx.dtype)}
            sched.set_timesteps(args.steps)   # order-2 DPM-Solver++ keeps state, so every tile resets it
            lat = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)
            with torch.no_grad():
                for i, t in enumerate(sched.timesteps):
                    eps = unet(sched.scale_model_input(lat.repeat(n, 1, 1, 1), t), t,
                               encoder_hidden_states=seq, added_cond_kwargs=cond, timestep_cond=None).sample
                    e_u = eps[-1:]
                    eg = e_u + gs * sum(eps[k:k + 1] - e_u for k in range(n - 1))
                    if args.sampler == "ddim":
                        abar = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
                        lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i,
                                                    x0=tweedie_mean(lat, abar, eg), eps=eg)
                    else:
                        lat = sched.step(eg, t, lat, return_dict=False)[0]
                image = decode_latents(ctx.models, lat).cpu()
            out.parent.mkdir(parents=True, exist_ok=True)
            write_decoded_image(image, out)
            out.with_suffix(".json").write_text(json.dumps(
                {"column": key, "conditions": [{"a": args.a, "b": args.b, "joint": args.joint}[c] for c in names],
                 "a": args.a, "b": args.b, "joint": args.joint, "seed": seed, "noise_from": NOISE_PAIR,
                 "sampler": type(sched).__name__, "steps": args.steps, "guidance": gs,
                 "size": [cell.width, cell.height], "dtype": str(ctx.dtype)}, indent=1))
            gc.collect()
            torch.cuda.empty_cache()
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
