#!/usr/bin/env python
"""Render SDXL's empty prompt on new seeds, for the dog classifier's "no dog" pile.

Classifier guidance starts from what the model draws with no prompt, and a classifier never shown
those pictures has no reliable answer about them. These are rendered at 25 steps because they are
used as clean images that training then noises itself, so the sampler's step count does not reach
the classifier. Seeds start at 1000, well clear of the held-out seeds 9 to 16.

Writes MOUNT_ROOT/outputs/dog_classifier/no_prompt/seed_NNNN.png, resuming past any that exist.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poe_repair import paths  # noqa: E402
from poe_repair.methods._sampling import initial_latents_for_pair, run_cfg, write_decoded_image  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl  # noqa: E402

OUT = paths.MOUNT_ROOT / "outputs/dog_classifier/no_prompt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(OUT)
    if u.free / u.total < 0.10:
        raise SystemExit("guard: /datasets nearly full")
    ctx = make_ctx(num_inference_steps=args.steps)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    cfg = RunConfig()
    for seed in range(args.first_seed, args.first_seed + args.n):
        path = OUT / f"seed_{seed:04d}.png"
        if path.exists():
            continue
        cell = PairSeedCell(pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / "no_prompt",
                            pair_slug="no_prompt", prompt_a="", prompt_b="", seed=seed,
                            regime="collision", height=1024, width=1024, grid_assets={})
        init, sigma = initial_latents_for_pair(cell=cell, models=ctx.models, device=ctx.device,
                                               dtype=ctx.dtype)
        out = run_cfg(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                      seq_cond=seq_e, pool_cond=pool_e, seq_e=seq_e, pool_e=pool_e,
                      guidance_scale=ctx.guidance_scale, num_inference_steps=args.steps,
                      height=1024, width=1024, euler_init_noise_sigma=sigma,
                      device=ctx.device, dtype=ctx.dtype)
        write_decoded_image(out.image, path)
        if seed % 25 == 0:
            print(f"seed {seed} done", flush=True)
    print(f"done: {len(list(OUT.glob('seed_*.png')))} pictures in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
