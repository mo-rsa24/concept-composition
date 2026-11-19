#!/usr/bin/env python
"""The joint prompt ("a cat and a dog", classifier-free guidance 7.5, 50 DDIM steps) on a list of seeds, one tile each.

Each seed starts from the same cached noise render_results.py uses for that seed (init_latents_for_cell), so the joint
tile sits beside plain PoE and the pushed tile of the same seed. Writes <rung2>/<out-name>/seedNNNN__joint.png at
1024 px. A comparison column, never training data: the seeds given here are evaluation seeds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--prompt", default="a cat and a dog")
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    out = RUNG2 / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    single.disk_guard(out)
    ctx = make_ctx(output_root=out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_j, pool_j = enc(args.prompt)
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    t0 = time.perf_counter()
    for seed in args.seeds:
        tile = out / f"seed{seed:04d}__joint.png"
        if tile.exists():
            continue
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
        img = (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(img).save(tile)
        print(f"seed {seed}: joint drawn ({time.perf_counter() - t0:.0f} s)", flush=True)
    print("JOINT DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
