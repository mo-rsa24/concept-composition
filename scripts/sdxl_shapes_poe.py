"""Plain SDXL on the toy's two prompts: does PoE of "a red circle" x "a blue square" draw both
shapes, with and without a Langevin corrector?

Per seed, five 1024 x 1024 renders from one shared starting latent (50 DDIM steps, guidance 7.5,
the repo's defaults): "a red circle" alone, "a blue square" alone, plain PoE of the two, PoE with
the Langevin corrector (k = 20 steps at c = 3, the setting the SDXL corrector grid settled on,
run_poe_langevin), and the one prompt "a red circle and a blue square". No training, no adapter.

Usage: python scripts/sdxl_shapes_poe.py --seeds 0-3 --out /datasets/.../sdxl_shapes
"""
import argparse, shutil, sys, time
from pathlib import Path

import torch

from poe_repair.config import joint_prompt
from poe_repair.methods._poe_langevin import run_poe_langevin
from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image
from poe_repair.run import make_ctx
from poe_repair._sdxl.runtime import load_shared_latents
from poe_repair.composers._helpers import encode_prompt_sdxl

A, B = "a red circle", "a blue square"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--c", type=float, default=3.0)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.seeds.split("-"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 5e9:
        sys.exit(f"disk guard: under 5 GB free on {out}")
    ctx = make_ctx()
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc(""); seq_a, pool_a = enc(A); seq_b, pool_b = enc(B)
    seq_j, pool_j = enc(joint_prompt(A, B))
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    print(f"guidance {ctx.guidance_scale}, steps {ctx.num_inference_steps}, joint '{joint_prompt(A, B)}'", flush=True)
    for seed in range(lo, hi + 1):
        t0 = time.time()
        lat, sig = load_shared_latents(pair_dir=out / "no_pair", seed=seed, device=ctx.device, dtype=ctx.dtype,
                                       models=ctx.models, height=1024, width=1024)
        runs = {
            "circle": lambda: run_cfg(init_latents=lat, seq_cond=seq_a, pool_cond=pool_a, euler_init_noise_sigma=sig, **common),
            "square": lambda: run_cfg(init_latents=lat, seq_cond=seq_b, pool_cond=pool_b, euler_init_noise_sigma=sig, **common),
            "poe": lambda: run_cfg_poe(init_latents=lat, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                                       euler_init_noise_sigma=sig, **common),
            "poe_corrector": lambda: run_poe_langevin(init_latents=lat, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b,
                                                      pool_b=pool_b, euler_init_noise_sigma=sig, k=a.k, c=a.c,
                                                      noise_seed=seed, **common),
            "joint": lambda: run_cfg(init_latents=lat, seq_cond=seq_j, pool_cond=pool_j, euler_init_noise_sigma=sig, **common),
        }
        for name, fn in runs.items():
            p = out / f"seed_{seed:04d}_{name}.png"
            if p.exists(): continue
            with torch.no_grad():
                o = fn()
            write_decoded_image(o.image[0], p)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
