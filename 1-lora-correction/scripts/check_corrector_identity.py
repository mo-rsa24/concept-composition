"""At k = 0 the two new correctors must be the plain samplers, on real SDXL.

Runs one cell four times and compares final latents with torch.equal:

    run_joint_langevin(k=0)  against  run_cfg with the joint embedding
    run_poe_uhmc(k=0)        against  run_cfg_poe

and prints one line per pair, ending in PASS or MISS. The laptop-side twin with a stand-in UNet
is tests/test_corrector_drivers_identity.py.

    python scripts/check_corrector_identity.py --pair a_cat__x__a_dog --seed 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._corrector_drivers import run_joint_langevin, run_poe_uhmc  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402


def verdict(name: str, ref: torch.Tensor, out: torch.Tensor) -> bool:
    same = torch.equal(ref, out)
    diff = (ref.float() - out.float()).abs().max().item()
    print(f"{name:<44} max |diff| = {diff:.3e}   {'PASS' if same else 'MISS'}")
    return same


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    cell = cell_from_slug(args.pair, args.seed)
    ctx = make_ctx()
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    base = dict(init_latents=init_latents, models=ctx.models, guidance_scale=ctx.guidance_scale,
                num_inference_steps=ctx.num_inference_steps, height=cell.height, width=cell.width,
                euler_init_noise_sigma=euler_sigma, device=ctx.device, dtype=ctx.dtype)
    poe = dict(seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
               seq_e=emb["seq_e"], pool_e=emb["pool_e"])
    print(f"pair {args.pair}  seed {args.seed}  {ctx.num_inference_steps} DDIM steps  "
          f"guidance {ctx.guidance_scale}  {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    ok = verdict(
        "joint + Langevin, k = 0  vs  joint (run_cfg)",
        run_cfg(scheduler=ctx.scheduler, seq_cond=seq_j, pool_cond=pool_j, seq_e=emb["seq_e"],
                pool_e=emb["pool_e"], **base).latents,
        run_joint_langevin(scheduler=ctx.scheduler, seq_j=seq_j, pool_j=pool_j,
                           seq_e=emb["seq_e"], pool_e=emb["pool_e"], k=0, c=3.0,
                           noise_seed=args.seed, decode=False, **base).latents,
    )
    ok &= verdict(
        "PoE + unadjusted HMC, k = 0  vs  PoE (run_cfg_poe)",
        run_cfg_poe(scheduler=ctx.scheduler, **poe, **base).latents,
        run_poe_uhmc(scheduler=ctx.scheduler, **poe, k=0, c=3.0, n_leapfrog=5,
                     noise_seed=args.seed, decode=False, **base).latents,
    )
    print(f"\n{'PASS' if ok else 'MISS'}: both drivers at k = 0   ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
