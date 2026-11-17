#!/usr/bin/env python
"""Plan 23's drawings: the joint prompt (real) and the product (fake), keeping their half-drawn states.

For every pair of ``kim_pairs.json`` (80 two-things pairs, cat x dog among them, the 7 held-out pairs
left out) and each of its 12 seeds (1 to 8, 17 to 20; never 9 to 16), two drawings from the seed's
own starting noise: the joint prompt with ordinary guidance, and the plain product of the two
prompts. Every 5th sampler state (steps 0, 5, ..., 45) and the final latent are saved as one
(11, 4, 128, 128) fp16 tensor, beside a 256 px picture for looking at.

Workers share the list through claims/<pair> (an atomic mkdir), so any number of cards can run it.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/kim/:
    states/<pair>__seedNN__<joint|product>.pt, small/<same>.png, timesteps.json, claims/
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

PAIRS = Path(__file__).resolve().parent / "kim_pairs.json"
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/kim"
KEEP = list(range(0, 50, 5))


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    spec = json.loads(PAIRS.read_text())
    if any(9 <= s <= 16 for s in spec["seeds"]):
        raise SystemExit("guard: seeds 9 to 16 are held out")
    if set(spec["pairs"]) & set(spec["held_out_pairs"]):
        raise SystemExit("guard: a held-out pair is in the drawing list")
    worker = f"{socket.gethostname().split('.')[0]}-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
    for d in ("states", "small", "claims"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    single.disk_guard(OUT)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    print(f"{worker}: {torch.cuda.get_device_name(0)}, {len(spec['pairs'])} pairs", flush=True)
    for pair, p in spec["pairs"].items():
        if (OUT / "claims" / pair / "done").exists():
            continue
        try:
            os.mkdir(OUT / "claims" / pair)
        except FileExistsError:
            continue
        (OUT / "claims" / pair / "worker").write_text(worker)
        seq_j, pool_j = enc(p["joint_prompt"])
        seq_a, pool_a = enc(p["prompt_a"])
        seq_b, pool_b = enc(p["prompt_b"])
        t0 = time.perf_counter()
        for seed in spec["seeds"]:
            for mode in ("joint", "product"):
                init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
                if mode == "joint":
                    o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
                else:
                    o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_a, pool_a=pool_a,
                                    seq_b=seq_b, pool_b=pool_b, **common)
                tr = o.tracker
                states = torch.cat([tr.trajectories[KEEP], tr.trajectories[-1:]])[:, 0].to(torch.float16)
                stem = f"{pair}__seed{seed:02d}__{mode}"
                torch.save(states, OUT / "states" / f"{stem}.pt")
                img = (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(img).resize((256, 256), Image.BICUBIC).save(OUT / "small" / f"{stem}.png")
                if not (OUT / "timesteps.json").exists():
                    (OUT / "timesteps.json").write_text(json.dumps(
                        {"kept_steps": KEEP + [50], "timesteps": [int(tr.timesteps[k]) for k in KEEP] + [0]}))
        (OUT / "claims" / pair / "done").write_text(worker)
        print(f"{worker} {pair}: {2 * len(spec['seeds'])} drawings in {time.perf_counter() - t0:.0f} s", flush=True)
    print("KIM DRAWINGS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
