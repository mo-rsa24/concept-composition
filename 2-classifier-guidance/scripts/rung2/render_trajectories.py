#!/usr/bin/env python
"""Plan 25's runs: cat x dog drawn by plain PoE and by the joint prompt, keeping the early half-drawn states.

The readout checker's push changes where the animals are; a push through the decoded picture does not
(checker-loop.md, "Pushing with the fine-tuned CLIP checker"). The readout checker learnt only from finished
pictures with noise added. These runs give it the sampler's own early states, each to be labelled by how
its run finished (label_trajectories.py), so it can learn what a state heading for a clean cat and dog looks
like at steps 0 to 10.

For each seed (default 2001 to 3200; never 9 to 16, never a training or evaluation seed of the cat x dog
lists, which use 1 to 1,250), two runs from the seed's own starting noise: plain PoE ("a cat" x "a dog",
classifier-free guidance 7.5 each) and the joint prompt "a cat and a dog". Saved per run: the states at
sampler steps 0 to 20 and 25, 30, 35, 40, 45, and the final latent, as one (27, 4, 128, 128) fp16 tensor, and
the finished picture at 512 x 512.

Workers share the seeds in blocks of 25 through claims/<block> (an atomic mkdir). The mkdir claim can be
skipped on NFS: after a run, list claims without a done file, remove them and run again.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/trajectories/:
    states/seedNNNN__<product|joint>.pt, pictures/seedNNNN__<product|joint>.png, timesteps.json, claims/
"""
from __future__ import annotations

import argparse
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

OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/trajectories"
KEEP = list(range(0, 21)) + [25, 30, 35, 40, 45]
BLOCK = 25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2001)
    ap.add_argument("--last", type=int, default=3200)
    ap.add_argument("--modes", nargs="+", default=["product", "joint"], choices=["product", "joint"],
                    help="which runs to draw per seed; plain PoE never ended clean on 1,200 seeds, so more clean examples "
                         "come only from joint runs")
    args = ap.parse_args()
    seeds = list(range(args.first, args.last + 1))
    if any(s <= 1250 for s in seeds):
        raise SystemExit("guard: seeds 1 to 1,250 belong to the held-out sheet or the cat x dog lists")
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    worker = f"{socket.gethostname().split('.')[0]}-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
    for d in ("states", "pictures", "claims"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    single.disk_guard(OUT)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_j, pool_j = enc("a cat and a dog")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    print(f"{worker}: {torch.cuda.get_device_name(0)}, seeds {seeds[0]} to {seeds[-1]}", flush=True)
    blocks = [seeds[i:i + BLOCK] for i in range(0, len(seeds), BLOCK)]
    for block in blocks:
        name = f"{block[0]}-{block[-1]}"
        if (OUT / "claims" / name / "done").exists():
            continue
        try:
            os.mkdir(OUT / "claims" / name)
        except FileExistsError:
            continue
        (OUT / "claims" / name / "worker").write_text(worker)
        t0 = time.perf_counter()
        for seed in block:
            for mode in args.modes:
                stem = f"seed{seed:04d}__{mode}"
                if (OUT / "states" / f"{stem}.pt").exists():
                    continue
                init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
                if mode == "joint":
                    o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
                else:
                    o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, seq_a=seq_a, pool_a=pool_a,
                                    seq_b=seq_b, pool_b=pool_b, **common)
                tr = o.tracker
                states = torch.cat([tr.trajectories[KEEP], tr.trajectories[-1:]])[:, 0].to(torch.float16)
                torch.save(states, OUT / "states" / f"{stem}.pt")
                img = (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(img).resize((512, 512), Image.BICUBIC).save(OUT / "pictures" / f"{stem}.png")
                if not (OUT / "timesteps.json").exists():
                    (OUT / "timesteps.json").write_text(json.dumps(
                        {"kept_steps": KEEP + [50], "timesteps": [int(tr.timesteps[k]) for k in KEEP] + [0]}))
        (OUT / "claims" / name / "done").write_text(worker)
        print(f"{worker} {name}: {len(args.modes) * len(block)} runs in {time.perf_counter() - t0:.0f} s", flush=True)
    print("TRAJECTORIES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
