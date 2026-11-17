#!/usr/bin/env python
"""Draw more joint-prompt pictures for the checker's data: held-out pairs for evaluation, cat x dog for training.

Plan 23 gave 14% clean pairs from the joint prompt (136 of 960), too few for an evaluation set that
can measure a 0.75 bar (20 clean pairs reads +-0.11) and too few clean SDXL cat x dog pictures (4)
beside 110 photos. This draws a pair list x seed list with the joint prompt only, the same sampler
and settings as plan 23 (render_kim.py), and keeps the finished latent and a 768 px JPEG for labelling
through the Claude API.

    --spec more_heldout.json  7 held-out pairs, seeds 101 to 160 (evaluation only, never training)
    --spec more_catdog.json   cat x dog, seeds 101 to 250 (training)

Seeds 9 to 16 are refused (the result sheet's). Held-out pairs are refused unless the spec says
"use": "evaluation". Workers share the work through claims/<pair>__<first seed> (an atomic mkdir),
10 seeds per claim, so any number of cards can run it.

Writes MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/more/<spec name>/:
    final/<pair>__seedNNN__joint.pt (1, 4, 128, 128) fp16, finals/<same>.jpg, claims/
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
from poe_repair.methods._sampling import run_cfg  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

HERE = Path(__file__).resolve().parent
BLOCK = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    spec = json.loads(args.spec.read_text())
    held = set(json.loads((HERE / "training_set_v3.json").read_text())["held_out_pairs"])
    if any(9 <= s <= 16 for s in spec["seeds"]):
        raise SystemExit("guard: seeds 9 to 16 are the result sheet's")
    if set(spec["pairs"]) & held and spec.get("use") != "evaluation":
        raise SystemExit("guard: held-out pairs may only be drawn for evaluation")
    out = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/more" / args.spec.stem
    for d in ("final", "finals", "claims"):
        (out / d).mkdir(parents=True, exist_ok=True)
    single.disk_guard(out)
    worker = f"{socket.gethostname().split('.')[0]}-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
    ctx = make_ctx(output_root=out / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    blocks = [(pair, spec["seeds"][i:i + BLOCK]) for pair in spec["pairs"] for i in range(0, len(spec["seeds"]), BLOCK)]
    print(f"{worker}: {torch.cuda.get_device_name(0)}, {len(blocks)} blocks of up to {BLOCK} seeds", flush=True)
    for pair, seeds in blocks:
        claim = out / "claims" / f"{pair}__{seeds[0]:03d}"
        if (claim / "done").exists():
            continue
        try:
            os.mkdir(claim)
        except FileExistsError:
            continue
        (claim / "worker").write_text(worker)
        seq_j, pool_j = enc(spec["pairs"][pair])
        t0 = time.perf_counter()
        for seed in seeds:
            stem = f"{pair}__seed{seed:03d}__joint"
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
            torch.save(o.tracker.trajectories[-1:][:, 0].to(torch.float16), out / "final" / f"{stem}.pt")
            img = (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(img).resize((768, 768), Image.BICUBIC).save(out / "finals" / f"{stem}.jpg", quality=92)
        (claim / "done").write_text(worker)
        print(f"{worker} {pair} seeds {seeds[0]}-{seeds[-1]}: {len(seeds)} drawings in {time.perf_counter() - t0:.0f} s", flush=True)
    print("MORE DRAWINGS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
