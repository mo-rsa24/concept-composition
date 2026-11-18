#!/usr/bin/env python
"""Plan 30 step 1: joint-prompt runs for many pairs, keeping the early states, to train one checker across pairs.

The cat x dog checker learnt from about 500 clean joint-prompt drawings of one pair. Plan 23 has 80 pairs but only 136
clean joint drawings across 48 of them (a median of 2 per pair). This draws 40 joint-prompt runs per pair ("A and B",
ordinary classifier-free guidance 7.5) for the pairs of kim_pairs.json, leaving out cat x dog (2,200 runs already) and
lion x meerkat (a test pair); kim_pairs.json's 7 held-out pairs are not in its list. Seeds 5001 to 5040 (never 9 to 16;
the starting noise per seed is shared across pairs). Saved per run, as render_trajectories.py does: the states at sampler
steps 0 to 20 and 25 to 45 by fives, plus the final latent, as one (27, 4, 128, 128) fp16 tensor, and the finished
picture at 512 x 512, for labelling.

Workers share the pairs through claims/<pair> (an atomic mkdir). After a run, list claims without a done file, remove
them and run again (the NFS mkdir claim can be skipped).

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/pair-runs/:
    states/<pair>__seedNNNN__joint.pt, pictures/<same>.png, timesteps.json, claims/
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

PAIRS = Path(__file__).resolve().parent / "kim_pairs.json"
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/pair-runs"
KEEP = list(range(0, 21)) + [25, 30, 35, 40, 45]
LEAVE_OUT = {"a_cat__x__a_dog", "a_lion__x__a_meerkat"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=5001)
    ap.add_argument("--last", type=int, default=5040)
    args = ap.parse_args()
    seeds = list(range(args.first, args.last + 1))
    if any(9 <= s <= 16 for s in seeds):
        raise SystemExit("guard: seeds 9 to 16 are held out")
    spec = json.loads(PAIRS.read_text())
    pairs = {k: v for k, v in spec["pairs"].items() if k not in LEAVE_OUT and k not in set(spec["held_out_pairs"])}
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    worker = f"{socket.gethostname().split('.')[0]}-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
    for d in ("states", "pictures", "claims"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    single.disk_guard(OUT)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    print(f"{worker}: {torch.cuda.get_device_name(0)}, {len(pairs)} pairs x {len(seeds)} seeds", flush=True)
    for pair, p in pairs.items():
        if (OUT / "claims" / pair / "done").exists():
            continue
        try:
            os.mkdir(OUT / "claims" / pair)
        except FileExistsError:
            continue
        (OUT / "claims" / pair / "worker").write_text(worker)
        seq_j, pool_j = enc(p["joint_prompt"])
        t0 = time.perf_counter()
        for seed in seeds:
            stem = f"{pair}__seed{seed:04d}__joint"
            if (OUT / "states" / f"{stem}.pt").exists():
                continue
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma, seq_cond=seq_j, pool_cond=pool_j, **common)
            tr = o.tracker
            torch.save(torch.cat([tr.trajectories[KEEP], tr.trajectories[-1:]])[:, 0].to(torch.float16),
                       OUT / "states" / f"{stem}.pt")
            img = (o.image[0].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(img).resize((512, 512), Image.BICUBIC).save(OUT / "pictures" / f"{stem}.png")
            if not (OUT / "timesteps.json").exists():
                (OUT / "timesteps.json").write_text(json.dumps(
                    {"kept_steps": KEEP + [50], "timesteps": [int(tr.timesteps[k]) for k in KEEP] + [0]}))
        (OUT / "claims" / pair / "done").write_text(worker)
        print(f"{worker} {pair}: {len(seeds)} runs in {time.perf_counter() - t0:.0f} s", flush=True)
    print("PAIR RUNS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
