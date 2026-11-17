#!/usr/bin/env python
"""Rung 2's compute piece: time plain and guided product renders on one card, and set N.

Renders the cat x dog product on seeds 1 to 4, then the same seeds pushed at strength 30, one
after another on one card. The head is plan 18's dog classifier (run dog-01, step 3,000), standing
in for rung 2's discriminator because both use the same trunk, so a guided step costs the same.
One untimed plain render goes first so start-up costs are not counted.

N for best-of-N is the guided render's median time over the plain render's, rounded, at least 1:
N plain renders then cost one guided render's GPU time.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/compute/:
    tiles/seedNN__plain.png, tiles/seedNN__s30.png, timing.json
"""
from __future__ import annotations

import json
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._classifier_guided import run_poe_classifier_guided  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

SEEDS = (1, 2, 3, 4)
STRENGTH = 30.0
STAND_IN = ("dog-01", 3000)
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/compute"


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card; co3 on a Blackwell runs on the CPU in silence")
    card = torch.cuda.get_device_name(0)
    print(f"{socket.gethostname()}: {card}, torch {torch.__version__}", flush=True)
    single.disk_guard(OUT)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc("a cat")
    seq_b, pool_b = enc("a dog")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_a=seq_a, pool_a=pool_a,
                  seq_b=seq_b, pool_b=pool_b, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    def plain(seed):
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        return run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma, **common)

    plain(SEEDS[0])  # start-up costs, not counted
    plain_s = {}
    for seed in SEEDS:
        o, dt = timed(lambda: plain(seed))
        write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__plain.png")
        plain_s[seed] = dt
        print(f"plain   seed {seed}: {dt:.1f} s", flush=True)

    head, ckpt = single.load_dog_classifier(ctx, seq_e, pool_e, *STAND_IN)
    torch.cuda.reset_peak_memory_stats()
    guided_s, capped = {}, {}

    def guided(seed):
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        return run_poe_classifier_guided(init_latents=init, euler_init_noise_sigma=sigma,
                                         head=head, strength=STRENGTH, **common)

    for seed in SEEDS:
        o, dt = timed(lambda: guided(seed))
        write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__s{STRENGTH:g}.png")
        guided_s[seed] = dt
        capped[seed] = sum(o.extras["capped_per_step"]) / len(o.extras["capped_per_step"])
        print(f"guided  seed {seed}: {dt:.1f} s, cap fired on {100 * capped[seed]:.0f}% of steps",
              flush=True)

    p, g = statistics.median(plain_s.values()), statistics.median(guided_s.values())
    n = max(1, round(g / p))
    record = {
        "host": socket.gethostname(), "card": card, "torch": torch.__version__,
        "steps": ctx.num_inference_steps, "strength": STRENGTH, "stand_in_head": str(ckpt),
        "plain_seconds": plain_s, "guided_seconds": guided_s,
        "plain_median_s": p, "guided_median_s": g, "guided_over_plain": g / p,
        "best_of_n": n, "cap_fired_fraction": capped,
        "peak_memory_gb_guided": torch.cuda.max_memory_allocated() / 1e9,
    }
    (OUT / "timing.json").write_text(json.dumps(record, indent=2))
    print(f"plain median {p:.1f} s, guided median {g:.1f} s, ratio {g / p:.2f}, N = {n}", flush=True)
    print("TIMING DONE", OUT / "timing.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
