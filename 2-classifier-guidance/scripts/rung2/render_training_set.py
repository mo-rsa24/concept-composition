#!/usr/bin/env python
"""Rung 2's training set: six renders per seed on seeds 1 to 8, for the reader to call by eye.

Columns, every one from the seed's own starting noise (the cached cat x dog cell's x_T):
    joint       "a cat and a dog", ordinary guidance
    left_right  "a cat on the left and a dog on the right"
    apart       "a cat and a dog sitting apart"
    lone_cat    "a cat"
    lone_dog    "a dog"
    product     the plain product of "a cat" and "a dog"

The first three are candidate positives, kept only where the reader calls one cat and one dog,
apart and intact. lone_cat, lone_dog and product are the negatives. The final latent of each
render is saved beside its picture, because the discriminator trains on noised latents, not on
pixels. No seed above 8 is accepted: seeds 9 to 16 are held out.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/training_set/:
    tiles/seedNN__<column>.png, latents/seedNN__<column>.pt, training-set-sheet.png, manifest.json
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402

OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/training_set"
HELD_OUT = range(9, 17)
PROMPTS = {
    "joint": "a cat and a dog",
    "left_right": "a cat on the left and a dog on the right",
    "apart": "a cat and a dog sitting apart",
    "lone_cat": "a cat",
    "lone_dog": "a dog",
}
COLUMNS = [("\"a cat and a dog\"", "joint"),
           ("\"a cat on the left\nand a dog on the right\"", "left_right"),
           ("\"a cat and a dog\nsitting apart\"", "apart"),
           ("\"a cat\"", "lone_cat"), ("\"a dog\"", "lone_dog"),
           ("plain product\n\"a cat\" x \"a dog\"", "product")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    args = ap.parse_args()
    if any(s in HELD_OUT for s in args.seeds):
        raise SystemExit("guard: seeds 9 to 16 are held out and never enter training")
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card; co3 on a Blackwell runs on the CPU in silence")
    print(f"{socket.gethostname()}: {torch.cuda.get_device_name(0)}, seeds {args.seeds}", flush=True)
    single.disk_guard(OUT)
    for d in ("tiles", "latents"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=OUT / "raw")
    enc = {k: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
           for k, p in {**PROMPTS, "empty": ""}.items()}
    seq_e, pool_e = enc["empty"]
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                  height=1024, width=1024, device=ctx.device, dtype=ctx.dtype)
    manifest = {"seeds": args.seeds, "prompts": PROMPTS, "product": ["a cat", "a dog"],
                "host": socket.gethostname(), "renders": []}

    for seed in args.seeds:
        t0 = time.perf_counter()
        for key in PROMPTS:
            init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
            seq_c, pool_c = enc[key]
            o = run_cfg(init_latents=init, euler_init_noise_sigma=sigma,
                        seq_cond=seq_c, pool_cond=pool_c, **common)
            write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__{key}.png")
            torch.save(o.latents.detach().cpu(), OUT / "latents" / f"seed{seed:02d}__{key}.pt")
            manifest["renders"].append({"seed": seed, "column": key, "prompt": PROMPTS[key]})
        init, sigma = init_latents_for_cell(single.cell_for(seed), ctx)
        o = run_cfg_poe(init_latents=init, euler_init_noise_sigma=sigma,
                        seq_a=enc["lone_cat"][0], pool_a=enc["lone_cat"][1],
                        seq_b=enc["lone_dog"][0], pool_b=enc["lone_dog"][1], **common)
        write_decoded_image(o.image, OUT / "tiles" / f"seed{seed:02d}__product.png")
        torch.save(o.latents.detach().cpu(), OUT / "latents" / f"seed{seed:02d}__product.pt")
        manifest["renders"].append({"seed": seed, "column": "product", "prompt": "a cat x a dog"})
        print(f"seed {seed}: 6 renders in {time.perf_counter() - t0:.0f} s", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    single.SEEDS, single.OUT = tuple(args.seeds), OUT
    single.sheet(COLUMNS, OUT / "training-set-sheet.png",
                 "Rung 2 training set: seeds 1 to 8, each column from the seed's own starting noise")
    print("TRAINING SET DONE", OUT / "training-set-sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
