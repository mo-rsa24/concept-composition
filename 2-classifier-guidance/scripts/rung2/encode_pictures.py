#!/usr/bin/env python
"""Encode plan 22's local pictures into SDXL latents, the space the sampler and the discriminator use.

Reads every tile of ``training_set_v3.json`` whose source is "local", loads its 1024 px picture
from MOUNT_ROOT, encodes it with SDXL's VAE as plan 18 did (pixels to [-1, 1], times the VAE's
scaling factor) but takes the posterior mean rather than a sample, so a picture always gives the
same latent, and saves it as a (1, 4, 128, 128) tensor beside the rung 2 latents' shape.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from train_presence_head import load_picture  # noqa: E402

TILES = Path(__file__).resolve().parent / "training_set_v3.json"


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    tiles = [t for t in json.loads(TILES.read_text())["tiles"] if t["source"] == "local"]
    ctx = make_ctx(output_root=paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/dataset-v3/raw")
    vae = ctx.models["vae"]
    scaling = float(vae.config.scaling_factor)
    done, bad, t0 = 0, [], time.time()
    for t in tiles:
        out = paths.MOUNT_ROOT / t["latent"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            done += 1
            continue
        pixels = load_picture(paths.MOUNT_ROOT / t["picture"], 1024).to(device=ctx.device, dtype=vae.dtype)
        with torch.no_grad():
            z = vae.encode(pixels).latent_dist.mean * scaling
        if not torch.isfinite(z).all():
            bad.append(t["id"])
            continue
        torch.save(z.to(torch.float16).cpu(), out)
        done += 1
    print(f"encoded {done} of {len(tiles)} in {time.time() - t0:.0f} s; non-finite: {bad}", flush=True)
    print("ENCODE DONE" if not bad else "ENCODE HAD NON-FINITE LATENTS", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
