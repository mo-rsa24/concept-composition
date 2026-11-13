#!/usr/bin/env python
"""Encode a pile of pictures to SDXL latents in parallel pieces, then merge them into one cache.

The trainers build their latent cache one picture at a time on the training card, which for 6,644
pictures is about 50 minutes before training starts. Encoding needs only SDXL's image encoder, which
fits on a 12 GB batch node, so the pile is cut into contiguous pieces, each encoded on its own node,
and merged in order into the file the trainer reads. The trainer then finds the cache and skips its
own encode.

Each piece encodes exactly as ``build_latent_cache`` does: the picture centre-cropped and resized,
the encoder in full precision (SDXL's encoder declares that it needs it), one random draw from the
latent distribution, scaled, stored in half precision.

    --shard I --of K   encode piece I of K of the rows, to <cache dir>/latent_shards/shard_I.pt
    --merge K          join pieces 0 to K-1 in order into <cache dir>/latent_cache.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair.config import RunConfig  # noqa: E402
from train_presence_head import load_picture  # noqa: E402

RESOLUTION = 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True, help="the pile's rows.json")
    ap.add_argument("--cache-dir", required=True, help="the run folder the trainer reads its cache from")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--of", type=int)
    ap.add_argument("--merge", type=int)
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    shard_dir = cache_dir / "latent_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = [torch.load(shard_dir / f"shard_{i}.pt", map_location="cpu") for i in range(args.merge)]
        cache = {"latents": torch.cat([p["latents"] for p in parts]),
                 "rows": [r for p in parts for r in p["rows"]]}
        torch.save(cache, cache_dir / "latent_cache.pt")
        print(f"merged {args.merge} pieces: {tuple(cache['latents'].shape)} -> {cache_dir / 'latent_cache.pt'}")
        return 0

    from diffusers import AutoencoderKL
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA")
    rows = json.loads(Path(args.rows).read_text())
    per = (len(rows) + args.of - 1) // args.of
    mine = rows[args.shard * per:(args.shard + 1) * per]
    vae = AutoencoderKL.from_pretrained(RunConfig().model_id, subfolder="vae",
                                        torch_dtype=torch.float32, use_safetensors=True).to("cuda")
    vae.requires_grad_(False)
    scaling = float(vae.config.scaling_factor)
    latents, kept, started = [], [], time.time()
    for row in mine:
        try:
            px = load_picture(row["path"], RESOLUTION).to(device="cuda", dtype=vae.dtype)
            with torch.no_grad():
                z = vae.encode(px).latent_dist.sample() * scaling
        except Exception as exc:
            print(f"  skipped {row['path']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        latents.append(z.squeeze(0).to(torch.float16).cpu())
        kept.append(row)
    torch.save({"latents": torch.stack(latents), "rows": kept}, shard_dir / f"shard_{args.shard}.pt")
    print(f"piece {args.shard} of {args.of}: {len(kept)} of {len(mine)} encoded in "
          f"{time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
