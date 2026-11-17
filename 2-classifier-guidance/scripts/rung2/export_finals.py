#!/usr/bin/env python
"""Decode plan 23's finished drawings to 768 px JPEGs for labelling by eye or through the Claude API.

Reads the final latent (the last of the 11 saved states) of every drawing under kim/states/, decodes
it with SDXL's VAE and writes kim/finals/<stem>.jpg. Skips drawings already exported and keeps
polling until plan 23's drawing workers have finished and every drawing is out.
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from poe_repair import paths  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import decode_latents  # noqa: E402

KIM = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/kim"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    mine = lambda stem: zlib.crc32(stem.encode()) % args.of == args.part
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    out = KIM / "finals"
    out.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=KIM / "raw")
    done_total = 0
    while True:
        todo = [f for f in sorted((KIM / "states").glob("*.pt"))
                if mine(f.stem) and not (out / f"{f.stem}.jpg").exists()]
        for f in todo:
            try:
                z = torch.load(f)[-1:].to(ctx.device, ctx.dtype)
            except Exception:
                continue                                  # still being written; next pass
            with torch.no_grad():
                img = decode_latents(ctx.models, z)[0].float().clamp(0, 1)
            arr = (img * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr).resize((768, 768), Image.BICUBIC).save(out / f"{f.stem}.jpg", quality=92)
            done_total += 1
        all_claimed_done = len(list((KIM / "claims").glob("*/done"))) >= 80
        if all_claimed_done and not todo:
            break
        print(f"exported {done_total}; waiting for more drawings", flush=True)
        time.sleep(60)
    print(f"EXPORT DONE {len(list(out.glob('*.jpg')))} pictures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
