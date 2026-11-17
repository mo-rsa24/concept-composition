#!/usr/bin/env python
"""Encode every v4 training picture three times: once plain and twice changed, for v4c's checker.

The changes are drawn from one fixed distribution for clean pairs and rejects alike, so colour,
sharpness, compression and left-right layout stop telling the sides apart (read map caveat C3: a
photo against a render, or one picture's colours, must not be learnable):
    mirrored left-right half the time; brightness and contrast x U(0.75, 1.25); saturation
    x U(0.6, 1.4); hue shifted by U(-0.03, 0.03) of a turn; Gaussian blur of radius U(0.5, 2) px on
    half; saved as JPEG at quality U(35, 90) and read back.

Every copy, the plain one included, goes picture -> VAE, so plan 23's drawings are re-encoded from
their exported JPEGs like every other source. Seeded per picture, so a rerun gives the same copies.

Writes MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/dataset-v4/aug/<tile id>__a{0,1,2}.pt
as (1, 4, 128, 128) fp16. --part/--of splits the pictures across cards.
"""
from __future__ import annotations

import argparse
import colorsys
import io
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from poe_repair import paths  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
HERE = Path(__file__).resolve().parent
COPIES = 3


def picture_path(t: dict) -> Path:
    if t["source"] == "kim":
        return RUNG2 / "kim/finals" / f"{t['id'][4:]}.jpg"
    if "picture" in t:
        return paths.MOUNT_ROOT / t["picture"]
    return RUNG2 / "training_set/tiles" / f"{t['id']}.png"


def square(im: Image.Image, size: int = 1024) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)


def change(im: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.25))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.6, 1.4))
    shift = rng.uniform(-0.03, 0.03)
    hsv = np.asarray(im.convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(round(shift * 255))) % 256).astype(np.uint8)
    im = Image.fromarray(hsv, "HSV").convert("RGB")
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=rng.randint(35, 90))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=HERE / "training_set_v4.json")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    tiles = [t for t in json.loads(args.manifest.read_text())["tiles"]
             if not t.get("check") and zlib.crc32(t["id"].encode()) % args.of == args.part]
    out = RUNG2 / "dataset-v4/aug"
    out.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(output_root=RUNG2 / "dataset-v4/raw")
    vae = ctx.models["vae"]
    scaling = float(vae.config.scaling_factor)
    done, missing, bad, t0 = 0, [], [], time.time()
    for t in tiles:
        targets = [out / f"{t['id']}__a{k}.pt" for k in range(COPIES)]
        if all(p.exists() for p in targets):
            done += 1
            continue
        try:
            base = square(Image.open(picture_path(t)).convert("RGB"))
        except FileNotFoundError:
            missing.append(t["id"])
            continue
        rng = random.Random(zlib.crc32(t["id"].encode()))
        for k, target in enumerate(targets):
            im = base if k == 0 else change(base, rng)
            x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).float().permute(2, 0, 1)[None] / 127.5 - 1.0
            with torch.no_grad():
                z = vae.encode(x.to(ctx.device, vae.dtype)).latent_dist.mean * scaling
            if not torch.isfinite(z).all():
                bad.append(f"{t['id']}__a{k}")
                continue
            torch.save(z.to(torch.float16).cpu(), target)
        done += 1
        if done % 200 == 0:
            print(f"{done} of {len(tiles)} pictures ({time.time() - t0:.0f} s)", flush=True)
    print(f"encoded {done} of {len(tiles)} pictures x {COPIES} in {time.time() - t0:.0f} s; "
          f"missing pictures {len(missing)} {missing[:5]}; non-finite {bad[:5]}", flush=True)
    print("AUGMENT DONE" if not missing and not bad else "AUGMENT INCOMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
