#!/usr/bin/env python
"""Draw the pilot's extra training photos with Stable Diffusion 3.5 medium.

COCO cannot supply what this experiment needs at the quality it needs. Tightening the box-fraction
floor so the animal fills the frame leaves 2,140 cat-only and 1,432 dog-only photos, which is
plenty, but only 34 photos holding both animals, and the both-animal set is the one the whole
comparison rests on. So the single-animal sets are tightened and the both-animal set is topped up
with generated images.

Why a different model family. SDXL is what the pilot fine-tunes and scores. Training it on its own
output would feed back whatever SDXL already believes about cats and dogs together, which is the
quantity under test. SD 3.5 medium is a separate architecture with separate training, so what it
draws is outside evidence rather than SDXL's own prior.

Why fp16 here. The Quadro RTX 8000 is Turing, which has fp16 tensor cores and no bf16 ones, so
bf16 runs through emulation and is several times slower. SD 3.5 medium is stable in fp16.

    ~/envs-sd35/bin/python 3-understanding-compositionality/scripts/census/inclusive_captions_generate.py \
      --prompt both --n 400 --start-seed 0 --out-dir /datasets/.../generated/both

Every image is named by its prompt key and seed, so the set is reproducible and a rejected image
can be named. Nothing here screens the images; that is the scorer's job in the step after.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
PROMPTS = {
    "cat": "a photo of a cat",
    "dog": "a photo of a dog",
    "both": "a photo of a cat and a dog",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    ap.add_argument("--no-t5", action="store_true",
                    help="drop the T5 text encoder; faster and lighter, and worse at putting two "
                         "named animals in one frame, which is exactly what the both set needs")
    args = ap.parse_args()

    from diffusers import StableDiffusion3Pipeline

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    extra = {"text_encoder_3": None, "tokenizer_3": None} if args.no_t5 else {}
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, **extra).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"{MODEL_ID} loaded in {args.dtype}"
          f"{' without T5' if args.no_t5 else ''}", flush=True)

    prompt = PROMPTS[args.prompt]
    started = time.time()
    written = 0
    for i in range(args.n):
        seed = args.start_seed + i
        target = out_dir / f"{args.prompt}_{seed:05d}.png"
        if target.exists():
            continue
        image = pipe(
            prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
            height=args.resolution, width=args.resolution,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        image.save(target)
        written += 1
        if written == 1 or written % 10 == 0:
            each = (time.time() - started) / written
            print(f"{written} written, {each:.1f}s each, "
                  f"{(args.n - i - 1) * each / 60:.0f} min left", flush=True)

    print(f"done: {written} new images in {out_dir}, "
          f"{(time.time() - started) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
