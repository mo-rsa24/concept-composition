#!/usr/bin/env python
"""SDXL's product estimate of the finished picture, decoded, for every cat x dog training and evaluation picture.

A checker that judges finished pictures can push the product sampler only through these estimates: at sampler
step s the sampler's guess of the finished latent is (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t),
with eps the product's noise guess (prompts "a cat", "a dog" and empty, classifier-free guidance 7.5 on each,
eps_cat + eps_dog - eps_null). This draws those guesses so a checker can be trained and scored on them.

Each picture's finished latent is noised to step s (one fixed draw per picture and step), the product's noise
guess is taken, the estimate decoded by the VAE and saved at 448 x 448. The picture keeps its own label: the
question asked of an estimate is "does the picture this is heading for count as a clean cat and dog?".
    training_set_catdog_more23.json (1,603 pictures): two steps per picture, drawn uniformly from 10 to 30
    eval_set_catdog_v2.json (220 pictures): steps 0, 5, 10, 15, 20, 25, 30 and 40, for a curve over steps
The estimate comes from a forward-noised finished picture, not from the product's own trajectory.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/estimates/:
    pictures/<tile id>__s<step>.jpg, index-<shard>.jsonl (one row per picture: id, step, split, label, kind, path)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from poe_repair import paths  # noqa: E402
from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean  # noqa: E402
from poe_repair.methods._sampling import add_time_ids  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import decode_latents, encode_prompt_sdxl  # noqa: E402
from train_readout_pair import load_latent  # noqa: E402

EVAL_STEPS = [0, 5, 10, 15, 20, 25, 30, 40]
TRAIN_LO, TRAIN_HI, TRAIN_PER = 10, 30, 2
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/estimates"


def jobs():
    train = json.loads((HERE / "training_set_catdog_more23.json").read_text())["tiles"]
    evals = json.loads((HERE / "eval_set_catdog_v2.json").read_text())["tiles"]
    out = []
    for t in train:
        rng = random.Random(t["id"])
        for s in sorted(rng.sample(range(TRAIN_LO, TRAIN_HI + 1), TRAIN_PER)):
            out.append((t, s, "train"))
    for t in evals:
        for s in EVAL_STEPS:
            out.append((t, s, "eval"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    work = jobs()[args.shard::args.of]
    (OUT / "pictures").mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard} of {args.of}: {len(work)} estimates", flush=True)

    ctx = make_ctx(output_root=OUT / "raw")
    unet = ctx.models["unet"]
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    (sa, pa), (sb, pb), (se, pe) = enc("a cat"), enc("a dog"), enc("")
    seq, pool = torch.cat([sa, sb, se]), torch.cat([pa, pb, pe])
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=3,
                                                          device=ctx.device, dtype=ctx.dtype)}
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_t = ctx.scheduler.timesteps.clone()
    index = open(OUT / f"index-{args.shard}.jsonl", "a")
    t0, done = time.time(), 0
    with torch.no_grad():
        for tile, s, split in work:
            path = OUT / "pictures" / f"{tile['id']}__s{s:02d}.jpg"
            row = {"id": tile["id"], "step": s, "split": split, "label": tile["label"], "kind": tile["kind"],
                   "path": str(path.relative_to(paths.MOUNT_ROOT))}
            if not path.exists():
                seed = int(hashlib.sha256(f"{tile['id']}|{s}".encode()).hexdigest()[:8], 16)
                noise = torch.randn((1, 4, 128, 128), generator=torch.Generator().manual_seed(seed))
                t = sampler_t[s]
                ab = ctx.scheduler.alphas_cumprod[int(t)].to(ctx.device, torch.float32)
                z0 = load_latent(tile).float().to(ctx.device)
                x = ctx.scheduler.add_noise(z0, noise.to(ctx.device), t.reshape(1).to(ctx.device))
                e = unet(x.repeat(3, 1, 1, 1).to(ctx.dtype), t.to(ctx.device), encoder_hidden_states=seq,
                         added_cond_kwargs=cond).sample.float()
                ea, eb, eu = e.chunk(3)
                eps = poe_eps(guided_eps(ea, eu, ctx.guidance_scale), guided_eps(eb, eu, ctx.guidance_scale), eu)
                x0 = tweedie_mean(x, ab, eps)
                img = decode_latents(ctx.models, x0.to(ctx.dtype))[0].float().clamp(0, 1)
                pil = Image.fromarray((img * 255).round().byte().permute(1, 2, 0).cpu().numpy())
                pil.resize((448, 448), Image.BICUBIC).save(path, quality=95)
            index.write(json.dumps(row) + "\n")
            index.flush()
            done += 1
            if done % 200 == 0:
                print(f"  {done} of {len(work)} ({time.time() - t0:.0f} s)", flush=True)
    print("ESTIMATES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
