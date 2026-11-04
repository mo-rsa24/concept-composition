#!/usr/bin/env python
"""Train an SDXL LoRA on photographs and captions, which no other trainer in this repo does.

Every other trainer here learns from cached noise tensors toward a correction target: no
photographs, no captions, no text encoder in the loop. This experiment changes the captions and
nothing else, so it needs the ordinary denoising objective with a real image on one side and a
real caption on the other.

It is built against this repo's own SDXL loader rather than the diffusers LoRA example, so the
training path uses the same text encoders, VAE and UNet that every sampling script here uses, and
needs no network call on a compute node.

What one step does:
  pick a photo uniformly, ask the run's caption rule what to call it (the rule rolls blank 10% of
  the time on every photo, which is the cell the whole experiment rests on), encode the photo to
  latents through the VAE, encode the caption through both text encoders, add noise at a uniformly
  drawn timestep, ask the UNet for the noise, and take the mean squared error against the noise
  that was actually added. Only the LoRA weights move.

The two runs differ in one place, the `--run` flag, which selects which caption rule is used:
  A   a both-animal photo is always captioned "a photo of a cat and a dog"
  B   it is "a photo of a cat" 40% of the time, "a photo of a dog" 40%, jointly 10%

Writes, under MOUNT_ROOT/outputs/inclusive_captions/<run>/:
    step_<n>/            the LoRA adapter at that step, loadable with PeftModel.from_pretrained
    loss.jsonl           one row per step: step, timestep, caption, loss

Run it on a GPU node:
    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
      -m poe_repair.experiments.inclusive_captions.trainer --run B --steps 2000

The smoke run, which is the first thing to do with it:
    ... --run B --steps 20 --photos 3 --resolution 512 --no-wandb
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from poe_repair import paths
from poe_repair._sdxl.runtime import (
    encode_prompt_sdxl,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
)
from poe_repair.config import DEFAULT_MODEL_ID
from poe_repair.experiments.inclusive_captions import captions as cap

FAMILY = "outputs/inclusive_captions"
MANIFEST = paths.MOUNT_ROOT / "outputs/inclusive_captions_pool_v2/manifest.json"

# The LoRA sits on the UNet's attention projections only, which is where every other adapter in
# this repo sits, so a result here is comparable with the rest of the project.
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]

# Three, not five. Each sheet costs four times what it did at 512 square, and the
# checkpoints that carry the argument are the untrained one, one early and the end.
CHECKPOINT_STEPS = (0, 500, 2000)


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")


def load_photo(path: str, resolution: int) -> torch.Tensor:
    """One photo as a [-1, 1] tensor, centre-cropped square then resized.

    Centre crop rather than a squash, because a squashed cat is a different animal to the model
    and the caption still says cat.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        side = min(im.size)
        left = (im.width - side) // 2
        top = (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize(
            (resolution, resolution), Image.LANCZOS
        )
        arr = torch.from_numpy(np.asarray(im, dtype="float32") / 127.5 - 1.0)
    return arr.permute(2, 0, 1).unsqueeze(0)


def add_time_ids(resolution: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """SDXL's size conditioning: original size, crop corner, target size.

    The photos are centre-cropped to a square at `resolution`, so original and target are the
    same and the crop corner is the origin. Telling the model otherwise is a second axis nobody
    asked to vary.
    """
    return torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]], device=device, dtype=dtype
    )


def build_lora(unet, rank: int, alpha: int):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=LORA_TARGETS, lora_dropout=0.0,
                        bias="none")
    model = get_peft_model(unet, config)
    # The base weights stay in their loaded precision and the adapter trains in fp32, because a
    # rank-32 update accumulated in fp16 loses the small early steps this experiment measures.
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", choices=cap.RUNS, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--resolution", type=int, default=1024)
    # Rank 16, with alpha matched to it so the adapter's scale stays 1. Halving the rank saves
    # 0.37 GB of the run's 29.8, because almost all of that is activations at 1,024 square rather
    # than adapter state, so this is a choice about the size of the correction being learnt and
    # not a way to fit on a smaller card.
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    # 3e-5, measured over three rates on the same 550 photos at the same 250 steps. At 1e-4 the
    # renders lose fur texture and photographic lighting and come back as smeared pastel shapes.
    # At 1e-5 the texture returns but four of the eight sheet seeds still draw tiled pattern, a
    # laptop or a picture frame, as the untrained model does at 512 square. At 3e-5 all eight
    # seeds draw animals and the joint prompt draws a cat beside a dog on most of them, which is
    # the model having learnt what these photographs look like rather than only how they are lit.
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--photos", type=int, default=0,
                    help="train on only N photos, taken one per set in turn so every caption branch "
                         "of the rule gets exercised; 0 means all of them. For the smoke run.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--wandb-project", default="poe-repair-inclusive-captions")
    ap.add_argument("--wandb-entity", default="prime_lab")
    ap.add_argument("--wandb-id", default=None,
                    help="a fixed run id, so the sheet renderer logs into the same run")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else paths.MOUNT_ROOT / FAMILY / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    _disk_guard(out_dir)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"no photo manifest at {manifest_path}; run scripts/census/inclusive_captions_pool_v2.py")
    manifest = json.loads(manifest_path.read_text())
    photos = cap.build_photos(args.run, manifest=manifest, seed=args.seed)
    if args.photos:
        # Round-robin across the sets rather than the first N, because the first N are all
        # cat-only and the both-animal branch is the one cell the two runs differ in. A smoke run
        # that never draws a both-animal photo proves the loop runs and nothing about the rule.
        by_set: dict[str, list[cap.Photo]] = {}
        for photo in photos:
            by_set.setdefault(photo.photo_set, []).append(photo)
        queues = list(by_set.values())
        picked: list[cap.Photo] = []
        while len(picked) < args.photos and any(queues):
            for queue in queues:
                if queue and len(picked) < args.photos:
                    picked.append(queue.pop(0))
        photos = picked
    print("photos by set: " + ", ".join(
        f"{s} {sum(1 for p in photos if p.photo_set == s)}"
        for s in dict.fromkeys(p.photo_set for p in photos)), flush=True)
    print(f"run {args.run}: {len(photos)} photos, {args.steps} steps at {args.resolution} square",
          flush=True)

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    vae, unet = models["vae"], models["unet"]
    for module in (vae, models["text_encoder"], models["text_encoder_2"]):
        module.requires_grad_(False)
    unet.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet = build_lora(unet, args.rank, args.alpha)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    print(f"{sum(p.numel() for p in trainable):,} trainable parameters", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    rule = cap.CaptionRule(run=args.run, rng=random.Random(args.seed))
    picker = random.Random(args.seed + 1)
    time_ids = add_time_ids(args.resolution, device=device, dtype=dtype)
    scaling = float(vae.config.scaling_factor)
    n_train_steps = int(scheduler.config.num_train_timesteps)

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               id=args.wandb_id, resume="allow",
                               name=f"run-{args.run}", config=vars(args))

    loss_log = (out_dir / "loss.jsonl").open("a")
    started = time.time()
    for step in range(args.steps + 1):
        if step in CHECKPOINT_STEPS and step <= args.steps:
            target = out_dir / f"step_{step:05d}"
            unet.save_pretrained(str(target))
            print(f"saved {target.name}", flush=True)
        if step == args.steps:
            break

        photo = picker.choice(photos)
        caption = rule(photo)
        pixels = load_photo(photo.path, args.resolution).to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * scaling
        latents = latents.to(dtype=dtype)
        prompt_embeds, pooled = encode_prompt_sdxl(caption, models=models, device=device, dtype=dtype)

        noise = torch.randn_like(latents)
        timestep = torch.randint(0, n_train_steps, (1,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timestep)
        predicted = unet(
            noisy, timestep, encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
        loss = F.mse_loss(predicted.float(), noise.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        row = {"step": step, "timestep": int(timestep.item()), "caption": caption,
               "loss": float(loss.item())}
        loss_log.write(json.dumps(row) + "\n")
        loss_log.flush()
        if wandb_run is not None:
            wandb_run.log({"train/loss": row["loss"], "train/timestep": row["timestep"]}, step=step)
        if step % 25 == 0:
            print(f"step {step:5d}  loss {row['loss']:.4f}  t {row['timestep']:4d}  "
                  f"{caption or '<blank>'}", flush=True)

    loss_log.close()
    print(f"done in {time.time() - started:.0f}s, adapters under {out_dir}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
