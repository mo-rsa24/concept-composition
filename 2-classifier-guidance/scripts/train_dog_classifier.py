#!/usr/bin/env python
"""Train the dog classifier the paper's way, check it can see before it steers, and watch it steer.

The trunk is SDXL's down and middle blocks on their own UNet copy, trained through LoRA adapters;
the head is an attention pool. One sigmoid answers "is there a dog". Training draws timesteps from
the 50 the sampler visits and keeps every batch half dog, half no dog.

The check before steering, written before the run and the same rule plan 15 used: on held-out
pictures, balanced accuracy at least MIN_BALANCED_ACCURACY on at least MIN_PASSING_EARLY_STEPS of
the first 20 sampler steps. Written to gate.json whether it passes or not.

The probe pushes the empty prompt toward "dog" on held-out seed 9 with plan 17's sampler, so the
change can be watched in W&B while it trains.

Writes, under MOUNT_ROOT/outputs/dog_classifier/<run>/:
    latent_cache.pt, split.json, checkpoints/, probe/, accuracy.json, gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.experiments.presence_head.dog_classifier import (  # noqa: E402
    AttentionPoolHead, DogClassifier, attach_trunk_adapters,
)
from poe_repair.experiments.presence_head.strong_head import UNetMidFeatures  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from train_presence_head import balanced_accuracy, build_latent_cache, disk_guard  # noqa: E402

# The species is a switch: the same recipe trains a cat classifier from the cat piles.
SPECIES = "dog"

MIN_BALANCED_ACCURACY = 0.75
MIN_PASSING_EARLY_STEPS = 10
EARLY_STEPS = 20
GATE_FRACTION = 0.15
# Checking 20 noise levels over the whole held-out set takes about 25 minutes, so the check reads
# at most this many, taken evenly from every source.
MAX_GATE_ROWS = 320
SPLIT_SEED = 0
PROBE_SEED, PROBE_STRENGTH = 9, 10.0
WANDB_PROJECT = "poe-repair-animals-compose"


@torch.no_grad()
def evaluate(model, latents, labels, idx, timesteps, steps, ctx, gen):
    out = []
    for s in steps:
        t_val, preds = timesteps[s], []
        for start in range(0, len(idx), 8):
            z0 = latents[idx[start:start + 8]].float().to(ctx.device)
            t = t_val.repeat(z0.shape[0]).to(ctx.device)
            noisy = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
            preds.append(torch.sigmoid(model(noisy, t))[:, 0].cpu())
        out.append({"sampler_step": int(s), "timestep": int(t_val),
                    f"{SPECIES}_present": balanced_accuracy(torch.cat(preds), labels[idx])})
    return out


def probe(model, ctx, seq_e, pool_e, out_dir, step, wandb_run):
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.methods._classifier_guided import run_cfg_classifier_guided
    from poe_repair.methods._sampling import write_decoded_image
    from poe_repair.runtime import PairSeedCell
    cell = PairSeedCell(pair_dir=RunConfig().paths.pilot_dir / f"seed_{PROBE_SEED}" / "a_cat__x__a_dog",
                        pair_slug="a_cat__x__a_dog", prompt_a="a cat", prompt_b="a dog",
                        seed=PROBE_SEED, regime="collision", height=1024, width=1024, grid_assets={})
    init, sigma = init_latents_for_cell(cell, ctx)
    out = run_cfg_classifier_guided(
        init_latents=init, models=ctx.models, scheduler=ctx.scheduler, seq_cond=seq_e,
        pool_cond=pool_e, seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps, height=1024, width=1024,
        euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype,
        head=model, strength=PROBE_STRENGTH)
    path = out_dir / "probe" / f"step_{step:06d}.png"
    write_decoded_image(out.image, path)
    rel = [p / e for p, e in zip(out.extras["push_norm_per_step"], out.extras["eps_norm_per_step"])]
    print(f"  probe {path}  push over prediction median {sorted(rel)[len(rel) // 2]:.3f}", flush=True)
    if wandb_run is not None:
        import wandb
        wandb_run.log({f"probe/no_prompt_pushed_to_{SPECIES}": wandb.Image(str(path))}, step=step)


def main() -> int:
    from diffusers import UNet2DConditionModel
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="dog-01")
    ap.add_argument("--species", choices=("dog", "cat"), default="dog")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=750)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0, help="for a smoke run only")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    global SPECIES
    SPECIES = args.species
    family = paths.MOUNT_ROOT / f"outputs/{SPECIES}_classifier"
    rows_path = family / "piles/rows.json"

    out_dir = family / args.run
    disk_guard(out_dir)
    for sub in ("checkpoints", "probe"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; on a Blackwell node co3 runs on the CPU in silence")
    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    ctx = make_ctx(output_root=out_dir / "probe")
    for name in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[name].requires_grad_(False)
        ctx.models[name].eval()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    clf_unet = UNet2DConditionModel.from_pretrained(
        RunConfig().model_id, subfolder="unet", torch_dtype=ctx.dtype, use_safetensors=True).to(ctx.device)
    clf_unet.enable_gradient_checkpointing()
    clf_unet, lora_params = attach_trunk_adapters(clf_unet)
    head = AttentionPoolHead().to(ctx.device)
    model = DogClassifier(UNetMidFeatures(clf_unet, seq_e, pool_e), head)
    print(f"trainable: {sum(p.numel() for p in lora_params):,} adapter, "
          f"{sum(p.numel() for p in head.parameters()):,} head", flush=True)
    opt = torch.optim.AdamW([{"params": lora_params, "lr": 1e-4},
                             {"params": head.parameters(), "lr": 3e-4}], weight_decay=1e-2)

    rows = json.loads(rows_path.read_text())
    if args.max_rows:
        rows = random.Random(SPLIT_SEED).sample(rows, args.max_rows)
    cache = build_latent_cache(rows, ctx, out_dir / "latent_cache.pt")
    rows, latents = cache["rows"], cache["latents"]
    labels = torch.tensor([float(r[SPECIES]) for r in rows])

    rng = random.Random(SPLIT_SEED)
    by_source: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_source.setdefault(r["source"], []).append(i)
    gate_idx = []
    for src, idx in sorted(by_source.items()):
        idx = idx[:]
        rng.shuffle(idx)
        gate_idx += idx[:max(1, int(GATE_FRACTION * len(idx)))]
    gate_set = set(gate_idx)
    if len(gate_idx) > MAX_GATE_ROWS:
        keep = []
        for src, idx in sorted(by_source.items()):
            held = [i for i in idx if i in gate_set]
            keep += held[:max(1, round(MAX_GATE_ROWS * len(held) / len(gate_idx)))]
        gate_idx = keep
    gate_idx = sorted(gate_idx)
    dog_train = [i for i in range(len(rows)) if i not in gate_set and labels[i] == 1]
    nodog_train = [i for i in range(len(rows)) if i not in gate_set and labels[i] == 0]
    split = {"n_rows": len(rows), "n_gate": len(gate_idx), "n_dog_train": len(dog_train),
             "n_no_dog_train": len(nodog_train)}
    (out_dir / "split.json").write_text(json.dumps({**split, "gate_idx": gate_idx}))
    print(json.dumps(split), flush=True)

    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    timesteps = ctx.scheduler.timesteps.clone().cpu()
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=WANDB_PROJECT, name=args.run, config={
                **split, "steps": args.steps, "probe_strength": PROBE_STRENGTH,
                "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
                "min_passing_early_steps": MIN_PASSING_EARLY_STEPS})
        except Exception as exc:
            print(f"wandb off: {exc}", flush=True)

    log, started = [], time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            steps_to_eval = range(EARLY_STEPS) if step == args.steps else range(0, EARLY_STEPS, 2)
            acc = evaluate(model, latents, labels, gate_idx, timesteps, steps_to_eval, ctx,
                           torch.Generator().manual_seed(7))
            log.append({"train_step": step, "per_step": acc})
            (out_dir / "accuracy.json").write_text(json.dumps(log, indent=2))
            mean = sum(r[f"{SPECIES}_present"] for r in acc) / len(acc)
            print(f"step {step:5d}  early-step balanced accuracy, {SPECIES} {mean:.3f}  "
                  f"({time.time() - started:.0f}s)", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"gate/{SPECIES}": mean}, step=step)
            torch.save({"head": head.state_dict(),
                        "lora": {n: p.detach().cpu() for n, p in clf_unet.named_parameters() if "lora_" in n},
                        "step": step}, out_dir / "checkpoints" / f"step_{step:06d}.pt")
            model.train()
        if not args.no_probe and step > 0 and (step % args.probe_every == 0 or step == args.steps):
            model.eval()
            probe(model, ctx, seq_e, pool_e, out_dir, step, wandb_run)
            model.train()
        if step == args.steps:
            break
        idx = random.choices(dog_train, k=4) + random.choices(nodog_train, k=4)
        z0 = latents[idx].float().to(ctx.device)
        y = labels[idx].to(ctx.device)
        t = timesteps[torch.randint(0, len(timesteps), (len(idx),))].to(ctx.device)
        noisy = ctx.scheduler.add_noise(z0, torch.randn_like(z0), t)
        loss = F.binary_cross_entropy_with_logits(model(noisy, t)[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - started:.0f}s)", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item()}, step=step)

    final = log[-1]["per_step"]
    passing = [r["sampler_step"] for r in final if r[f"{SPECIES}_present"] >= MIN_BALANCED_ACCURACY]
    gate = {"rule": f"at least {MIN_PASSING_EARLY_STEPS} of the first {EARLY_STEPS} sampler steps with "
                    f"balanced accuracy at least {MIN_BALANCED_ACCURACY} on {SPECIES} present",
            "per_step": final, "passing_steps": passing,
            "passed": len(passing) >= MIN_PASSING_EARLY_STEPS}
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))
    print(f"GATE {'PASSED' if gate['passed'] else 'FAILED'}: {len(passing)} of {EARLY_STEPS}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
