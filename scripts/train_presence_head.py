#!/usr/bin/env python
"""Train the two-output presence head, and watch composition through it while it trains.

SDXL is frozen. Only the head trains. It reads a noisy latent and its timestep and answers two
independent questions, "a cat instance is present" and "a dog instance is present", each with its
own sigmoid and its own loss, so the two never share a normaliser.

The labelling rule is imported from poe_repair.experiments.presence_head.labelling, never
restated here. Its one load-bearing line: a fused animal is labelled neither animal present,
because one box is never both animals. That is where the meaning of "both" enters the loss, and
nothing in a caption carries it.

The probe is why this script renders at all. Every PROBE_EVERY steps it draws held-out cat x dog
seeds through the classifier-guided composer at two weights and logs the sheet to W&B, so
composition can be watched appearing rather than read once at the end. Four renders is a picture,
never a rate.

Writes, under MOUNT_ROOT/outputs/presence_head/<run>/:
    latent_cache.pt      one latent per labelled picture, encoded once
    checkpoints/         the head at each probe
    probe/               the probe sheets, also logged to W&B
    accuracy_by_step.json  balanced accuracy per label at every sampling step, per probe
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poe_repair import paths  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.experiments.presence_head import labelling as L  # noqa: E402
from poe_repair.experiments.presence_head.head import PresenceHead, LABEL_NAMES  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell  # noqa: E402

LABELLED_SET = paths.MOUNT_ROOT / "outputs/presence_head/labelled_set/rows.json"
FAMILY = "outputs/presence_head"

# A fused animal is 50 of 1,028 pictures and it is the class the whole idea rests on, so it is
# oversampled to a fixed share of every batch rather than left at its natural rate. The share is
# a constant here so it cannot be tuned after seeing a render.
FUSED_SHARE_OF_BATCH = 0.25
HELD_OUT_FRACTION = 0.15
SPLIT_SEED = 0

PROBE_EVERY = 250
PROBE_SEEDS = (9, 10)
PROBE_WEIGHTS = (0.5, 2.0)
PROBE_PAIR = "a_cat__x__a_dog"
PROBE_PROMPTS = ("a cat", "a dog")

RESOLUTION = 1024
MIN_FREE_DISK_PERCENT = 10
WANDB_PROJECT = "poe-repair-animals-compose"


def disk_guard(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_percent = 100.0 * usage.free / usage.total
    print(f"disk guard: {root} has {free_percent:.1f}% free", flush=True)
    if free_percent < MIN_FREE_DISK_PERCENT:
        raise SystemExit(f"guard: {root} is {100 - free_percent:.1f}% full")


def load_picture(path: str, resolution: int) -> torch.Tensor:
    """One picture as a [-1, 1] tensor, centre-cropped square then resized."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize(
            (resolution, resolution), Image.BICUBIC)
        x = torch.from_numpy(
            (torch.frombuffer(im.tobytes(), dtype=torch.uint8)
             .reshape(resolution, resolution, 3).numpy())).float()
    return (x.permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0)


def build_latent_cache(rows, ctx, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"latent cache already at {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")
    vae = ctx.models["vae"]
    scaling = float(vae.config.scaling_factor)
    latents, kept, started = [], [], time.time()
    for i, row in enumerate(rows):
        try:
            pixels = load_picture(row["path"], RESOLUTION).to(
                device=ctx.device, dtype=vae.dtype)
            with torch.no_grad():
                z = vae.encode(pixels).latent_dist.sample() * scaling
        except Exception as exc:
            print(f"  skipped {row['path']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        latents.append(z.squeeze(0).to(torch.float16).cpu())
        kept.append(row)
        if i % 100 == 0:
            print(f"  encoded {i}/{len(rows)}  ({time.time() - started:.0f}s)", flush=True)
    cache = {"latents": torch.stack(latents), "rows": kept}
    torch.save(cache, cache_path)
    print(f"latent cache: {cache['latents'].shape} -> {cache_path}", flush=True)
    return cache


def split_indices(rows, fraction: float, seed: int):
    """Hold out a fraction of each label group, so every group is represented on both sides."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (r["source"], r["cat_present"], r["dog_present"], "fused" in r["reason"])
        groups.setdefault(key, []).append(i)
    rng = random.Random(seed)
    train, held = [], []
    for key, idx in sorted(groups.items()):
        idx = idx[:]
        rng.shuffle(idx)
        n_held = max(1, int(round(fraction * len(idx)))) if len(idx) > 2 else 0
        held.extend(idx[:n_held])
        train.extend(idx[n_held:])
    return sorted(train), sorted(held)


def balanced_accuracy(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean of the true-positive rate and the true-negative rate, so a skewed set cannot flatter."""
    pos, neg = truth > 0.5, truth <= 0.5
    tpr = (pred[pos] > 0.5).float().mean().item() if pos.any() else float("nan")
    tnr = (pred[neg] <= 0.5).float().mean().item() if neg.any() else float("nan")
    parts = [v for v in (tpr, tnr) if not math.isnan(v)]
    return sum(parts) / len(parts) if parts else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="presence-head-01")
    ap.add_argument("--steps", type=int, default=7500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--probe-every", type=int, default=PROBE_EVERY)
    ap.add_argument("--no-probe", action="store_true", help="train without rendering, for a smoke run")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY / args.run
    disk_guard(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "probe").mkdir(parents=True, exist_ok=True)

    print(f"node {socket.gethostname()}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    ctx = make_ctx(output_root=out_dir / "probe")
    if not torch.cuda.is_available():
        raise SystemExit("guard: no CUDA; co3 on a Blackwell node runs on the CPU in silence")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    for module in (ctx.models["vae"], ctx.models["unet"],
                   ctx.models["text_encoder"], ctx.models["text_encoder_2"]):
        module.requires_grad_(False)
        module.eval()

    rows = json.loads(LABELLED_SET.read_text())
    cache = build_latent_cache(rows, ctx, out_dir / "latent_cache.pt")
    rows, latents = cache["rows"], cache["latents"]
    train_idx, held_idx = split_indices(rows, HELD_OUT_FRACTION, SPLIT_SEED)
    fused_train = [i for i in train_idx if "fused" in rows[i]["reason"]]
    other_train = [i for i in train_idx if "fused" not in rows[i]["reason"]]
    print(f"train {len(train_idx)} ({len(fused_train)} fused), held out {len(held_idx)}", flush=True)

    targets = torch.tensor(
        [[float(r["cat_present"]), float(r["dog_present"])] for r in rows], dtype=torch.float32)

    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_timesteps = ctx.scheduler.timesteps.clone().to("cpu")
    print(f"training over the {len(sampler_timesteps)} timesteps the sampler visits, "
          f"{int(sampler_timesteps.min())} to {int(sampler_timesteps.max())}", flush=True)

    head = PresenceHead().to(ctx.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    print(f"head parameters: {head.num_parameters:,}", flush=True)

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=WANDB_PROJECT, name=args.run, config={
                "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                "fused_share_of_batch": FUSED_SHARE_OF_BATCH,
                "held_out_fraction": HELD_OUT_FRACTION,
                "rule": L.RULE_IN_WORDS, "rule_constants": L.RULE_CONSTANTS,
                "n_train": len(train_idx), "n_fused_train": len(fused_train),
                "n_held_out": len(held_idx),
            })
        except Exception as exc:
            print(f"wandb off: {type(exc).__name__}: {exc}", flush=True)

    rng = random.Random(1234)
    n_fused_per_batch = max(1, int(round(FUSED_SHARE_OF_BATCH * args.batch_size)))
    accuracy_log: list[dict] = []
    started = time.time()

    for step in range(args.steps + 1):
        if step % args.probe_every == 0:
            head.eval()
            acc = evaluate(head, latents, targets, held_idx, sampler_timesteps, ctx)
            accuracy_log.append({"train_step": step, **acc})
            (out_dir / "accuracy_by_step.json").write_text(json.dumps(accuracy_log, indent=2))
            ckpt = out_dir / "checkpoints" / f"head_step_{step:06d}.pt"
            torch.save({"state_dict": head.state_dict(), "step": step}, ckpt)
            print(f"step {step:6d}  held-out balanced accuracy  "
                  f"cat {acc['mean_balanced_accuracy']['cat_present']:.3f}  "
                  f"dog {acc['mean_balanced_accuracy']['dog_present']:.3f}  "
                  f"({time.time() - started:.0f}s)", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"eval/balanced_accuracy/{k}": v
                               for k, v in acc["mean_balanced_accuracy"].items()}, step=step)
            if not args.no_probe and step > 0:
                probe(head, ctx, out_dir, step, wandb_run)
            head.train()
        if step == args.steps:
            break

        batch_idx = (rng.choices(fused_train, k=n_fused_per_batch) if fused_train else []) + \
                    rng.choices(other_train, k=args.batch_size - n_fused_per_batch)
        z0 = latents[batch_idx].to(device=ctx.device, dtype=torch.float32)
        y = targets[batch_idx].to(ctx.device)
        t = sampler_timesteps[torch.randint(0, len(sampler_timesteps), (len(batch_idx),))]
        noise = torch.randn_like(z0)
        noisy = ctx.scheduler.add_noise(z0, noise, t.to(ctx.device))
        logits = head(noisy, t.to(ctx.device))
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item()}, step=step)

    print(f"done in {time.time() - started:.0f}s", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


@torch.no_grad()
def evaluate(head, latents, targets, held_idx, sampler_timesteps, ctx) -> dict:
    """Balanced accuracy per label at every timestep the sampler visits."""
    z0 = latents[held_idx].to(device=ctx.device, dtype=torch.float32)
    y = targets[held_idx].to(ctx.device)
    per_step = []
    for t_value in sampler_timesteps:
        t = t_value.repeat(len(held_idx)).to(ctx.device)
        noise = torch.randn_like(z0)
        noisy = ctx.scheduler.add_noise(z0, noise, t)
        p = torch.sigmoid(head(noisy, t))
        per_step.append({
            "timestep": int(t_value),
            **{name: balanced_accuracy(p[:, i], y[:, i]) for i, name in enumerate(LABEL_NAMES)},
        })
    mean = {name: sum(r[name] for r in per_step) / len(per_step) for name in LABEL_NAMES}
    return {"per_sampling_step": per_step, "mean_balanced_accuracy": mean,
            "n_held_out": len(held_idx)}


def probe(head, ctx, out_dir: Path, step: int, wandb_run) -> None:
    """Render held-out seeds through the guided composer so composition can be watched arriving."""
    from poe_repair.composers import poe_classifier_guided
    cfg = RunConfig()
    tiles = {}
    for seed in PROBE_SEEDS:
        cell = PairSeedCell(
            pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / PROBE_PAIR,
            pair_slug=PROBE_PAIR, prompt_a=PROBE_PROMPTS[0], prompt_b=PROBE_PROMPTS[1],
            seed=seed, regime="collision", height=1024, width=1024, grid_assets={})
        for weight in PROBE_WEIGHTS:
            try:
                tiles[(seed, weight)] = poe_classifier_guided.run(
                    cell, ctx, head=head, weight=weight, train_step=step,
                    exp_name=f"probe/step_{step:06d}", overwrite=True)
            except Exception as exc:
                print(f"  probe seed {seed} weight {weight} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    if not tiles:
        return
    fig, axes = plt.subplots(len(PROBE_SEEDS), len(PROBE_WEIGHTS),
                             figsize=(len(PROBE_WEIGHTS) * 3.2, len(PROBE_SEEDS) * 3.5))
    axes = axes.reshape(len(PROBE_SEEDS), len(PROBE_WEIGHTS))
    for r, seed in enumerate(PROBE_SEEDS):
        for c, weight in enumerate(PROBE_WEIGHTS):
            ax = axes[r][c]
            path = tiles.get((seed, weight))
            if path is not None:
                with Image.open(path) as im:
                    ax.imshow(im.convert("RGB").resize((320, 320)))
            ax.set_title(f"seed {seed}, push {weight}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"the product with the presence head pushing it, after {step} training steps",
                 fontsize=12)
    fig.tight_layout()
    sheet = out_dir / "probe" / f"sheet_step_{step:06d}.png"
    fig.savefig(sheet, dpi=100)
    plt.close(fig)
    print(f"  probe sheet {sheet}", flush=True)
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({"probe/sheet": wandb.Image(str(sheet))}, step=step)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
