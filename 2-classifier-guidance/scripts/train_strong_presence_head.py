#!/usr/bin/env python
"""Train the strong inclusive presence head, check it can see before it steers, and watch it steer.

SDXL is frozen. The head reads SDXL's own middle block on the empty prompt and answers yes or no
for each of the 200 species in the cleaned piles, independently. The labels are the user's
inclusive rule plus the one line that makes "both" mean two animals: a picture holding a cat and a
dog is yes to both; a smooshed animal is no to both; a single cat is yes to cat only.

The check before steering, written before the run. On held-out pictures involving a cat or a dog,
smooshes included, the head passes at a sampler step if its balanced accuracy is at least
MIN_BALANCED_ACCURACY on both "cat present" and "dog present". It passes overall if at least
MIN_PASSING_EARLY_STEPS of the first 20 sampler steps pass, since the first 20 are where the
layout is decided. The weak head scored 0.68 on cat and 0.56 on dog averaged over those steps.
The push is then switched on only over the steps that passed, so it never pushes where it cannot
see. The gate's verdict goes to gate.json whether it passes or not.

The probe renders held-out cat x dog seeds through the guided product while training, so the
change can be watched. Two seeds is a picture, never a rate.

Writes, under MOUNT_ROOT/outputs/strong_presence_head/<run>/:
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.config import RunConfig  # noqa: E402
from poe_repair.experiments.presence_head.strong_head import (  # noqa: E402
    SpeciesHead, StrongPresenceHead, UNetMidFeatures,
)
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl  # noqa: E402
from train_presence_head import balanced_accuracy, build_latent_cache, disk_guard  # noqa: E402

PILES = paths.MOUNT_ROOT / "outputs/cooccurrence_head/piles/piles_clean.json"
FAMILY = "outputs/strong_presence_head"
TARGETS = ("cat", "dog")

# The check before steering. In the source so it cannot move after the curve is seen.
MIN_BALANCED_ACCURACY = 0.75
MIN_PASSING_EARLY_STEPS = 10
EARLY_STEPS = 20

# Every batch of 8 is built from fixed shares, so the hard cases for the two steered species
# are seen at a fixed rate rather than at their natural 1 in 20.
BATCH_SHARES = {"both_catdog": 2, "both_other": 1, "neither_catdog": 2, "neither_other": 1,
                "single": 2}
GATE_FRACTION = 0.25
SPLIT_SEED = 0

PROBE_SEEDS = (9, 10)
PROBE_WEIGHT = 1.0
WANDB_PROJECT = "poe-repair-animals-compose"


def involves_target(row) -> bool:
    if row["source"] == "coco":
        return True
    a, b = row["pair"].split("__x__")
    from cooccurrence_label_cache import noun
    return bool({noun(a), noun(b)} & set(TARGETS))


def bucket(row) -> str:
    if row["pile"] == "single":
        return "single"
    side = "catdog" if involves_target(row) else "other"
    return f"{row['pile']}_{side}"


@torch.no_grad()
def evaluate(model, latents, targets, idx, timesteps, steps, ctx, gen) -> list[dict]:
    """Balanced accuracy on the two targets at each requested sampler step, with fixed noise."""
    out = []
    z0_all = latents[idx].float()
    y_all = targets[idx]
    for s in steps:
        t_val = timesteps[s]
        preds = []
        for start in range(0, len(idx), 8):
            z0 = z0_all[start:start + 8].to(ctx.device)
            noise = torch.randn(z0.shape, generator=gen).to(ctx.device)
            t = t_val.repeat(z0.shape[0]).to(ctx.device)
            preds.append(torch.sigmoid(model(ctx.scheduler.add_noise(z0, noise, t), t)).cpu())
        p = torch.cat(preds)
        row = {"sampler_step": int(s), "timestep": int(t_val)}
        for name in TARGETS:
            j = model.species.index(name)
            row[f"{name}_present"] = balanced_accuracy(p[:, j], y_all[:, j])
        out.append(row)
    return out


def probe(model, ctx, out_dir, step, window, wandb_run) -> None:
    from poe_repair.composers import poe_classifier_guided
    cfg = RunConfig()
    tiles = []
    for seed in PROBE_SEEDS:
        cell = PairSeedCell(pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / "a_cat__x__a_dog",
                            pair_slug="a_cat__x__a_dog", prompt_a="a cat", prompt_b="a dog",
                            seed=seed, regime="collision", height=1024, width=1024, grid_assets={})
        path = poe_classifier_guided.run(cell, ctx, head=model, weight=PROBE_WEIGHT,
                                         guidance_window=window, train_step=step,
                                         exp_name=f"probe/step_{step:06d}", overwrite=True)
        tiles.append((seed, path))
    fig, axes = plt.subplots(1, len(tiles), figsize=(len(tiles) * 3.6, 4.0))
    for ax, (seed, path) in zip(axes, tiles):
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB").resize((360, 360)))
        ax.set_title(f"seed {seed}", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"product pushed by the strong helper (strength {PROBE_WEIGHT}), "
                 f"after {step} training steps", fontsize=12)
    fig.tight_layout()
    sheet = out_dir / "probe" / f"sheet_step_{step:06d}.png"
    fig.savefig(sheet, dpi=100)
    plt.close(fig)
    print(f"  probe sheet {sheet}", flush=True)
    if wandb_run is not None:
        import wandb
        wandb_run.log({"probe/sheet": wandb.Image(str(sheet))}, step=step)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="strong-01")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0, help="for a smoke run only")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out_dir = paths.MOUNT_ROOT / FAMILY / args.run
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

    rows = json.loads(PILES.read_text())
    if args.max_rows:
        rows = random.Random(SPLIT_SEED).sample(rows, args.max_rows)
    cache = build_latent_cache(rows, ctx, out_dir / "latent_cache.pt")
    rows, latents = cache["rows"], cache["latents"]
    species = sorted({s for r in rows for s in r["present"]} | set(TARGETS))
    targets = torch.zeros(len(rows), len(species))
    for i, r in enumerate(rows):
        for s in r["present"]:
            targets[i, species.index(s)] = 1.0

    rng = random.Random(SPLIT_SEED)
    target_rows = [i for i, r in enumerate(rows) if involves_target(r)]
    rng.shuffle(target_rows)
    gate_idx = sorted(target_rows[:int(GATE_FRACTION * len(target_rows))])
    gate_set = set(gate_idx)
    train_idx = [i for i in range(len(rows)) if i not in gate_set]
    buckets: dict[str, list[int]] = {k: [] for k in BATCH_SHARES}
    for i in train_idx:
        buckets[bucket(rows[i])].append(i)
    split = {"n_rows": len(rows), "n_species": len(species), "n_gate": len(gate_idx),
             "n_train": len(train_idx), "train_buckets": {k: len(v) for k, v in buckets.items()}}
    (out_dir / "split.json").write_text(json.dumps({**split, "gate_idx": gate_idx, "species": species}))
    print(json.dumps(split), flush=True)
    empty = [k for k, v in buckets.items() if not v]
    if empty:
        raise SystemExit(f"guard: batch buckets with nothing to draw from: {empty}")

    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    model = StrongPresenceHead(UNetMidFeatures(ctx.models["unet"], seq_e, pool_e),
                               SpeciesHead(len(species)).to(ctx.device), species, TARGETS)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-2)
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    timesteps = ctx.scheduler.timesteps.clone().cpu()

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=WANDB_PROJECT, name=args.run, config={
                **split, "lr": args.lr, "steps": args.steps, "batch_shares": BATCH_SHARES,
                "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
                "min_passing_early_steps": MIN_PASSING_EARLY_STEPS})
        except Exception as exc:
            print(f"wandb off: {exc}", flush=True)

    accuracy_log, started = [], time.time()
    window = (0, EARLY_STEPS)
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.head.eval()
            steps_to_eval = range(EARLY_STEPS) if step == args.steps else range(0, EARLY_STEPS, 2)
            acc = evaluate(model, latents, targets, gate_idx, timesteps, steps_to_eval, ctx,
                           torch.Generator().manual_seed(7))
            accuracy_log.append({"train_step": step, "per_step": acc})
            (out_dir / "accuracy.json").write_text(json.dumps(accuracy_log, indent=2))
            mean = {n: sum(r[f"{n}_present"] for r in acc) / len(acc) for n in TARGETS}
            print(f"step {step:5d}  early-step balanced accuracy  cat {mean['cat']:.3f}  "
                  f"dog {mean['dog']:.3f}  ({time.time() - started:.0f}s)", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"gate/{n}": v for n, v in mean.items()}, step=step)
            torch.save({"state_dict": model.head.state_dict(), "species": species, "step": step},
                       out_dir / "checkpoints" / f"head_step_{step:06d}.pt")
            model.head.train()
        if (not args.no_probe and step > 0 and step % args.probe_every == 0) or \
                (not args.no_probe and step == args.steps):
            model.head.eval()
            probe(model, ctx, out_dir, step, window, wandb_run)
            model.head.train()
        if step == args.steps:
            break

        idx = [i for k, n in BATCH_SHARES.items() for i in random.choices(buckets[k], k=n)]
        z0 = latents[idx].to(device=ctx.device, dtype=torch.float32)
        y = targets[idx].to(ctx.device)
        t = timesteps[torch.randint(0, len(timesteps), (len(idx),))].to(ctx.device)
        noisy = ctx.scheduler.add_noise(z0, torch.randn_like(z0), t)
        with torch.no_grad():
            feat = model.features(noisy, t)
        loss = F.binary_cross_entropy_with_logits(model.head(feat), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item()}, step=step)

    final = accuracy_log[-1]["per_step"]
    passing = [r["sampler_step"] for r in final
               if all(r[f"{n}_present"] >= MIN_BALANCED_ACCURACY for n in TARGETS)]
    gate = {
        "rule": f"at least {MIN_PASSING_EARLY_STEPS} of the first {EARLY_STEPS} sampler steps with "
                f"balanced accuracy at least {MIN_BALANCED_ACCURACY} on both cat and dog present",
        "per_step": final, "passing_steps": passing,
        "passed": len(passing) >= MIN_PASSING_EARLY_STEPS,
        "guidance_window": [min(passing), max(passing) + 1] if passing else None,
        "weak_head_for_comparison": {"cat_present": 0.677, "dog_present": 0.563},
    }
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))
    print(f"GATE {'PASSED' if gate['passed'] else 'FAILED'}: {len(passing)} of {EARLY_STEPS} early "
          f"steps pass; guidance window {gate['guidance_window']}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
