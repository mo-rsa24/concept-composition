#!/usr/bin/env python
"""Train rung 2's pair-aware discriminator's head on the checked training set, and check it.

Reads the 41 tiles in ``training_set.json`` (positives, and intact negatives) as the final latents
the training-set renders saved. Each training step takes 4 positives and 4 negatives, noises each
to a timestep drawn from the sampler's 50, reads SDXL's frozen middle-block features without a
gradient, and trains only the head with binary cross-entropy on the logit.

The check, written before the run (plan 21, task 1.3): on the tiles of the --check-seeds, each
noised 4 times with a fixed generator, balanced accuracy at least MIN_BALANCED_ACCURACY on at
least MIN_PASSING_EARLY_STEPS of the first EARLY_STEPS sampler steps. Run once holding out seeds 7
and 8; the final discriminator then trains on all eight seeds with no check seeds, and its
accuracy on fresh noise of its own tiles is reported as fit only.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/discriminator/<run>/:
    head_step_NNNNNN.pt, accuracy.json, gate.json, split.json
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from train_presence_head import balanced_accuracy  # noqa: E402
import classifier_guidance_from_the_paper as single  # noqa: E402
from discriminator import PairDiscriminator, head_parameter_count  # noqa: E402

MIN_BALANCED_ACCURACY = 0.75
MIN_PASSING_EARLY_STEPS = 10
EARLY_STEPS = 20
CHECK_DRAWS = 4
TRAINING_SET = Path(__file__).resolve().parent / "training_set.json"
LATENTS = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/training_set/latents"
OUT_ROOT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/discriminator"


@torch.no_grad()
def accuracy_per_step(model, z, y, timesteps, steps, ctx, gen):
    out = []
    for s in steps:
        t_val, preds, truth = timesteps[s], [], []
        for _ in range(CHECK_DRAWS):
            for start in range(0, len(z), 8):
                z0 = z[start:start + 8].to(ctx.device)
                t = t_val.repeat(z0.shape[0]).to(ctx.device)
                noisy = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
                preds.append(torch.sigmoid(model(noisy, t)).float().cpu())
                truth.append(y[start:start + 8])
        out.append({"sampler_step": int(s), "timestep": int(t_val),
                    "balanced_accuracy": balanced_accuracy(torch.cat(preds), torch.cat(truth))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--check-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--tiles", type=Path, default=TRAINING_SET)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card; co3 on a Blackwell runs on the CPU in silence")
    print(f"{socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
    out = OUT_ROOT / args.run
    single.disk_guard(out)
    out.mkdir(parents=True, exist_ok=True)

    tiles = json.loads(args.tiles.read_text())["tiles"]
    if any(t["seed"] >= 9 for t in tiles):
        raise SystemExit("guard: a tile from seeds 9 to 16 is in the training set")
    train = [t for t in tiles if t["seed"] not in args.check_seeds]
    check = [t for t in tiles if t["seed"] in args.check_seeds] or train

    def load(ts):
        z = torch.cat([torch.load(LATENTS / f"{t['latent']}.pt").float() for t in ts])
        return z, torch.tensor([float(t["label"]) for t in ts])

    z_train, y_train = load(train)
    z_check, y_check = load(check)
    split = {"tiles_file": str(args.tiles), "train_seeds": sorted({t["seed"] for t in train}), "check_seeds": args.check_seeds,
             "train_positives": int(y_train.sum()), "train_negatives": int((1 - y_train).sum()),
             "check_positives": int(y_check.sum()), "check_negatives": int((1 - y_check).sum()),
             "check_is_held_out": bool(args.check_seeds)}
    (out / "split.json").write_text(json.dumps(split, indent=2))
    print(json.dumps(split), flush=True)

    ctx = make_ctx(output_root=out / "raw")
    for name in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[name].requires_grad_(False)
        ctx.models[name].eval()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    model = PairDiscriminator(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
    model.head.float()
    print(f"trainable: {head_parameter_count(model):,} head weights; the UNet is frozen", flush=True)
    opt = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=1e-2)
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    timesteps = ctx.scheduler.timesteps.clone().cpu()
    pos = [i for i in range(len(train)) if y_train[i] == 1]
    neg = [i for i in range(len(train)) if y_train[i] == 0]

    log, started = [], time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.head.eval()
            steps_to_eval = range(EARLY_STEPS) if step == args.steps else range(0, EARLY_STEPS, 4)
            acc = accuracy_per_step(model, z_check, y_check, timesteps, steps_to_eval, ctx,
                                    torch.Generator().manual_seed(7))
            log.append({"train_step": step, "per_step": acc})
            (out / "accuracy.json").write_text(json.dumps(log, indent=2))
            mean = sum(r["balanced_accuracy"] for r in acc) / len(acc)
            print(f"step {step:5d}  early-step balanced accuracy {mean:.3f}  ({time.time() - started:.0f} s)",
                  flush=True)
            torch.save({"head": model.head.state_dict(), "step": step, "split": split},
                       out / f"head_step_{step:06d}.pt")
            model.head.train()
        if step == args.steps:
            break
        idx = random.choices(pos, k=4) + random.choices(neg, k=4)
        z0 = z_train[idx].to(ctx.device)
        y = y_train[idx].to(ctx.device)
        t = timesteps[torch.randint(0, len(timesteps), (len(idx),))].to(ctx.device)
        noisy = ctx.scheduler.add_noise(z0, torch.randn_like(z0), t)
        with torch.no_grad():
            feat = model.features(noisy.to(ctx.dtype), t)
        loss = F.binary_cross_entropy_with_logits(model.head_logit(feat.float()), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - started:.0f} s)", flush=True)

    final = log[-1]["per_step"]
    passing = [r["sampler_step"] for r in final if r["balanced_accuracy"] >= MIN_BALANCED_ACCURACY]
    gate = {"rule": f"at least {MIN_PASSING_EARLY_STEPS} of the first {EARLY_STEPS} sampler steps with "
                    f"balanced accuracy at least {MIN_BALANCED_ACCURACY}, {CHECK_DRAWS} noise draws per tile",
            "check_is_held_out": split["check_is_held_out"], "per_step": final,
            "passing_steps": passing, "passed": len(passing) >= MIN_PASSING_EARLY_STEPS}
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    word = "held-out" if split["check_is_held_out"] else "fit only, not held out"
    print(f"GATE ({word}) {'PASSED' if gate['passed'] else 'FAILED'}: {len(passing)} of {EARLY_STEPS}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
