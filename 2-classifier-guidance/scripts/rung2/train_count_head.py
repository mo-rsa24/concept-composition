#!/usr/bin/env python
"""A Readout-Guidance-style count head: from SDXL's features of a half-drawn picture, a map with one blob per body.

Readout Guidance (Luo et al., CVPR 2024, arXiv 2312.02150) trains small heads on a frozen diffusion
model's features from several decoder layers. Here the head reads SDXL's middle block and its first
two upsampling blocks (32, 64 and 128 px feature grids), pools each to 32 x 32, and predicts a
32 x 32 density map. The target comes from GroundingDINO's boxes on the finished drawing: one
Gaussian blob per box centre, each summing to 1, so the map sums to the number of bodies.

Training examples are plan 23's saved states (every 5th step of every joint and product drawing,
plus the final latent), each paired with its own drawing's finished-picture target. The UNet is
frozen and read on the empty prompt.

The check, written before training: on plan 22's held-out hand-called tiles (7 whole pairs plus
cat x dog seeds 7 and 8), each noised to the first 20 sampler steps 4 times, a tile is called plural
when the map sums to 1.5 or more. Main bar: good pairs against singles and hybrids, balanced accuracy
at least 0.75 on at least 10 of 20 steps. Also reported: good pairs against every negative (the plan
22 bar), which wrong pairs such as two dogs are expected to fail, since they are plural.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/count_head/<run>/:
    head_step_NNNNNN.pt, check.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.methods._sampling import add_time_ids  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
HERE = Path(__file__).resolve().parent
GRID = 32
PLURAL_AT = 1.5
BAR, MIN_STEPS, EARLY = 0.75, 10, 20


class Readout(nn.Module):
    """SDXL's mid block and up blocks 0 and 1, read through forward hooks on a frozen UNet."""

    def __init__(self, unet, seq_e, pool_e):
        super().__init__()
        self.unet, self.feats = unet, {}
        self.register_buffer("seq_e", seq_e, persistent=False)
        self.register_buffer("pool_e", pool_e, persistent=False)
        for name, mod in (("mid", unet.mid_block), ("up0", unet.up_blocks[0]), ("up1", unet.up_blocks[1])):
            mod.register_forward_hook(lambda m, i, o, n=name: self.feats.__setitem__(n, o if torch.is_tensor(o) else o[0]))

    def forward(self, x_t, t, seq=None, pool=None):
        """seq, pool: per-picture prompt embeddings (B, 77, 2048) and (B, 1280); the stored prompt when None."""
        b = x_t.shape[0]
        seq = self.seq_e.expand(b, -1, -1) if seq is None else seq
        pool = self.pool_e.expand(b, -1) if pool is None else pool
        cond = {"text_embeds": pool,
                "time_ids": add_time_ids(height=1024, width=1024, batch_size=b, device=x_t.device, dtype=x_t.dtype)}
        self.unet(x_t, t, encoder_hidden_states=seq, added_cond_kwargs=cond)
        return [self.feats[k] for k in ("mid", "up0", "up1")]


class CountHead(nn.Module):
    def __init__(self, channels=(1280, 1280, 640), width=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, width, 1), nn.GroupNorm(16, width), nn.SiLU())
                                   for c in channels])
        self.out = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 1, 1))

    def forward(self, feats):
        h = sum(p(F.adaptive_avg_pool2d(f.float(), GRID)) for p, f in zip(self.proj, feats))
        return F.softplus(self.out(h))[:, 0]                     # (B, 32, 32) density, sums to the count


def target_map(boxes, size=1024):
    yy, xx = torch.meshgrid(torch.arange(GRID).float() + 0.5, torch.arange(GRID).float() + 0.5, indexing="ij")
    m = torch.zeros(GRID, GRID)
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2 / size * GRID, (y0 + y1) / 2 / size * GRID
        sig = max(1.0, 0.15 * max(x1 - x0, y1 - y0) / size * GRID)
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
        m += g / g.sum()
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="count-v1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    out = RUNG2 / "count_head" / args.run
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads((RUNG2 / "detector/labels.json").read_text())
    kept = json.loads((RUNG2 / "kim/timesteps.json").read_text())["timesteps"]
    stems = sorted(labels)
    targets = {s: target_map(labels[s]["boxes"]) for s in stems}
    print(f"{len(stems)} drawings; counts {dict(sorted(__import__('collections').Counter(labels[s]['count'] for s in stems).items()))}",
          flush=True)

    ctx = make_ctx(output_root=out / "raw")
    for n in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[n].requires_grad_(False)
        ctx.models[n].eval()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    reader = Readout(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
    head = CountHead().to(ctx.device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_t = ctx.scheduler.timesteps.clone().cpu()

    tiles = [t for t in json.loads((HERE / "training_set_v3.json").read_text())["tiles"] if t["check"]]
    z_check = torch.cat([torch.load(paths.MOUNT_ROOT / t["latent"] if "/" in t["latent"] else
                                    RUNG2 / "training_set/latents" / f"{t['latent']}.pt").float() for t in tiles])
    kind = [t["kind"] if t["label"] == 0 else "positive" for t in tiles]

    @torch.no_grad()
    def check(steps):
        rows, gen = [], torch.Generator().manual_seed(7)
        for s in steps:
            t_val, counts = sampler_t[s], [[] for _ in tiles]
            for _ in range(4):
                for i in range(0, len(tiles), 8):
                    z0 = z_check[i:i + 8].to(ctx.device)
                    t = t_val.repeat(z0.shape[0]).to(ctx.device)
                    noisy = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
                    c = head(reader(noisy.to(ctx.dtype), t)).sum((1, 2)).cpu()
                    for j, v in enumerate(c.tolist()):
                        counts[i + j].append(v)
            mean = [sum(c) / len(c) for c in counts]
            def bal(neg_kinds):
                pos = [m >= PLURAL_AT for m, k in zip(mean, kind) if k == "positive"]
                neg = [m < PLURAL_AT for m, k in zip(mean, kind) if k in neg_kinds]
                return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2 if pos and neg else float("nan")
            by = {k: round(sum(m for m, kk in zip(mean, kind) if kk == k) / max(1, kind.count(k)), 2) for k in set(kind)}
            rows.append({"sampler_step": int(s), "main_balanced_accuracy": bal({"single", "hybrid"}),
                         "all_negatives_balanced_accuracy": bal(set(kind) - {"positive"}), "mean_count_by_kind": by})
        return rows

    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            head.eval()
            rows = check(range(EARLY) if step == args.steps else range(0, EARLY, 4))
            mean = sum(r["main_balanced_accuracy"] for r in rows) / len(rows)
            print(f"step {step:5d}  main check {mean:.3f}  counts by kind at step {rows[-1]['sampler_step']}: "
                  f"{rows[-1]['mean_count_by_kind']}  ({time.time() - started:.0f} s)", flush=True)
            torch.save({"head": head.state_dict(), "step": step}, out / f"head_step_{step:06d}.pt")
            head.train()
            if step == args.steps:
                passing = [r["sampler_step"] for r in rows if r["main_balanced_accuracy"] >= BAR]
                verdict = {"rule": f"good pairs against singles and hybrids, balanced accuracy >= {BAR} on >= {MIN_STEPS} "
                                   f"of the first {EARLY} steps; plural when the map sums to >= {PLURAL_AT}",
                           "per_step": rows, "passing_steps": passing, "passed": len(passing) >= MIN_STEPS}
                (out / "check.json").write_text(json.dumps(verdict, indent=1))
                print(f"COUNT CHECK {'PASSED' if verdict['passed'] else 'FAILED'}: {len(passing)} of {EARLY}", flush=True)
                break
        batch = random.choices(stems, k=8)
        ks = [random.randrange(len(kept)) for _ in batch]
        x = torch.stack([torch.load(RUNG2 / "kim/states" / f"{s}.pt")[k] for s, k in zip(batch, ks)]).to(ctx.device, ctx.dtype)
        t = torch.tensor([max(kept[k], 1) for k in ks], device=ctx.device)
        y = torch.stack([targets[s] for s in batch]).to(ctx.device)
        with torch.no_grad():
            feats = reader(x, t)
        pred = head(feats)
        loss = 1e3 * F.mse_loss(pred, y) + 0.1 * F.mse_loss(pred.sum((1, 2)), y.sum((1, 2)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - started:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
