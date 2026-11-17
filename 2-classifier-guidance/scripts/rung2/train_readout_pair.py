#!/usr/bin/env python
"""Plan 23's discriminator, Readout-Guidance style: a clean joint-prompt pair against the product's drawing.

Kim et al.'s set-up with the read map's caveat C4 met: the real set is joint-prompt drawings the
Claude API called a faithful pair (exactly one of each named thing, no hybrid, an intact photo),
and the fake set is the product's drawings from the same pairs and noise. Joint drawings the API
rejected are left out. Examples are the drawings' own saved states (every 5th step and the final
latent), so the head learns at the noise levels where it will push.

The head follows Readout Guidance (Luo et al., CVPR 2024): it reads SDXL's frozen middle block and
first two upsampling blocks (``train_count_head.Readout``), projects each to 128 channels at 32 x 32,
sums them, and gives one logit after two convolutions and an attention pool.

The check, written before training, is plan 22's and plan 23's: on the held-out hand-called tiles
(7 whole pairs plus cat x dog seeds 7 and 8), each noised to the first 20 sampler steps 4 times,
balanced accuracy at least 0.75 on at least 10 of the 20 steps, good pairs against every negative.
Accuracy on singles and hybrids alone is reported beside it.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/readout_pair/<run>/:
    head_step_NNNNNN.pt, check.json, split.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poe_repair import paths  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from train_count_head import GRID, Readout  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
HERE = Path(__file__).resolve().parent
BAR, MIN_STEPS, EARLY = 0.75, 10, 20


class PairHead(nn.Module):
    def __init__(self, channels=(1280, 1280, 640), width=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, width, 1), nn.GroupNorm(16, width), nn.SiLU())
                                   for c in channels])
        self.mix = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())   # 32 -> 8
        self.pos = nn.Parameter(torch.randn(1, 65, width) / width ** 0.5)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.out = nn.Linear(width, 1)

    def forward(self, feats):
        h = sum(p(F.adaptive_avg_pool2d(f.float(), GRID)) for p, f in zip(self.proj, feats))
        tok = self.mix(h).flatten(2).transpose(1, 2)
        x = torch.cat([tok.mean(1, keepdim=True), tok], 1) + self.pos
        pooled, _ = self.attn(x[:, :1], x, x)
        return self.out(pooled[:, 0])[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="readout-pair-v1")
    ap.add_argument("--labels", type=Path, required=True, help="the API labels JSONL for plan 23's drawings")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    out = RUNG2 / "readout_pair" / args.run
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()]
    real = [r["id"] for r in rows if r.get("method") == "joint" and r.get("verdict") == "faithful pair"]
    fake = [r["id"] for r in rows if r.get("method") == "product" and "verdict" in r]
    real = [s for s in real if (RUNG2 / "kim/states" / f"{s}.pt").exists()]
    fake = [s for s in fake if (RUNG2 / "kim/states" / f"{s}.pt").exists()]
    split = {"real_clean_joint": len(real), "fake_product": len(fake),
             "joint_rejected_by_api": sum(1 for r in rows if r.get("method") == "joint" and r.get("verdict") != "faithful pair"),
             "pairs_with_a_clean_joint": len({s.split("__seed")[0] for s in real})}
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(json.dumps(split), flush=True)

    kept = json.loads((RUNG2 / "kim/timesteps.json").read_text())["timesteps"]
    ctx = make_ctx(output_root=out / "raw")
    for n in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[n].requires_grad_(False)
        ctx.models[n].eval()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    reader = Readout(ctx.models["unet"], seq_e, pool_e).to(ctx.device)
    head = PairHead().to(ctx.device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    sampler_t = ctx.scheduler.timesteps.clone().cpu()

    tiles = [t for t in json.loads((HERE / "training_set_v3.json").read_text())["tiles"] if t["check"]]
    z_check = torch.cat([torch.load(paths.MOUNT_ROOT / t["latent"] if "/" in t["latent"] else
                                    RUNG2 / "training_set/latents" / f"{t['latent']}.pt").float() for t in tiles])
    y_check = [t["label"] for t in tiles]
    kind = [t["kind"] if t["label"] == 0 else "positive" for t in tiles]

    @torch.no_grad()
    def check(steps):
        res, gen = [], torch.Generator().manual_seed(7)
        for s in steps:
            t_val, logits = sampler_t[s], [[] for _ in tiles]
            for _ in range(4):
                for i in range(0, len(tiles), 8):
                    z0 = z_check[i:i + 8].to(ctx.device)
                    t = t_val.repeat(z0.shape[0]).to(ctx.device)
                    noisy = ctx.scheduler.add_noise(z0, torch.randn(z0.shape, generator=gen).to(ctx.device), t)
                    for j, v in enumerate(head(reader(noisy.to(ctx.dtype), t)).cpu().tolist()):
                        logits[i + j].append(v)
            yes = [sum(l) / len(l) > 0 for l in logits]

            def bal(neg_kinds):
                pos = [p for p, k in zip(yes, kind) if k == "positive"]
                neg = [not p for p, k in zip(yes, kind) if k in neg_kinds]
                return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2 if pos and neg else float("nan")
            res.append({"sampler_step": int(s), "balanced_accuracy": bal(set(kind) - {"positive"}),
                        "singles_and_hybrids_balanced_accuracy": bal({"single", "hybrid"})})
        return res

    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            head.eval()
            rows_c = check(range(EARLY) if step == args.steps else range(0, EARLY, 4))
            mean = sum(r["balanced_accuracy"] for r in rows_c) / len(rows_c)
            print(f"step {step:5d}  held-out balanced accuracy {mean:.3f}  ({time.time() - started:.0f} s)", flush=True)
            torch.save({"head": head.state_dict(), "step": step, "split": split}, out / f"head_step_{step:06d}.pt")
            head.train()
            if step == args.steps:
                passing = [r["sampler_step"] for r in rows_c if r["balanced_accuracy"] >= BAR]
                verdict = {"rule": f"good pairs against every negative, balanced accuracy >= {BAR} on >= {MIN_STEPS} "
                                   f"of the first {EARLY} steps", "per_step": rows_c,
                           "passing_steps": passing, "passed": len(passing) >= MIN_STEPS}
                (out / "check.json").write_text(json.dumps(verdict, indent=1))
                print(f"READOUT CHECK {'PASSED' if verdict['passed'] else 'FAILED'}: {len(passing)} of {EARLY}", flush=True)
                break
        names = random.choices(real, k=4) + random.choices(fake, k=4)
        ks = [random.randrange(len(kept)) for _ in names]
        x = torch.stack([torch.load(RUNG2 / "kim/states" / f"{s}.pt")[k] for s, k in zip(names, ks)]).to(ctx.device, ctx.dtype)
        t = torch.tensor([max(kept[k], 1) for k in ks], device=ctx.device)
        y = torch.tensor([1.0] * 4 + [0.0] * 4, device=ctx.device)
        with torch.no_grad():
            feats = reader(x, t)
        loss = F.binary_cross_entropy_with_logits(head(feats), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - started:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
