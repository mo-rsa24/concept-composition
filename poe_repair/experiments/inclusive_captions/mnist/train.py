"""Train the MNIST stage's models, one per (dial, training seed), on fresh canvases every batch.

  --smoke          200 steps at each channel width in --widths, prints the step time and the time
                   15 trainings would take; writes nothing but the printout
  (default)        the 15 trainings: dial 0, 0.1, 0.3, 0.6, 1 x training seeds 1, 2, 3, the same step
                   count for all; each saves data/mnist/checkpoints/dial-<d>-seed-<s>.pt (the EMA
                   weights) and appends one line to data/mnist/checkpoints/trainings.jsonl

The training seed sets the starting weights and the canvas draws, so seed 1 starts from the same
weights at every dial. Sampling never touches this generator.

Usage (from the repo root):
    python -I poe_repair/experiments/inclusive_captions/mnist/train.py --smoke --widths 48 64
    nohup python -I poe_repair/experiments/inclusive_captions/mnist/train.py --width 64 > log 2>&1 &
"""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvases import DIALS, CanvasMaker, ShareCanvasMaker  # noqa: E402
from model import T, UNet, make_abar  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "data" / "mnist" / "checkpoints"
STEPS, BATCH, LR, EMA = 30_000, 128, 2e-4, 0.999
SEEDS = (1, 2, 3)


def train_one(dial, seed, width, steps, dev="cuda", log_every=5000, share=None):
    torch.manual_seed(seed)
    net = UNet(width).to(dev)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0, fused=True)
    fwd = torch.compile(net, mode="max-autotune-no-cudagraphs")
    ema_p, net_p = list(ema.parameters()), list(net.parameters())
    cm = CanvasMaker(dial, dev, seed=seed) if share is None else ShareCanvasMaker(share, dev, seed=seed)
    abar = make_abar(dev)
    g = torch.Generator(device=dev).manual_seed(10_000 + seed)
    t0, losses = time.time(), []
    for step in range(1, steps + 1):
        x0, lab, _ = cm.batch(BATCH)
        k = torch.randint(1, T + 1, (BATCH,), device=dev, generator=g)
        a = abar[k][:, None, None, None]
        eps = torch.randn(x0.shape, device=dev, generator=g)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = fwd(xt, k, lab)
        loss = ((pred.float() - eps) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, net_p, 1 - EMA)
        losses.append(loss.item()) if step % 50 == 0 else None
        if step % log_every == 0:
            print(f"  dial {dial:g} seed {seed} step {step}/{steps} loss {sum(losses[-20:]) / len(losses[-20:]):.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    return ema, time.time() - t0, (sum(losses[-20:]) / max(1, len(losses[-20:])))


def smoke(widths, steps=200):
    n_params = {}
    for w in widths:
        torch.cuda.reset_peak_memory_stats()
        train_one(0.3, 1, w, 20, log_every=10**9)                       # warm-up
        _, secs, loss = train_one(0.3, 1, w, steps, log_every=10**9)
        n_params[w] = sum(p.numel() for p in UNet(w).parameters())
        per = secs / steps
        print(f"width {w}: {n_params[w] / 1e6:.2f} M parameters · {per * 1000:.1f} ms a step at batch {BATCH} · "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB · loss after {steps} steps {loss:.4f} · "
              f"one training of {STEPS:,} steps {per * STEPS / 60:.1f} min · 15 trainings "
              f"{per * STEPS * 15 / 3600:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--widths", type=int, nargs="+", default=[48, 64])
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--only", type=float, nargs=2, metavar=("DIAL", "SEED"),
                    help="train one model only, e.g. --only 0.3 1")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--share", type=float, help="plan 03: train one share model, with --seed; saves share-<s>-seed-<t>.pt")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.smoke:
        smoke(a.widths)
        return
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2**30
    assert free > 5, f"only {free:.1f} GiB free on the filesystem holding {out}"
    if a.share is not None:
        path = out / f"share-{a.share:g}-seed-{a.seed}.pt"
        if path.exists():
            print(f"skip {path.name}: already saved", flush=True)
            return
        ema, secs, loss = train_one(0.0, a.seed, a.width, a.steps, share=a.share)
        torch.save({"state": ema.state_dict(), "width": a.width, "share": a.share, "seed": a.seed,
                    "steps": a.steps, "batch": BATCH}, path)
        row = {"share": a.share, "seed": a.seed, "width": a.width, "steps": a.steps, "seconds": round(secs),
               "final_loss": round(loss, 5), "path": str(path.resolve())}
        with open(out / "trainings.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"saved {path.name} in {secs / 60:.1f} min, loss {loss:.4f}", flush=True)
        return
    jobs = [(a.only[0], int(a.only[1]))] if a.only else [(d, s) for d in DIALS for s in SEEDS]
    for dial, seed in jobs:
            path = out / f"dial-{dial:g}-seed-{seed}.pt"
            if path.exists():
                print(f"skip {path.name}: already saved", flush=True)
                continue
            ema, secs, loss = train_one(dial, seed, a.width, a.steps)
            torch.save({"state": ema.state_dict(), "width": a.width, "dial": dial, "seed": seed,
                        "steps": a.steps, "batch": BATCH}, path)
            row = {"dial": dial, "seed": seed, "width": a.width, "steps": a.steps, "seconds": round(secs),
                   "final_loss": round(loss, 5), "path": str(path.resolve().relative_to(REPO))}
            with open(out / "trainings.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"saved {path.name} in {secs / 60:.1f} min, loss {loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
