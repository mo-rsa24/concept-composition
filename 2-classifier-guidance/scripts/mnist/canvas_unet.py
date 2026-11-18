"""A class-conditional pixel U-Net on 32 x 64 MNIST canvases, its trainer, and the DDIM product sampler.

Route step 7b, plan 24. One U-Net per two-digit share (0, 0.01, 0.1). Labels are the digits 0 to 9
plus a null label (10) taken on 10% of examples. A two-digit canvas enters training once per digit
it holds, so the expert for 7 means "a canvas with a 7 on it, sometimes beside another digit".
Canvases are built fresh each batch by `canvas_sheet.Canvases`; the five held-out pairs never
appear together on a training canvas.

The sampler is DDIM, 50 steps, eta 0, as on SDXL. The product of two digit experts a and b is
eps_null + g * ((eps_a - eps_null) + (eps_b - eps_null)); at g = 1 this is the score of
p(x | a) p(x | b) / p(x), Bradley et al.'s Bayes composition. A push function, when given, is added
per sample and capped per sample at `cap` times the product prediction's norm, the per-sample form
of `poe_repair.methods._classifier_guided.capped_push`.

Usage:
    python canvas_unet.py train --mnist <raw dir> --share 0.01 --out <dir> [--steps 40000]
    python canvas_unet.py train --mnist <raw dir> --share 0.1 --out <dir> --time 200
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_sheet import HELD_OUT, PAIRS, Canvases, H, W, load_mnist  # noqa: E402

NULL = 10
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
ABAR = torch.cumprod(1 - BETAS, 0).float()
SEEN = [p for p in PAIRS if p not in HELD_OUT]


# ---------------------------------------------------------------- the network

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, dropout=0.1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.e(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n, self.a = nn.GroupNorm(32, c), nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.n(x).flatten(2).transpose(1, 2)
        return x + self.a(s, s, s, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)


class CanvasUNet(nn.Module):
    """32 x 64 -> 16 x 32 -> 8 x 16 and back, two res blocks per level, attention at 8 x 16."""

    def __init__(self, ch=64, mults=(1, 2, 2), n_classes=11):
        super().__init__()
        emb = ch * 4
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cemb = nn.Embedding(n_classes, emb)
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.down, skips, c = nn.ModuleList(), [ch], ch
        for i, m in enumerate(mults):
            for _ in range(2):
                self.down.append(ResBlock(c, ch * m, emb))
                c = ch * m
                skips.append(c)
            if i < len(mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb), Attn(c), ResBlock(c, c, emb)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(3):
                self.up.append(ResBlock(c + skips.pop(), ch * m, emb))
                c = ch * m
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, y, features=False):
        """Noise prediction; with features=True also the decoder's output at each resolution."""
        emb = self.temb(timestep_embedding(t, self.ch)) + self.cemb(y)
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        feats = []
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], 1), emb)
            elif isinstance(m, nn.Upsample):
                feats.append(h)
                h = m(h)
            else:
                h = m(h)
        feats.append(h)
        eps = self.out(h)
        return (eps, feats) if features else eps


# ---------------------------------------------------------------- the data

def training_batch(cv, share, n, rng):
    """n (canvas, label) examples; a two-digit canvas gives one example per digit it holds."""
    xs, ys, two = [], [], 0
    while len(xs) < n:
        if rng.random() < share:
            a, b = SEEN[rng.integers(len(SEEN))]
            c = cv.two(a, b)
            xs += [c, c]
            ys += [a, b]
            two += 1
        else:
            d = int(rng.integers(10))
            xs.append(cv.one(d))
            ys.append(d)
    x = torch.from_numpy(np.stack(xs[:n])).float().div_(127.5).sub_(1).unsqueeze(1)
    y = torch.tensor(ys[:n])
    y[torch.from_numpy(rng.random(n) < 0.1)] = NULL
    return x, y, two


# ---------------------------------------------------------------- the sampler

def ddim_timesteps(n=50):
    return torch.linspace(T - 1, 0, n).round().long()


@torch.no_grad()
def product_eps(model, x, t, labels, g=1.0):
    """eps_null + g * sum over labels of (eps_label - eps_null), one forward over the stacked batch."""
    b = x.shape[0]
    ys = [torch.full((b,), NULL, device=x.device)] + [torch.full((b,), l, device=x.device) for l in labels]
    out = model(x.repeat(len(ys), 1, 1, 1), t.repeat(len(ys)), torch.cat(ys)).chunk(len(ys))
    return out[0] + g * sum(e - out[0] for e in out[1:])


def per_sample_cap(push, eps, cap):
    """Scale each sample's push to at most cap times that sample's eps norm; report which fired."""
    if cap is None:
        return push, torch.zeros(push.shape[0], dtype=torch.bool, device=push.device)
    pn = push.flatten(1).norm(dim=1)
    lim = cap * eps.flatten(1).norm(dim=1)
    fired = pn > lim
    scale = torch.where(fired, lim / pn.clamp_min(1e-12), torch.ones_like(pn))
    return push * scale[:, None, None, None], fired


def start_noise(seeds, device):
    """One fixed 1 x 32 x 64 starting noise per seed, the same in every column."""
    return torch.stack([torch.randn(1, H, W, generator=torch.Generator().manual_seed(int(s)))
                        for s in seeds]).to(device)


def sample(model, seeds, labels, g=1.0, steps=50, push_fn=None, cap=0.1, device="cuda"):
    """DDIM eta 0. push_fn(x, t, eps) -> push of eps's shape, added to eps after the cap.

    Returns the finished canvases in [-1, 1] and, per step, the fraction of samples where the cap fired.
    """
    x = start_noise(seeds, device)
    ts = ddim_timesteps(steps).to(device)
    abar = ABAR.to(device)
    fired_log = []
    for i, t in enumerate(ts):
        tt = t.repeat(x.shape[0])
        eps = product_eps(model, x, tt, labels, g)
        if push_fn is not None:
            push, fired = per_sample_cap(push_fn(x, tt, eps), eps, cap)
            eps = eps + push
            fired_log.append(fired.float().mean().item())
        a = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x, fired_log


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()


# ---------------------------------------------------------------- training

def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    images, labels = load_mnist(args.mnist)
    rng = np.random.default_rng(args.seed)
    cv = Canvases(images, labels, rng)
    torch.manual_seed(args.seed)
    model = CanvasUNet(args.ch).to(dev)
    ema = CanvasUNet(args.ch).to(dev).eval()
    ema.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    out = args.out / f"share-{args.share}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"share {args.share}: {n_params:,} weights, ch {args.ch}, batch {args.batch}, held out {HELD_OUT}",
          flush=True)

    total = args.time if args.time else args.steps
    warm = 20 if args.time else 0
    t_data = t_step = 0.0
    log, two_total = [], 0
    for step in range(1, total + warm + 1):
        t0 = time.perf_counter()
        x, y, two = training_batch(cv, args.share, args.batch, rng)
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        two_total += two
        t1 = time.perf_counter()
        t = torch.randint(0, T, (x.shape[0],), device=dev)
        noise = torch.randn_like(x)
        a = ABAR.to(dev)[t][:, None, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            pred = model(a.sqrt() * x + (1 - a).sqrt() * noise, t, y)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step > warm:
            t_data += t1 - t0
            t_step += t2 - t1
        if not args.time and (step % 500 == 0 or step == total):
            log.append({"step": step, "loss": loss.item(), "two_digit_canvases": two_total})
            print(json.dumps(log[-1]), flush=True)
        if not args.time and (step % args.save_every == 0 or step == total):
            torch.save({"ema": ema.state_dict(), "step": step, "args": vars(args) | {"mnist": str(args.mnist),
                        "out": str(args.out)}, "held_out": HELD_OUT}, out / "unet.pt")

    if args.time:
        per = (t_data + t_step) / args.time
        res = {"share": args.share, "device": torch.cuda.get_device_name(), "ch": args.ch, "batch": args.batch,
               "amp": args.amp,
               "weights": n_params, "timed_steps": args.time, "ms_per_step": round(1000 * per, 1),
               "ms_data": round(1000 * t_data / args.time, 1), "ms_gpu": round(1000 * t_step / args.time, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "projected_min_for": args.steps, "projected_min": round(per * args.steps / 60, 1)}
        print(json.dumps(res), flush=True)
        tag = torch.cuda.get_device_name().replace(" ", "_") + ("-amp" if args.amp else "")
        (out / f"timing-{tag}.json").write_text(json.dumps(res, indent=1))
        return
    (out / "train_log.json").write_text(json.dumps(log, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--mnist", type=Path, required=True)
    tr.add_argument("--share", type=float, required=True)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--steps", type=int, default=40000)
    tr.add_argument("--batch", type=int, default=128)
    tr.add_argument("--ch", type=int, default=64)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--ema", type=float, default=0.999)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--save-every", type=int, default=5000)
    tr.add_argument("--time", type=int, default=0, help="time this many steps after 20 warm-up steps, then stop")
    tr.add_argument("--amp", action="store_true", help="forward pass in bfloat16 autocast; weights and loss stay fp32")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)


if __name__ == "__main__":
    main()
