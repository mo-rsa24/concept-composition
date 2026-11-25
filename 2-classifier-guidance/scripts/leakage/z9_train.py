#!/usr/bin/env python3
"""Loop 12 step Z9: the digit-and-colour judge and one finished expert per data holder.

Design: artifacts/notes/progress/leakage/09-dcfm-colored-mnist/design.md.

  judge   a small CNN with a digit head and a colour head, trained on real coloured MNIST training digits over all 100
          combinations (never on a generated picture); reports accuracy on real test digits of the novel combinations
  expert  Paper 1's digit U-Net blocks (canvas_unet.py) at 3 x 32 x 32, conditioned on a digit and a colour through two
          added embeddings; each label is set to none (index 10) on 10% of examples, independently. --time N times N steps
          and stops; otherwise trains --steps and saves the EMA weights.

  python z9_train.py judge  --mnist <raw> --split <json> --out <dir>
  python z9_train.py expert --mnist <raw> --split <json> --holder 0 --out <dir> [--steps 40000 | --time 200]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mnist"))
from canvas_unet import ABAR, T, Attn, ResBlock, timestep_embedding  # noqa: E402
from z9_data import Stream, colourise, load_mnist, reference  # noqa: E402

NONE = 10


class ColourUNet(nn.Module):
    """32 -> 16 -> 8 and back, two res blocks per level, attention at 8 x 8; labels: digit and colour, 10 = none."""

    def __init__(self, ch=64, mults=(1, 2, 2)):
        super().__init__()
        emb = ch * 4
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.demb, self.cemb = nn.Embedding(11, emb), nn.Embedding(11, emb)
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.down, skips, c = nn.ModuleList(), [ch], ch
        for i, m in enumerate(mults):
            for _ in range(2):
                self.down.append(ResBlock(c, ch * m, emb)); c = ch * m; skips.append(c)
            if i < len(mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb), Attn(c), ResBlock(c, c, emb)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(3):
                self.up.append(ResBlock(c + skips.pop(), ch * m, emb)); c = ch * m
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest")); self.up.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x, t, d, c, features=False):
        emb = self.temb(timestep_embedding(t, self.ch)) + self.demb(d) + self.cemb(c)
        h = self.inp(x); hs = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h); hs.append(h)
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        feats = []
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], 1), emb)
            elif isinstance(m, nn.Upsample):
                feats.append(h); h = m(h)
            else:
                h = m(h)
        feats.append(h)
        eps = self.out(h)
        return (eps, feats) if features else eps


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2))
        self.digit, self.colour = nn.Linear(128, 10), nn.Linear(128, 10)

    def forward(self, x):
        h = self.body(x)
        return self.digit(h), self.colour(h)


def train_judge(args, split):
    dev = "cuda"
    images, labels = load_mnist(args.mnist, "train")
    timg, tlab = load_mnist(args.mnist, "test")
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    judge = Judge().to(dev)
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(images), 256)
        col = rng.integers(0, 10, 256)
        x = torch.from_numpy(colourise(images[idx], col)).to(dev)
        x = x + 0.05 * torch.randn_like(x)  # a little noise so finished samples' grain is not out of range
        ld, lc = judge(x)
        loss = F.cross_entropy(ld, torch.from_numpy(labels[idx].astype(np.int64)).to(dev)) + \
            F.cross_entropy(lc, torch.from_numpy(col.astype(np.int64)).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4)}), flush=True)
    judge.eval()
    res = {}
    for name in ("novel", "seen"):
        x, d, c = reference(timg, tlab, [tuple(k) for k in split[name]], 100, seed=7)
        right = []
        with torch.no_grad():
            for i in range(0, len(x), 1000):
                ld, lc = judge(torch.from_numpy(x[i:i + 1000]).to(dev))
                right.append(((ld.argmax(1).cpu().numpy() == d[i:i + 1000]) & (lc.argmax(1).cpu().numpy() == c[i:i + 1000])))
        right = np.concatenate(right)
        res[f"{name}_test_accuracy"] = float(right.mean())
        res[f"{name}_per_combination"] = {f"{dd}-{cc}": float(right[j * 100:(j + 1) * 100].mean()) for j, (dd, cc) in enumerate(split[name])}
    res["passes_95"] = res["novel_test_accuracy"] >= 0.95
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"judge": judge.state_dict(), "steps": args.steps}, args.out / "judge.pt")
    (args.out / "judge.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("combination")}), flush=True)


def train_expert(args, split):
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    dev = "cuda"
    images, labels = load_mnist(args.mnist, "train")
    stream = Stream(images, labels, split["holders"][args.holder], seed=100 + args.holder)
    torch.manual_seed(100 + args.holder)
    model = ColourUNet(args.ch).to(dev)
    ema = ColourUNet(args.ch).to(dev).eval(); ema.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(200 + args.holder)
    total = args.time or args.steps
    out = args.out / f"holder-{args.holder}"; out.mkdir(parents=True, exist_ok=True)
    log, t_sum, warm = [], 0.0, 20 if args.time else 0
    for step in range(1, total + warm + 1):
        t0 = time.perf_counter()
        x, d, c = stream.batch(args.batch)
        d = np.where(rng.random(args.batch) < 0.1, NONE, d); c = np.where(rng.random(args.batch) < 0.1, NONE, c)
        x = torch.from_numpy(x).to(dev)
        d, c = torch.from_numpy(d.astype(np.int64)).to(dev), torch.from_numpy(c.astype(np.int64)).to(dev)
        t = torch.randint(0, T, (x.shape[0],), device=dev)
        noise = torch.randn_like(x)
        a = ABAR.to(dev)[t][:, None, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(a.sqrt() * x + (1 - a).sqrt() * noise, t, d, c)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        if args.time:
            torch.cuda.synchronize()
            if step > warm:
                t_sum += time.perf_counter() - t0
            continue
        if step % 1000 == 0 or step == total:
            log.append({"step": step, "loss": round(loss.item(), 5)}); print(json.dumps(log[-1]), flush=True)
        if step % 10_000 == 0 or step == total:
            torch.save({"ema": ema.state_dict(), "step": step, "holder": args.holder, "combos": split["holders"][args.holder],
                        "ch": args.ch, "batch": args.batch}, out / "unet.pt")
    if args.time:
        per = t_sum / args.time
        res = {"device": torch.cuda.get_device_name(), "ms_per_step": round(1000 * per, 2), "batch": args.batch,
               "projected_minutes_for_40000": round(per * 40000 / 60, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        (out / "timing.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)
        return
    (out / "train_log.json").write_text(json.dumps(log, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("what", choices=["judge", "expert"])
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holder", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    if args.what == "judge":
        args.steps = args.steps if args.steps != 40_000 else 4000
        train_judge(args, split)
    else:
        train_expert(args, split)


if __name__ == "__main__":
    main()
