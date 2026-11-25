#!/usr/bin/env python3
"""Loop 12 step Z9: sampling and scoring, shared by every arm.

Design: artifacts/notes/progress/leakage/09-dcfm-colored-mnist/design.md.

The composed prediction (arm 1, DCFM's Eq. 5 at weight 1):
    eps = eps_none + (eps_d - eps_none) + (eps_c - eps_none)
eps_none is the mean of both experts with both labels none; eps_d (eps_c) is the mean, over the holders whose pictures include
digit d (colour c), of their prediction with that label and the other none. DDIM, 50 steps, sampler noise 1.0 (Paper 1's kept
setting); seed k starts from the same 3 x 32 x 32 noise in every arm, and the sampler's fresh noise comes from a per-seed stream.
A push, when given, is capped per picture at `cap` times the prediction's length and added (canvas_unet.per_sample_cap).

  plain    arm 1 on the novel and seen combinations, 100 pictures each; the judge's accuracy; control 3 (each expert alone on its
           own combinations, 20 each); FID against real test digits, with the real-against-real floor
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mnist"))
from canvas_unet import ABAR, ddim_timesteps, per_sample_cap  # noqa: E402
from z9_data import load_mnist, reference  # noqa: E402
from z9_train import NONE, ColourUNet, Judge  # noqa: E402

SEEDS = list(range(100))


class Experts:
    def __init__(self, folder, split, dev):
        self.models = []
        for k in (0, 1):
            ck = torch.load(Path(folder) / f"holder-{k}" / "unet.pt", map_location=dev, weights_only=False)
            m = ColourUNet(ck["ch"]).to(dev).eval(); m.load_state_dict(ck["ema"]); m.requires_grad_(False)
            self.models.append(m)
        self.digits = [set(d for d, _ in h) for h in split["holders"]]
        self.colours = [set(c for _, c in h) for h in split["holders"]]
        self.dev = dev

    def eps(self, x, t, d, c):
        """Composed prediction for one (d, c) shared by the whole batch."""
        b = x.shape[0]
        full = lambda v: torch.full((b,), v, device=self.dev, dtype=torch.long)
        none = [m(x, t, full(NONE), full(NONE)) for m in self.models]
        e0 = sum(none) / 2
        ed = [m(x, t, full(d), full(NONE)) for k, m in enumerate(self.models) if d in self.digits[k]]
        ec = [m(x, t, full(NONE), full(c)) for k, m in enumerate(self.models) if c in self.colours[k]]
        return e0 + (sum(ed) / len(ed) - e0) + (sum(ec) / len(ec) - e0)

    def eps_one(self, k, x, t, d, c):
        b = x.shape[0]
        return self.models[k](x, t, torch.full((b,), d, device=self.dev), torch.full((b,), c, device=self.dev))


def start_noise(seeds, dev):
    return torch.stack([torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(int(s))) for s in seeds]).to(dev)


def sample(eps_fn, seeds, dev, push_fn=None, cap=0.1, steps=50, eta=1.0, record=None):
    """DDIM with sampler noise eta; push_fn(x, t, eps) -> push. record(i, x, t) is called before each step."""
    x = start_noise(seeds, dev)
    gens = [torch.Generator().manual_seed(1_000_003 + int(s)) for s in seeds]
    ts = ddim_timesteps(steps).to(dev)
    abar = ABAR.to(dev)
    applied = []
    for i, t in enumerate(ts):
        tt = t.repeat(x.shape[0])
        if record is not None:
            record(i, x, tt)
        with torch.no_grad():
            eps = eps_fn(x, tt)
        if push_fn is not None:
            p, _ = per_sample_cap(push_fn(x, tt, eps), eps, cap)
            applied.append(p.flatten(1).norm(dim=1).cpu())
            eps = eps + p
        a = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        if eta > 0 and i + 1 < len(ts):
            sig = eta * ((1 - a_prev) / (1 - a)).sqrt() * (1 - a / a_prev).sqrt()
            z = torch.stack([torch.randn(x.shape[1:], generator=g) for g in gens]).to(dev)
            x = a_prev.sqrt() * x0 + (1 - a_prev - sig ** 2).clamp_min(0).sqrt() * eps + sig * z
        else:
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x.clamp(-1, 1), (torch.stack(applied, 1) if applied else None)


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()


def load_judge(path, dev):
    j = Judge().to(dev).eval(); j.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["judge"]); return j


@torch.no_grad()
def judge_right(judge, imgs_u8, d, c, dev):
    x = torch.from_numpy(imgs_u8).float().to(dev) / 127.5 - 1
    out = []
    for i in range(0, len(x), 1000):
        ld, lc = judge(x[i:i + 1000])
        out.append(((ld.argmax(1).cpu().numpy() == d[i:i + 1000]) & (lc.argmax(1).cpu().numpy() == c[i:i + 1000])))
    return np.concatenate(out)


def inception(dev):
    from pytorch_fid.inception import InceptionV3
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()


@torch.no_grad()
def fid_stats(net, imgs_u8, dev):
    feats = []
    for i in range(0, len(imgs_u8), 500):
        x = torch.from_numpy(imgs_u8[i:i + 500]).float().to(dev) / 255.0
        feats.append(net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    f = np.concatenate(feats).astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


def fid(a, b):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(a[0], a[1], b[0], b[1]))


def draw_set(experts, combos, dev, push_for=None, cap=0.1, per=100):
    """per pictures of each combination; push_for(d, c) -> push_fn or None. Returns uint8 pictures, digits, colours, push lengths."""
    imgs, dd, cc, lengths = [], [], [], []
    for d, c in combos:
        pf = push_for(d, c) if push_for else None
        x, applied = sample(lambda xx, tt: experts.eps(xx, tt, d, c), SEEDS[:per], dev, push_fn=pf, cap=cap)
        imgs.append(to_uint8(x)); dd += [d] * per; cc += [c] * per
        if applied is not None:
            lengths.append(applied.numpy())
    return np.concatenate(imgs), np.array(dd), np.array(cc), (np.concatenate(lengths) if lengths else None)


def plain(args, split, dev):
    experts = Experts(args.experts, split, dev)
    judge = load_judge(args.judge, dev)
    out = args.out / "plain"; out.mkdir(parents=True, exist_ok=True)
    res = {}
    # control 3: each expert alone on its own combinations
    own = {}
    for k in (0, 1):
        right = []
        for d, c in split["holders"][k]:
            x, _ = sample(lambda xx, tt: experts.eps_one(k, xx, tt, d, c), SEEDS[:20], dev)
            right.append(judge_right(judge, to_uint8(x), np.full(20, d), np.full(20, c), dev))
        own[k] = float(np.concatenate(right).mean())
    res["control_3_expert_alone_right"] = own
    res["control_3_passes"] = all(v >= 0.8 for v in own.values())
    print(json.dumps({"control_3": own}), flush=True)
    timg, tlab = load_mnist(args.mnist, "test")
    net = inception(dev)
    for name in ("novel", "seen"):
        t0 = time.time()
        imgs, d, c, _ = draw_set(experts, [tuple(k) for k in split[name]], dev)
        np.savez_compressed(out / f"{name}.npz", images=imgs, digits=d, colours=c)
        right = judge_right(judge, imgs, d, c, dev)
        np.save(out / f"{name}-right.npy", right)
        real, _, _ = reference(timg, tlab, [tuple(k) for k in split[name]], 200, seed=7)
        blocks = to_uint8(torch.from_numpy(real)).reshape(len(split[name]), 200, 3, 32, 32)
        real_u8, real_other = blocks[:, :100].reshape(-1, 3, 32, 32), blocks[:, 100:].reshape(-1, 3, 32, 32)
        s_real = fid_stats(net, real_u8, dev)
        res[name] = {"accuracy": float(right.mean()), "fid": fid(fid_stats(net, imgs, dev), s_real), "pictures": int(len(imgs)),
                     "per_combination": {f"{a}-{b}": float(right[j * 100:(j + 1) * 100].mean()) for j, (a, b) in enumerate(split[name])},
                     "seconds": round(time.time() - t0, 1)}
        if name == "novel":
            res[name]["fid_real_against_real"] = fid(fid_stats(net, real_other, dev), s_real)  # two disjoint halves of 100 per combination
        print(json.dumps({name: {k: v for k, v in res[name].items() if k != "per_combination"}}), flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def log_wandb(name, config, summary):
    try:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=name, group="loop-12-z9", config=config,
                         settings=wandb.Settings(init_timeout=60))
        run.summary.update(summary); rid = run.id; run.finish(); return rid
    except Exception as e:
        print(f"wandb not logged: {e}", flush=True); return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["plain"])
    ap.add_argument("--mnist", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--experts", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    dev = "cuda"
    res = plain(args, split, dev)
    summ = {"novel_accuracy": res["novel"]["accuracy"], "novel_fid": res["novel"]["fid"], "seen_accuracy": res["seen"]["accuracy"],
            "seen_fid": res["seen"]["fid"], "fid_floor": res["novel"]["fid_real_against_real"]}
    print(json.dumps({"wandb_run": log_wandb(f"loop-12-z9-{args.stage}", {"stage": args.stage}, summ)}), flush=True)


if __name__ == "__main__":
    main()
