"""Step 7b's checker and its separate-label control, on the frozen canvas U-Net (plan 24 task 1.5).

Both read the frozen U-Net's decoder features (8 x 16 and 16 x 32, asked with the null label so the
features do not depend on any pair) through a small head, with the step's log signal-to-noise ratio
as sixteen sine and cosine features, as on the toy.

- The pair-aware checker also reads the asked pair (a and b embedded and summed, so order does not
  matter) and outputs one log-odds that the canvas becomes two clear, right digits. Its push is
  -s * sqrt(1 - abar_t) * grad log(d / (1 - d)), zero where |logit| passes Kim et al.'s clip.
- The separate-label control outputs ten "digit d present" logits and never sees the pair. Its
  push is -s * sqrt(1 - abar_t) * grad [log sigmoid(z_a) + log sigmoid(z_b)], the toy's form.

Data, every draw from seed 1,000 up (seeds 0 to 63 are evaluation only; `draws` refuses lower):
- drawn: plain product of the share-0.01 and share-0.1 models on the 40 seen pairs, seeds 1,000 to
  1,039, each called by the judge; "two clear, right digits" is a positive, anything else a
  negative. Both models feed both labels, so the model a canvas came from is no shortcut.
- pasted, from MNIST's training digits: positives are two digits of a seen pair side by side;
  negatives are the asked pair on one spot, one asked digit alone, and a wrong pair.
- batches are half positive, half negative; within each label half drawn, half pasted where the
  drawn pool allows. The same random shift, stroke thickening or thinning, and blur go on both.
- held-out check: the five held-out pairs, pasted from MNIST's test digits plus their plain-product
  draws; balanced accuracy at sampler steps 0 to 19, averaged, is the sweep's accuracy.

Sweep knobs, one per run: --data-frac (share of every pool kept) and --flip (share of training
labels flipped, fixed when the set is built).

Usage:
    python checker.py draws --models <dir> --judge <pt> --out <draws.npz>
    python checker.py train --kind pair|separate --models <dir> --draws <npz> --mnist <raw> --out <dir>
                            [--data-frac 1.0] [--flip 0.0] [--steps 4000]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_sheet import D, H, W, Canvases, load_mnist  # noqa: E402
from canvas_unet import ABAR, HELD_OUT, NULL, SEEN, CanvasUNet, ddim_timesteps, sample, to_uint8  # noqa: E402

GUIDED_SHARE = 0.01                     # the model the bar is read on (walk map decision b)
DRAW_SEEDS = range(1000, 1040)
LOGIT_CLIP = float(np.log((1 - 1e-5) / 1e-5))
STEPS = ddim_timesteps(50)
_ab = ABAR[STEPS].double()
_prev = torch.cat([_ab[1:], torch.ones(1, dtype=torch.float64)])
_lam = (1 - _ab / _prev).clamp_min(1e-8) / (1 - _ab)     # Kim's beta / (1 - abar), on the 50-step grid
LAMBDA = (_lam / _lam.mean()).float()
STEP_P = torch.tensor([2.0] * 10 + [1.0] * 40)
STEP_P = STEP_P / STEP_P.sum()           # steps 0 to 9 of 50 drawn twice as often


def load_unet(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = CanvasUNet(ck["args"]["ch"]).to(dev).eval()
    net.load_state_dict(ck["ema"])
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def time_features(t):
    a = ABAR.to(t.device)[t]
    lsnr = torch.log(a / (1 - a))[:, None]
    f = torch.exp(torch.linspace(0, 3, 8, device=t.device))[None] * lsnr / 10
    return torch.cat([f.sin(), f.cos()], 1)


class Head(nn.Module):
    def __init__(self, out, pair_aware, film=False, expert=False):
        super().__init__()
        self.pair_aware, self.use_film, self.expert = pair_aware, film, expert
        c = 256 if expert else 128          # expert: features asked with each digit, side by side
        self.film = nn.Linear(64, 2 * 128) if film else None
        self.down = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.body = nn.Sequential(nn.Conv2d(2 * c, 128, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.pair = nn.Embedding(10, 64) if pair_aware else None
        self.mlp = nn.Sequential(nn.Linear(256 + 16 + (64 if pair_aware else 0), 256), nn.SiLU(),
                                 nn.Linear(256, out))

    def forward(self, feats, t, pair=None):
        h = torch.cat([feats[0], self.down(feats[1])], 1)
        pe = self.pair(pair[:, 0]) + self.pair(pair[:, 1]) if self.pair_aware else None
        if self.use_film:
            g, b = self.film(pe).chunk(2, 1)
            h = self.body[1](self.body[0](h))
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
            h = self.body[3](self.body[2](h))
        else:
            h = self.body(h)
        z = [h.mean((2, 3)), h.amax((2, 3)), time_features(t)]
        if self.pair_aware:
            z.append(pe)
        return self.mlp(torch.cat(z, 1))


def features(unet, x, t):
    return unet(x, t, torch.full((x.shape[0],), NULL, device=x.device), features=True)[1]


def features_pair(unet, x, t, pair):
    """The decoder features asked with digit a and with digit b, concatenated per level."""
    f = unet(torch.cat([x, x]), torch.cat([t, t]), torch.cat([pair[:, 0], pair[:, 1]]), features=True)[1]
    return [torch.cat(level.chunk(2), 1) for level in f]


def head_feats(unet, head, x, t, pair):
    return features_pair(unet, x, t, pair) if getattr(head, "expert", False) else features(unet, x, t)


# ---------------------------------------------------------------- the pushes

def pair_push(unet, head, a, b, s):
    def push(x, t, eps):
        if s == 0:
            return torch.zeros_like(eps)
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            pair = torch.tensor([[a, b]], device=x.device).expand(x.shape[0], 2)
            logit = head(head_feats(unet, head, xx, t, pair), t, pair)[:, 0]
            g, = torch.autograd.grad(logit.sum(), xx)
        inside = (logit.detach().abs() < LOGIT_CLIP).float()[:, None, None, None]
        return -s * (1 - ABAR.to(x.device)[t]).sqrt()[:, None, None, None] * g * inside
    return push


def separate_push(unet, head, a, b, s):
    def push(x, t, eps):
        if s == 0:
            return torch.zeros_like(eps)
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            z = head(features(unet, xx, t), t)
            g, = torch.autograd.grad((F.logsigmoid(z[:, a]) + F.logsigmoid(z[:, b])).sum(), xx)
        return -s * (1 - ABAR.to(x.device)[t]).sqrt()[:, None, None, None] * g
    return push


@torch.no_grad()
def score_clean(unet, head, x, a, b):
    """The pair-aware checker's log-odds on finished canvases, at the last sampler step (best-of-N)."""
    t = torch.full((x.shape[0],), int(STEPS[-1]), device=x.device)
    pair = torch.tensor([[a, b]], device=x.device).expand(x.shape[0], 2)
    return head(head_feats(unet, head, x, t, pair), t, pair)[:, 0]


# ---------------------------------------------------------------- the draws

def draws(args):
    from judge import judge, load as load_judge
    assert min(DRAW_SEEDS) >= 1000, "checker draws come from seed 1,000 up"
    dev = "cuda"
    jn = load_judge(args.judge, dev)
    xs, pairs, src, calls = [], [], [], []
    for share in (0.01, 0.1):
        unet = load_unet(args.models / f"share-{share}" / "unet.pt", dev)
        for a, b in SEEN + HELD_OUT:
            x = to_uint8(sample(unet, DRAW_SEEDS, [a, b], device=dev)[0])
            for c in x:
                xs.append(c), pairs.append((a, b)), src.append(share)
                calls.append(judge(jn, c, (a, b), dev)[0])
        print(f"share {share}: {len(xs)} draws so far", flush=True)
    calls = np.array(calls)
    held = np.array([tuple(p) in HELD_OUT for p in pairs])
    np.savez_compressed(args.out, x=np.stack(xs), pair=np.array(pairs), share=np.array(src), call=calls,
                        held=held, seeds=np.array(list(DRAW_SEEDS)))
    pos = calls == "two clear, right digits"
    print(json.dumps({"draws": len(xs), "seen positives": int((pos & ~held).sum()),
                      "seen negatives": int((~pos & ~held).sum()), "held-out positives": int((pos & held).sum()),
                      "held-out negatives": int((~pos & held).sum())}), flush=True)


# ---------------------------------------------------------------- the pasted canvases

def pasted(cv, rng, n, pairs, kind):
    """n pasted canvases, their asked pair, and which digits each holds (a fused pair holds both).

    kind: two, fused, single, wrong.
    """
    xs, ps, present = [], [], np.zeros((n, 10), np.float32)
    for _ in range(n):
        a, b = pairs[rng.integers(len(pairs))]
        if kind == "two":
            c = cv.two(a, b)
            present[len(xs), [a, b]] = 1
        elif kind == "fused":
            c = np.zeros((H, W), np.uint8)
            x0, y0 = int(rng.integers(8, W - D - 8)), int(rng.integers(0, H - D + 1))
            dx, dy = int(rng.integers(-5, 6)), int(rng.integers(-2, 3))
            c[y0:y0 + D, x0:x0 + D] = cv.digit(a)
            y2 = min(max(y0 + dy, 0), H - D)
            c[y2:y2 + D, x0 + dx:x0 + dx + D] = np.maximum(c[y2:y2 + D, x0 + dx:x0 + dx + D], cv.digit(b))
            present[len(xs), [a, b]] = 1
        elif kind == "single":
            alone = a if rng.random() < 0.5 else b
            c = cv.one(alone)
            present[len(xs), alone] = 1
        else:
            other = int(rng.integers(10))
            while other == b:
                other = int(rng.integers(10))
            c = cv.two(a, other)
            present[len(xs), [a, other]] = 1
        xs.append(c), ps.append((a, b))
    return np.stack(xs), np.array(ps), present


def guard(x, gen):
    """The same random shift, stroke thickening or thinning, and blur, on every label."""
    b = x.shape[0]
    shift = torch.randint(-2, 3, (2,), generator=gen).tolist()
    x = torch.roll(x, shifts=shift, dims=(2, 3))
    r = torch.rand(b, generator=gen).to(x.device)[:, None, None, None]
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    blur = F.avg_pool2d(x, 3, 1, 1)
    x = torch.where(r < 0.2, thick, torch.where(r < 0.4, thin, torch.where(r < 0.6, blur, x)))
    return x


# ---------------------------------------------------------------- training

def build_set(args, rng):
    images, labels = load_mnist(args.mnist)
    cv = Canvases(images, labels, rng)
    d = np.load(args.draws)
    excl = set(args.exclude_pairs)
    seen = ~d["held"] & np.array([tuple(p) not in excl for p in d["pair"]])
    train_pairs = [p for p in SEEN if p not in excl]
    pos_d = seen & (d["call"] == "two clear, right digits")
    neg_d = seen & (d["call"] != "two clear, right digits")
    n = 8000
    two = pasted(cv, rng, n, train_pairs, "two")
    w = args.wrong_weight
    share = {"fused": 1, "single": 1, "wrong": w}
    negs = [pasted(cv, rng, int(n * share[k] / (2 + w)), train_pairs, k) for k in ("fused", "single", "wrong")]
    nodigits = lambda m: np.zeros((m, 10), np.float32)
    sets = {
        "pos_drawn": (d["x"][pos_d], d["pair"][pos_d], nodigits(pos_d.sum())),
        "pos_pasted": two,
        "neg_drawn": (d["x"][neg_d], d["pair"][neg_d], nodigits(neg_d.sum())),
        "neg_pasted": tuple(np.concatenate([g[i] for g in negs]) for i in range(3)),
    }
    if args.kind == "separate":          # presence is exact only on pasted canvases
        sets["pos_drawn"] = tuple(v[:0] for v in sets["pos_drawn"])
        sets["neg_drawn"] = tuple(v[:0] for v in sets["neg_drawn"])
    kept = {}
    for k, v in sets.items():
        m = max(1, int(round(len(v[0]) * args.data_frac))) if len(v[0]) else 0
        idx = rng.permutation(len(v[0]))[:m]
        kept[k] = tuple(a[idx] for a in v)
    return kept, d


def held_set(args, d):
    """The held-out check: pasted held-out pairs from MNIST's test digits, plus their drawn canvases."""
    timg = np.fromfile(args.mnist / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
    tlbl = np.fromfile(args.mnist / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    rng = np.random.default_rng(7)
    cv = Canvases(timg, tlbl, rng)
    pos, pp, _ = pasted(cv, rng, 400, HELD_OUT, "two")
    negs = [pasted(cv, rng, 100, HELD_OUT, k) for k in ("fused", "single", "wrong")]
    held = d["held"]
    dpos = held & (d["call"] == "two clear, right digits")
    dneg = held & (d["call"] != "two clear, right digits")
    x = np.concatenate([pos, d["x"][dpos]] + [g[0] for g in negs] + [d["x"][dneg]])
    p = np.concatenate([pp, d["pair"][dpos]] + [g[1] for g in negs] + [d["pair"][dneg]])
    y = np.concatenate([np.ones(len(pos) + dpos.sum()), np.zeros(300 + dneg.sum())])
    return x, p, y


def to_x(a, dev):
    return torch.from_numpy(a).to(dev).float().div(127.5).sub(1).unsqueeze(1)


def train(args):
    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    unet = load_unet(args.models / f"share-{GUIDED_SHARE}" / "unet.pt", dev)
    kept, d = build_set(args, rng)
    pair_aware = args.kind == "pair"
    head = Head(1 if pair_aware else 10, pair_aware, film=args.film, expert=args.expert_feats and pair_aware).to(dev)
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    flips = {k: rng.random(len(v[0])) < args.flip for k, v in kept.items()}
    sizes = {k: len(v[0]) for k, v in kept.items()}
    print(json.dumps({"kind": args.kind, "data_frac": args.data_frac, "flip": args.flip, "pool_sizes": sizes}),
          flush=True)

    def take(name, m):
        x, p, pres = kept[name]
        i = rng.integers(len(x), size=m)
        return x[i], p[i], flips[name][i], pres[i]

    half = args.batch // 2
    for step in range(1, args.steps + 1):
        parts = []
        for label in ("pos", "neg"):
            nd = half // 2 if sizes[f"{label}_drawn"] else 0
            for name, m in ((f"{label}_drawn", nd), (f"{label}_pasted", half - nd)):
                if m:
                    x, p, fl, pres = take(name, m)
                    y = np.full(m, 1.0 if label == "pos" else 0.0)
                    parts.append((x, p, np.where(fl, 1 - y, y), pres))
        if args.wrong_ask and pair_aware:
            px = np.concatenate([q[0] for q in parts if q[2].mean() > 0.5])
            pp = np.concatenate([q[1] for q in parts if q[2].mean() > 0.5]).copy()
            for i in range(len(pp)):
                other = int(rng.integers(10))
                while other in (pp[i][0], pp[i][1]):
                    other = int(rng.integers(10))
                pp[i] = (pp[i][0], other) if rng.random() < 0.5 else (other, pp[i][1])
            parts.append((px, pp, np.zeros(len(px)), np.zeros((len(px), 10), np.float32)))
        x0 = guard(to_x(np.concatenate([q[0] for q in parts]), dev), gen)
        pair = torch.from_numpy(np.concatenate([q[1] for q in parts])).to(dev).long()
        y = torch.from_numpy(np.concatenate([q[2] for q in parts])).to(dev).float()
        k = torch.multinomial(STEP_P, len(y), replacement=True, generator=gen)
        t = STEPS[k].to(dev)
        a = ABAR.to(dev)[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, generator=gen).to(dev)
        with torch.no_grad():
            feats = head_feats(unet, head, xt, t, pair) if pair_aware else features(unet, xt, t)
        w = LAMBDA.to(dev)[k.to(dev)]
        if pair_aware:
            logit = head(feats, t, pair)[:, 0]
            loss = (w * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()
        else:
            target = torch.from_numpy(np.concatenate([q[3] for q in parts])).to(dev)
            z = head(feats, t)
            loss = (w[:, None] * F.binary_cross_entropy_with_logits(z, target, reduction="none")).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4)}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "kind": args.kind, "film": args.film, "expert": head.expert,
                "args": {k: str(v) for k, v in vars(args).items()}},
               args.out / "head.pt")
    res = {"kind": args.kind, "data_frac": args.data_frac, "flip": args.flip, "pool_sizes": sizes}
    if pair_aware:
        res |= held_accuracy(args, unet, head, d)
    (args.out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


@torch.no_grad()
def held_accuracy(args, unet, head, d):
    dev = "cuda"
    x, p, y = held_set(args, d)
    x0 = to_x(x, dev)
    pair = torch.from_numpy(p).to(dev).long()
    gen = torch.Generator().manual_seed(123)
    per_step = []
    for k in range(20):
        t = torch.full((len(y),), int(STEPS[k]), device=dev)
        a = ABAR.to(dev)[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, generator=gen).to(dev)
        logit = torch.cat([head(head_feats(unet, head, xt[i:i + 512], t[i:i + 512], pair[i:i + 512]), t[i:i + 512],
                                pair[i:i + 512])[:, 0]
                           for i in range(0, len(y), 512)]).cpu().numpy()
        pred = logit > 0
        tpr = float(pred[y == 1].mean())
        tnr = float((~pred[y == 0]).mean())
        per_step.append(round((tpr + tnr) / 2, 4))
    return {"held_out_balanced_accuracy_steps_0_19": round(float(np.mean(per_step)), 4),
            "held_out_balanced_accuracy_per_step": per_step,
            "held_out_set": {"positives": int((y == 1).sum()), "negatives": int((y == 0).sum())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draws", "train"])
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--judge", type=Path)
    ap.add_argument("--draws", type=Path)
    ap.add_argument("--mnist", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["pair", "separate"], default="pair")
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--flip", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-pairs", type=lambda v: [tuple(int(d) for d in p.split(",")) for p in v.split()],
                    default=[], help='pairs left out of checker training, e.g. "0,6 1,2 6,9" (loop 2)')
    ap.add_argument("--wrong-ask", action="store_true", help="each positive also asked for a wrong pair, as a negative")
    ap.add_argument("--wrong-weight", type=float, default=1.0, help="wrong-pair share of pasted negatives, against 1 each")
    ap.add_argument("--film", action="store_true", help="the asked pair modulates the head's convolutions")
    ap.add_argument("--expert-feats", action="store_true", help="read the features asked with each asked digit (round 8)")
    args = ap.parse_args()
    draws(args) if args.cmd == "draws" else train(args)


if __name__ == "__main__":
    main()
