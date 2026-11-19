"""Plan 32: checkers that read the noisy canvas itself, so they belong to neither digit expert.

pair      one log-odds that the canvas becomes the asked pair, clear, one digit in each half; trained on
          pasted canvases (the asked pair apart against fused overlays, lone digits, wrong pairs), then on
          its own pushed renders labelled by the third judge (--draws).
separate  ten "is digit d there?" answers, on pasted canvases with exact presence (a fused pair holds both).
ink       one network that reads one 32 x 32 half and answers "is there a digit here?", applied to the left
          and right halves; trained on halves of one-digit canvases and blank halves only, never on a canvas
          with two digits.

The 7b recipe otherwise: 4,000 steps of 256, Adam 0.001, steps drawn from the 50 sampler steps with steps
0 to 9 twice as often, the loss weighted by Kim's beta / (1 - abar), the same shift, thickening, thinning
and blur on every example. The five evaluation pairs never enter a checker's training.

Usage: python pixel_checker.py train --kind pair|separate|ink --mnist <raw dir> --out <dir> [--draws <npz> ...]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_sheet import Canvases, D, H, W  # noqa: E402
from canvas_unet import ABAR, load_mnist  # noqa: E402
from checker import STEPS, LAMBDA, STEP_P, LOGIT_CLIP, time_features, guard  # noqa: E402

EVAL_PAIRS = [(1, 9), (3, 6), (4, 7), (2, 8), (0, 5)]
TRAIN_PAIRS = [(a, b) for a in range(10) for b in range(a + 1, 10) if (a, b) not in EVAL_PAIRS]
if os.environ.get("PC_NOCROSS"):          # plan 33: no cross-range pair (a 0-4 digit with a 5-9 digit) in a checker's pasted data
    TRAIN_PAIRS = [(a, b) for a, b in TRAIN_PAIRS if not (a <= 4 < b)]


class PixelHead(nn.Module):
    """Noisy canvas (or half) and step in; out logits. The step and the pair scale and shift the middle maps."""

    def __init__(self, out, pair_aware=False):
        super().__init__()
        self.pair_aware = pair_aware
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU())
        self.c1, self.c2 = nn.Conv2d(128, 128, 3, padding=1), nn.Conv2d(128, 128, 3, padding=1)
        self.pair = nn.Embedding(10, 64) if pair_aware else None
        self.film = nn.Linear(16 + (64 if pair_aware else 0), 256)
        self.mlp = nn.Sequential(nn.Linear(256 + 16 + (64 if pair_aware else 0), 256), nn.SiLU(), nn.Linear(256, out))

    def forward(self, x, t, pair=None):
        cond = [time_features(t)]
        if self.pair_aware:
            cond.append(self.pair(pair[:, 0]) + self.pair(pair[:, 1]))
        cond = torch.cat(cond, 1)
        g, b = self.film(cond).chunk(2, 1)
        h = F.silu(self.c1(self.stem(x)))
        h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(self.c2(h))
        return self.mlp(torch.cat([h.mean((2, 3)), h.amax((2, 3)), cond], 1))


def ink_logits(head, x, t):
    """The ink checker's two answers: left half, right half."""
    return torch.cat([head(x[..., :32], t), head(x[..., 32:], t)], 1)


# ---------------------------------------------------------------- the pushes (same form as step 7b)

def _rate(t, x):
    return (1 - ABAR.to(x.device)[t]).sqrt()[:, None, None, None]


def pair_push(head, a, b, s):
    def push(x, t, eps):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            pair = torch.tensor([[a, b]], device=x.device).expand(x.shape[0], 2)
            logit = head(xx, t, pair)[:, 0]
            g, = torch.autograd.grad(logit.sum(), xx)
        inside = (logit.detach().abs() < LOGIT_CLIP).float()[:, None, None, None]
        return -s * _rate(t, x) * g * inside
    return push


def separate_push(head, a, b, s):
    def push(x, t, eps):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            z = head(xx, t)
            g, = torch.autograd.grad((F.logsigmoid(z[:, a]) + F.logsigmoid(z[:, b])).sum(), xx)
        return -s * _rate(t, x) * g
    return push


def ink_push(head, s):
    def push(x, t, eps):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            z = ink_logits(head, xx, t)
            g, = torch.autograd.grad(F.logsigmoid(z).sum(), xx)
        return -s * _rate(t, x) * g
    return push


def load_head(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    out = {"pair": 1, "separate": 10, "ink": 1}[ck["kind"]]
    h = PixelHead(out, ck["kind"] == "pair").to(dev).eval()
    h.load_state_dict(ck["head"])
    for p in h.parameters():
        p.requires_grad_(False)
    h.kind = ck["kind"]
    return h


# ---------------------------------------------------------------- training data

def pasted(cv, rng, n, kind):
    """n pasted canvases in plan 32's layout, their asked pair, and which digits each holds."""
    xs, ps, present = [], [], np.zeros((n, 10), np.float32)
    for i in range(n):
        a, b = TRAIN_PAIRS[rng.integers(len(TRAIN_PAIRS))]
        if kind == "two":
            c = cv.two(a, b); present[i, [a, b]] = 1
        elif kind == "fused":
            c = cv.overlaid(a, b) if rng.random() < 0.3 else np.zeros((H, W), np.uint8)
            if not c.any():
                side = int(rng.integers(2))
                c = np.maximum(cv.one_half(a, side), cv.one_half(b, side))
            present[i, [a, b]] = 1
        elif kind == "single":
            alone = a if rng.random() < 0.5 else b
            c = cv.one_half(alone, int(rng.integers(2))); present[i, alone] = 1
        else:
            other = int(rng.integers(10))
            while other in (a, b):
                other = int(rng.integers(10))
            c = cv.two(a, other); present[i, [a, other]] = 1
        xs.append(c); ps.append((a, b))
    return np.stack(xs), np.array(ps), present


def halves(cv, rng, n):
    """Halves of one-digit canvases (label 1) and blank halves (label 0), 32 x 32 each."""
    xs, ys = [], []
    for _ in range(n):
        side = int(rng.integers(2))
        c = cv.one_half(int(rng.integers(10)), side)
        xs += [c[:, :32], c[:, 32:]]
        ys += [float(side == 0), float(side == 1)]
    return np.stack(xs), np.array(ys, np.float32)


def to_x(a, dev):
    return torch.from_numpy(a).float().to(dev).div(127.5).sub(1).unsqueeze(1)


def build(kind, cv, rng, draws):
    if kind == "ink":
        return {"halves": halves(cv, rng, 8000)}
    n = 8000
    pos = pasted(cv, rng, n, "two")
    negs = [pasted(cv, rng, n // 3, k) for k in ("fused", "single", "wrong")]
    neg = tuple(np.concatenate([g[i] for g in negs]) for i in range(3))
    sets = {"pos_pasted": pos, "neg_pasted": neg}
    if kind == "pair" and draws:
        xs, ps, ok = [], [], []
        for p in draws:
            d = np.load(p)
            keep = np.array([tuple(sorted(q)) not in EVAL_PAIRS for q in d["pair"]])
            xs.append(d["x"][keep]); ps.append(d["pair"][keep]); ok.append(d["ok"][keep])
        x, pr, ok = np.concatenate(xs), np.concatenate(ps), np.concatenate(ok).astype(bool)
        z = np.zeros((len(x), 10), np.float32)
        sets["pos_drawn"] = (x[ok], pr[ok], z[ok])
        sets["neg_drawn"] = (x[~ok], pr[~ok], z[~ok])
    return sets


@torch.no_grad()
def held_out(kind, head, dev, mnist, seed=7):
    """Accuracy on canvases pasted from MNIST test digits, over the first 20 sampler steps and all 50."""
    rng = np.random.default_rng(seed)
    timg = np.fromfile(mnist / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, D, D)
    tlbl = np.fromfile(mnist / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    cv = Canvases(timg, tlbl, rng)
    res = {}
    for name, ks in (("steps 0-19", range(20)), ("steps 0-49", range(50))):
        hits = []
        for k in ks:
            t = STEPS[k].repeat(128).to(dev)
            a = ABAR.to(dev)[t][:, None, None, None]
            if kind == "ink":
                x, y = halves(cv, rng, 64)
                xt = a.sqrt() * to_x(x, dev) + (1 - a).sqrt() * torch.randn(128, 1, H, 32, device=dev)
                hits.append(((head(xt, t)[:, 0] > 0).float().cpu().numpy() == y).mean())
            else:
                global TRAIN_PAIRS
                keep = TRAIN_PAIRS
                TRAIN_PAIRS = EVAL_PAIRS
                pos, neg = pasted(cv, rng, 64, "two"), pasted(cv, rng, 64, "fused")
                TRAIN_PAIRS = keep
                x = np.concatenate([pos[0], neg[0]]); pr = np.concatenate([pos[1], neg[1]])
                xt = a.sqrt() * to_x(x, dev) + (1 - a).sqrt() * torch.randn(128, 1, H, W, device=dev)
                if kind == "pair":
                    pred = (head(xt, t, torch.from_numpy(pr).to(dev).long())[:, 0] > 0).float().cpu().numpy()
                    y = np.r_[np.ones(64), np.zeros(64)]
                    hits.append(0.5 * ((pred[:64] == 1).mean() + (pred[64:] == 0).mean()))
                else:
                    z = head(xt, t)
                    both = torch.stack([z[i, pr[i]].min() for i in range(128)]) > 0
                    hits.append(both[:64].float().mean().item())   # both asked digits read present on apart pairs
        res[name] = round(float(np.mean(hits)), 3)
    return res


def train(args):
    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    images, labels = load_mnist(args.mnist)
    cv = Canvases(images, labels, rng)
    sets = build(args.kind, cv, rng, args.draws)
    sizes = {k: len(v[0]) for k, v in sets.items()}
    print(json.dumps({"kind": args.kind, "pool_sizes": sizes, "draws": [str(p) for p in args.draws]}), flush=True)
    head = PixelHead({"pair": 1, "separate": 10, "ink": 1}[args.kind], args.kind == "pair").to(dev)
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.kind == "ink":
            x, y = sets["halves"]
            i = rng.integers(len(x), size=args.batch)
            x0, y = guard(to_x(x[i], dev), gen), torch.from_numpy(y[i]).to(dev)
            pair = None
        else:
            parts, half = [], args.batch // 2
            for lab in ("pos", "neg"):
                nd = half // 2 if sizes.get(f"{lab}_drawn") else 0
                for name, m in ((f"{lab}_drawn", nd), (f"{lab}_pasted", half - nd)):
                    if m:
                        x, p, pres = sets[name]
                        i = rng.integers(len(x), size=m)
                        parts.append((x[i], p[i], np.full(m, float(lab == "pos")), pres[i]))
            x0 = guard(to_x(np.concatenate([q[0] for q in parts]), dev), gen)
            pair = torch.from_numpy(np.concatenate([q[1] for q in parts])).to(dev).long()
            y = torch.from_numpy(np.concatenate([q[2] for q in parts])).to(dev).float()
            pres = torch.from_numpy(np.concatenate([q[3] for q in parts])).to(dev)
        k = torch.multinomial(STEP_P, len(y), replacement=True, generator=gen)
        t = STEPS[k].to(dev)
        a = ABAR.to(dev)[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(x0.shape, generator=gen).to(dev)
        w = LAMBDA.to(dev)[k.to(dev)]
        if args.kind == "pair":
            loss = (w * F.binary_cross_entropy_with_logits(head(xt, t, pair)[:, 0], y, reduction="none")).mean()
        elif args.kind == "separate":
            loss = (w[:, None] * F.binary_cross_entropy_with_logits(head(xt, t), pres, reduction="none")).mean()
        else:
            loss = (w * F.binary_cross_entropy_with_logits(head(xt, t)[:, 0], y, reduction="none")).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}),
                  flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "kind": args.kind, "args": {k: str(v) for k, v in vars(args).items()}},
               args.out / "head.pt")
    head.eval()
    res = {"kind": args.kind, "pool_sizes": sizes, "weights": sum(p.numel() for p in head.parameters()),
           "held_out": held_out(args.kind, head, dev, args.mnist), "minutes": round((time.time() - t0) / 60, 1)}
    (args.out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--kind", choices=["pair", "separate", "ink"], required=True)
    tr.add_argument("--mnist", type=Path, required=True)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--draws", type=Path, nargs="*", default=[])
    tr.add_argument("--steps", type=int, default=4000)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
