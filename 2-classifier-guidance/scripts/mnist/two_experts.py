"""Plan 32: sample two digit experts that never met, composed over a blank-canvas base, count and draw sheets.

The product is eps_blank + (eps_A(a) - eps_blank) + (eps_B(b) - eps_blank), with
eps_blank(x_t) = (x_t + sqrt(abar_t)) / sqrt(1 - abar_t), the exact noise prediction for a canvas that is -1
everywhere. DDIM in 50 steps, eta 1.0, each seed's own fixed noise; pushes as in step 7b, summed and capped
at 0.1 of the product's size per canvas.

The count is the third judge's (step 7b, loop 1): a tile is "two clear, right, apart" when it holds exactly
two blobs, one in each half, whose top digits over classes 0 to 9 are the asked pair, each at least 0.9 sure.

Usage:
  python two_experts.py alone --expert <unet.pt> --digits 0-4 --name e04-either --out <dir>
  python two_experts.py cells --a <unet.pt> --b <unet.pt> --push none|zero|random|both|ink --name <cell> --out <dir>
         [--pair-head <pt>] [--sep-head <pt>] [--ink-head <pt>]
  python two_experts.py draws --a <unet.pt> --b <unet.pt> --push both --pair-head <pt> --sep-head <pt>
         --draws-out <npz> [--seed0 1000 --n 12]        pushed renders on the training pairs, judged, for the
                                                       pair-aware checker's on-policy pass
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from canvas_unet import ABAR, sample, to_uint8  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge, split_wide  # noqa: E402
from judge_sheet import blobs, crop28  # noqa: E402
import pixel_checker as pc  # noqa: E402

EVAL_SEEDS = list(range(64))
CALLS = ["two clear, right, apart", "same half", "one blob", "two, not both clear and right", "damaged"]
COL = {"two clear, right, apart": "#1d7a3e", "same half": "#c86a0a", "one blob": "#c0262d",
       "two, not both clear and right": "#6b7a88", "damaged": "#1f2a33"}


def blank_eps(x, t):
    a = ABAR.to(x.device)[t][:, None, None, None]
    return (x + a.sqrt()) / (1 - a).sqrt()


def floor_eps(x, t, tau):
    """The blank canvas given spread tau: N(-1, tau^2) noised to t. tau 0 is blank_eps, whose score diverges as t -> 0."""
    a = ABAR.to(x.device)[t][:, None, None, None]
    return (1 - a).sqrt() * (x + a.sqrt()) / (a * tau ** 2 + 1 - a)


def two_eps(A, B, a, b, base="floor", tau=0.3):
    """base: blank (point mass), floor (blank with spread tau), avgnull (mean of the two experts' no-digit predictions)."""
    def f(x, t):
        ya = torch.full((x.shape[0],), a, device=x.device)
        yb = torch.full((x.shape[0],), b, device=x.device)
        if base == "avgnull":
            n = torch.full((x.shape[0],), 10, device=x.device)
            ref = 0.5 * (A(x, t, n) + B(x, t, n))
        else:
            ref = blank_eps(x, t) if base == "blank" else floor_eps(x, t, tau)
        return A(x, t, ya) + B(x, t, yb) - ref
    return f


def region_eps(A, B, a, b, sides, crop):
    """Region composition (plan 33, E0): each seed has a side; expert A owns one half and B the other, and the halves are joined.
    crop False: each expert sees the whole canvas and only its half of the prediction is kept.
    crop True: each expert sees only its half of the canvas (a 32 x 32 crop)."""
    sides_t = torch.tensor(sides)
    def f(x, t):
        dev = x.device
        left = (sides_t == 0).to(dev)[:, None, None, None]           # True: A takes the left half
        ya = torch.full((x.shape[0],), a, device=dev); yb = torch.full((x.shape[0],), b, device=dev)
        if crop:
            xl, xr = x[..., :32], x[..., 32:]
            eA_l, eA_r = A(xl, t, ya), A(xr, t, ya)
            eB_l, eB_r = B(xl, t, yb), B(xr, t, yb)
            e_left = torch.where(left, eA_l, eB_l); e_right = torch.where(left, eB_r, eA_r)
            return torch.cat([e_left, e_right], dim=-1)
        eA, eB = A(x, t, ya), B(x, t, yb)
        m = torch.zeros_like(x); m[..., :32] = 1
        pick_a = torch.where(left, m, 1 - m)                          # 1 where A's prediction is used
        return pick_a * eA + (1 - pick_a) * eB
    return f


def union_eps(A, B, U, a, b):
    """E1: the two experts over a base trained on the union of both ranges (the step 7b share-0 model's no-digit prediction)."""
    def f(x, t):
        ya = torch.full((x.shape[0],), a, device=x.device); yb = torch.full((x.shape[0],), b, device=x.device)
        n = torch.full((x.shape[0],), 10, device=x.device)
        return A(x, t, ya) + B(x, t, yb) - U(x, t, n)
    return f



def build_eps(args, A, B, a, b, U, seeds=None):
    if args.compose in ("region", "crop"):
        sides = [int(np.random.default_rng(1000 + s).integers(2)) for s in (EVAL_SEEDS if seeds is None else seeds)]
        return region_eps(A, B, a, b, sides, args.compose == "crop")
    if args.base == "union":
        return union_eps(A, B, U, a, b)
    return two_eps(A, B, a, b, args.base, args.tau)


def one_eps(A, d):
    return lambda x, t: A(x, t, torch.full((x.shape[0],), d, device=x.device))


# ---------------------------------------------------------------- the pushes

def make_push(kind, a, b, heads, s=10.0):
    if kind == "none":
        return None
    if kind == "zero":
        return lambda x, t, eps: torch.zeros_like(eps)
    if kind == "ink":
        return pc.ink_push(heads["ink"], s)
    pp, sp = pc.pair_push(heads["pair"], a, b, s), pc.separate_push(heads["sep"], a, b, s)
    both = lambda x, t, eps: pp(x, t, eps) + sp(x, t, eps)
    if kind == "both":
        return both
    gen = torch.Generator().manual_seed(12345)

    def random(x, t, eps):
        ref = both(x, t, eps)
        r = torch.randn(x.shape, generator=gen).to(x.device)
        return r * (ref.flatten(1).norm(dim=1) / r.flatten(1).norm(dim=1))[:, None, None, None]
    return random


# ---------------------------------------------------------------- counting

@torch.no_grad()
def reads(net, canvas, dev):
    parts = []
    for ys, xs in blobs(canvas):
        parts += split_wide(canvas, ys, xs)
    return [(float(xs.mean()), torch.softmax(net(crop28(canvas, ys, xs).to(dev)), 1)[0].cpu().numpy())
            for ys, xs in parts]


def call_one_tolerant(rd, digit):
    """Clutter-tolerant own-digit call (plan 33): exactly one blob read at least 0.9 as the asked digit, other blobs allowed."""
    return sum(1 for _, p in rd if int((p[:10] / p[:10].sum()).argmax()) == digit and (p[:10] / p[:10].sum()).max() >= 0.9) == 1


def call_two(rd, pair):
    if len(rd) == 0 or len(rd) > 2:
        return "damaged"
    if len(rd) == 1:
        return "one blob"
    if sorted(int(x >= 32) for x, _ in rd) != [0, 1]:
        return "same half"
    d = [p[:10] / p[:10].sum() for _, p in rd]
    if sorted(int(q.argmax()) for q in d) == sorted(pair) and all(q.max() >= 0.9 for q in d):
        return "two clear, right, apart"
    return "two, not both clear and right"


def call_one(rd, digit):
    if len(rd) != 1:
        return False
    q = rd[0][1][:10] / rd[0][1][:10].sum()
    return int(q.argmax()) == digit and q.max() >= 0.9


def call_tolerant(rd, pair):
    """Clutter-tolerant (plan 33, E2): one blob read at least 0.9 as the asked digit a and one as b, in different halves, other blobs allowed."""
    a, b = pair
    def hits(d):
        return [x >= 32 for x, p in rd if (p[:10] / p[:10].sum())[d] >= 0.9 and int((p[:10] / p[:10].sum()).argmax()) == d]
    ha, hb = hits(a), hits(b)
    return len(ha) == 1 and len(hb) == 1 and ha[0] != hb[0]


def sheet(canv, calls, title, path):
    fig, axs = plt.subplots(8, 8, figsize=(16, 5.6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(canv[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        c = COL.get(calls[i], "#1d7a3e" if calls[i] is True else "#c0262d")
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2.2)
        ax.text(1, 6, str(i), color="#9aa5b1", fontsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=70); plt.close(fig)


def parse_pairs(s):
    return [tuple(int(v) for v in p.split("-")) for p in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["alone", "cells", "draws"])
    ap.add_argument("--expert", type=Path); ap.add_argument("--digits", default="0-4")
    ap.add_argument("--a", type=Path); ap.add_argument("--b", type=Path)
    ap.add_argument("--push", default="none", choices=["none", "zero", "random", "both", "ink"])
    ap.add_argument("--pairs", default="1-9,3-6,4-7,2-8,0-5")
    ap.add_argument("--pair-head", type=Path); ap.add_argument("--sep-head", type=Path); ap.add_argument("--ink-head", type=Path)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--name", default=""); ap.add_argument("--out", type=Path)
    ap.add_argument("--draws-out", type=Path); ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--cap", type=float, default=0.1); ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--lang-k", type=int, default=0); ap.add_argument("--lang-c", type=float, default=0.0)
    ap.add_argument("--base", default="floor", choices=["blank", "floor", "avgnull", "union"])
    ap.add_argument("--compose", default="product", choices=["product", "region", "crop"])
    ap.add_argument("--union", type=Path)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()
    dev = "cuda"
    net = load_judge(args.judge, dev)

    if args.mode == "alone":
        A = load_unet(args.expert, dev)
        lo, hi = (int(v) for v in args.digits.split("-"))
        out = args.out / "alone" / args.name
        out.mkdir(parents=True, exist_ok=True)
        counts, canv, tol = {}, {}, {}
        for d in range(lo, hi + 1):
            x = to_uint8(sample(None, EVAL_SEEDS, [], eps_fn=one_eps(A, d), device=dev, eta=1.0)[0])
            rd = [reads(net, c, dev) for c in x]
            ok = [bool(call_one(r, d)) for r in rd]
            tol[str(d)] = int(sum(call_one_tolerant(r, d) for r in rd))
            counts[str(d)] = int(sum(ok)); canv[str(d)] = x
            sheet(x, ok, f"{args.name} alone, asked {d}: {sum(ok)} of 64 one clear {d}", out / f"alone-{d}.png")
            print(args.name, d, counts[str(d)], flush=True)
        np.savez_compressed(out / "canvases.npz", **canv)
        (out / "counts.json").write_text(json.dumps(counts, indent=1))
        (out / "counts_tolerant.json").write_text(json.dumps(tol, indent=1))
        print(args.name, "tolerant", json.dumps(tol), flush=True)
        return

    A, B = load_unet(args.a, dev), load_unet(args.b, dev)
    U = load_unet(args.union, dev) if args.union else None
    heads = {}
    if args.pair_head: heads["pair"] = pc.load_head(args.pair_head, dev)
    if args.sep_head: heads["sep"] = pc.load_head(args.sep_head, dev)
    if args.ink_head: heads["ink"] = pc.load_head(args.ink_head, dev)

    if args.mode == "draws":
        rng = np.random.default_rng(args.seed0)
        xs, ps, oks = [], [], []
        for a, b in parse_pairs(args.pairs):
            seeds = list(range(args.seed0, args.seed0 + args.n))
            assert min(seeds) >= 1000, "draws come from seed 1,000 up"
            x = to_uint8(sample(None, seeds, [], eps_fn=build_eps(args, A, B, a, b, U, seeds), push_fn=make_push(args.push, a, b, heads, args.s),
                                device=dev, eta=1.0)[0])
            ok = [call_two(reads(net, c, dev), (a, b)) == CALLS[0] for c in x]
            xs.append(x); ps += [(a, b)] * len(x); oks += ok
            print(a, b, sum(ok), "of", len(x), flush=True)
            args.seed0 += args.n
        args.draws_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.draws_out, x=np.concatenate(xs), pair=np.array(ps), ok=np.array(oks))
        print(json.dumps({"draws": len(oks), "ok": int(sum(oks))}))
        return

    out = args.out / "cells" / args.name
    out.mkdir(parents=True, exist_ok=True)
    counts, canv = {}, {}
    for a, b in parse_pairs(args.pairs):
        x = to_uint8(sample(None, EVAL_SEEDS, [], eps_fn=build_eps(args, A, B, a, b, U),
                            push_fn=make_push(args.push, a, b, heads, args.s), cap=args.cap, device=dev, eta=args.eta,
                            langevin=(args.lang_k, args.lang_c))[0])
        rds = [reads(net, c, dev) for c in x]
        calls = [call_two(r, (a, b)) for r in rds]
        tally = {k: calls.count(k) for k in CALLS}
        tally["tolerant"] = int(sum(call_tolerant(r, (a, b)) for r in rds))
        counts[f"{a}-{b}"] = tally; canv[f"{a}-{b}"] = x
        sheet(x, calls, f"{args.name}, asked {a} and {b}: {tally[CALLS[0]]} of 64 two clear, right, apart "
              f"(green); same half {tally['same half']} (orange), one blob {tally['one blob']} (red)", out / f"sheet-{a}-{b}.png")
        print(args.name, a, b, json.dumps(tally), flush=True)
    total = sum(v[CALLS[0]] for v in counts.values())
    tol_total = sum(v["tolerant"] for v in counts.values())
    np.savez_compressed(out / "canvases.npz", **canv)
    (out / "counts.json").write_text(json.dumps({"cell": args.name, "push": args.push, "s": args.s, "base": args.base, "tau": args.tau,
                                                  "total_of_320": total, "tolerant_total_of_320": tol_total, "per_pair": counts}, indent=1))
    print(json.dumps({"cell": args.name, "total_of_320": total, "tolerant_total_of_320": tol_total}), flush=True)


if __name__ == "__main__":
    main()
