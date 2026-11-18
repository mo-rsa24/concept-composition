"""The five-version sheet: plain product against four pushes, seeds 0 to 63 (plan 24 task 1.6).

On the share-0.01 U-Net, for 7 & 8 and the five held-out pairs, each canvas called by the judge:
1. plain product (no push)
2. the pair-aware checker's push at s = 0, 1, 3, 10, 30
3. the separate-label push ("a present" plus "b present") at the same s
4. a random push: per sample and step, a fixed random direction scaled to the size the pair-aware
   push had after the cap at that step and s
5. best-of-N: N plain-product draws per seed, the pair-aware checker picking the one it scores
   highest on the finished canvas; N is a guided draw's measured cost in plain draws, rounded up, and N = 16 is
   reported beside it. Extra draws use seeds 100,000 up, never a checker's training seeds.
At s = 0 both guided columns must match plain product byte for byte; the script stops if not.
The cap is 0.1 of the product prediction's norm per sample (`--cap` for the cap sweep). Each
control is read at its own best s against the bar; the pair-aware push at its best s.

Writes counts.json, calls.csv, canvases.npz, cap.json and sheet-<pair>.png under --out.

Usage:
    python five_versions.py --models <dir> --checkers <dir> --judge <pt> --out <dir> [--pair-head pair-full] [--cap 0.1]
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_unet import HELD_OUT, per_sample_cap, sample, to_uint8  # noqa: E402
from checker import GUIDED_SHARE, Head, load_unet, pair_push, score_clean, separate_push  # noqa: E402
from judge import LADDER, judge, load as load_judge  # noqa: E402

PAIRS = [(7, 8)] + HELD_OUT
SEEDS = list(range(64))
S = [0.0, 1.0, 3.0, 10.0, 30.0]
RIGHT = "two clear, right digits"


def load_head(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    pair_aware = ck["kind"] == "pair"
    h = Head(1 if pair_aware else 10, pair_aware).to(dev).eval()
    h.load_state_dict(ck["head"])
    for p in h.parameters():
        p.requires_grad_(False)
    return h


class Recorder:
    """Wraps a push: applies the per-sample cap itself, records the capped norms and where it fired."""

    def __init__(self, fn, cap):
        self.fn, self.cap, self.norms, self.fired = fn, cap, [], []

    def __call__(self, x, t, eps):
        p, fired = per_sample_cap(self.fn(x, t, eps), eps, self.cap)
        self.norms.append(p.flatten(1).norm(dim=1).detach())
        self.fired.append(fired.float().mean().item())
        return p


def random_push(norms, seeds, dev):
    """A fixed random direction per seed and step, scaled to the recorded pair-aware push norm."""
    step = {"i": 0}
    gens = [torch.Generator().manual_seed(50_000 + s) for s in seeds]

    def push(x, t, eps):
        i = step["i"]
        step["i"] += 1
        d = torch.stack([torch.randn(x.shape[1:], generator=g) for g in gens]).to(dev)
        d = d / d.flatten(1).norm(dim=1)[:, None, None, None]
        return d * norms[i][:, None, None, None]
    return push


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--checkers", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair-head", default="pair-full")
    ap.add_argument("--separate-head", default="separate")
    ap.add_argument("--cap", type=float, default=0.1)
    ap.add_argument("--pairs", default="all", help="'all' or '78' (the cap sweep reads 7 & 8 only)")
    ap.add_argument("--only-s", type=float, default=None, help="pair-aware push at this s only (cap sweep)")
    ap.add_argument("--pair-only", action="store_true", help="plain and the pair-aware push only (sweep 1)")
    args = ap.parse_args()
    dev = "cuda"
    unet = load_unet(args.models / f"share-{GUIDED_SHARE}" / "unet.pt", dev)
    hp = load_head(args.checkers / args.pair_head / "head.pt", dev)
    hs = load_head(args.checkers / args.separate_head / "head.pt", dev)
    jn = load_judge(args.judge, dev)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = PAIRS if args.pairs == "all" else [(7, 8)]
    s_list = S if args.only_s is None else [args.only_s]

    # cost: one plain and one guided batch of 64 on 7 & 8, to match best-of-N's N
    sample(unet, SEEDS[:8], [7, 8], push_fn=pair_push(unet, hp, 7, 8, 3.0), device=dev)   # warm-up
    torch.cuda.synchronize(); t0 = time.perf_counter()
    sample(unet, SEEDS, [7, 8], device=dev)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    sample(unet, SEEDS, [7, 8], push_fn=pair_push(unet, hp, 7, 8, 3.0), cap=args.cap, device=dev)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    n_match = max(1, math.ceil((t2 - t1) / (t1 - t0)))   # rounded up, in best-of-N's favour, as rung 2
    cost = {"plain_s_per_64": round(t1 - t0, 2), "guided_s_per_64": round(t2 - t1, 2), "N_matched": n_match}
    print(json.dumps(cost), flush=True)

    canv, rows, counts, cap_log = {}, [], {}, {}

    def record(col, a, b, x):
        x8 = to_uint8(x)
        canv[f"{col}|{a}{b}"] = x8
        tally = {k: 0 for k in LADDER}
        for s, c in zip(SEEDS, x8):
            call, reads = judge(jn, c, (a, b), dev)
            tally[call] += 1
            rows.append([col, f"{a}&{b}", s, call])
        counts[f"{col}|{a}&{b}"] = tally
        print(f"{a}&{b} {col}: {tally[RIGHT]} of 64 two clear, right digits", flush=True)
        return x8

    for a, b in pairs:
        plain = record("plain", a, b, sample(unet, SEEDS, [a, b], device=dev)[0])
        for s in s_list:
            rec = Recorder(pair_push(unet, hp, a, b, s), args.cap)
            xg = record(f"pair s={s:g}", a, b, sample(unet, SEEDS, [a, b], push_fn=rec, cap=None, device=dev)[0])
            cap_log[f"pair s={s:g}|{a}&{b}"] = {"fired_share_per_step": [round(f, 3) for f in rec.fired],
                                               "steps_fired_on_half_or_more": int(sum(f >= 0.5 for f in rec.fired))}
            if s == 0:
                assert (xg == plain).all(), "s = 0 is not byte-identical to plain product"
            if args.pair_only:
                continue
            sep = Recorder(separate_push(unet, hs, a, b, s), args.cap)
            xs = record(f"separate s={s:g}", a, b, sample(unet, SEEDS, [a, b], push_fn=sep, cap=None, device=dev)[0])
            if s == 0:
                assert (xs == plain).all(), "s = 0 is not byte-identical to plain product"
                print("s = 0: both guided columns byte-identical to plain product", flush=True)
                continue
            record(f"random s={s:g}", a, b, sample(unet, SEEDS, [a, b], push_fn=random_push(rec.norms, SEEDS, dev),
                                                   cap=args.cap, device=dev)[0])
        if args.only_s is not None or args.pair_only:
            continue
        for n in sorted({n_match, 16}):
            best = []
            for s in SEEDS:
                seeds = [s] + [100_000 + 100 * s + j for j in range(1, n)]
                x = sample(unet, seeds, [a, b], device=dev)[0]
                best.append(x[int(score_clean(unet, hp, x, a, b).argmax())])
            record(f"best-of-{n}", a, b, torch.stack(best))

    np.savez_compressed(args.out / "canvases.npz", **canv)
    with open(args.out / "calls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "pair", "seed", "judge_call"])
        w.writerows(rows)
    (args.out / "cap.json").write_text(json.dumps(cap_log, indent=1))
    (args.out / "counts.json").write_text(json.dumps({"cost": cost, "cap": args.cap, "pair_head": args.pair_head, "separate_head": args.separate_head,
                                                      "counts": counts}, indent=1))
    if args.only_s is None and not args.pair_only:
        for a, b in pairs:
            draw(args.out, a, b, canv, counts, n_match)


def draw(out, a, b, canv, counts, n_match):
    def best_s(kind):
        return max((s for s in S[1:]), key=lambda s: counts[f"{kind} s={s:g}|{a}&{b}"][RIGHT])
    sp = best_s("pair")
    cols = [("plain product", "plain"), (f"pair-aware, s = {sp:g} (its best)", f"pair s={sp:g}"),
            (f"separate-label, s = {best_s('separate'):g} (its best)", f"separate s={best_s('separate'):g}"),
            (f"random push, s = {best_s('random'):g} (its best)", f"random s={best_s('random'):g}"),
            (f"best-of-{n_match} (matched cost)", f"best-of-{n_match}")]
    fig, axs = plt.subplots(len(cols), 16, figsize=(26, 1.25 * len(cols) + 1.2))
    for r, (label, col) in enumerate(cols):
        n = counts[f"{col}|{a}&{b}"][RIGHT]
        for k in range(16):
            ax = axs[r, k]
            ax.imshow(canv[f"{col}|{a}{b}"][k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
        axs[r, 0].set_ylabel(f"{label}\n{n} of 64", rotation=0, ha="right", va="center", fontsize=11)
    for k in range(16):
        axs[0, k].set_title(f"seed {k}", fontsize=9)
    fig.suptitle(f"{a} & {b} on the share-0.01 model: the first 16 of 64 seeds per version; "
                 f"the count is the judge's two clear, right digits out of 64", fontsize=13, weight="bold")
    fig.savefig(out / f"sheet-{a}{b}.png", dpi=70, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
