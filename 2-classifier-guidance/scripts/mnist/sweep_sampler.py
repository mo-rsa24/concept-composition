"""Loop 2 rounds 6 and 7 (plan 24): the push's strength by sampler step, and a partly random sampler.

Renders the three tuning pairs (0 & 6, 1 & 2, 6 & 9), seeds 0 to 63, on the share-0.01 model with a
named checker's pair-aware push, under each setting, and counts them with the third judge.
Settings, written before any render:
  const:<s>                     strength s at every step (the reference)
  sched:<early>,<late>,<K>      strength early for the first K of 50 steps, then late
  eta:<eta>@<s>                 DDIM with eta, strength s at every step
  sep:<eta>@<s>+<s2>            eta, the push at s plus the separate-label push at s2 (round 6)
  ens:<eta>@<s>                 eta, the push averaged over --head and --head2, each at s (round 7)
The cap stays 0.1 of the product prediction's norm, per sample.

Usage: python sweep_sampler.py --models <dir> --checkers <dir> --judge3 <pt> --head <name> --out <dir> <setting> ...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_unet import sample, to_uint8
from checker import GUIDED_SHARE, load_unet, pair_push, separate_push
from five_versions import load_head
from judge import load
from judge_tune import read_all, call

PAIRS = [(0, 6), (1, 2), (6, 9)]
SEEDS = list(range(64))


def scheduled(fn, early, late, k):
    step = {"i": 0}

    def push(x, t, eps):
        s = early if step["i"] < k else late
        step["i"] += 1
        return s * fn(x, t, eps) if s else torch.zeros_like(eps)
    return push


def main():
    ap = argparse.ArgumentParser()
    for a in ("--models", "--checkers", "--judge3", "--out"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--head2", default=None, help="second checker, for ens: settings")
    ap.add_argument("--pairs", default=None, help='pairs to render, e.g. "0-3_1-9" or "0,3 1,9" (default: the three tuning pairs)')
    ap.add_argument("settings", nargs="+")
    a = ap.parse_args()
    dev = "cuda"
    unet = load_unet(a.models / f"share-{GUIDED_SHARE}" / "unet.pt", dev)
    head = load_head(a.checkers / a.head / "head.pt", dev)
    j3 = load(a.judge3, dev)
    sep_head = load_head(a.checkers / "separate" / "head.pt", dev)
    head2 = load_head(a.checkers / a.head2 / "head.pt", dev) if a.head2 else None
    a.out.mkdir(parents=True, exist_ok=True)
    counts, canv = {}, {}
    pairs = [tuple(int(d) for d in q.replace("-", ",").split(",")) for q in a.pairs.replace("_", " ").split()] \
        if a.pairs else PAIRS
    assert all(len(q) == 2 for q in pairs), pairs
    for st in a.settings:
        kind, val = st.split(":")
        for p in pairs:
            if kind == "const":
                fn, eta = pair_push(unet, head, *p, float(val)), 0.0
            elif kind == "sched":
                e, l, k = val.split(",")
                fn, eta = scheduled(pair_push(unet, head, *p, 1.0), float(e), float(l), int(k)), 0.0
            elif kind == "sep":
                e, rest = val.split("@")
                s1, s2 = rest.split("+")
                f1, f2 = pair_push(unet, head, *p, float(s1)), separate_push(unet, sep_head, *p, float(s2))
                fn, eta = (lambda f1, f2: lambda x, t, eps: f1(x, t, eps) + f2(x, t, eps))(f1, f2), float(e)
            elif kind == "ens":
                e, s = val.split("@")
                f1, f2 = pair_push(unet, head, *p, float(s)), pair_push(unet, head2, *p, float(s))
                fn, eta = (lambda f1, f2: lambda x, t, eps: 0.5 * (f1(x, t, eps) + f2(x, t, eps)))(f1, f2), float(e)
            else:
                e, s = val.split("@")
                fn, eta = pair_push(unet, head, *p, float(s)), float(e)
            x = to_uint8(sample(unet, SEEDS, list(p), push_fn=fn, cap=0.1, device=dev, eta=eta)[0])
            key = f"{st}|{p[0]}{p[1]}"
            canv[key] = x
            counts[key] = int(sum(call(r, p, 1.01, 0.9) for r in read_all(j3, x)))
        total = sum(counts[f"{st}|{p[0]}{p[1]}"] for p in pairs)
        counts[f"{st}|total of {64 * len(pairs)}"] = total
        print(st, [counts[f"{st}|{p[0]}{p[1]}"] for p in pairs], "total", total, flush=True)
    np.savez_compressed(a.out / "canvases.npz", **canv)
    (a.out / "counts_v3.json").write_text(json.dumps({"head": a.head, "counts": counts}, indent=1))


if __name__ == "__main__":
    main()
