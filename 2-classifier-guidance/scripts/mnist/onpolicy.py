"""Loop 2 round 9 (plan 24): add the push's own renders to the checker's training canvases.

Renders a named checker's pair-aware push on the checker's own training pairs (the seen pairs minus
the three tuning pairs and 7 & 8 is kept), 12 seeds each from seed 2,000 up (never 0 to 63, never a
held-out or tuning pair), labels each canvas with the third judge, and writes a new draws file: the
old draws plus these, labelled "two clear, right digits" or "not". The next checker trains on it.

Usage: python onpolicy.py --models <dir> --checkers <dir> --judge3 <pt> --head <name> --s <strength>
                          --draws <in.npz> --out <out.npz> [--seed0 2000]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canvas_unet import SEEN, sample, to_uint8
from checker import GUIDED_SHARE, load_unet, pair_push
from five_versions import load_head
from judge import load
from judge_tune import read_all, call

TUNING = [(0, 6), (1, 2), (6, 9)]


def main():
    ap = argparse.ArgumentParser()
    for a in ("--models", "--checkers", "--judge3", "--draws", "--out"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--per-pair", type=int, default=12, help="pushed pictures per training pair")
    a = ap.parse_args()
    assert a.seed0 >= 1000
    dev = "cuda"
    unet = load_unet(a.models / f"share-{GUIDED_SHARE}" / "unet.pt", dev)
    head = load_head(a.checkers / a.head / "head.pt", dev)
    j3 = load(a.judge3, dev)
    d = dict(np.load(a.draws))
    xs, ps, calls = [], [], []
    for p in [q for q in SEEN if q not in TUNING]:
        seeds = list(range(a.seed0, a.seed0 + a.per_pair))
        x = to_uint8(sample(unet, seeds, list(p), push_fn=pair_push(unet, head, *p, a.s), cap=0.1, device=dev)[0])
        for r, c in zip(read_all(j3, x), x):
            xs.append(c); ps.append(p)
            calls.append("two clear, right digits" if call(r, p, 1.01, 0.9) else "not")
    n = len(xs)
    d["x"] = np.concatenate([d["x"], np.stack(xs)])
    d["pair"] = np.concatenate([d["pair"], np.array(ps)])
    d["call"] = np.concatenate([d["call"], np.array(calls)])
    d["held"] = np.concatenate([d["held"], np.zeros(n, bool)])
    d["share"] = np.concatenate([d["share"], np.full(n, -1.0)])      # -1 marks a pushed render
    d.pop("call_v1", None)
    np.savez_compressed(a.out, **d)
    pos = sum(c == "two clear, right digits" for c in calls)
    print(json.dumps({"pushed_renders": n, "judged_two_clear_right": pos, "head": a.head, "s": a.s}), flush=True)


if __name__ == "__main__":
    main()
