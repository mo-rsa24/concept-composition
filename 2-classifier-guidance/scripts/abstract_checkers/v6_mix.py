"""Question 95's checks before training on pasted canvases and sampler states together
(abstract-checkers/10-reading-identity/design.md, "Training on pasted canvases and sampler states together").

Stages:
    clean-check  the digit judge on the one-step clean estimate at sampler steps 27 to 36 of the saved digits 0 to 4
                 states (resample/train-states.npz); a diagnostic only, never the checker
    settle       board question 97, rerun 1: at which sampling step the sampler has settled which digits it draws.
                 V3's plain and pushed rows on digits 5 to 9 at seeds 1000 to 1031, the judge on the clean estimate
                 at all 50 steps, compared with the judge on the finished picture

Usage:
    python v6_mix.py clean-check --model <unet.pt> --judge <judge.pt> --states <train-states.npz> --out <folder>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist")]
import two_experts as te  # noqa: E402
import v6_baselines as vb  # noqa: E402
from canvas_unet import ABAR, ddim_timesteps, product_eps, sample  # noqa: E402
import v3_compose as v3  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402
from v3_compose import CLEAN, to_uint8  # noqa: E402

# Fixed before the run (design, "Training on pasted canvases and sampler states together", check 1).
MIN_CLEAN_BA = 0.75
SETTLE_SEEDS = list(range(1000, 1032))   # V3's final seeds
SETTLE_SHARE = 0.9                        # the step where 90% of canvases read the digits they end with
STEPS = vb.IDENTITY_STEPS
CHUNK = 256


@torch.no_grad()
def clean_check(model, net, data, dev):
    n = len(data["y"])
    pred = np.zeros((n, len(STEPS)), bool)
    calls = [dict() for _ in STEPS]
    abar = ABAR.to(dev)
    for j, t in enumerate(STEPS):
        for (a, bb) in sorted({tuple(p) for p in data["pair"].tolist()}):
            idx = np.flatnonzero((data["pair"][:, 0] == a) & (data["pair"][:, 1] == bb))
            for i in range(0, len(idx), CHUNK):
                k = idx[i:i + CHUNK]
                x = torch.from_numpy(data["x"][k, j]).float().to(dev)
                tt = torch.full((len(k),), t, device=dev, dtype=torch.long)
                eps = product_eps(model, x, tt, [a, bb], g=1.0)
                ab = abar[t]
                x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                canv = to_uint8(x0)
                for m, c in zip(k, canv):
                    call = te.call_two(te.reads(net, c, dev), (a, bb))
                    calls[j][call] = calls[j].get(call, 0) + 1
                    pred[m, j] = call == CLEAN
    y = data["y"]
    per_step = [vb.balanced(pred[:, j], y) for j in range(len(STEPS))]
    mean = float(np.mean(per_step))
    return {"canvases": int(n), "label_1": int(y.sum()), "label_0": int((y == 0).sum()), "steps": STEPS,
            "per_step": per_step, "mean_ba": mean,
            "called_asked_given_1": [float(pred[y == 1, j].mean()) for j in range(len(STEPS))],
            "called_asked_given_0": [float(pred[y == 0, j].mean()) for j in range(len(STEPS))],
            "calls_per_step": calls,
            "verdict": "identity visible" if mean >= MIN_CLEAN_BA else "identity not plainly visible: option B"}


def digit_set(net, canvas, dev):
    """The judge's most likely digit for each blob, as a sorted tuple."""
    return tuple(sorted(int((p[:10] / p[:10].sum()).argmax()) for _, p in te.reads(net, canvas, dev)))


def settle(model, chead, net, dev):
    ts = [int(t) for t in ddim_timesteps(50)]
    abar = ABAR.to(dev)
    out = {}
    for row in ("plain", "push"):
        match = np.zeros((0, len(ts)), bool)
        first = []
        for a, bb in v3.UNSEEN:
            x0s = {}
            inner = (v3.make_push("push", 0.4, chead, model, a, bb, 4242 + 10 * a + bb) if row == "push"
                     else (lambda x, t, eps: torch.zeros_like(eps)))

            def record(x, t, eps):
                ab = abar[t[0]]
                x0s[int(t[0])] = to_uint8((x - (1 - ab).sqrt() * eps) / ab.sqrt())
                return inner(x, t, eps)
            x, _ = sample(model, SETTLE_SEEDS, [a, bb], g=1.0, steps=50, cap=1.0, push_fn=record)
            final = to_uint8(x)
            m = np.zeros((len(SETTLE_SEEDS), len(ts)), bool)
            for k in range(len(SETTLE_SEEDS)):
                end = digit_set(net, final[k], dev)
                for j, t in enumerate(ts):
                    m[k, j] = digit_set(net, x0s[t][k], dev) == end
                stay = [j for j in range(len(ts)) if m[k, j:].all()]
                first.append(stay[0] if stay else len(ts))
            match = np.concatenate([match, m])
        share = match.mean(0)
        first = np.array(first)
        reach = next((j for j in range(len(ts)) if share[j] >= SETTLE_SHARE), None)
        out[row] = {"canvases": int(len(first)), "share_matching_final_per_step": share.round(4).tolist(),
                    "first_step_share_at_least_0.9": reach,
                    "settled_from_step_median": float(np.median(first)),
                    "settled_by_step_15": float((first <= 15).mean()), "settled_by_step_26": float((first <= 26).mean()),
                    "settled_by_step_36": float((first <= 36).mean())}
        print(json.dumps({row: {k: v for k, v in out[row].items() if k != "share_matching_final_per_step"}}), flush=True)
    return {"steps": ts, "seeds": [SETTLE_SEEDS[0], SETTLE_SEEDS[-1]], "rows": out,
            "note": "a canvas is settled from the first step after which the judge's digits on the clean estimate always match its finished picture"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["clean-check", "settle"])
    for flag in ("model", "judge", "out"):
        ap.add_argument(f"--{flag}", type=Path, required=True)
    ap.add_argument("--states", type=Path)
    ap.add_argument("--count-head", dest="count_head", type=Path)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model = load_unet(args.model, dev)
    net = load_judge(args.judge, dev)
    if args.stage == "settle":
        chead, _ = v3.load_head(args.count_head, dev)
        r = settle(model, chead, net, dev)
        r["seconds"] = round(time.time() - start, 1)
        (args.out / "settle.json").write_text(json.dumps(r, indent=1))
        return 0
    with np.load(args.states) as z:
        data = {k: z[k] for k in ("x", "y", "pair")}
    print(json.dumps({"stage": args.stage, "states": str(args.states), "canvases": len(data["y"]),
                      "device": torch.cuda.get_device_name()}), flush=True)
    r = clean_check(model, net, data, dev)
    r["seconds"] = round(time.time() - start, 1)
    (args.out / "clean-check.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "calls_per_step"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
