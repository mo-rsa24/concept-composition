"""Question 95's checks before training on pasted canvases and sampler states together
(abstract-checkers/10-reading-identity/design.md, "Training on pasted canvases and sampler states together").

Stages:
    clean-check  the digit judge on the one-step clean estimate at sampler steps 27 to 36 of the saved digits 0 to 4
                 states (resample/train-states.npz); a diagnostic only, never the checker

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
from canvas_unet import ABAR, product_eps  # noqa: E402
from checker import load_unet  # noqa: E402
from judge import load as load_judge  # noqa: E402
from v3_compose import CLEAN, to_uint8  # noqa: E402

# Fixed before the run (design, "Training on pasted canvases and sampler states together", check 1).
MIN_CLEAN_BA = 0.75
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["clean-check"])
    for flag in ("model", "judge", "states", "out"):
        ap.add_argument(f"--{flag}", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model = load_unet(args.model, dev)
    net = load_judge(args.judge, dev)
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
