#!/usr/bin/env python3
"""The overfit rung for Paper 1's two checkers: each must memorise 16 canvases from the expert pool (step M1).

Eight canvases hold a pair apart and eight hold a fused pair. They are read at the sampler's last step, where almost no noise
is left, so a failure here means the network or the loss is broken. A canvas counts as right for the separate checker only
when all ten of its "is this digit there?" answers are right.

Usage: PC_NOCROSS=1 python checker_overfit.py --pool <dir> --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnist"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_checker  # noqa: E402,F401  (keeps cross-range pairs out of the pasted data)
import pixel_checker as pc  # noqa: E402
from canvas_sheet import Canvases  # noqa: E402
from canvas_unet import ABAR, load_mnist  # noqa: E402
from checker import STEPS  # noqa: E402

BAR = 15


def fit(kind, x0, pair, y, present, device, steps):
    head = pc.PixelHead(1 if kind == "pair" else 10, kind == "pair").to(device)
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    t = STEPS[-1].repeat(len(x0)).to(device)
    a = ABAR.to(device)[t][:, None, None, None]
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0)).to(device)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    first = None
    for _ in range(steps):
        out = head(xt, t, pair) if kind == "pair" else head(xt, t)
        loss = F.binary_cross_entropy_with_logits(out[:, 0], y) if kind == "pair" \
            else F.binary_cross_entropy_with_logits(out, present)
        first = loss.item() if first is None else first
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        out = head(xt, t, pair) if kind == "pair" else head(xt, t)
        right = ((out[:, 0] > 0).float() == y) if kind == "pair" else ((out > 0).float() == present).all(1)
    return {"first_loss": round(first, 4), "last_loss": round(loss.item(), 4), "right_of_16": int(right.sum()),
            "passed": int(right.sum()) >= BAR}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    images, labels = load_mnist(args.pool)
    cv = Canvases(images, labels, rng)
    pos, neg = pc.pasted(cv, rng, 8, "two"), pc.pasted(cv, rng, 8, "fused")
    x0 = pc.to_x(np.concatenate([pos[0], neg[0]]), device)
    pair = torch.from_numpy(np.concatenate([pos[1], neg[1]])).long().to(device)
    present = torch.from_numpy(np.concatenate([pos[2], neg[2]])).to(device)
    y = torch.cat([torch.ones(8), torch.zeros(8)]).to(device)
    result = {kind: fit(kind, x0, pair, y, present, device, args.steps) for kind in ("pair", "separate")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    if not all(r["passed"] for r in result.values()):
        raise SystemExit("overfit rung failed")


if __name__ == "__main__":
    main()
