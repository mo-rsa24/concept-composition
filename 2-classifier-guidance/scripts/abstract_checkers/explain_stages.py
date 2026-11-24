"""Every in-between output of the saved count checker on the zoom ladder's example canvases, for the miniature page.

For each example canvas and timestep: the noisy canvas, the frozen model's three noise predictions, the two maps, the
output of every layer of the checker (each convolution and SiLU, the timestep's scale and shift, the pooled numbers),
the logit and the push gradient. The forward pass is recomputed layer by layer and checked against the head's own output.

Usage:
    python explain_stages.py --model <unet.pt> --head <count-head-map.pt> --examples <folder> --out <npz>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mnist")]
from canvas_unet import ABAR, NULL  # noqa: E402
from checker import load_unet, time_features  # noqa: E402
from v2_learn import EXAMPLES, EXPLAIN_TS, PIXELS  # noqa: E402
from v3_compose import load_head  # noqa: E402


def layers(head, m, t):
    """PixelHead's forward pass, step by step (pixel_checker.py lines 47 to 56)."""
    out, s = {}, m
    for i, layer in enumerate(head.stem):
        s = layer(s)
        out[f"stem{i}_{type(layer).__name__.lower()}"] = s
    out["c1_conv"] = head.c1(s)
    h = F.silu(out["c1_conv"])
    out["c1_silu"] = h
    cond = time_features(t)
    gamma, beta = head.film(cond).chunk(2, 1)
    out["film_gamma"], out["film_beta"], out["time_features"] = gamma, beta, cond
    h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
    out["film_out"] = h
    out["c2_conv"] = head.c2(h)
    h = F.silu(out["c2_conv"])
    out["c2_silu"] = h
    out["pooled"] = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
    out["logit"] = head.mlp(torch.cat([out["pooled"], cond], 1))[:, 0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    model = load_unet(args.model, dev)
    head, ck = load_head(args.head, dev)
    noise = torch.from_numpy(np.load(args.examples / "display-noise.npy")).float().to(dev)[None, None]
    store, worst = {}, 0.0
    for name in EXAMPLES:
        a, b = (int(v) for v in name.split("_")[-2:])
        x0 = torch.from_numpy(np.load(args.examples / f"{name}-clean.npy")).float().to(dev)[None, None]
        pair = torch.tensor([[a, b]], device=dev)
        for step in EXPLAIN_TS:
            t = torch.tensor([step], device=dev)
            ab = ABAR.to(dev)[t][:, None, None, None]
            with torch.enable_grad():
                xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * noise).requires_grad_(True)
                y = torch.cat([torch.full((1,), NULL, device=dev), pair[:, 0], pair[:, 1]])
                e0, ea, eb = model(xt.repeat(3, 1, 1, 1), t.repeat(3), y).chunk(3)
                maps = []
                for d in (ea - e0, eb - e0):
                    maps.append(d.abs() / d.abs().flatten(1).sum(1).clamp_min(1e-12)[:, None, None, None] * PIXELS)
                m = torch.cat(maps, 1)
                acts = layers(head, m, t)
                g, = torch.autograd.grad(acts["logit"].sum(), xt)
            direct = head(m, t, pair)[:, 0]
            worst = max(worst, float((direct - acts["logit"]).abs().max()))
            key = f"{name}_t{step}"
            row = {"x0": x0, "x_t": xt, "eps_null": e0, "eps_a": ea, "eps_b": eb, "maps": m, "grad": g, **acts}
            for k, v in row.items():
                store[f"{key}__{k}"] = v.detach().float().cpu().numpy()[0].astype(np.float16 if v.dim() == 4 else np.float32)
            print(json.dumps({"example": name, "t": step, "logit": float(acts["logit"]),
                              "probability": float(torch.sigmoid(acts["logit"])), "grad_norm": float(g.norm())}), flush=True)
    if worst > 1e-4:
        raise SystemExit(f"the layer-by-layer forward pass differs from the head's own by {worst}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(EXAMPLES), timesteps=np.array(EXPLAIN_TS), **store)
    print(json.dumps({"saved": str(args.out), "arrays": len(store), "largest_forward_gap": worst,
                      "head_window": ck.get("window")}), flush=True)


if __name__ == "__main__":
    main()
