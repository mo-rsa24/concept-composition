"""Plan 20, piece 4: the released 256 px unconditional model on seeds 1 to 16, three columns.

  product        cond_fn = scale * grad(log sig(cat) + log sig(dog)), the separate-label classifier's
                 two pushes together (the product, since the model itself is unconditional)
  + pair w=1     the product plus w * grad(clamp(logit, -11.5, 11.5)) from the pair-aware classifier,
  + pair w=2     the log-odds clipped as Kim et al., so past the clip its push is 0

Seed i draws its starting noise from torch.Generator seeded i; the per-step noise comes from the
global generator seeded 0 before each column, and cond_fn draws no random numbers, so tile i in every
column is the same draw pushed differently. 250 respaced steps, as the README and as the product's
draws the pair-aware classifier was trained against.

Writes one PNG per tile at full resolution, a sheet, a calls CSV with an empty `call` column, and a
per-step log of seed 1: the separate-label push's norm, the pair-aware push's norm, and the product's
own mean step, each as applied to the mean.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import UNCOND_256, W, released_classifier  # noqa: E402
from guided_diffusion.script_util import args_to_dict, create_model_and_diffusion, model_and_diffusion_defaults  # noqa: E402

CLIP = 11.5
COLUMNS = [("product", 0.0), ("pair_w1", 1.0), ("pair_w2", 2.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--separate", required=True); p.add_argument("--pair", required=True)
    p.add_argument("--out", required=True, help="tiles, sheet, CSV and logs go here")
    p.add_argument("--scale", type=float, default=10.0); p.add_argument("--steps", default="250")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    a = p.parse_args()
    assert th.cuda.is_available()
    os.makedirs(f"{a.out}/tiles", exist_ok=True)

    md = model_and_diffusion_defaults(); md.update(UNCOND_256); md["timestep_respacing"] = a.steps
    model, diff = create_model_and_diffusion(**args_to_dict(argparse.Namespace(**md), md.keys()))
    model.load_state_dict(th.load(f"{W}/256x256_diffusion_uncond.pt", map_location="cpu", weights_only=True))
    model.cuda().convert_to_fp16(); model.eval()
    sep, _ = released_classifier(256, 2); sep.load_state_dict(th.load(a.separate, map_location="cpu", weights_only=True)); sep.cuda().eval()
    pair, _ = released_classifier(256, 1); pair.load_state_dict(th.load(a.pair, map_location="cpu", weights_only=True)); pair.cuda().eval()

    st = {"w": 0.0, "log": None}

    def cond_fn(x, t, y=None):
        with th.enable_grad():
            xi = x.detach().requires_grad_(True)
            g_sep = th.autograd.grad(F.logsigmoid(sep(xi, t)).sum(), xi)[0] * a.scale
            g_pair = th.zeros_like(g_sep)
            if st["w"]:
                xj = x.detach().requires_grad_(True)
                g_pair = th.autograd.grad(pair(xj, t)[:, 0].clamp(-CLIP, CLIP).sum(), xj)[0] * st["w"]
        if st["log"] is not None:
            st["log"]["_g"] = (g_sep, g_pair)
        return g_sep + g_pair

    up = diff.condition_mean

    def logged(cfn, pmv, x, t, model_kwargs=None):
        new = up(cfn, pmv, x, t, model_kwargs=model_kwargs)
        if st["log"] is not None:
            g_sep, g_pair = st["log"].pop("_g"); var = pmv["variance"].float()
            st["log"]["rows"].append({"t": int(t[0]), "separate_push": float((var * g_sep.float()).norm()),
                                      "pair_push": float((var * g_pair.float()).norm()),
                                      "product_step": float((pmv["mean"].float() - x.float()).norm())})
        return new

    diff.condition_mean = logged
    rows, logs, t0 = [], {}, time.time()
    for col, w in COLUMNS:
        st["w"] = w
        for s in a.seeds:
            g = th.Generator(device="cuda").manual_seed(s)
            noise = th.randn((1, 3, 256, 256), generator=g, device="cuda")
            th.manual_seed(0)
            st["log"] = {"rows": []} if s == a.seeds[0] else None
            x = diff.p_sample_loop(lambda x, t, y=None: model(x, t, None), (1, 3, 256, 256), noise=noise,
                                   clip_denoised=True, model_kwargs={}, cond_fn=cond_fn, device="cuda")
            u8 = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)[0].permute(1, 2, 0).cpu().numpy()
            name = f"{col}_seed{s:02d}.png"
            Image.fromarray(u8).save(f"{a.out}/tiles/{name}")
            with th.no_grad():
                tz = th.zeros(1, dtype=th.long, device="cuda")
                pc, pd = th.sigmoid(sep(x, tz))[0].tolist(); lo = float(pair(x, tz)[0, 0])
            rows.append({"seed": s, "column": col, "tile": name, "sep_p_cat": round(pc, 3), "sep_p_dog": round(pd, 3),
                         "pair_logit": round(lo, 2), "call": ""})
            if st["log"] is not None:
                logs[col] = st["log"]["rows"]
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)

    with open(f"{a.out}/calls.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump({"scale": a.scale, "steps": a.steps, "clip": CLIP, "seed_logged": a.seeds[0], "per_step": logs},
              open(f"{a.out}/push_per_step.json", "w"))

    tile, pad, cap = 160, 6, 18
    sheet = Image.new("RGB", (pad + len(COLUMNS) * (tile + pad) + 60, cap + pad + len(a.seeds) * (tile + pad)), "white")
    d = ImageDraw.Draw(sheet)
    for j, (col, _) in enumerate(COLUMNS):
        d.text((60 + pad + j * (tile + pad), 4), col, fill="black")
        for i, s in enumerate(a.seeds):
            if j == 0:
                d.text((6, cap + pad + i * (tile + pad) + tile // 2), f"seed {s}", fill="black")
            im = Image.open(f"{a.out}/tiles/{col}_seed{s:02d}.png").resize((tile, tile))
            sheet.paste(im, (60 + pad + j * (tile + pad), cap + pad + i * (tile + pad)))
    sheet.save(f"{a.out}/sheet-released-256.png")
    print("DONE", len(rows), "tiles", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
