"""Reproduce classifier guidance (Dhariwal and Nichol 2021, arXiv 2105.05233) on the released
256 px unconditional ImageNet model and 256 px noisy classifier, without editing the upstream
repository.

The upstream `classifier_sample.py` needs mpi4py and blobfile through `dist_util`; sampling needs
neither, so this wrapper imports only `script_util` and loads the weights with `torch.load`.
One sample per seed (batch of 1): seed i starts from the same noise at every scale, so a seed's
tiles at scale 0, 1 and 10 are the same draw pushed by different amounts.

Modes
  identity  seeds 0..n-1: plain sampling (no cond_fn) and classifier_sample's path at scale 0,
            plus a second plain run on the first seed as the determinism control. Writes sha256
            of the uint8 render and of the float32 sample for each.
  guide     for each class and scale, seeds 0..n-1; the released classifier's call at t=0 on
            every render; one seed's per-step raw gradient norm and applied push norm.

Outputs: npz to --out (on /datasets), sheets/CSV/JSON to --results (the repo's results folder).
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F

WANT = {207: "golden retriever", 263: "Pembroke Welsh corgi", 281: "tabby cat"}

MODEL_FLAGS = dict(
    attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, image_size=256,
    learn_sigma=True, noise_schedule="linear", num_channels=256, num_head_channels=64,
    num_res_blocks=2, resblock_updown=True, use_fp16=True, use_scale_shift_norm=True,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="the upstream guided-diffusion clone")
    p.add_argument("--weights", required=True, help="folder holding the released .pt files")
    p.add_argument("--out", required=True, help="where npz arrays go (on /datasets)")
    p.add_argument("--results", required=True, help="where sheets, CSV and JSON go")
    p.add_argument("--mode", choices=["identity", "guide"], required=True)
    p.add_argument("--classes", type=int, nargs="*", default=[263])
    p.add_argument("--scales", type=float, nargs="*", default=[1.0, 10.0])
    p.add_argument("--n_seeds", type=int, default=16)
    p.add_argument("--steps", default="250", help="timestep_respacing, as the README's SAMPLE_FLAGS")
    p.add_argument("--log_seed", type=int, default=0, help="seed whose per-step push is logged")
    return p.parse_args()


def class_names():
    try:
        from torchvision.models import ResNet50_Weights
        return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
    except Exception:
        return [str(i) for i in range(1000)]


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    a = parse()
    assert th.cuda.is_available(), "no CUDA device: refusing to sample on the CPU"
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    sys.path.insert(0, a.repo)
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults, classifier_defaults, create_model_and_diffusion,
        create_classifier, args_to_dict,
    )

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    dev = th.device("cuda")
    names = class_names()

    md = model_and_diffusion_defaults()
    md.update(MODEL_FLAGS)
    md["timestep_respacing"] = a.steps
    model, diffusion = create_model_and_diffusion(**args_to_dict(argparse.Namespace(**md), md.keys()))
    model.load_state_dict(th.load(os.path.join(a.weights, "256x256_diffusion_uncond.pt"), map_location="cpu", weights_only=True))
    model.to(dev).convert_to_fp16()
    model.eval()

    cd = classifier_defaults()
    cd["image_size"] = 256
    classifier = create_classifier(**cd)
    classifier.load_state_dict(th.load(os.path.join(a.weights, "256x256_classifier.pt"), map_location="cpu", weights_only=True))
    classifier.to(dev).eval()

    state = {"scale": 0.0, "log": None}

    def cond_fn(x, t, y=None):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logp = F.log_softmax(classifier(x_in, t), dim=-1)
            sel = logp[range(len(logp)), y.view(-1)]
            g = th.autograd.grad(sel.sum(), x_in)[0]
        if state["log"] is not None:
            state["log"]["_g"] = float(g.float().norm())
        return g * state["scale"]  # exactly classifier_sample.py's cond_fn

    def model_fn(x, t, y=None):
        return model(x, t, None)  # unconditional model, as classifier_sample.py with class_cond False

    upstream_condition_mean = diffusion.condition_mean

    def logged_condition_mean(cfn, p_mean_var, x, t, model_kwargs=None):
        new = upstream_condition_mean(cfn, p_mean_var, x, t, model_kwargs=model_kwargs)
        if state["log"] is not None:
            state["log"]["rows"].append({
                "t": int(t[0]), "raw_grad_norm": state["log"].pop("_g"),
                "applied_push_norm": float((new - p_mean_var["mean"].float()).norm()),
                "mean_norm": float(p_mean_var["mean"].float().norm()),
                "variance_mean": float(p_mean_var["variance"].float().mean()),
            })
        return new

    def sample(seed, cls, guided):
        th.manual_seed(seed)
        y = th.full((1,), cls, device=dev, dtype=th.long)
        t0 = time.time()
        x = diffusion.p_sample_loop(model_fn, (1, 3, 256, 256), clip_denoised=True,
                                    model_kwargs={"y": y}, cond_fn=cond_fn if guided else None,
                                    device=dev)
        u8 = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous()
        return x.float().cpu().numpy(), u8.cpu().numpy()[0], time.time() - t0

    @th.no_grad()
    def call(u8):
        x = th.from_numpy(u8).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
        p = F.softmax(classifier(x, th.zeros(1, dtype=th.long, device=dev)).float(), -1)[0]
        top = th.topk(p, 5)
        return [(int(i), float(v)) for v, i in zip(top.values, top.indices)], p

    meta = {"repo_commit": os.popen(f"git -C {a.repo} rev-parse HEAD").read().strip(),
            "steps": a.steps, "device": th.cuda.get_device_name(0), "torch": th.__version__,
            "model_flags": MODEL_FLAGS, "classifier": cd}

    if a.mode == "identity":
        rows = []
        for s in range(a.n_seeds):
            runs = [("plain", False), ("scale0", True)] + ([("plain_again", False)] if s == 0 else [])
            for tag, guided in runs:
                state["scale"] = 0.0
                f32, u8, dt = sample(s, 263, guided)
                np.savez(os.path.join(a.out, f"identity_seed{s}_{tag}.npz"), f32=f32, u8=u8)
                rows.append({"seed": s, "run": tag, "sha256_uint8": sha(u8), "sha256_float32": sha(f32), "seconds": round(dt, 1)})
                if tag != "plain_again":
                    from PIL import Image
                    Image.fromarray(u8).save(os.path.join(a.results, f"identity_seed{s}_{tag}.png"))
                print(rows[-1], flush=True)
        with open(os.path.join(a.results, "identity_checksums.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        json.dump({**meta, "rows": rows}, open(os.path.join(a.results, "identity_checksums.json"), "w"), indent=1)
        return

    diffusion.condition_mean = logged_condition_mean  # instance attribute; upstream file untouched
    calls = []
    for cls in a.classes:
        for scale in a.scales:
            state["scale"] = scale
            for s in range(a.n_seeds):
                state["log"] = {"rows": []} if s == a.log_seed else None
                f32, u8, dt = sample(s, cls, True)
                top, p = call(u8)
                np.savez(os.path.join(a.out, f"guide_c{cls}_s{scale:g}_seed{s}.npz"), f32=f32, u8=u8)
                from PIL import Image
                Image.fromarray(u8).save(os.path.join(a.results, f"c{cls}_s{scale:g}_seed{s}.png"))
                calls.append({"class_asked": cls, "class_name": WANT.get(cls, names[cls]), "scale": scale, "seed": s,
                              "top1": top[0][0], "top1_name": names[top[0][0]], "top1_p": round(top[0][1], 4),
                              "p_asked": round(float(p[cls]), 4), "hit": int(top[0][0] == cls), "seconds": round(dt, 1)})
                print(calls[-1], flush=True)
                if state["log"] is not None:
                    json.dump({**meta, "class": cls, "scale": scale, "seed": s, "per_step": state["log"]["rows"]},
                              open(os.path.join(a.results, f"push_per_step_c{cls}_s{scale:g}_seed{s}.json"), "w"), indent=1)
                    state["log"] = None
    path = os.path.join(a.results, f"classifier_calls_{'_'.join(map(str, a.classes))}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(calls[0]))
        w.writeheader(); w.writerows(calls)
    json.dump(meta, open(os.path.join(a.results, f"run_meta_{'_'.join(map(str, a.classes))}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
